import os
import time
import cv2
import logging
from typing import List, Dict, Optional
from videocuts.config import Config
from videocuts.utils.system import cache_is_fresh
from videocuts.utils.font import verify_font_available, print_font_installation_guide, get_font_path_with_fallback
from videocuts.audio.transcription import transcribe_audio, parse_srt_to_segments
from videocuts.audio.analysis import extract_audio_normalized
from videocuts.audio.diarization import run_diarization
from videocuts.models.transcript import Transcript
import json

# Epic 3 - Candidate Discovery Pipeline
from videocuts.candidates.timeline import build_timeline, save_timeline
from videocuts.candidates.generator import generate_candidates, save_candidates_raw
from videocuts.candidates.features import compute_all_features, save_candidates_features
from videocuts.candidates.scorer import score_all_candidates, save_candidates_scored
from videocuts.candidates.llm_reranker import rerank_with_llm, save_llm_results
from videocuts.candidates.selection import select_final_clips, save_clips_selected
from videocuts.candidates.refinement import refine_clip_boundaries, save_clips_refined

from videocuts.video.tracking import analyze_clip_faces, analyze_best_layout, determine_layout_segments, crop_x_expression_for_segments
from videocuts.caption.generators import write_clip_ass
from videocuts.caption.hook import detect_hook_phrase
from videocuts.video.processor import create_vertical_with_subs, create_multi_layout_clip

def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # App logger
    app_logger = logging.getLogger("videocuts")
    app_logger.setLevel(level)
    app_logger.propagate = False
    
    # Clean up existing handlers to avoid duplication (for long-running processes)
    if app_logger.hasHandlers():
        app_logger.handlers.clear()
        
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    app_logger.addHandler(ch)
    
    # Root logger (suppress others)
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

def run_pipeline(cfg: Config):
    """Run the full video processing pipeline with provided configuration."""
    input_video = cfg.paths.input_video
    srt_file = cfg.paths.srt_file
    audio_wav = cfg.paths.audio_16k_mono
    audio_metadata_path = cfg.paths.audio_metadata
    transcript_json_path = cfg.paths.transcript_json
    output_dir = cfg.paths.output_dir
    
    # 0. Setup Logging
    setup_logging(cfg.debug)
    logger = logging.getLogger(__name__)
    
    # 1. Setup & Project directory
    os.makedirs(cfg.paths.project_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    video_mtime = os.path.getmtime(input_video)
    
    logger.info("Checking font availability...")
    if not verify_font_available(cfg.caption.font_name, cfg.caption.font_path):
        logger.warning(f"Caption font '{cfg.caption.font_name}' - NOT FOUND (fallback will be used)")
        print_font_installation_guide(cfg.caption.font_name)
    
    # 2. Audio Extraction (before transcription for proper input format)
    if not cfg.force_audio_extraction and cache_is_fresh(audio_wav, video_mtime):
        logger.info(f"Reusing cached audio '{audio_wav}'.")
    else:
        metadata = extract_audio_normalized(input_video, audio_wav)
        # Save metadata for reproducibility
        with open(audio_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Audio metadata saved: {metadata}")
    
    # 3. Transcription (uses normalized audio, outputs JSON)
    if not cfg.force_transcribe and cache_is_fresh(transcript_json_path, video_mtime):
        logger.info(f"Reusing cached transcript '{transcript_json_path}'.")
        transcript = Transcript.load(transcript_json_path)
        segments = transcript.to_legacy_segments()
        detected_language = transcript.language
    else:
        logger.info(f"Transcribing with provider='{cfg.transcription.provider}'...")
        transcript = transcribe_audio(
            audio_wav,
            cfg.transcription,
            output_json_path=transcript_json_path
        )
        segments = transcript.to_legacy_segments()
        detected_language = transcript.language
        logger.info(f"Detected language: '{detected_language}'")
    
    # ==========================================================================
    # Epic 3 - Clip Candidate Discovery Pipeline
    # ==========================================================================
    
    # 4. Diarization (speaker identification)
    diarization_path = cfg.paths.diarization_json
    if cfg.diarization.enabled:
        if not cache_is_fresh(diarization_path, video_mtime):
            logger.info("Running speaker diarization...")
            diarization_result = run_diarization(audio_wav, cfg.diarization, diarization_path)
            logger.info(f"Diarization complete: {diarization_result.num_speakers} speakers")
        else:
            logger.info(f"Reusing cached diarization '{diarization_path}'")
    else:
        diarization_path = None
        logger.info("Diarization disabled")
    
    # 5. Timeline Building (enriched segments with audio features)
    timeline_path = cfg.paths.timeline_json
    if not cache_is_fresh(timeline_path, video_mtime):
        logger.info("Building analysis-ready timeline...")
        timeline = build_timeline(
            transcript_json_path, 
            audio_wav, 
            diarization_path,
            cfg
        )
        save_timeline(timeline, timeline_path)
    else:
        logger.info(f"Reusing cached timeline '{timeline_path}'")
        from videocuts.candidates.models import Timeline
        timeline = Timeline.load(timeline_path)
    
    # 6. Candidate Generation (sliding window with snapping)
    logger.info("Generating clip candidates...")
    raw_candidates = generate_candidates(timeline, cfg)
    save_candidates_raw(raw_candidates, cfg.paths.candidates_raw_json)
    
    if not raw_candidates:
        logger.error("No candidates generated. Video may be too short.")
        return []
    
    # 7. Feature Computation (hook/middle/end structure)
    logger.info("Computing candidate features...")
    featured_candidates = compute_all_features(raw_candidates, timeline, cfg)
    save_candidates_features(featured_candidates, cfg.paths.candidates_features_json)
    
    # 8. Heuristic Scoring (deterministic ranking)
    logger.info("Scoring candidates with heuristics...")
    scored_candidates = score_all_candidates(featured_candidates, cfg)
    save_candidates_scored(scored_candidates, cfg.paths.candidates_scored_json)
    
    eligible_count = sum(1 for c in scored_candidates if c.eligible)
    if eligible_count == 0:
        logger.error("No eligible candidates after filtering.")
        return []
    
    logger.info(f"Scored {len(scored_candidates)} candidates, {eligible_count} eligible")
    
    # 9. Optional LLM Reranking (for top candidates)
    llm_results = None
    if cfg.llm.enabled:
        logger.info("Running LLM reranking on top candidates...")
        llm_results = rerank_with_llm(scored_candidates, timeline, cfg)
        if llm_results:
            save_llm_results(llm_results, cfg.paths.candidates_llm_json)
            logger.info(f"LLM reranked {len(llm_results)} candidates")
    
    # 10. Final Selection (with diversity enforcement)
    logger.info("Selecting final clips...")
    selected_clips = select_final_clips(scored_candidates, llm_results, cfg)
    save_clips_selected(selected_clips, cfg.paths.clips_selected_json)
    
    if not selected_clips:
        logger.error("No clips selected for rendering.")
        return []
    
    # 11. Boundary Refinement (clean starts/ends)
    logger.info("Refining clip boundaries...")
    refined_clips = refine_clip_boundaries(selected_clips, timeline, cfg)
    save_clips_refined(refined_clips, cfg.paths.clips_refined_json)
    
    logger.info(f"Selected {len(refined_clips)} clips for rendering")
    
    # Log selected clips
    if cfg.debug:
        logger.debug("--- Selected Clips ---")
        for clip in refined_clips:
            logger.debug(f"  {clip.candidate_id}: {clip.start_s_after:.1f}s - {clip.end_s_after:.1f}s")
            logger.debug(f"    Title: {clip.title}")
            logger.debug(f"    Score: {clip.final_score:.1f}")
            if clip.changes:
                logger.debug(f"    Changes: {', '.join(clip.changes)}")
        logger.debug("----------------------")
    
    # Limit CPU usage if requested
    if cfg.cpu_limit > 0:
        import torch
        torch.set_num_threads(cfg.cpu_limit)
        os.environ["OMP_NUM_THREADS"] = str(cfg.cpu_limit)
        cv2.setNumThreads(1)
        logger.info(f"Restricted CPU usage to {cfg.cpu_limit} threads")

    generated_clips = []

    # 12. Clip Rendering
    for idx, clip in enumerate(refined_clips):
        start = clip.start_s_after
        end = clip.end_s_after
        duration = end - start
        if duration <= 0:
            continue
        
        # Get text for this clip
        clip_segments = [s for s in segments if s["start"] >= start and s["end"] <= end]
        clip_full_text = " ".join([s["text"].strip() for s in clip_segments])
        
        output_name = f"clip_{idx + 1}.mp4"
        output_path = os.path.join(output_dir, output_name)
        ass_path = os.path.join(output_dir, f"clip_{idx + 1}_subs.ass")
        
        write_clip_ass(segments, start, end, ass_path, cfg)
        
        # Face Analysis & Layout
        face_analysis = analyze_clip_faces(input_video, start, end, cfg)
        spk_segs = face_analysis["speaker_segments"]
        
        layout_segments = []
        layout_analysis = analyze_best_layout(
            face_analysis["multi_samples"], 
            face_analysis["track_detections"], 
            start, end, cfg
        )
        recommended = layout_analysis["recommended_layout"]
        effective_layout = recommended if cfg.layout.mode == "auto" else cfg.layout.mode
        
        if effective_layout != "single":
            layout_segments = determine_layout_segments(
                face_analysis["multi_samples"], 
                face_analysis["track_detections"], 
                start, end, cfg
            )
        
        # Hook Text (from LLM or fallback)
        hook_text = clip.hook_line or clip.title
        if not hook_text and cfg.hook.enabled:
            hook_info = detect_hook_phrase(segments, start, end, cfg, keywords=[])
            if hook_info:
                hook_text = hook_info["text"]
        
        # Create Clip
        if effective_layout != "single" and layout_segments:
            create_multi_layout_clip(
                input_video, ass_path, output_path, start, duration,
                layout_segments, spk_segs, start, end, cfg, hook_text
            )
        else:
            crop_expr = crop_x_expression_for_segments(spk_segs, start, end, cfg.video.target_width)
            create_vertical_with_subs(
                input_video, ass_path, output_path, start, duration,
                cfg, crop_expr, hook_text
            )
        
        logger.info(f"Generated {output_name}: {clip.title}")
        
        generated_clips.append({
            "index": idx + 1,
            "path": output_path,
            "title": clip.title,
            "description": clip.hook_line or "",
            "start": start,
            "end": end,
            "viral_score": clip.final_score,
            "hook_text": hook_text,
            "transcript": clip_full_text,
            "clip_full_transcript": clip_full_text,
            "candidate_id": clip.candidate_id,
        })

    return generated_clips
