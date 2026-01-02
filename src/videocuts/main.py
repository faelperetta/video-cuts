import os
import time
import cv2
import logging
from typing import List, Dict, Optional
from videocuts.config import Config, get_niche_config
from videocuts.utils.system import cache_is_fresh
from videocuts.utils.font import verify_font_available, print_font_installation_guide, get_font_path_with_fallback
from videocuts.audio.transcription import transcribe_video, parse_srt_to_segments
from videocuts.audio.analysis import extract_audio, compute_rms_per_segment, load_sentiment_model
from videocuts.highlights.selector import score_segments_for_highlights, select_highlight_intervals
from videocuts.llm.selector import detect_llm_availability, select_highlight_intervals_llm
from videocuts.video.tracking import analyze_clip_faces, analyze_best_layout, determine_layout_segments, crop_x_expression_for_segments
from videocuts.caption.generators import write_clip_ass
from videocuts.caption.hook import detect_hook_phrase
from videocuts.video.processor import create_vertical_with_subs, create_multi_layout_clip

def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.WARNING) # Suppress debug from other libs
    
    # App logger
    app_logger = logging.getLogger("videocuts")
    app_logger.setLevel(level)
    app_logger.addHandler(ch)
    app_logger.propagate = False

def run_pipeline(cfg: Config):
    """Run the full video processing pipeline with provided configuration."""
    input_video = cfg.paths.input_video
    srt_file = cfg.paths.srt_file
    audio_wav = cfg.paths.audio_wav
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
    
    # 2. Transcription
    if not cfg.force_transcribe and cache_is_fresh(srt_file, video_mtime):
        logger.info(f"Reusing cached transcription '{srt_file}'.")
        segments = parse_srt_to_segments(srt_file)
        detected_language = "unknown"
    else:
        segments, detected_language = transcribe_video(input_video, srt_file, model_size=cfg.model.whisper_size, language=cfg.model.whisper_language)
    
    # 3. Audio Extraction
    if not cfg.force_audio_extraction and cache_is_fresh(audio_wav, video_mtime):
        logger.info(f"Reusing cached audio '{audio_wav}'.")
    else:
        extract_audio(input_video, audio_wav)
        
    # 4. Highlight Selection
    if cfg.llm.enabled and detect_llm_availability():
        logger.info("Using LLM for clip identification...")
        highlight_intervals = select_highlight_intervals_llm(segments, cfg.llm.prompt_template_path, cfg)
    else:
        logger.info("Using local scoring for clip identification...")
        rms_values = compute_rms_per_segment(audio_wav, segments)
        tokenizer, sentiment_model = load_sentiment_model(cfg.model.sentiment_model)
        niche_cfg = get_niche_config(cfg.content_type)
        scored_segments = score_segments_for_highlights(segments, rms_values, tokenizer, sentiment_model, niche_cfg["HOOK_KEYWORDS"], cfg)
        highlight_intervals = select_highlight_intervals(scored_segments, segments, cfg, segments[-1]["end"])
    
    if cfg.debug:
        logger.debug("--- Highlight Clips Details ---")
        for i, clip in enumerate(highlight_intervals):
            title = clip.get("hook") or clip.get("seed_text", "")[:30] + "..."
            desc = clip.get("summary") or clip.get("seed_text") or "N/A"
            score = clip.get("viral_score") or clip.get("score")
            tags = ", ".join(clip.get("hashtags", [])) if clip.get("hashtags") else "N/A"
            logger.debug(f"Clip {i+1}:")
            logger.debug(f"  Title: {title}")
            logger.debug(f"  Desc:  {desc}")
            logger.debug(f"  Score: {score}")
            logger.debug(f"  Tags:  {tags}")
            time_str = f"{clip['start']:.2f}s - {clip['end']:.2f}s"
            if "raw_start" in clip and "raw_end" in clip:
                if abs(clip["raw_start"] - clip["start"]) > 0.1 or abs(clip["raw_end"] - clip["end"]) > 0.1:
                    time_str += f" (LLM intended: {clip['raw_start']:.2f}s - {clip['raw_end']:.2f}s)"
            logger.debug(f"  Time:  {time_str}")
        logger.debug("-------------------------------")

    if not highlight_intervals:
        logger.error("No highlight intervals selected.")
        return

    # 5. Clip Generation
    for idx, interval in enumerate(highlight_intervals):
        start, end = interval["start"], interval["end"]
        duration = end - start
        if duration <= 0: continue
        
        output_name = f"clip_{idx + 1}.mp4"
        output_path = os.path.join(output_dir, output_name)
        ass_path = os.path.join(output_dir, f"clip_{idx + 1}_subs.ass")
        
        write_clip_ass(segments, start, end, ass_path, cfg)
        
        # Face Analysis & Layout
        face_analysis = analyze_clip_faces(input_video, start, end, cfg)
        spk_segs = face_analysis["speaker_segments"]
        
        layout_segments = []
        layout_analysis = analyze_best_layout(face_analysis["multi_samples"], face_analysis["track_detections"], start, end, cfg)
        recommended = layout_analysis["recommended_layout"]
        effective_layout = recommended if cfg.layout.mode == "auto" else cfg.layout.mode
        
        if effective_layout != "single":
            layout_segments = determine_layout_segments(face_analysis["multi_samples"], face_analysis["track_detections"], start, end, cfg)
        
        # Hook Detection
        hook_text = interval.get("hook")
        if not hook_text and cfg.hook.enabled:
            hook_keywords = get_niche_config(cfg.content_type)["HOOK_KEYWORDS"]
            hook_info = detect_hook_phrase(segments, start, end, cfg)
            if hook_info: hook_text = hook_info["text"]
            
        # Create Clip
        if effective_layout != "single" and layout_segments:
            create_multi_layout_clip(input_video, ass_path, output_path, start, duration, layout_segments, spk_segs, start, end, cfg, hook_text)
        else:
            crop_expr = crop_x_expression_for_segments(spk_segs, start, end, cfg.video.target_width)
            create_vertical_with_subs(input_video, ass_path, output_path, start, duration, cfg, crop_expr, hook_text)
            
        logger.info(f"Generated {output_name}")
