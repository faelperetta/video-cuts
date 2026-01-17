"""Speaker diarization module using pyannote-audio (Epic 3 - US-3.0).

This module provides speaker diarization functionality to identify
who is speaking when in an audio file. It uses the pyannote-audio
library which provides state-of-the-art speaker diarization.
"""
from __future__ import annotations
import os
import logging
from typing import List, Optional, Dict, Any

from videocuts.config import DiarizationConfig
from videocuts.candidates.models import (
    DiarizationResult,
    SpeakerTurn,
    OverlapRegion,
)

logger = logging.getLogger(__name__)


def run_diarization(
    audio_path: str,
    cfg: DiarizationConfig,
    output_path: Optional[str] = None
) -> DiarizationResult:
    """
    Run speaker diarization on an audio file.
    
    Args:
        audio_path: Path to audio file (16kHz mono WAV recommended)
        cfg: Diarization configuration
        output_path: Optional path to save diarization JSON
    
    Returns:
        DiarizationResult with speaker turns and overlap regions
    """
    if not cfg.enabled:
        logger.info("Diarization disabled, returning empty result")
        return DiarizationResult()
    
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as e:
        logger.warning(f"pyannote-audio not available: {e}")
        logger.warning("Falling back to single-speaker mode")
        return _fallback_single_speaker(audio_path)
    
    # Determine device
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading diarization model '{cfg.model}' on {device}...")
    
    try:
        # Load the pyannote pipeline
        # Note: Requires HuggingFace token with pyannote access
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        pipeline = Pipeline.from_pretrained(
            cfg.model,
            use_auth_token=hf_token
        )
        pipeline.to(torch.device(device))
        
        logger.info(f"Running diarization on '{audio_path}'...")
        
        # Run diarization with speaker count hints
        diarization = pipeline(
            audio_path,
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers
        )
        
        # Extract speaker turns and overlaps
        result = _parse_diarization_output(diarization)
        
        logger.info(f"Found {result.num_speakers} speakers with {len(result.speaker_turns)} turns")
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        logger.warning("Falling back to single-speaker mode")
        result = _fallback_single_speaker(audio_path)
    
    # Save if path provided
    if output_path:
        result.save(output_path)
        logger.info(f"Saved diarization to '{output_path}'")
    
    return result


def _parse_diarization_output(diarization) -> DiarizationResult:
    """
    Parse pyannote diarization output into our data model.
    
    Args:
        diarization: pyannote.core.Annotation object
    
    Returns:
        DiarizationResult with parsed turns and overlaps
    """
    speaker_turns = []
    speakers_seen = set()
    
    # Iterate through diarization timeline
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speakers_seen.add(speaker)
        speaker_turns.append(SpeakerTurn(
            speaker=speaker,
            start_s=segment.start,
            end_s=segment.end,
            confidence=1.0  # pyannote doesn't provide per-segment confidence
        ))
    
    # Sort by start time
    speaker_turns.sort(key=lambda t: t.start_s)
    
    # Detect overlapping speech
    overlaps = _detect_overlaps(speaker_turns)
    
    # Calculate duration
    duration_s = max((t.end_s for t in speaker_turns), default=0.0)
    
    return DiarizationResult(
        speaker_turns=speaker_turns,
        overlaps=overlaps,
        num_speakers=len(speakers_seen),
        duration_s=duration_s
    )


def _detect_overlaps(turns: List[SpeakerTurn]) -> List[OverlapRegion]:
    """
    Detect regions where multiple speakers overlap.
    
    Args:
        turns: List of speaker turns sorted by start time
    
    Returns:
        List of overlap regions
    """
    overlaps = []
    n = len(turns)
    
    for i in range(n):
        for j in range(i + 1, n):
            turn_a = turns[i]
            turn_b = turns[j]
            
            # Check for overlap
            overlap_start = max(turn_a.start_s, turn_b.start_s)
            overlap_end = min(turn_a.end_s, turn_b.end_s)
            
            if overlap_start < overlap_end:
                # Found an overlap
                overlaps.append(OverlapRegion(
                    start_s=overlap_start,
                    end_s=overlap_end,
                    speakers=[turn_a.speaker, turn_b.speaker]
                ))
    
    # Merge adjacent overlaps
    overlaps.sort(key=lambda o: o.start_s)
    merged = []
    for overlap in overlaps:
        if merged and merged[-1].end_s >= overlap.start_s:
            # Extend existing overlap
            merged[-1].end_s = max(merged[-1].end_s, overlap.end_s)
            for spk in overlap.speakers:
                if spk not in merged[-1].speakers:
                    merged[-1].speakers.append(spk)
        else:
            merged.append(overlap)
    
    return merged


def _fallback_single_speaker(audio_path: str) -> DiarizationResult:
    """
    Fallback when diarization is unavailable: assume single speaker.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        DiarizationResult with single speaker covering entire audio
    """
    import subprocess
    import json
    
    # Get audio duration using ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        duration_s = float(probe_data["streams"][0].get("duration", 0))
    except Exception:
        duration_s = 0.0
    
    if duration_s > 0:
        return DiarizationResult(
            speaker_turns=[SpeakerTurn(
                speaker="SPEAKER_00",
                start_s=0.0,
                end_s=duration_s,
                confidence=0.5  # Low confidence since we're guessing
            )],
            overlaps=[],
            num_speakers=1,
            duration_s=duration_s
        )
    
    return DiarizationResult()


def assign_speakers_to_segments(
    segments: List[Dict],
    diarization: DiarizationResult
) -> List[Dict]:
    """
    Assign speaker labels to transcript segments based on diarization.
    
    Args:
        segments: List of transcript segments with start/end times
        diarization: Diarization result with speaker turns
    
    Returns:
        Segments enriched with speaker, speaker_confidence, has_overlap
    """
    enriched = []
    
    for seg in segments:
        seg_start = seg.get("start_s", seg.get("start", 0))
        seg_end = seg.get("end_s", seg.get("end", 0))
        
        # Find the speaker with most overlap
        best_speaker = None
        best_overlap = 0.0
        total_overlap = 0.0
        
        for turn in diarization.speaker_turns:
            overlap_start = max(seg_start, turn.start_s)
            overlap_end = min(seg_end, turn.end_s)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = turn.speaker
            
            total_overlap += overlap_duration
        
        # Calculate confidence based on how much of segment is covered
        seg_duration = max(seg_end - seg_start, 0.001)
        confidence = best_overlap / seg_duration if best_overlap > 0 else 0.0
        
        # Check for overlapping speech
        has_overlap = False
        for overlap_region in diarization.overlaps:
            if overlap_region.start_s < seg_end and overlap_region.end_s > seg_start:
                has_overlap = True
                break
        
        # Create enriched segment
        enriched_seg = dict(seg)
        enriched_seg.update({
            "speaker": best_speaker,
            "speaker_confidence": round(confidence, 3),
            "has_overlap": has_overlap
        })
        enriched.append(enriched_seg)
    
    return enriched
