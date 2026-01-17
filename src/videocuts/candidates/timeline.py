"""Timeline builder module (Epic 3 - US-3.0).

This module builds an analysis-ready timeline from:
- transcript.json (segments with word timings)
- diarization.json (speaker information)
- audio_16k_mono.wav (audio features)

The timeline enriches each segment with:
- Speaker information and confidence
- RMS energy (loudness)
- Speech rate (words per second)
- Silence before/after
"""
from __future__ import annotations
import os
import json
import logging
from typing import List, Dict, Optional, Tuple

import torch
import torchaudio

from videocuts.config import Config, TimelineConfig
from videocuts.models.transcript import Transcript
from videocuts.candidates.models import (
    Timeline,
    TimelineSegment,
    AudioFeatures,
    DiarizationResult,
)
# Note: assign_speakers_to_segments imported inside build_timeline() to avoid circular import

logger = logging.getLogger(__name__)


def build_timeline(
    transcript_path: str,
    audio_path: str,
    diarization_path: Optional[str],
    cfg: Config
) -> Timeline:
    """
    Build an analysis-ready timeline from transcript, audio, and diarization.
    
    Args:
        transcript_path: Path to transcript.json
        audio_path: Path to audio file (16kHz mono WAV)
        diarization_path: Path to diarization.json (optional)
        cfg: Configuration object
    
    Returns:
        Timeline with enriched segments and audio features
    """
    logger.info("Building analysis-ready timeline...")
    
    # 1. Load transcript
    transcript = Transcript.load(transcript_path)
    segments = transcript.to_legacy_segments()
    
    logger.info(f"Loaded {len(segments)} transcript segments")
    
    # 2. Load or skip diarization
    diarization = None
    if diarization_path and os.path.exists(diarization_path):
        diarization = DiarizationResult.load(diarization_path)
        logger.info(f"Loaded diarization with {diarization.num_speakers} speakers")
    
    # 3. Assign speakers to segments
    if diarization and diarization.speaker_turns:
        # Import here to avoid circular import
        from videocuts.audio.diarization import assign_speakers_to_segments
        segments = assign_speakers_to_segments(segments, diarization)
    else:
        # Default speaker assignment
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
            seg["speaker_confidence"] = 0.5
            seg["has_overlap"] = False
    
    # 4. Compute audio features per segment
    segments = _compute_segment_audio_features(segments, audio_path, cfg.timeline)
    
    # 5. Compute silence before/after
    segments = _compute_silence_boundaries(segments)
    
    # 6. Compute global audio features
    audio_features = _compute_global_audio_features(audio_path, cfg.timeline)
    
    # 7. Build timeline segments
    timeline_segments = [
        TimelineSegment(
            id=seg.get("id", i),
            start_s=seg.get("start_s", seg.get("start", 0)),
            end_s=seg.get("end_s", seg.get("end", 0)),
            text=seg.get("text", ""),
            speaker=seg.get("speaker"),
            speaker_confidence=seg.get("speaker_confidence", 0.0),
            has_overlap=seg.get("has_overlap", False),
            rms_energy=seg.get("rms_energy", 0.0),
            speech_rate_wps=seg.get("speech_rate_wps", 0.0),
            silence_before_s=seg.get("silence_before_s", 0.0),
            silence_after_s=seg.get("silence_after_s", 0.0),
        )
        for i, seg in enumerate(segments)
    ]
    
    timeline = Timeline(
        duration_s=transcript.duration_s,
        segments=timeline_segments,
        audio_features=audio_features,
    )
    
    logger.info(f"Built timeline with {len(timeline_segments)} segments, duration={timeline.duration_s:.2f}s")
    
    return timeline


def save_timeline(timeline: Timeline, output_path: str) -> None:
    """Save timeline to JSON file."""
    timeline.save(output_path)
    logger.info(f"Saved timeline to '{output_path}'")


def _compute_segment_audio_features(
    segments: List[Dict],
    audio_path: str,
    cfg: TimelineConfig
) -> List[Dict]:
    """
    Compute RMS energy and speech rate for each segment.
    
    Args:
        segments: List of transcript segments
        audio_path: Path to audio file
        cfg: Timeline configuration
    
    Returns:
        Segments enriched with rms_energy and speech_rate_wps
    """
    logger.info("Computing audio features per segment...")
    
    # Load audio
    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    except Exception as e:
        logger.warning(f"Could not load audio for feature extraction: {e}")
        # Return segments with default values
        for seg in segments:
            seg["rms_energy"] = 0.0
            seg["speech_rate_wps"] = _compute_speech_rate(seg)
        return segments
    
    for seg in segments:
        start_s = seg.get("start_s", seg.get("start", 0))
        end_s = seg.get("end_s", seg.get("end", 0))
        
        # Compute RMS for segment
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        end_sample = min(end_sample, waveform.shape[1])
        
        if end_sample > start_sample:
            seg_wave = waveform[:, start_sample:end_sample]
            rms = torch.sqrt((seg_wave ** 2).mean()).item()
        else:
            rms = 0.0
        
        seg["rms_energy"] = round(rms, 6)
        
        # Compute speech rate
        seg["speech_rate_wps"] = _compute_speech_rate(seg)
    
    return segments


def _compute_speech_rate(seg: Dict) -> float:
    """
    Compute speech rate in words per second.
    
    Args:
        seg: Segment with text and timing
    
    Returns:
        Words per second
    """
    text = seg.get("text", "").strip()
    start_s = seg.get("start_s", seg.get("start", 0))
    end_s = seg.get("end_s", seg.get("end", 0))
    duration = max(end_s - start_s, 0.001)
    
    # Count words
    words = [w for w in text.split() if w.strip()]
    word_count = len(words)
    
    return round(word_count / duration, 2)


def _compute_silence_boundaries(segments: List[Dict]) -> List[Dict]:
    """
    Compute silence before and after each segment.
    
    Args:
        segments: List of segments sorted by time
    
    Returns:
        Segments enriched with silence_before_s and silence_after_s
    """
    n = len(segments)
    
    for i, seg in enumerate(segments):
        start_s = seg.get("start_s", seg.get("start", 0))
        end_s = seg.get("end_s", seg.get("end", 0))
        
        # Silence before (gap from previous segment)
        if i > 0:
            prev_end = segments[i - 1].get("end_s", segments[i - 1].get("end", 0))
            silence_before = max(start_s - prev_end, 0)
        else:
            silence_before = start_s  # Silence from video start
        
        # Silence after (gap to next segment)
        if i < n - 1:
            next_start = segments[i + 1].get("start_s", segments[i + 1].get("start", 0))
            silence_after = max(next_start - end_s, 0)
        else:
            silence_after = 0  # End of video, will be updated later
        
        seg["silence_before_s"] = round(silence_before, 3)
        seg["silence_after_s"] = round(silence_after, 3)
    
    return segments


def _compute_global_audio_features(
    audio_path: str,
    cfg: TimelineConfig
) -> AudioFeatures:
    """
    Compute global audio features for the entire audio file.
    
    Args:
        audio_path: Path to audio file
        cfg: Timeline configuration
    
    Returns:
        AudioFeatures with global RMS, per-second RMS, and peaks
    """
    logger.info("Computing global audio features...")
    
    try:
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # mono, 1D
    except Exception as e:
        logger.warning(f"Could not load audio for global features: {e}")
        return AudioFeatures()
    
    duration_s = waveform.shape[0] / sr
    
    # Global RMS
    global_rms = torch.sqrt((waveform ** 2).mean()).item()
    
    # Per-second RMS
    rms_per_second = []
    window_samples = int(cfg.rms_window_s * sr)
    hop_samples = int(cfg.rms_hop_s * sr)
    
    for i in range(0, waveform.shape[0], sr):  # Every second
        end_i = min(i + sr, waveform.shape[0])
        chunk = waveform[i:end_i]
        if chunk.shape[0] > 0:
            rms = torch.sqrt((chunk ** 2).mean()).item()
            rms_per_second.append(round(rms, 6))
    
    # Find peaks (local maxima in RMS)
    peaks = []
    for i in range(1, len(rms_per_second) - 1):
        if rms_per_second[i] > rms_per_second[i - 1] and rms_per_second[i] > rms_per_second[i + 1]:
            if rms_per_second[i] > global_rms * 1.2:  # 20% above average
                peaks.append(float(i))  # Second index
    
    return AudioFeatures(
        global_rms=round(global_rms, 6),
        rms_per_second=rms_per_second,
        peaks=peaks,
        duration_s=round(duration_s, 3),
    )


def save_audio_features(features: AudioFeatures, output_path: str) -> None:
    """Save audio features to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features.to_dict(), f, indent=2)
    logger.info(f"Saved audio features to '{output_path}'")
