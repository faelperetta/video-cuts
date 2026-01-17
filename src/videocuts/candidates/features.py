"""Feature computation for clip candidates (Epic 3 - US-3.2).

This module computes hook/middle/end structure features for each candidate
to enable scoring and ranking.

Features computed:
- Hook region (first 3s): question, open_loop, contrast
- Close region (last 4s): resolution, sentence_boundary
- Speaker: dominance_ratio, switch_count
- Audio: overlap_ratio, energy_peak_ratio
- Silence: at_start, at_end
"""
from __future__ import annotations
import re
import logging
from typing import List, Tuple, Optional

from videocuts.config import Config
from videocuts.candidates.models import (
    CandidateWindow,
    CandidateFeatures,
    Timeline,
    TimelineSegment,
)
from videocuts.candidates.generator import get_segments_in_window, get_text_in_window

logger = logging.getLogger(__name__)

# Hook detection patterns
HOOK_INTERROGATIVES = [
    "what", "why", "how", "when", "where", "who", "which",
    "do you", "have you", "can you", "would you", "could you",
    "is it", "are you", "was it", "were you",
    # Portuguese
    "o que", "por que", "como", "quando", "onde", "quem",
    "você sabe", "você já", "você pode",
    # Spanish
    "qué", "por qué", "cómo", "cuándo", "dónde", "quién",
]

OPEN_LOOP_PHRASES = [
    "here's the thing", "here is the thing",
    "the thing is", "the secret is", "the truth is",
    "wait", "but wait", "hold on", "listen",
    "you won't believe", "nobody talks about",
    "the problem is", "the crazy thing is",
    "let me tell you", "i'm going to tell you",
    # Portuguese
    "olha só", "escuta", "o negócio é", 
    "a verdade é", "ninguém fala sobre",
    # Spanish
    "mira", "escucha", "el secreto es",
    "la verdad es", "nadie habla de",
]

CONTRAST_MARKERS = [
    "but", "however", "actually", "in fact",
    "on the other hand", "contrary to",
    "the opposite", "instead of", "unlike",
    # Portuguese
    "mas", "porém", "na verdade", "de fato",
    "pelo contrário", "ao contrário",
    # Spanish
    "pero", "sin embargo", "en realidad", "de hecho",
    "por el contrario",
]

RESOLUTION_PHRASES = [
    "that's why", "this is why", "so that's",
    "the point is", "the lesson is", "the takeaway is",
    "in summary", "to summarize", "to conclude",
    "so yeah", "and that's", "that's the key",
    "at the end of the day", "bottom line",
    # Portuguese
    "por isso", "é por isso", "então é isso",
    "o ponto é", "a lição é", "resumindo",
    "no final das contas",
    # Spanish
    "por eso", "es por eso", "entonces",
    "el punto es", "la lección es", "en resumen",
    "al final del día",
]


def compute_candidate_features(
    candidate: CandidateWindow,
    timeline: Timeline,
    cfg: Config
) -> CandidateFeatures:
    """
    Compute all structure features for a candidate.
    
    Args:
        candidate: Candidate window
        timeline: Timeline with enriched segments
        cfg: Configuration
    
    Returns:
        CandidateFeatures with all computed features
    """
    segments = get_segments_in_window(
        timeline, candidate.start_s, candidate.end_s
    )
    
    if not segments:
        return CandidateFeatures(
            candidate_id=candidate.candidate_id,
            start_s=candidate.start_s,
            end_s=candidate.end_s,
            duration_s=candidate.duration_s,
        )
    
    # Get text for different regions
    hook_end = candidate.start_s + 3.0
    close_start = candidate.end_s - 4.0
    
    hook_text = get_text_in_window(timeline, candidate.start_s, hook_end)
    close_text = get_text_in_window(timeline, close_start, candidate.end_s)
    full_text = get_text_in_window(timeline, candidate.start_s, candidate.end_s)
    
    # Get segments for specific regions
    close_segments = get_segments_in_window(timeline, close_start, candidate.end_s)
    
    # Hook features
    hook_question = detect_hook_question(hook_text)
    hook_open_loop = detect_hook_open_loop(hook_text)
    hook_contrast = detect_hook_contrast(hook_text)
    
    # Close features
    close_resolution = detect_close_resolution(close_text)
    ends_on_sentence = check_sentence_boundary(close_segments[-1] if close_segments else None)
    ends_with_question = "?" in close_text[-50:] if close_text else False
    
    # Speaker features
    speaker_dominance, speaker_switches = compute_speaker_stats(
        segments, candidate.start_s, candidate.end_s
    )
    
    # Audio features
    overlap_ratio = compute_overlap_ratio(segments, candidate.duration_s)
    energy_peak = compute_energy_peak_ratio(segments, timeline)
    speech_coverage = compute_speech_coverage(
        segments, candidate.start_s, candidate.end_s
    )
    
    # Silence features
    silence_start = compute_silence_at_start(segments, candidate.start_s)
    silence_end = compute_silence_at_end(segments, candidate.end_s)
    
    # Boundary quality
    starts_mid = check_starts_mid_sentence(segments, candidate.start_s)
    
    return CandidateFeatures(
        candidate_id=candidate.candidate_id,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        duration_s=candidate.duration_s,
        hook_question=hook_question,
        hook_open_loop=hook_open_loop,
        hook_contrast=hook_contrast,
        close_resolution=close_resolution,
        ends_on_sentence_boundary=ends_on_sentence,
        ends_with_question=ends_with_question,
        speaker_dominance_ratio=round(speaker_dominance, 3),
        speaker_switch_count=speaker_switches,
        overlap_ratio=round(overlap_ratio, 3),
        energy_peak_ratio=round(energy_peak, 3),
        speech_coverage=round(speech_coverage, 3),
        silence_at_start_s=round(silence_start, 3),
        silence_at_end_s=round(silence_end, 3),
        starts_mid_sentence=starts_mid,
    )


def compute_all_features(
    candidates: List[CandidateWindow],
    timeline: Timeline,
    cfg: Config
) -> List[CandidateFeatures]:
    """
    Compute features for all candidates.
    
    Args:
        candidates: List of candidate windows
        timeline: Timeline with enriched segments
        cfg: Configuration
    
    Returns:
        List of candidates with computed features
    """
    logger.info(f"Computing features for {len(candidates)} candidates...")
    
    features = []
    for candidate in candidates:
        feat = compute_candidate_features(candidate, timeline, cfg)
        features.append(feat)
    
    logger.info(f"Computed features for {len(features)} candidates")
    return features


# =============================================================================
# Hook Detection
# =============================================================================

def detect_hook_question(text: str) -> bool:
    """Check if text starts with a question or interrogative."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Check for question mark in first sentence
    first_sentence = text_lower.split(".")[0] if "." in text_lower else text_lower
    if "?" in first_sentence:
        return True
    
    # Check for interrogative words at start
    for interrogative in HOOK_INTERROGATIVES:
        if text_lower.startswith(interrogative):
            return True
        if text_lower.startswith(f"so {interrogative}"):
            return True
    
    return False


def detect_hook_open_loop(text: str) -> bool:
    """Check if text contains an open loop phrase."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    for phrase in OPEN_LOOP_PHRASES:
        if phrase in text_lower:
            return True
    
    return False


def detect_hook_contrast(text: str) -> bool:
    """Check if text contains contrast markers."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    for marker in CONTRAST_MARKERS:
        # Check for marker at word boundary
        pattern = rf"\b{re.escape(marker)}\b"
        if re.search(pattern, text_lower):
            return True
    
    return False


# =============================================================================
# Close Detection
# =============================================================================

def detect_close_resolution(text: str) -> bool:
    """Check if text contains resolution/conclusion phrases."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    for phrase in RESOLUTION_PHRASES:
        if phrase in text_lower:
            return True
    
    return False


def check_sentence_boundary(segment: Optional[TimelineSegment]) -> bool:
    """Check if segment ends on a sentence boundary."""
    if not segment or not segment.text:
        return False
    
    text = segment.text.strip()
    
    # Check for sentence-ending punctuation
    if text.endswith((".", "!", "?", "...", "。", "！", "？")):
        return True
    
    # Check for common ending patterns
    if text.endswith(("right", "yeah", "you know", "okay", "so")):
        return True
    
    return False


# =============================================================================
# Speaker Features
# =============================================================================

def compute_speaker_stats(
    segments: List[TimelineSegment],
    start_s: float,
    end_s: float
) -> Tuple[float, int]:
    """
    Compute speaker dominance ratio and switch count.
    
    Args:
        segments: Segments in the window
        start_s: Window start
        end_s: Window end
    
    Returns:
        (dominance_ratio, switch_count)
    """
    if not segments:
        return 0.0, 0
    
    window_duration = max(end_s - start_s, 0.001)
    
    # Count time per speaker
    speaker_times = {}
    for seg in segments:
        speaker = seg.speaker or "UNKNOWN"
        # Clip segment to window boundaries
        seg_start = max(seg.start_s, start_s)
        seg_end = min(seg.end_s, end_s)
        duration = max(seg_end - seg_start, 0)
        
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
    
    if not speaker_times:
        return 0.0, 0
    
    # Dominance ratio = max speaker time / window duration
    max_time = max(speaker_times.values())
    dominance_ratio = max_time / window_duration
    
    # Count speaker switches
    switches = 0
    prev_speaker = None
    for seg in segments:
        speaker = seg.speaker or "UNKNOWN"
        if prev_speaker is not None and speaker != prev_speaker:
            switches += 1
        prev_speaker = speaker
    
    return min(dominance_ratio, 1.0), switches


# =============================================================================
# Audio Features
# =============================================================================

def compute_overlap_ratio(
    segments: List[TimelineSegment],
    window_duration: float
) -> float:
    """Compute fraction of window with overlapping speech."""
    if not segments or window_duration <= 0:
        return 0.0
    
    overlap_duration = sum(
        seg.end_s - seg.start_s
        for seg in segments
        if seg.has_overlap
    )
    
    return min(overlap_duration / window_duration, 1.0)


def compute_energy_peak_ratio(
    segments: List[TimelineSegment],
    timeline: Timeline
) -> float:
    """Compute peak RMS / median RMS in window."""
    if not segments:
        return 1.0
    
    energies = [seg.rms_energy for seg in segments if seg.rms_energy > 0]
    
    if not energies:
        return 1.0
    
    peak = max(energies)
    median = sorted(energies)[len(energies) // 2]
    
    if median <= 0:
        return 1.0
    
    return peak / median


def compute_speech_coverage(
    segments: List[TimelineSegment],
    start_s: float,
    end_s: float
) -> float:
    """Compute fraction of window covered by speech."""
    window_duration = max(end_s - start_s, 0.001)
    
    if not segments:
        return 0.0
    
    speech_duration = 0.0
    for seg in segments:
        # Clip to window
        seg_start = max(seg.start_s, start_s)
        seg_end = min(seg.end_s, end_s)
        speech_duration += max(seg_end - seg_start, 0)
    
    return min(speech_duration / window_duration, 1.0)


# =============================================================================
# Silence Features
# =============================================================================

def compute_silence_at_start(
    segments: List[TimelineSegment],
    window_start: float
) -> float:
    """Compute silence duration at the start of window."""
    if not segments:
        return 0.0
    
    first_seg = min(segments, key=lambda s: s.start_s)
    return max(first_seg.start_s - window_start, 0)


def compute_silence_at_end(
    segments: List[TimelineSegment],
    window_end: float
) -> float:
    """Compute silence duration at the end of window."""
    if not segments:
        return 0.0
    
    last_seg = max(segments, key=lambda s: s.end_s)
    return max(window_end - last_seg.end_s, 0)


def check_starts_mid_sentence(
    segments: List[TimelineSegment],
    window_start: float
) -> bool:
    """Check if window starts mid-sentence."""
    if not segments:
        return False
    
    # Find segment that contains window start
    for seg in segments:
        if seg.start_s <= window_start < seg.end_s:
            # Window start is in the middle of this segment
            # Check if previous text ends with sentence boundary
            prev_text = seg.text[:int((window_start - seg.start_s) * 10)]
            if prev_text and not prev_text.strip().endswith((".", "!", "?", "...", ",")):
                return True
            break
    
    return False


def save_candidates_features(
    features: List[CandidateFeatures],
    output_path: str
) -> None:
    """Save candidates with features to JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    data = {
        "candidates": [f.to_dict() for f in features]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(features)} candidate features to '{output_path}'")
