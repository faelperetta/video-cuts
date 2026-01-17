"""Clip boundary refinement (Epic 3 - US-3.6).

This module refines clip boundaries to avoid awkward cuts,
extending starts/ends to capture setup/resolution phrases.

Refinement steps:
1. Extend start backward by up to 1.5s if it captures a setup phrase
   and does not add silence > 0.8s
2. Extend end forward by up to 2.0s if it captures a resolution phrase
   and does not exceed max_len_s
3. Ensure no leading silence > 0.8s
4. Ensure no trailing silence > 1.0s
5. End on sentence boundary when possible
"""
from __future__ import annotations
import logging
from typing import List, Optional

from videocuts.config import Config
from videocuts.candidates.models import (
    SelectedClip,
    RefinedClip,
    Timeline,
    TimelineSegment,
)
from videocuts.candidates.generator import get_segments_in_window
from videocuts.candidates.features import (
    detect_hook_open_loop,
    detect_close_resolution,
    RESOLUTION_PHRASES,
)

logger = logging.getLogger(__name__)


# Refinement parameters
MAX_EXTEND_START = 1.5  # seconds
MAX_EXTEND_END = 2.0    # seconds
MAX_SILENCE_START = 0.8 # seconds
MAX_SILENCE_END = 1.0   # seconds


def refine_clip_boundaries(
    selected_clips: List[SelectedClip],
    timeline: Timeline,
    cfg: Config
) -> List[RefinedClip]:
    """
    Refine boundaries for all selected clips.
    
    Args:
        selected_clips: List of selected clips
        timeline: Timeline with enriched segments
        cfg: Configuration
    
    Returns:
        List of refined clips with adjusted boundaries
    """
    logger.info(f"Refining boundaries for {len(selected_clips)} clips...")
    
    max_len = cfg.candidate.max_len_s
    refined = []
    
    for clip in selected_clips:
        refined_clip = refine_single_clip(clip, timeline, max_len)
        refined.append(refined_clip)
    
    logger.info(f"Refined {len(refined)} clip boundaries")
    return refined


def refine_single_clip(
    clip: SelectedClip,
    timeline: Timeline,
    max_len_s: float
) -> RefinedClip:
    """
    Refine boundaries for a single clip.
    
    Args:
        clip: Selected clip
        timeline: Timeline with segments
        max_len_s: Maximum clip duration
    
    Returns:
        RefinedClip with adjusted boundaries
    """
    original_start = clip.start_s
    original_end = clip.end_s
    new_start = original_start
    new_end = original_end
    changes = []
    
    # Get segments around the clip
    extended_segments = get_segments_in_window(
        timeline, 
        original_start - MAX_EXTEND_START - 1.0,
        original_end + MAX_EXTEND_END + 1.0
    )
    
    # 1. Try to extend start backward for setup phrase
    extended_start, start_change = extend_start_for_setup(
        original_start, extended_segments, timeline
    )
    if extended_start is not None:
        new_start = extended_start
        if start_change:
            changes.append(start_change)
    
    # 2. Try to extend end forward for resolution
    extended_end, end_change = extend_end_for_resolution(
        original_end, extended_segments, timeline, max_len_s, new_start
    )
    if extended_end is not None:
        new_end = extended_end
        if end_change:
            changes.append(end_change)
    
    # 3. Validate silence constraints
    new_start, new_end, silence_changes = validate_silence_constraints(
        new_start, new_end, timeline
    )
    changes.extend(silence_changes)
    
    # 4. Try to end on sentence boundary
    adjusted_end, boundary_change = snap_to_sentence_boundary(
        new_end, timeline, max_dist=1.0
    )
    if adjusted_end is not None and adjusted_end != new_end:
        new_end = adjusted_end
        if boundary_change:
            changes.append(boundary_change)
    
    # Calculate new duration
    new_duration = new_end - new_start
    
    return RefinedClip(
        candidate_id=clip.candidate_id,
        start_s_before=round(original_start, 3),
        end_s_before=round(original_end, 3),
        start_s_after=round(new_start, 3),
        end_s_after=round(new_end, 3),
        duration_s=round(new_duration, 3),
        changes=changes,
        final_rank=clip.final_rank,
        final_score=clip.final_score,
        title=clip.title,
        hook_line=clip.hook_line,
    )


def extend_start_for_setup(
    current_start: float,
    segments: List[TimelineSegment],
    timeline: Timeline
) -> tuple[Optional[float], Optional[str]]:
    """
    Try to extend start backward to capture a setup phrase.
    
    Args:
        current_start: Current clip start
        segments: Nearby segments
        timeline: Timeline
    
    Returns:
        (new_start, change_reason) or (None, None) if no change
    """
    # Find segments just before current start
    before_segments = [
        s for s in segments
        if s.end_s <= current_start and s.end_s > current_start - MAX_EXTEND_START
    ]
    
    if not before_segments:
        return None, None
    
    # Sort by end time descending (closest first)
    before_segments.sort(key=lambda s: s.end_s, reverse=True)
    
    for seg in before_segments:
        # Check if this segment has setup content
        text = seg.text.lower()
        
        has_setup = any(phrase in text for phrase in [
            "let me", "here's", "the thing is", "so basically",
            "what happened", "the reason", "you see",
            "olha", "então", "basicamente", "o que aconteceu",
            "mira", "entonces", "lo que pasó",
        ])
        
        if has_setup:
            # Calculate silence that would be added
            silence = current_start - seg.end_s
            if silence <= MAX_SILENCE_START:
                return seg.start_s, "extended_start_setup"
    
    return None, None


def extend_end_for_resolution(
    current_end: float,
    segments: List[TimelineSegment],
    timeline: Timeline,
    max_len_s: float,
    start_s: float
) -> tuple[Optional[float], Optional[str]]:
    """
    Try to extend end forward to capture a resolution phrase.
    
    Args:
        current_end: Current clip end
        segments: Nearby segments
        timeline: Timeline
        max_len_s: Maximum allowed clip length
        start_s: Current clip start
    
    Returns:
        (new_end, change_reason) or (None, None) if no change
    """
    # Find segments just after current end
    after_segments = [
        s for s in segments
        if s.start_s >= current_end and s.start_s < current_end + MAX_EXTEND_END
    ]
    
    if not after_segments:
        return None, None
    
    # Sort by start time (earliest first)
    after_segments.sort(key=lambda s: s.start_s)
    
    for seg in after_segments:
        # Check if this segment has resolution content
        text = seg.text.lower()
        
        has_resolution = any(phrase in text for phrase in RESOLUTION_PHRASES)
        
        if has_resolution:
            # Check if extending would exceed max length
            new_duration = seg.end_s - start_s
            if new_duration <= max_len_s:
                # Calculate silence that would be added
                silence = seg.start_s - current_end
                if silence <= MAX_SILENCE_END:
                    return seg.end_s, "extended_end_resolution"
    
    return None, None


def validate_silence_constraints(
    start_s: float,
    end_s: float,
    timeline: Timeline
) -> tuple[float, float, List[str]]:
    """
    Validate and adjust for silence constraints.
    
    Args:
        start_s: Clip start
        end_s: Clip end
        timeline: Timeline
    
    Returns:
        (adjusted_start, adjusted_end, changes)
    """
    changes = []
    segments = get_segments_in_window(timeline, start_s, end_s)
    
    if not segments:
        return start_s, end_s, changes
    
    # Check leading silence
    first_seg = min(segments, key=lambda s: s.start_s)
    leading_silence = first_seg.start_s - start_s
    
    if leading_silence > MAX_SILENCE_START:
        # Move start forward to reduce silence
        start_s = first_seg.start_s - MAX_SILENCE_START
        changes.append(f"reduced_leading_silence_to_{MAX_SILENCE_START}s")
    
    # Check trailing silence
    last_seg = max(segments, key=lambda s: s.end_s)
    trailing_silence = end_s - last_seg.end_s
    
    if trailing_silence > MAX_SILENCE_END:
        # Move end backward to reduce silence
        end_s = last_seg.end_s + MAX_SILENCE_END
        changes.append(f"reduced_trailing_silence_to_{MAX_SILENCE_END}s")
    
    return start_s, end_s, changes


def snap_to_sentence_boundary(
    end_s: float,
    timeline: Timeline,
    max_dist: float = 1.0
) -> tuple[Optional[float], Optional[str]]:
    """
    Try to snap end time to a sentence boundary.
    
    Args:
        end_s: Current end time
        timeline: Timeline
        max_dist: Maximum distance to snap
    
    Returns:
        (adjusted_end, change_reason) or (None, None) if no change
    """
    # Find segments near end
    nearby = [
        s for s in timeline.segments
        if abs(s.end_s - end_s) <= max_dist
    ]
    
    if not nearby:
        return None, None
    
    # Find closest segment that ends on sentence boundary
    for seg in sorted(nearby, key=lambda s: abs(s.end_s - end_s)):
        text = seg.text.strip()
        if text.endswith((".", "!", "?", "...", "。", "！", "？")):
            if seg.end_s != end_s:
                return seg.end_s, "snapped_to_sentence_boundary"
    
    return None, None


def save_clips_refined(
    clips: List[RefinedClip],
    output_path: str
) -> None:
    """Save refined clips to JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    data = {
        "clips": [c.to_dict() for c in clips]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(clips)} refined clips to '{output_path}'")
