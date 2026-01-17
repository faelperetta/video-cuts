"""Candidate window generator (Epic 3 - US-3.1).

This module generates raw candidate clip windows using a sliding window
approach with transcript-segment snapping.

Algorithm:
1. Create sliding windows over time with step_s interval
2. For each window, propose [t, t + target_len]
3. Snap start/end to nearest segment boundary within tolerance
4. Discard if duration < min or > max after snapping
"""
from __future__ import annotations
import logging
from typing import List, Optional

from videocuts.config import Config, CandidateConfig
from videocuts.candidates.models import CandidateWindow, Timeline, TimelineSegment

logger = logging.getLogger(__name__)


def generate_candidates(
    timeline: Timeline,
    cfg: Config
) -> List[CandidateWindow]:
    """
    Generate raw candidate windows using sliding window approach.
    
    Args:
        timeline: Analysis-ready timeline with segments
        cfg: Configuration object
    
    Returns:
        List of candidate windows satisfying duration constraints
    """
    candidate_cfg = cfg.candidate
    
    logger.info(
        f"Generating candidates: target={candidate_cfg.target_len_s}s, "
        f"min={candidate_cfg.min_len_s}s, max={candidate_cfg.max_len_s}s, "
        f"step={candidate_cfg.step_s}s"
    )
    
    if not timeline.segments:
        logger.warning("No segments in timeline, cannot generate candidates")
        return []
    
    duration = timeline.duration_s
    candidates = []
    candidate_id = 0
    
    # Sliding window over timeline
    t = 0.0
    while t < duration - candidate_cfg.min_len_s:
        # Generate candidates of varying lengths
        # Start from min_len up to max_len with a step (e.g. 10s)
        # We also include the specific target_len_s if standard
        
        scan_step = 10.0 # Scan every 10s of duration
        lengths_to_scan = sorted(list(set(
            [candidate_cfg.target_len_s] + 
            list(range(int(candidate_cfg.min_len_s), int(candidate_cfg.max_len_s) + 1, int(scan_step))) +
            [candidate_cfg.max_len_s]
        )))
        
        for length in lengths_to_scan:
            if length < candidate_cfg.min_len_s or length > candidate_cfg.max_len_s:
                continue
                
            # Propose window
            proposed_start = t
            proposed_end = t + length
            
            # Snap to segment boundaries
            snapped_start = snap_to_segment_boundary(
                proposed_start, 
                timeline.segments,
                candidate_cfg.snap_tolerance_s,
                prefer="after"  # Prefer starting at segment start
            )
            snapped_end = snap_to_segment_boundary(
                proposed_end,
                timeline.segments,
                candidate_cfg.snap_tolerance_s,
                prefer="before"  # Prefer ending at segment end
            )
            
            # Calculate duration after snapping
            candidate_duration = snapped_end - snapped_start
            
            # Validate duration constraints
            if validate_candidate(
                snapped_start, 
                snapped_end,
                candidate_cfg.min_len_s,
                candidate_cfg.max_len_s
            ):
                candidate_id += 1
                candidates.append(CandidateWindow(
                    candidate_id=f"c_{candidate_id:04d}",
                    start_s=round(snapped_start, 3),
                    end_s=round(snapped_end, 3),
                    duration_s=round(candidate_duration, 3)
                ))
        
        t += candidate_cfg.step_s
    
    logger.info(f"Generated {len(candidates)} candidates from {duration:.1f}s video")
    
    # Remove duplicates (same start/end after snapping)
    candidates = _remove_duplicate_candidates(candidates)
    logger.info(f"After deduplication: {len(candidates)} unique candidates")
    
    return candidates


def snap_to_segment_boundary(
    time_s: float,
    segments: List[TimelineSegment],
    tolerance_s: float = 1.2,
    prefer: str = "nearest"
) -> float:
    """
    Snap a time to the nearest segment boundary within tolerance.
    
    Args:
        time_s: Time to snap
        segments: List of timeline segments
        tolerance_s: Maximum distance to snap
        prefer: "nearest", "before", or "after"
    
    Returns:
        Snapped time (or original if no boundary within tolerance)
    """
    best_boundary = time_s
    best_distance = tolerance_s + 1  # Start beyond tolerance
    
    for seg in segments:
        # Check segment start boundary
        start_dist = abs(seg.start_s - time_s)
        if start_dist <= tolerance_s:
            if prefer == "after" and seg.start_s >= time_s:
                if start_dist < best_distance:
                    best_distance = start_dist
                    best_boundary = seg.start_s
            elif prefer == "before" and seg.start_s <= time_s:
                if start_dist < best_distance:
                    best_distance = start_dist
                    best_boundary = seg.start_s
            elif prefer == "nearest":
                if start_dist < best_distance:
                    best_distance = start_dist
                    best_boundary = seg.start_s
        
        # Check segment end boundary
        end_dist = abs(seg.end_s - time_s)
        if end_dist <= tolerance_s:
            if prefer == "after" and seg.end_s >= time_s:
                if end_dist < best_distance:
                    best_distance = end_dist
                    best_boundary = seg.end_s
            elif prefer == "before" and seg.end_s <= time_s:
                if end_dist < best_distance:
                    best_distance = end_dist
                    best_boundary = seg.end_s
            elif prefer == "nearest":
                if end_dist < best_distance:
                    best_distance = end_dist
                    best_boundary = seg.end_s
    
    return best_boundary


def validate_candidate(
    start_s: float,
    end_s: float,
    min_len_s: float,
    max_len_s: float
) -> bool:
    """
    Validate candidate meets duration constraints.
    
    Args:
        start_s: Start time
        end_s: End time
        min_len_s: Minimum duration
        max_len_s: Maximum duration
    
    Returns:
        True if valid, False otherwise
    """
    duration = end_s - start_s
    return min_len_s <= duration <= max_len_s


def _remove_duplicate_candidates(
    candidates: List[CandidateWindow]
) -> List[CandidateWindow]:
    """
    Remove duplicate candidates with same start/end times.
    
    Args:
        candidates: List of candidates
    
    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []
    
    for c in candidates:
        key = (round(c.start_s, 1), round(c.end_s, 1))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique


def get_segments_in_window(
    timeline: Timeline,
    start_s: float,
    end_s: float
) -> List[TimelineSegment]:
    """
    Get all segments that overlap with a time window.
    
    Args:
        timeline: Timeline with segments
        start_s: Window start
        end_s: Window end
    
    Returns:
        List of segments overlapping the window
    """
    return [
        seg for seg in timeline.segments
        if seg.end_s > start_s and seg.start_s < end_s
    ]


def get_text_in_window(
    timeline: Timeline,
    start_s: float,
    end_s: float
) -> str:
    """
    Get concatenated text from all segments in a time window.
    
    Args:
        timeline: Timeline with segments
        start_s: Window start
        end_s: Window end
    
    Returns:
        Concatenated text from overlapping segments
    """
    segments = get_segments_in_window(timeline, start_s, end_s)
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


def save_candidates_raw(
    candidates: List[CandidateWindow],
    output_path: str
) -> None:
    """Save raw candidates to JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    data = {
        "candidates": [c.to_dict() for c in candidates]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(candidates)} raw candidates to '{output_path}'")
