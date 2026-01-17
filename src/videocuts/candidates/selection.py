"""Final clip selection with diversity enforcement (Epic 3 - US-3.5).

This module selects the best candidates for rendering, ensuring
that selected clips don't overlap too much.

Selection logic:
1. Rank by llm_score (if available), then heuristic score
2. Enforce diversity: no two clips with >35% time overlap
3. Output top_k_render clips (default 5)
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional

from videocuts.config import Config
from videocuts.candidates.models import ScoredCandidate, SelectedClip, LLMResult

logger = logging.getLogger(__name__)


def select_final_clips(
    scored_candidates: List[ScoredCandidate],
    llm_results: Optional[Dict[str, LLMResult]],
    cfg: Config
) -> List[SelectedClip]:
    """
    Select final clips for rendering with diversity enforcement.
    
    Args:
        scored_candidates: Candidates sorted by heuristic score
        llm_results: Optional LLM rerank results (candidate_id -> LLMResult)
        cfg: Configuration with selection parameters
    
    Returns:
        List of selected clips, ranked by final score
    """
    top_k = cfg.candidate.top_k_render
    max_overlap = cfg.candidate.max_overlap_ratio
    
    # Filter to eligible candidates only
    eligible = [c for c in scored_candidates if c.eligible]
    
    if not eligible:
        logger.warning("No eligible candidates for selection")
        return []
    
    logger.info(f"Selecting top {top_k} from {len(eligible)} eligible candidates")
    
    # Compute final score (LLM score takes priority if available)
    def get_final_score(c: ScoredCandidate) -> float:
        if llm_results and c.candidate_id in llm_results:
            llm_result = llm_results[c.candidate_id]
            # Combine: LLM score (0-10) * 10 + heuristic score (0-100)
            return llm_result.llm_score * 10 + c.score
        return c.score
    
    # Sort by final score descending
    eligible.sort(key=get_final_score, reverse=True)
    
    # Select with diversity enforcement
    selected = []
    for candidate in eligible:
        if len(selected) >= top_k:
            break
        
        # Check overlap with already selected clips
        if _has_excessive_overlap(candidate, selected, max_overlap):
            continue
        
        # Get LLM result if available
        llm_result = llm_results.get(candidate.candidate_id) if llm_results else None
        
        # Generate title
        if llm_result and llm_result.title:
            title = llm_result.title
        else:
            # Fallback: generate from transcript
            title = _generate_fallback_title(candidate)
        
        selected.append(SelectedClip(
            candidate_id=candidate.candidate_id,
            start_s=candidate.start_s,
            end_s=candidate.end_s,
            duration_s=candidate.duration_s,
            final_rank=len(selected) + 1,
            final_score=round(get_final_score(candidate), 2),
            title=title,
            hook_line=llm_result.hook_line if llm_result else None,
            llm_score=llm_result.llm_score if llm_result else None,
        ))
    
    logger.info(f"Selected {len(selected)} clips for rendering")
    
    # Sort by final rank
    selected.sort(key=lambda c: c.final_rank)
    
    return selected


def _has_excessive_overlap(
    candidate: ScoredCandidate,
    selected: List[SelectedClip],
    max_overlap: float
) -> bool:
    """Check if candidate overlaps too much with any selected clip."""
    for clip in selected:
        overlap = compute_overlap_ratio(
            candidate.start_s, candidate.end_s,
            clip.start_s, clip.end_s
        )
        if overlap > max_overlap:
            return True
    return False


def compute_overlap_ratio(
    start1: float, end1: float,
    start2: float, end2: float
) -> float:
    """
    Compute overlap ratio between two time ranges.
    
    Returns the overlap duration divided by the shorter clip duration.
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(overlap_end - overlap_start, 0)
    
    if overlap_duration == 0:
        return 0.0
    
    duration1 = end1 - start1
    duration2 = end2 - start2
    min_duration = max(min(duration1, duration2), 0.001)
    
    return overlap_duration / min_duration


def _generate_fallback_title(candidate: ScoredCandidate, max_words: int = 8) -> str:
    """
    Generate fallback title from candidate features/reasons.
    
    Args:
        candidate: Scored candidate with reasons
        max_words: Maximum words in title
    
    Returns:
        Generated title string
    """
    # Try to create a title from the scoring reasons
    reasons = candidate.reasons or []
    
    if "hook_question" in reasons:
        return "A Question Worth Asking"
    if "clean_resolution" in reasons:
        return "The Key Takeaway"
    if "hook_open_loop" in reasons:
        return "Wait For This"
    if "strong_energy_peak" in reasons:
        return "High Energy Moment"
    
    # Default fallback
    return f"Clip at {int(candidate.start_s // 60)}:{int(candidate.start_s % 60):02d}"


def enforce_diversity(
    candidates: List[ScoredCandidate],
    max_overlap: float = 0.35
) -> List[ScoredCandidate]:
    """
    Filter candidates to ensure diversity (no excessive overlap).
    
    Args:
        candidates: Candidates sorted by score descending
        max_overlap: Maximum allowed overlap ratio
    
    Returns:
        Filtered list with diverse candidates
    """
    diverse = []
    
    for candidate in candidates:
        has_conflict = False
        for selected in diverse:
            overlap = compute_overlap_ratio(
                candidate.start_s, candidate.end_s,
                selected.start_s, selected.end_s
            )
            if overlap > max_overlap:
                has_conflict = True
                break
        
        if not has_conflict:
            diverse.append(candidate)
    
    return diverse


def save_clips_selected(
    clips: List[SelectedClip],
    output_path: str
) -> None:
    """Save selected clips to JSON file."""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    data = {
        "selected": [c.to_dict() for c in clips]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(clips)} selected clips to '{output_path}'")
