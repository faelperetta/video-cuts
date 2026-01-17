"""Heuristic scorer for clip candidates (Epic 3 - US-3.3).

This module scores candidates using deterministic heuristics
without requiring LLM calls. It applies hard filters to discard
invalid candidates and computes a weighted score (0-100).

Hard Filters (discard if any true):
- starts_mid_sentence
- silence_at_start_s > 0.8
- silence_at_end_s > 1.0
- speaker_dominance_ratio < 0.70
- overlap_ratio > 0.25
- speech_coverage < 0.75

Scoring Components (0-100):
- hook_score (0-30): question, open_loop, contrast
- continuity_score (0-25): speaker dominance, switches
- energy_score (0-20): energy peak ratio
- closure_score (0-25): resolution, sentence boundary
"""
from __future__ import annotations
import logging
from typing import List, Tuple

from videocuts.config import Config
from videocuts.candidates.models import CandidateFeatures, ScoredCandidate

logger = logging.getLogger(__name__)


# Default thresholds
DEFAULT_MAX_SILENCE_START = 0.8
DEFAULT_MAX_SILENCE_END = 1.0
DEFAULT_MIN_SPEAKER_DOMINANCE = 0.70
DEFAULT_MAX_OVERLAP = 0.25
DEFAULT_MIN_SPEECH_COVERAGE = 0.75


def apply_hard_filters(
    features: CandidateFeatures,
    cfg: Config = None
) -> Tuple[bool, List[str]]:
    """
    Apply hard filters to determine if candidate is eligible.
    
    Args:
        features: Candidate features
        cfg: Configuration (optional, uses defaults if None)
    
    Returns:
        (passes, disqualify_reasons) - passes=True if all filters pass
    """
    reasons = []
    
    # Filter 1: Starts mid-sentence
    if features.starts_mid_sentence:
        reasons.append("starts_mid_sentence")
    
    # Filter 2: Too much silence at start
    if features.silence_at_start_s > DEFAULT_MAX_SILENCE_START:
        reasons.append(f"silence_start_too_long ({features.silence_at_start_s:.2f}s > {DEFAULT_MAX_SILENCE_START}s)")
    
    # Filter 3: Too much silence at end
    if features.silence_at_end_s > DEFAULT_MAX_SILENCE_END:
        reasons.append(f"silence_end_too_long ({features.silence_at_end_s:.2f}s > {DEFAULT_MAX_SILENCE_END}s)")
    
    # Filter 4: Low speaker dominance (too much back-and-forth)
    if features.speaker_dominance_ratio < DEFAULT_MIN_SPEAKER_DOMINANCE:
        reasons.append(f"low_speaker_dominance ({features.speaker_dominance_ratio:.2f} < {DEFAULT_MIN_SPEAKER_DOMINANCE})")
    
    # Filter 5: Too much overlapping speech
    if features.overlap_ratio > DEFAULT_MAX_OVERLAP:
        reasons.append(f"high_overlap_ratio ({features.overlap_ratio:.2f} > {DEFAULT_MAX_OVERLAP})")
    
    # Filter 6: Too little speech coverage
    if features.speech_coverage < DEFAULT_MIN_SPEECH_COVERAGE:
        reasons.append(f"low_speech_coverage ({features.speech_coverage:.2f} < {DEFAULT_MIN_SPEECH_COVERAGE})")
    
    passes = len(reasons) == 0
    return passes, reasons


def compute_score(features: CandidateFeatures) -> Tuple[float, List[str]]:
    """
    Compute weighted score (0-100) for a candidate.
    
    Args:
        features: Candidate features
    
    Returns:
        (score, reasons) - score between 0-100 and list of scoring reasons
    """
    reasons = []
    
    # Hook Score (0-30)
    hook_score = 0.0
    if features.hook_question:
        hook_score += 15.0
        reasons.append("hook_question")
    if features.hook_open_loop:
        hook_score += 10.0
        reasons.append("hook_open_loop")
    if features.hook_contrast:
        hook_score += 5.0
        reasons.append("hook_contrast")
    
    # Continuity Score (0-25)
    # +25 * clamp((speaker_dominance - 0.70) / 0.30, 0, 1)
    dominance_factor = _clamp((features.speaker_dominance_ratio - 0.70) / 0.30, 0, 1)
    continuity_score = 25.0 * dominance_factor
    
    # Penalty for speaker switches: -5 * min(switch_count, 3)
    switch_penalty = 5.0 * min(features.speaker_switch_count, 3)
    continuity_score = max(continuity_score - switch_penalty, 0)
    
    if dominance_factor > 0.5:
        reasons.append("high_speaker_dominance")
    if features.speaker_switch_count <= 1:
        reasons.append("few_speaker_switches")
    
    # Energy Score (0-20)
    # +20 * clamp((energy_peak - 1.0) / 1.0, 0, 1)
    energy_factor = _clamp((features.energy_peak_ratio - 1.0) / 1.0, 0, 1)
    energy_score = 20.0 * energy_factor
    
    if energy_factor > 0.5:
        reasons.append("strong_energy_peak")
    
    # Closure Score (0-25)
    closure_score = 0.0
    if features.close_resolution:
        closure_score += 15.0
        reasons.append("clean_resolution")
    if features.ends_on_sentence_boundary:
        closure_score += 10.0
        reasons.append("ends_on_sentence")
    
    # Penalty: -10 if ends with question (often implies unresolved)
    if features.ends_with_question:
        closure_score = max(closure_score - 10.0, 0)
        reasons.append("ends_with_question_penalty")
    
    # Total score
    total_score = hook_score + continuity_score + energy_score + closure_score
    
    # Clamp to 0-100
    total_score = _clamp(total_score, 0, 100)
    
    return round(total_score, 2), reasons


def score_candidate(features: CandidateFeatures, cfg: Config = None) -> ScoredCandidate:
    """
    Score a single candidate with filters and scoring.
    
    Args:
        features: Candidate features
        cfg: Configuration (optional)
    
    Returns:
        ScoredCandidate with eligibility and score
    """
    # Apply hard filters
    eligible, disqualify_reasons = apply_hard_filters(features, cfg)
    
    # Compute score (even for ineligible candidates for debugging)
    score, score_reasons = compute_score(features)
    
    # Compute component scores for transparency
    hook_score = 0.0
    if features.hook_question:
        hook_score += 15.0
    if features.hook_open_loop:
        hook_score += 10.0
    if features.hook_contrast:
        hook_score += 5.0
    
    dominance_factor = _clamp((features.speaker_dominance_ratio - 0.70) / 0.30, 0, 1)
    continuity_score = 25.0 * dominance_factor
    continuity_score = max(continuity_score - 5.0 * min(features.speaker_switch_count, 3), 0)
    
    energy_factor = _clamp((features.energy_peak_ratio - 1.0) / 1.0, 0, 1)
    energy_score = 20.0 * energy_factor
    
    closure_score = 0.0
    if features.close_resolution:
        closure_score += 15.0
    if features.ends_on_sentence_boundary:
        closure_score += 10.0
    if features.ends_with_question:
        closure_score = max(closure_score - 10.0, 0)
    
    return ScoredCandidate(
        candidate_id=features.candidate_id,
        start_s=features.start_s,
        end_s=features.end_s,
        duration_s=features.duration_s,
        eligible=eligible,
        score=score if eligible else 0.0,
        hook_score=round(hook_score, 2),
        continuity_score=round(continuity_score, 2),
        energy_score=round(energy_score, 2),
        closure_score=round(closure_score, 2),
        reasons=score_reasons,
        disqualify_reasons=disqualify_reasons,
        features=features,
    )


def score_all_candidates(
    features_list: List[CandidateFeatures],
    cfg: Config = None
) -> List[ScoredCandidate]:
    """
    Score all candidates and sort by score descending.
    
    Args:
        features_list: List of candidate features
        cfg: Configuration (optional)
    
    Returns:
        List of scored candidates, sorted by score descending
    """
    logger.info(f"Scoring {len(features_list)} candidates...")
    
    scored = []
    eligible_count = 0
    
    for features in features_list:
        candidate = score_candidate(features, cfg)
        scored.append(candidate)
        if candidate.eligible:
            eligible_count += 1
    
    # Sort by score descending (eligible first, then by score)
    scored.sort(key=lambda c: (c.eligible, c.score), reverse=True)
    
    logger.info(f"Scored {len(scored)} candidates: {eligible_count} eligible, {len(scored) - eligible_count} filtered out")
    
    if scored and scored[0].eligible:
        logger.info(f"Top score: {scored[0].score:.1f} ({scored[0].candidate_id})")
    
    return scored


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def save_candidates_scored(
    scored: List[ScoredCandidate],
    output_path: str,
    include_ineligible: bool = True
) -> None:
    """
    Save scored candidates to JSON file.
    
    Args:
        scored: List of scored candidates
        output_path: Output file path
        include_ineligible: Whether to include ineligible candidates
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    candidates_to_save = scored
    if not include_ineligible:
        candidates_to_save = [c for c in scored if c.eligible]
    
    data = {
        "candidates": [c.to_dict() for c in candidates_to_save]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(candidates_to_save)} scored candidates to '{output_path}'")
