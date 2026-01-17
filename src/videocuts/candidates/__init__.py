# Epic 3 - Clip Candidate Discovery Package
"""
This package implements the hybrid clip candidate discovery system.

Modules:
- models: Data models for candidates and features
- timeline: Timeline builder from transcript + audio + diarization
- generator: Sliding window candidate generation
- features: Hook/middle/end feature computation
- scorer: Deterministic heuristic scoring
- llm_reranker: Optional LLM reranking
- selection: Final clip selection with diversity
- refinement: Clip boundary refinement

Note: To avoid circular imports, import functions directly from submodules:
    from videocuts.candidates.timeline import build_timeline
    from videocuts.candidates.generator import generate_candidates
"""

# Only export models at package level (no circular import risk)
from videocuts.candidates.models import (
    CandidateWindow,
    CandidateFeatures,
    ScoredCandidate,
    SelectedClip,
    RefinedClip,
    Timeline,
    TimelineSegment,
    AudioFeatures,
    DiarizationResult,
    SpeakerTurn,
    OverlapRegion,
    LLMResult,
)

__all__ = [
    # Models
    "CandidateWindow",
    "CandidateFeatures", 
    "ScoredCandidate",
    "SelectedClip",
    "RefinedClip",
    "Timeline",
    "TimelineSegment",
    "AudioFeatures",
    "DiarizationResult",
    "SpeakerTurn",
    "OverlapRegion",
    "LLMResult",
]
