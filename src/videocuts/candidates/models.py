"""Data models for Epic 3 clip candidate discovery pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class CandidateWindow:
    """Raw candidate clip window (US-3.1)."""
    candidate_id: str
    start_s: float
    end_s: float
    duration_s: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateFeatures:
    """Computed features for a candidate (US-3.2)."""
    candidate_id: str
    start_s: float
    end_s: float
    duration_s: float
    
    # Hook region features (first 3s)
    hook_question: bool = False
    hook_open_loop: bool = False
    hook_contrast: bool = False
    
    # Close region features (last 4s)
    close_resolution: bool = False
    ends_on_sentence_boundary: bool = False
    ends_with_question: bool = False
    
    # Speaker features
    speaker_dominance_ratio: float = 0.0
    speaker_switch_count: int = 0
    
    # Audio features
    overlap_ratio: float = 0.0
    energy_peak_ratio: float = 1.0
    speech_coverage: float = 1.0
    
    # Silence features  
    silence_at_start_s: float = 0.0
    silence_at_end_s: float = 0.0
    
    # Boundary quality
    starts_mid_sentence: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredCandidate:
    """Scored candidate with eligibility status (US-3.3)."""
    candidate_id: str
    start_s: float
    end_s: float
    duration_s: float
    
    # Scoring
    eligible: bool = True
    score: float = 0.0
    
    # Component scores
    hook_score: float = 0.0
    continuity_score: float = 0.0
    energy_score: float = 0.0
    closure_score: float = 0.0
    
    # Human-readable reasons
    reasons: List[str] = field(default_factory=list)
    disqualify_reasons: List[str] = field(default_factory=list)
    
    # Original features (for LLM context)
    features: Optional[CandidateFeatures] = None
    
    def to_dict(self) -> dict:
        d = {
            "candidate_id": self.candidate_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "eligible": self.eligible,
            "score": self.score,
            "hook_score": self.hook_score,
            "continuity_score": self.continuity_score,
            "energy_score": self.energy_score,
            "closure_score": self.closure_score,
            "reasons": self.reasons,
            "disqualify_reasons": self.disqualify_reasons,
        }
        if self.features:
            d["features"] = self.features.to_dict()
        return d


@dataclass
class LLMResult:
    """LLM reranking result for a candidate (US-3.4)."""
    candidate_id: str
    llm_score: float  # 0-10
    title: str
    hook_line: str
    self_contained: bool
    has_clear_payoff: bool
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectedClip:
    """Final selected clip for rendering (US-3.5)."""
    candidate_id: str
    start_s: float
    end_s: float
    duration_s: float
    final_rank: int
    final_score: float
    title: str
    
    # LLM enhancements (optional)
    hook_line: Optional[str] = None
    llm_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RefinedClip:
    """Refined clip with adjusted boundaries (US-3.6)."""
    candidate_id: str
    start_s_before: float
    end_s_before: float
    start_s_after: float
    end_s_after: float
    duration_s: float
    changes: List[str] = field(default_factory=list)
    
    # Metadata
    final_rank: int = 0
    final_score: float = 0.0
    title: str = ""
    hook_line: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Timeline Models
# =============================================================================

@dataclass
class TimelineSegment:
    """Enriched transcript segment with audio features."""
    id: int
    start_s: float
    end_s: float
    text: str
    
    # Speaker info (from diarization)
    speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    has_overlap: bool = False
    
    # Audio features
    rms_energy: float = 0.0
    speech_rate_wps: float = 0.0  # words per second
    
    # Silence boundaries
    silence_before_s: float = 0.0
    silence_after_s: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AudioFeatures:
    """Global audio feature analysis."""
    global_rms: float = 0.0
    rms_per_second: List[float] = field(default_factory=list)
    peaks: List[float] = field(default_factory=list)
    duration_s: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Timeline:
    """Complete analysis-ready timeline (US-3.0)."""
    duration_s: float
    segments: List[TimelineSegment] = field(default_factory=list)
    audio_features: Optional[AudioFeatures] = None
    
    def to_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "segments": [s.to_dict() for s in self.segments],
            "audio_features": self.audio_features.to_dict() if self.audio_features else None,
        }
    
    def save(self, path: str) -> None:
        """Save timeline to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "Timeline":
        """Load timeline from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = [
            TimelineSegment(**{k: v for k, v in s.items()})
            for s in data.get("segments", [])
        ]
        
        audio_features = None
        if data.get("audio_features"):
            audio_features = AudioFeatures(**data["audio_features"])
        
        return cls(
            duration_s=data["duration_s"],
            segments=segments,
            audio_features=audio_features,
        )


# =============================================================================
# Diarization Models
# =============================================================================

@dataclass
class SpeakerTurn:
    """A speaker turn from diarization."""
    speaker: str
    start_s: float
    end_s: float
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OverlapRegion:
    """Region where multiple speakers overlap."""
    start_s: float
    end_s: float
    speakers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiarizationResult:
    """Complete diarization output."""
    speaker_turns: List[SpeakerTurn] = field(default_factory=list)
    overlaps: List[OverlapRegion] = field(default_factory=list)
    num_speakers: int = 0
    duration_s: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "speaker_turns": [t.to_dict() for t in self.speaker_turns],
            "overlaps": [o.to_dict() for o in self.overlaps],
            "num_speakers": self.num_speakers,
            "duration_s": self.duration_s,
        }
    
    def save(self, path: str) -> None:
        """Save diarization to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "DiarizationResult":
        """Load diarization from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        speaker_turns = [SpeakerTurn(**t) for t in data.get("speaker_turns", [])]
        overlaps = [OverlapRegion(**o) for o in data.get("overlaps", [])]
        
        return cls(
            speaker_turns=speaker_turns,
            overlaps=overlaps,
            num_speakers=data.get("num_speakers", 0),
            duration_s=data.get("duration_s", 0.0),
        )
