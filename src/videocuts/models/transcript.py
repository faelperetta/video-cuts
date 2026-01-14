"""Transcript data models matching Epic 2 JSON schema (US-2.1)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class Word:
    """Single word with timestamps."""
    start_s: float
    end_s: float
    word: str


@dataclass
class Segment:
    """Transcript segment with word-level detail."""
    id: int
    start_s: float
    end_s: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """
    Complete transcript with metadata.
    
    Matches the Epic 2 JSON schema:
    {
      "provider": "local_faster_whisper",
      "model": "large-v3",
      "language": "en",
      "duration_s": 3723.52,
      "segments": [...]
    }
    """
    provider: str
    model: str
    language: str
    duration_s: float
    segments: List[Segment] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save transcript to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "Transcript":
        """Load transcript from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = []
        for s in data.get("segments", []):
            words = [
                Word(
                    start_s=w["start_s"],
                    end_s=w["end_s"],
                    word=w["word"]
                )
                for w in s.get("words", [])
            ]
            segments.append(Segment(
                id=s["id"],
                start_s=s["start_s"],
                end_s=s["end_s"],
                text=s["text"],
                avg_logprob=s.get("avg_logprob", 0.0),
                no_speech_prob=s.get("no_speech_prob", 0.0),
                words=words
            ))
        
        return cls(
            provider=data["provider"],
            model=data["model"],
            language=data["language"],
            duration_s=data["duration_s"],
            segments=segments
        )
    
    def to_legacy_segments(self) -> List[dict]:
        """
        Convert to legacy segment format for backward compatibility.
        
        Returns list of dicts with: id, start, end, text
        """
        return [
            {
                "id": seg.id,
                "start": seg.start_s,
                "end": seg.end_s,
                "text": seg.text
            }
            for seg in self.segments
        ]
