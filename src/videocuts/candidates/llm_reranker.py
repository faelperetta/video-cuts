"""LLM reranker for top candidates (Epic 3 - US-3.4).

This module optionally reranks the top N heuristic candidates
using an LLM to assess "clip worthiness" and generate titles.

Only runs on top 12 candidates by default to control cost/latency.
The pipeline must work without this module.
"""
from __future__ import annotations
import os
import json
import re
import logging
from typing import List, Dict, Optional

from videocuts.config import Config
from videocuts.candidates.models import (
    ScoredCandidate,
    LLMResult,
    Timeline,
)
from videocuts.candidates.generator import get_text_in_window

logger = logging.getLogger(__name__)


# LLM rerank prompt template
RERANK_SYSTEM_PROMPT = """You are a short-form video editor assistant. Your job is to evaluate whether a transcript excerpt will perform well as a standalone TikTok/YouTube Shorts clip. Be strict: prefer clips with a strong hook, clear payoff, and clean ending. Avoid clips that require missing context."""

RERANK_USER_PROMPT_TEMPLATE = """Evaluate the following podcast clip candidate and return JSON only.

Candidate metadata:
- start_s: {start_s}
- end_s: {end_s}
- duration_s: {duration_s}
- speaker_dominance_ratio: {speaker_dominance_ratio}
- speaker_switch_count: {speaker_switch_count}
- overlap_ratio: {overlap_ratio}
- energy_peak_ratio: {energy_peak_ratio}

Transcript excerpt:
\"\"\"
{transcript_text}
\"\"\"

Requirements:
1) Determine if the clip has a complete start–middle–end arc and is understandable without external context.
2) Assign an llm_score from 0 to 10 (10 = very likely to perform well).
3) Write a short title (max 8 words) that describes the idea without misleading clickbait.
4) Write a hook_line (max 8 words) suitable as the first caption on screen.
5) Provide brief notes (max 3 bullets) why it is good/bad.
6) If the clip ends mid-thought, heavily penalize the score.

Return JSON with this exact schema:
{{
  "llm_score": number,
  "title": string,
  "hook_line": string,
  "self_contained": boolean,
  "has_clear_payoff": boolean,
  "notes": [string, string, string]
}}"""


def rerank_with_llm(
    scored_candidates: List[ScoredCandidate],
    timeline: Timeline,
    cfg: Config,
    top_n: Optional[int] = None
) -> Dict[str, LLMResult]:
    """
    Rerank top candidates using LLM.
    
    Args:
        scored_candidates: Candidates sorted by heuristic score
        timeline: Timeline for transcript text extraction
        cfg: Configuration with LLM settings
        top_n: Number of candidates to rerank (default from config)
    
    Returns:
        Dict mapping candidate_id to LLMResult
    """
    if not cfg.llm.enabled:
        logger.info("LLM reranking disabled")
        return {}
    
    # Check LLM availability
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM rerank")
        return {}
    
    top_n = top_n or cfg.llm.rerank_top_n
    
    # Get top N eligible candidates
    eligible = [c for c in scored_candidates if c.eligible][:top_n]
    
    if not eligible:
        logger.warning("No eligible candidates for LLM reranking")
        return {}
    
    logger.info(f"Reranking top {len(eligible)} candidates with LLM...")
    
    results = {}
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        logger.error("OpenAI package not installed")
        return {}
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return {}
    
    for candidate in eligible:
        try:
            result = _rerank_single_candidate(
                candidate, timeline, client, cfg
            )
            if result:
                results[candidate.candidate_id] = result
        except Exception as e:
            logger.warning(f"LLM rerank failed for {candidate.candidate_id}: {e}")
            continue
    
    logger.info(f"LLM reranked {len(results)} candidates")
    return results


def _rerank_single_candidate(
    candidate: ScoredCandidate,
    timeline: Timeline,
    client,
    cfg: Config
) -> Optional[LLMResult]:
    """
    Rerank a single candidate using LLM.
    
    Args:
        candidate: Scored candidate
        timeline: Timeline for text extraction
        client: OpenAI client
        cfg: Configuration
    
    Returns:
        LLMResult or None if failed
    """
    # Get transcript text for this candidate
    transcript_text = get_text_in_window(
        timeline, candidate.start_s, candidate.end_s
    )
    
    if not transcript_text.strip():
        logger.warning(f"No transcript text for {candidate.candidate_id}")
        return None
    
    # Get feature values (from stored features)
    features = candidate.features
    if not features:
        speaker_dominance_ratio = 0.8
        speaker_switch_count = 0
        overlap_ratio = 0.0
        energy_peak_ratio = 1.0
    else:
        speaker_dominance_ratio = features.speaker_dominance_ratio
        speaker_switch_count = features.speaker_switch_count
        overlap_ratio = features.overlap_ratio
        energy_peak_ratio = features.energy_peak_ratio
    
    # Build prompt
    user_prompt = RERANK_USER_PROMPT_TEMPLATE.format(
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        duration_s=candidate.duration_s,
        speaker_dominance_ratio=speaker_dominance_ratio,
        speaker_switch_count=speaker_switch_count,
        overlap_ratio=overlap_ratio,
        energy_peak_ratio=energy_peak_ratio,
        transcript_text=transcript_text[:3000]  # Limit transcript length
    )
    
    # Call LLM
    try:
        response = client.chat.completions.create(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=cfg.llm.rerank_temperature,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result_data = json.loads(content)
        
        return LLMResult(
            candidate_id=candidate.candidate_id,
            llm_score=float(result_data.get("llm_score", 5.0)),
            title=result_data.get("title", "")[:50],
            hook_line=result_data.get("hook_line", "")[:50],
            self_contained=result_data.get("self_contained", False),
            has_clear_payoff=result_data.get("has_clear_payoff", False),
            notes=result_data.get("notes", [])[:3]
        )
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for {candidate.candidate_id}: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM call failed for {candidate.candidate_id}: {e}")
        return None


def save_llm_results(
    results: Dict[str, LLMResult],
    output_path: str
) -> None:
    """Save LLM rerank results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    data = {
        "candidates": [r.to_dict() for r in results.values()]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results)} LLM results to '{output_path}'")
