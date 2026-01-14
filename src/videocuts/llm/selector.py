import os
import json
import re
import logging
from typing import List, Dict, Tuple, Optional
from videocuts.config import Config
from videocuts.highlights.selector import extend_interval_to_last_word

logger = logging.getLogger(__name__)

CLIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "niche": {"type": "string"},
        "video_themes": {"type": "array", "items": {"type": "string"}},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_number": {"type": "number"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "start_substring": {"type": "string"},
                    "end_substring": {"type": "string"},
                    "hook": {"type": "string"},
                    "summary": {"type": "string"},
                    "viral_score": {"type": "number"},
                    "idea_completion": {"type": "string", "enum": ["yes", "no"]},
                    "hashtags": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "clip_number", "start_seconds", "end_seconds", "start_substring", "end_substring", 
                    "hook", "summary", "viral_score", "idea_completion", "hashtags"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["language", "niche", "video_themes", "clips"],
    "additionalProperties": False
}

def detect_llm_availability() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path): raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f: return f.read()

def format_timestamp_simple(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def format_transcript_for_llm(segments: List[Dict]) -> str:
    return "\n".join([f"[{int(s['start'])}] {s['text'].strip()}" for s in segments])

def build_system_prompt(template: str, num: int, min_d: float, max_d: float, niche: Optional[str] = None) -> str:
    # "3. Extract ALL high-potential viral clips" is now hardcoded in prompt.md, no replacement needed for count.
    # But for duration, we might want to guide it if config overrides.
    # However, for exhaustive search, we stick to the prompt's "Flexible" instruction unless debugging.
    p = template
    # p = template.replace("5–8 high-potential viral clips", f"{num} high-potential viral clips") # REMOVED
    p = p.replace("15 and 180 seconds", f"{int(min_d)} and {int(max_d)} seconds") # REMOVED to rely on prompt's "15-180"
    
    if niche: p = p.replace("{{NICHE}}", niche) if "{{NICHE}}" in p else p.replace("You are an expert short-form video editor", f"You are an expert short-form video editor specializing in {niche} content")
    return p

def call_openai_for_clips(system_p: str, transcript: str, model: str = "gpt-4o-mini", max_t: int = 4000, temp: float = 0.7) -> str:
    from openai import OpenAI
    client = OpenAI()
    is_reasoning = model.startswith("o")
    params = {"model": model, "messages": [{"role": "system", "content": system_p}, {"role": "user", "content": f"Analyze this video transcript and extract the best viral clips:\n\n{transcript}"}], "response_format": {"type": "json_schema", "json_schema": {"name": "viral_clips_extraction", "strict": True, "schema": CLIPS_SCHEMA}}}
    if is_reasoning: params["max_completion_tokens"] = max(max_t, 25000)
    else: params.update({"max_tokens": max_t, "temperature": temp})
    return client.chat.completions.create(**params).choices[0].message.content

def find_segment_by_text(segments: List[Dict], substring: str, start_search: float, window: float = 60.0) -> Optional[Dict]:
    """Find the segment containing the substring near the predicted time."""
    if not substring: return None
    clean_sub = substring.lower().strip()
    # Search within a window around the detailed timestamp
    search_start = max(0, start_search - window)
    search_end = start_search + window
    
    candidates = [s for s in segments if s["start"] >= search_start and s["start"] <= search_end]
    
    # Exact match first
    for s in candidates:
        if clean_sub in s["text"].lower():
            return s
            
    # Fuzzy match could go here if needed, but strict substring is safer for now
    return None

def parse_llm_clip_response(resp: str, segments: List[Dict], min_l: float, max_l: float, pad: float) -> List[Dict]:
    match = re.search(r'```json\s*([\s\S]*?)\s*```', resp)
    json_str = match.group(1) if match else resp[resp.find('{'):resp.rfind('}')+1]
    try: 
        data = json.loads(json_str)
    except Exception as e: 
        logger.error(f"Failed to parse LLM JSON: {e}")
        logger.debug(f"Raw response: {resp}")
        return []
    
    clips, dur = data.get("clips", []), max((s["end"] for s in segments), default=0.0)
    intervals = []
    for c in clips:
        s_raw = c.get("start_seconds", 0)
        e_raw = c.get("end_seconds", 0)
        start_sub = c.get("start_substring", "")
        end_sub = c.get("end_substring", "")
        
        # Ensure we have floats
        try:
            s = float(s_raw)
            e = float(e_raw)
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp types from LLM: start={s_raw}, end={e_raw}. Skipping clip.")
            continue

        logger.info(f"LLM proposed: {s:.2f}s - {e:.2f}s")
        
        # Refine Start
        s_seg = find_segment_by_text(segments, start_sub, s)
        if s_seg:
            logger.info(f"  Matched start text '{start_sub}' to segment at {s_seg['start']:.2f}s")
            s = s_seg['start']
        else:
            logger.warning(f"  Start text '{start_sub}' not found near {s:.2f}s, using raw timestamp.")
            
        # Refine End
        e_seg = find_segment_by_text(segments, end_sub, e)
        if e_seg:
            logger.info(f"  Matched end text '{end_sub}' to segment at {e_seg['end']:.2f}s")
            e = e_seg['end']
        else:
            logger.warning(f"  End text '{end_sub}' not found near {e:.2f}s, using raw timestamp.")
            # Fallback alignment
            e_aligned = extend_interval_to_last_word(segments, s, min(e, dur), pad, dur)
            e = e_aligned

        vs = float(c.get("viral_score", 5.0))
        intervals.append({
            "start": s, 
            "end": e, 
            "raw_start": float(s_raw) if s_raw is not None else 0.0,
            "raw_end": float(e_raw) if e_raw is not None else 0.0,
            "score": vs, 
            "viral_score": vs, 
            "text": c.get("hook", ""), 
            "hook": c.get("hook", ""), 
            "summary": c.get("summary", ""), 
            "idea_completion": c.get("idea_completion", None), 
            "hashtags": c.get("hashtags", None)
        })
    return intervals

def split_transcript_into_parts(segments: List[Dict]) -> List[Tuple[str, str]]:
    """
    DEPRECATED: Use split_transcript_into_chunks() for better scalability.
    Kept for backward compatibility.
    """
    if not segments: return []
    dur = segments[-1]["end"]
    
    # Short videos: single pass
    if dur < 1800:
        logger.info(f"Video duration {dur:.1f}s < 30m. Using single pass for LLM selection.")
        return [("Full Video", format_transcript_for_llm(segments))]
        
    parts, ranges = [], [(0.0, 0.35), (0.30, 0.68), (0.63, 1.0)]
    for i, (sr, er) in enumerate(ranges):
        ps = [s for s in segments if s["start"] >= dur * sr and s["end"] <= dur * er]
        if ps: parts.append((f"Part {i+1}/{len(ranges)}", format_transcript_for_llm(ps)))
    return parts


# =============================================================================
# Scalable Chunking for Long Videos
# =============================================================================

def split_transcript_into_chunks(
    segments: List[Dict], 
    chunk_duration_s: float = 600.0,  # 10 minutes
    min_silence_gap: float = 1.5
) -> List[Dict]:
    """
    Split transcript into chunks by time boundaries, preferring natural pauses.
    
    Args:
        segments: List of transcript segments with start, end, text
        chunk_duration_s: Target duration per chunk in seconds (default: 10 min)
        min_silence_gap: Minimum gap between segments to consider a natural break
    
    Returns:
        List of chunk dicts: {chunk_index, start_s, end_s, segments, text}
    """
    if not segments:
        return []
    
    total_duration = segments[-1]["end"]
    chunks = []
    current_chunk_segments = []
    chunk_start = 0.0
    chunk_index = 0
    
    for i, seg in enumerate(segments):
        current_chunk_segments.append(seg)
        chunk_duration = seg["end"] - chunk_start
        
        # Check if we should end this chunk
        should_split = False
        
        if chunk_duration >= chunk_duration_s:
            # Look for natural break point (silence gap)
            if i + 1 < len(segments):
                gap = segments[i + 1]["start"] - seg["end"]
                if gap >= min_silence_gap:
                    should_split = True
                # Even without silence, split if way over target
                elif chunk_duration >= chunk_duration_s * 1.3:
                    should_split = True
            else:
                should_split = True  # Last segment
        
        if should_split and current_chunk_segments:
            chunk_text = " ".join([s["text"].strip() for s in current_chunk_segments])
            chunks.append({
                "chunk_index": chunk_index,
                "start_s": chunk_start,
                "end_s": seg["end"],
                "segments": current_chunk_segments.copy(),
                "text": chunk_text
            })
            chunk_index += 1
            chunk_start = segments[i + 1]["start"] if i + 1 < len(segments) else seg["end"]
            current_chunk_segments = []
    
    # Handle remaining segments
    if current_chunk_segments:
        chunk_text = " ".join([s["text"].strip() for s in current_chunk_segments])
        chunks.append({
            "chunk_index": chunk_index,
            "start_s": chunk_start,
            "end_s": current_chunk_segments[-1]["end"],
            "segments": current_chunk_segments,
            "text": chunk_text
        })
    
    logger.info(f"Split {total_duration:.0f}s video into {len(chunks)} chunks")
    return chunks


def summarize_chunk_for_llm(chunk: Dict, max_segments: int = 50) -> str:
    """
    Create a condensed summary of a chunk for LLM processing.
    
    Preserves the [SECONDS] format the LLM prompt expects for timestamp extraction.
    Returns format like:
    
    [CHUNK 3: 20:00-30:00]
    [1200] First segment text...
    [1205] Second segment text...
    """
    start_ts = format_timestamp_simple(chunk["start_s"])
    end_ts = format_timestamp_simple(chunk["end_s"])
    
    # Format segments with timestamps (same format as format_transcript_for_llm)
    segments = chunk.get("segments", [])
    
    # Limit segments if there are too many (prioritize evenly distributed samples)
    if len(segments) > max_segments:
        # Sample evenly across the chunk
        step = len(segments) / max_segments
        indices = [int(i * step) for i in range(max_segments)]
        segments = [segments[i] for i in indices]
    
    formatted_segments = [f"[{int(s['start'])}] {s['text'].strip()}" for s in segments]
    transcript_text = "\n".join(formatted_segments)
    
    return f"[CHUNK {chunk['chunk_index'] + 1}: {start_ts}-{end_ts}]\n{transcript_text}"


def format_chunks_for_llm(chunks: List[Dict]) -> str:
    """Format all chunks into a single string for LLM processing."""
    summaries = [summarize_chunk_for_llm(c) for c in chunks]
    return "\n\n".join(summaries)

def select_highlight_intervals_llm(
    segments: List[Dict],
    prompt_path: str,
    cfg: Config,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """Multi-pass LLM selection strategy to find clips with round-robin distribution."""
    try: template = load_prompt_template(prompt_path)
    except: return []
    
    system_p = build_system_prompt(template, cfg.highlight.num_highlights, cfg.highlight.min_length, cfg.highlight.max_length, cfg.content_type)
    
    # Organize candidates by part to allow rotation
    parts_candidates = []
    
    for name, transcript in split_transcript_into_parts(segments):
        resp = call_openai_for_clips(system_p, transcript, model=model)
        candidates = parse_llm_clip_response(resp, segments, cfg.highlight.min_length, cfg.highlight.max_length, cfg.highlight.last_word_pad)
        
        logger.info(f"LLM found {len(candidates)} candidates in {name}.")
        
        # Sort candidates within this part first
        def sort_key(x):
            is_complete = 1 if x.get("idea_completion", "no").lower() == "yes" else 0
            return (is_complete, x["score"])
        
        candidates.sort(key=sort_key, reverse=True)
        parts_candidates.append(candidates)
    
    seen = []
    unique = []
    
    # Implementation of Round-Robin selection across parts
    num_parts = len(parts_candidates)
    if num_parts == 0:
        return []
        
    part_indices = [0] * num_parts # Current pointer for each part's list
    
    logger.info(f"Rotating selection across {num_parts} video parts for top {cfg.highlight.num_highlights} clips...")

    while len(unique) < cfg.highlight.num_highlights:
        any_added_this_loop = False
        
        for p_idx in range(num_parts):
            # Check if this part still has candidates to pick from
            if part_indices[p_idx] < len(parts_candidates[p_idx]):
                candidate = parts_candidates[p_idx][part_indices[p_idx]]
                part_indices[p_idx] += 1
                any_added_this_loop = True
                
                # Check for temporal overlap (avoiding clips within 10s of each other)
                if not any(abs(candidate["start"] - s["start"]) < 10 for s in seen):
                    seen.append(candidate)
                    unique.append(candidate)
                    logger.info(f"  [Round-Robin] Selected clip starting at {candidate['start']:.2f}s from Part {p_idx+1}/{num_parts} (Score: {candidate['score']})")
                    if len(unique) >= cfg.highlight.num_highlights:
                        break
                else:
                    logger.debug(f"  [Round-Robin] Skipped overlapping clip at {candidate['start']:.2f}s from Part {p_idx+1}")
        
        # If we went through all parts and didn't find a single new candidate, we're done
        if not any_added_this_loop:
            break
            
    return unique


# =============================================================================
# Scalable LLM Selection (v2)
# =============================================================================

def select_highlight_intervals_llm_v2(
    segments: List[Dict],
    prompt_path: str,
    cfg: Config,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """
    Scalable LLM selection for long videos using hierarchical summarization.
    
    Strategy:
    - Video < 30 min: Single-pass with full transcript (same as v1)
    - Video 30 min - 2 hr: Chunked summaries, single LLM call
    - Video > 2 hr: Same as above but with smaller chunks
    """
    if not segments:
        return []
    
    total_duration = segments[-1]["end"]
    
    try:
        template = load_prompt_template(prompt_path)
    except Exception:
        logger.error(f"Failed to load prompt template from {prompt_path}")
        return []
    
    system_p = build_system_prompt(
        template, 
        cfg.highlight.num_highlights, 
        cfg.highlight.min_length, 
        cfg.highlight.max_length, 
        cfg.content_type
    )
    
    # Choose chunk duration based on video length
    if total_duration < 1800:  # < 30 minutes
        chunk_duration_s = 300  # 5 min chunks for short videos
        logger.info(f"Video {total_duration:.0f}s < 30min: Using 5-min chunks")
    elif total_duration < 7200:  # 30 min - 2 hours
        chunk_duration_s = 600  # 10 min chunks
        logger.info(f"Video {total_duration:.0f}s (30min-2hr): Using 10-min chunks")
    else:  # > 2 hours
        chunk_duration_s = 480  # 8 min chunks for very long videos
        logger.info(f"Video {total_duration:.0f}s (> 2hr): Using 8-min chunks")
    
    # Always use chunked approach with round-robin
    chunks = split_transcript_into_chunks(segments, chunk_duration_s=chunk_duration_s)
    transcript = format_chunks_for_llm(chunks)
    
    logger.info(f"Sending {len(chunks)} chunk summaries to LLM (~{len(transcript)} chars)")
    resp = call_openai_for_clips(system_p, transcript, model=model)
    candidates = parse_llm_clip_response(
        resp, segments,
        cfg.highlight.min_length, cfg.highlight.max_length,
        cfg.highlight.last_word_pad
    )
    
    logger.info(f"LLM returned {len(candidates)} clip candidates")
    
    # Round-robin across chunks (for all video lengths)
    # Group candidates by which chunk they belong to
    num_chunks = int(total_duration / chunk_duration_s) + 1
    
    chunk_candidates = [[] for _ in range(num_chunks)]
    for candidate in candidates:
        chunk_idx = min(int(candidate["start"] / chunk_duration_s), num_chunks - 1)
        chunk_candidates[chunk_idx].append(candidate)
    
    # Sort within each chunk by score
    def sort_key(x):
        is_complete = 1 if x.get("idea_completion", "no").lower() == "yes" else 0
        return (is_complete, x.get("score", 0))
    
    for chunk_list in chunk_candidates:
        chunk_list.sort(key=sort_key, reverse=True)
    
    # Round-robin selection across chunks
    unique = []
    chunk_indices = [0] * num_chunks  # Pointer for each chunk
    chunks_with_candidates = [i for i in range(num_chunks) if chunk_candidates[i]]
    
    logger.info(f"Round-robin selection across {len(chunks_with_candidates)} chunks with candidates")
    
    while len(unique) < cfg.highlight.num_highlights:
        any_added = False
        
        for chunk_idx in chunks_with_candidates:
            if chunk_indices[chunk_idx] < len(chunk_candidates[chunk_idx]):
                candidate = chunk_candidates[chunk_idx][chunk_indices[chunk_idx]]
                chunk_indices[chunk_idx] += 1
                any_added = True
                
                # Check for overlap with already selected clips
                if not any(abs(candidate["start"] - u["start"]) < 10 for u in unique):
                    unique.append(candidate)
                    chunk_time = f"{int(chunk_idx * chunk_duration_s / 60)}-{int((chunk_idx + 1) * chunk_duration_s / 60)}min"
                    logger.info(f"  [Chunk {chunk_idx + 1}] Selected clip at {candidate['start']:.2f}s (Score: {candidate.get('score', 0)})")
                    if len(unique) >= cfg.highlight.num_highlights:
                        break
        
        if not any_added:
            break
    
    return unique
