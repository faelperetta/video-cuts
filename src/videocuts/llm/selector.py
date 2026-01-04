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
    # p = p.replace("15 and 90 seconds", f"{int(min_d)} and {int(max_d)} seconds") # REMOVED to rely on prompt's "15-180"
    
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
    if not segments: return []
    dur = segments[-1]["end"]
    
    # Optimization: If video is < 30 mins (1800s), send the whole thing in 1 shot.
    # This saves 66% of LLM calls for short-medium videos and gives better global context.
    if dur < 1800:
        logger.info(f"Video duration {dur:.1f}s < 30m. Using single pass for LLM selection.")
        return [("Full Video", format_transcript_for_llm(segments))]
        
    parts, ranges = [], [(0.0, 0.35), (0.30, 0.68), (0.63, 1.0)]
    for i, (sr, er) in enumerate(ranges):
        ps = [s for s in segments if s["start"] >= dur * sr and s["end"] <= dur * er]
        if ps: parts.append((f"Part {i+1}/{len(ranges)}", format_transcript_for_llm(ps)))
    return parts

def select_highlight_intervals_llm(
    segments: List[Dict],
    prompt_path: str,
    cfg: Config,
    model: str = "gpt-4o-mini"
) -> List[Dict]:
    """Multi-pass LLM selection strategy to find clips."""
    try: template = load_prompt_template(prompt_path)
    except: return []
    
    system_p = build_system_prompt(template, cfg.highlight.num_highlights, cfg.highlight.min_length, cfg.highlight.max_length, cfg.content_type)
    all_candidates = []
    
    for name, transcript in split_transcript_into_parts(segments):
        resp = call_openai_for_clips(system_p, transcript, model=model)
        all_candidates.extend(parse_llm_clip_response(resp, segments, cfg.highlight.min_length, cfg.highlight.max_length, cfg.highlight.last_word_pad))
    
    seen = []
    unique = []
    # Sort first by explicit "yes" on completion, then by viral score
    def sort_key(x):
        is_complete = 1 if x.get("idea_completion", "no").lower() == "yes" else 0
        return (is_complete, x["score"])

    # Log total found before filtering
    logger.info(f"LLM found {len(all_candidates)} candidates. Filtering top {cfg.highlight.num_highlights}...")

    for c in sorted(all_candidates, key=sort_key, reverse=True):
        if not any(abs(c["start"] - s["start"]) < 10 for s in seen):
            seen.append(c)
            unique.append(c)
            if len(unique) >= cfg.highlight.num_highlights: break
    return unique
