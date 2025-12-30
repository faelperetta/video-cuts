import os
import json
import re
from typing import List, Dict, Tuple, Optional
from videocuts.config import Config
from videocuts.highlights.selector import extend_interval_to_last_word

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
                    "hook": {"type": "string"},
                    "summary": {"type": "string"},
                    "viral_score": {"type": "number"},
                    "idea_completion": {"type": "string", "enum": ["yes", "no"]},
                    "hashtags": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "clip_number", "start_seconds", "end_seconds", 
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
    return "\n".join([f"[{format_timestamp_simple(s['start'])}] {s['text'].strip()}" for s in segments])

def build_system_prompt(template: str, num: int, min_d: float, max_d: float, niche: Optional[str] = None) -> str:
    p = template.replace("5–8 high-potential viral clips", f"{num} high-potential viral clips")
    p = p.replace("15 and 90 seconds", f"{int(min_d)} and {int(max_d)} seconds")
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

def parse_llm_clip_response(resp: str, segments: List[Dict], min_l: float, max_l: float, pad: float) -> List[Dict]:
    match = re.search(r'```json\s*([\s\S]*?)\s*```', resp)
    json_str = match.group(1) if match else resp[resp.find('{'):resp.rfind('}')+1]
    try: data = json.loads(json_str)
    except: return []
    
    clips, dur = data.get("clips", []), max((s["end"] for s in segments), default=0.0)
    intervals = []
    for c in clips:
        s, e = float(c.get("start_seconds", 0)), float(c.get("end_seconds", 0))
        if e - s < min_l:
            defic = min_l - (e - s)
            s, e = max(s - defic / 2, 0), min(e + defic / 2, dur)
        elif e - s > max_l: e = s + max_l
        e = extend_interval_to_last_word(segments, s, min(e, dur), pad, dur)
        vs = float(c.get("viral_score", 5.0))
        intervals.append({"start": s, "end": e, "score": vs, "viral_score": vs, "text": c.get("hook", ""), "hook": c.get("hook", ""), "summary": c.get("summary", ""), "idea_completion": c.get("idea_completion", None), "hashtags": c.get("hashtags", None)})
    return intervals

def split_transcript_into_parts(segments: List[Dict]) -> List[Tuple[str, str]]:
    if not segments: return []
    dur = segments[-1]["end"]
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
    for c in sorted(all_candidates, key=lambda x: x["score"], reverse=True):
        if not any(abs(c["start"] - s["start"]) < 10 for s in seen):
            seen.append(c)
            unique.append(c)
            if len(unique) >= cfg.highlight.num_highlights: break
    return unique
