from typing import List, Dict, Optional
from videocuts.config import Config
from videocuts.audio.analysis import sentiment_intensity

def normalize(values: List[float]) -> List[float]:
    if not values: return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-8: return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]

def hook_keyword_bonus(text: str, hook_keywords: List[str]) -> float:
    t = text.lower()
    for phrase in hook_keywords:
        if phrase in t: return 1.0
    return 0.0

def score_segments_for_highlights(
    segments: List[Dict],
    rms_values: List[float],
    tokenizer,
    model,
    hook_keywords: List[str],
    cfg: Config
) -> List[Dict]:
    """Compute highlight scores for each segment based on emotion, energy, hook, and rate."""
    weights = cfg.get_niche_config(cfg.content_type)["WEIGHTS"]
    text_emotions, speech_rates, hook_bonuses = [], [], []

    for seg in segments:
        text = seg["text"].strip()
        duration = max(seg["end"] - seg["start"], 1e-2)
        text_emotions.append(sentiment_intensity(text, tokenizer, model))
        speech_rates.append(len(text.split()) / duration)
        hook_bonuses.append(hook_keyword_bonus(text, hook_keywords))

    rms_norm = normalize(rms_values)
    speech_rates_norm = normalize(speech_rates)

    scored = []
    for i, seg in enumerate(segments):
        score = (weights["emotion"] * text_emotions[i] +
                 weights["energy"] * rms_norm[i] +
                 weights["hook"] * hook_bonuses[i] +
                 weights["speech_rate"] * speech_rates_norm[i])
        
        new_seg = dict(seg)
        new_seg.update({"emotion": text_emotions[i], "energy": rms_norm[i], "speech_rate": speech_rates_norm[i], "hook_bonus": hook_bonuses[i], "score": score})
        scored.append(new_seg)

    return scored

def select_highlight_intervals(
    scored_segments: List[Dict],
    segments: List[Dict],
    cfg: Config,
    video_duration: float
) -> List[Dict]:
    """Heuristically select high-scoring intervals as clips."""
    num_h = cfg.highlight.num_highlights
    min_len = cfg.highlight.min_length
    max_len = cfg.highlight.max_length
    ctx_before = cfg.highlight.context_before
    ctx_after = cfg.highlight.context_after
    min_gap = cfg.highlight.min_gap
    pad = cfg.highlight.last_word_pad
    
    sorted_segs = sorted(scored_segments, key=lambda x: x["score"], reverse=True)
    selected = []
    
    for seed in sorted_segs:
        if len(selected) >= num_h: break
        
        start, end = seed["start"], seed["end"]
        if any(abs(start - s["start"]) < min_gap or abs(end - s["end"]) < min_gap for s in selected): continue
        if any(start < s["end"] + min_gap and end > s["start"] - min_gap for s in selected): continue
        
        interval = build_interval_for_segment(seed, segments, video_duration, min_len, max_len, ctx_before, ctx_after, pad)
        if not any(interval["start"] < s["end"] + min_gap and interval["end"] > s["start"] - min_gap for s in selected):
            selected.append(interval)
            
    return sorted(selected, key=lambda x: x["start"])

def build_interval_for_segment(
    seed: Dict,
    segments: List[Dict],
    video_duration: float,
    min_len: float,
    max_len: float,
    ctx_before: float,
    ctx_after: float,
    last_word_pad: float
) -> Dict:
    start = max(seed["start"] - ctx_before, 0.0)
    end = min(seed["end"] + ctx_after, video_duration)
    
    if end - start < min_len:
        deficit = min_len - (end - start)
        start = max(start - deficit / 2, 0.0)
        end = min(start + min_len, video_duration)
    if end - start > max_len:
        end = start + max_len
        
    final_end = extend_interval_to_last_word(segments, start, end, last_word_pad, video_duration)
    return {"start": start, "end": final_end, "score": seed["score"], "seed_text": seed["text"]}

def extend_interval_to_last_word(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    pad: float,
    video_duration: float
) -> float:
    best_end = clip_end
    for seg in segments:
        if seg["start"] < clip_end and seg["end"] > clip_end:
            best_end = min(seg["end"] + pad, video_duration)
            break
    return best_end

def intervals_conflict(start: float, end: float, selected: List[Dict], min_gap: float) -> bool:
    for s in selected:
        if start < s["end"] + min_gap and end > s["start"] - min_gap: return True
    return False
