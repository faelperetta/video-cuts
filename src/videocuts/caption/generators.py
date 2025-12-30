import os
import math
from typing import List, Dict
from videocuts.config import Config
from videocuts.utils.system import format_timestamp

def trim_text_to_clip(text: str, seg_start: float, seg_end: float, clip_end: float) -> str:
    """Heuristically drop unfinished sentences that extend beyond clip_end."""
    if seg_end <= clip_end + 1e-3:
        return text

    total = max(seg_end - seg_start, 1e-3)
    audible = max(min(clip_end, seg_end) - seg_start, 0.0)
    coverage = max(min(audible / total, 1.0), 0.0)

    if coverage <= 0.15:
        return ""

    approx_len = max(1, int(len(text) * coverage))
    truncated = text[:approx_len].rstrip()

    sentence_breaks = [truncated.rfind(marker) for marker in (".", "!", "?", ",")]
    best_break = max(sentence_breaks)
    if best_break >= max(len(truncated) * 0.35, 1):
        truncated = truncated[:best_break + 1]

    return truncated.strip()

def write_clip_srt(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    output_path: str
) -> None:
    """Create a trimmed SRT where timestamps are relative to `clip_start`."""
    entries = []
    counter = 1

    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        if seg_end < clip_start or seg_start > clip_end:
            continue

        adj_start = max(seg_start, clip_start) - clip_start
        adj_end = min(seg_end, clip_end) - clip_start
        if adj_end - adj_start < 1e-2:
            continue

        text = seg["text"].strip()
        if seg_end > clip_end + 1e-3:
            text = trim_text_to_clip(text, seg_start, seg_end, clip_end)

        if not text:
            continue

        entries.append((counter, adj_start, adj_end, text))
        counter += 1

    if not entries:
        entries = [(1, 0.0, min(clip_end - clip_start, 2.0), " ")]

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, start, end, text in entries:
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

def format_ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.cc)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass_header(cfg: Config) -> str:
    """Generate ASS file header with styles for animated captions."""
    target_w, target_h = cfg.video.target_width, cfg.video.target_height
    font_name, font_size = cfg.caption.font_name, cfg.caption.font_size
    prim, high = cfg.caption.primary_color, cfg.caption.highlight_color
    out, back = cfg.caption.outline_color, cfg.caption.back_color
    out_w, shad = cfg.caption.outline_width, cfg.caption.shadow_depth
    margin_v = cfg.caption.margin_v

    return f"""[Script Info]
Title: Animated Captions
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{prim},{high},{out},{back},1,0,0,0,100,100,0,0,1,{out_w},{shad},2,40,40,{margin_v},1
Style: Highlight,{font_name},{font_size},{high},{prim},{out},{back},1,0,0,0,100,100,0,0,1,{out_w},{shad},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def extract_words_for_clip(segments: List[Dict], clip_start: float, clip_end: float) -> List[Dict]:
    """Extract word-level timing data for a clip from Whisper segments."""
    words = []
    for seg in segments:
        seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
        if seg_end < clip_start or seg_start > clip_end:
            continue
        
        seg_words = seg.get("words", [])
        if seg_words:
            for word_info in seg_words:
                ws, we = word_info.get("start", seg_start), word_info.get("end", seg_end)
                wt = word_info.get("word", "").strip()
                if we < clip_start or ws > clip_end: continue
                if wt:
                    words.append({"start": max(ws - clip_start, 0), "end": min(we - clip_start, clip_end - clip_start), "text": wt})
        else:
            text_words = seg["text"].strip().split()
            if not text_words: continue
            cl_start, cl_end = max(seg_start, clip_start), min(seg_end, clip_end)
            dur = cl_end - cl_start
            if dur <= 0: continue
            w_dur = dur / len(text_words)
            for i, word in enumerate(text_words):
                words.append({"start": cl_start + i * w_dur - clip_start, "end": cl_start + (i+1) * w_dur - clip_start, "text": word})
    return words

def group_words_into_lines(words: List[Dict], words_per_line: int = 4) -> List[Dict]:
    if not words: return []
    lines, curr = [], []
    for word in words:
        curr.append(word)
        if len(curr) >= words_per_line:
            lines.append({"start": curr[0]["start"], "end": curr[-1]["end"], "words": curr.copy()})
            curr = []
    if curr:
        lines.append({"start": curr[0]["start"], "end": curr[-1]["end"], "words": curr})
    return lines

def write_clip_ass(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    output_path: str,
    cfg: Config
) -> None:
    """Create an ASS subtitle file with word-by-word highlighting animation."""
    words = extract_words_for_clip(segments, clip_start, clip_end)
    if not words:
        with open(output_path, "w", encoding="utf-8") as f: f.write(generate_ass_header(cfg))
        return
    
    lines = group_words_into_lines(words, cfg.caption.words_per_line)
    content = generate_ass_header(cfg)
    
    if cfg.caption.use_word_highlight:
        for line in lines:
            line_words = line["words"]
            for idx, current_word in enumerate(line_words):
                text_parts = []
                for i, w in enumerate(line_words):
                    if i > idx: continue
                    if i == idx:
                        text_parts.append(f"{{\\c{cfg.caption.highlight_color}\\fscx105\\fscy105}}{w['text'].upper()}{{\\c{cfg.caption.primary_color}\\fscx100\\fscy100}}")
                    else:
                        text_parts.append(w["text"].upper())
                line_text = " ".join(text_parts)
                content += f"Dialogue: 0,{format_ass_timestamp(current_word['start'])},{format_ass_timestamp(current_word['end'])},Default,,0,0,0,,{line_text}\n"
    else:
        for line in lines:
            line_text = " ".join(w["text"].upper() for w in line["words"])
            content += f"Dialogue: 0,{format_ass_timestamp(line['start'])},{format_ass_timestamp(line['end'])},Default,,0,0,0,,{line_text}\n"
    
    with open(output_path, "w", encoding="utf-8") as f: f.write(content)
