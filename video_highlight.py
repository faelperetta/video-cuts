from __future__ import annotations

import os
import math
import subprocess
import urllib.request
import tempfile
import glob
import argparse
from typing import List, Dict, Tuple, Optional
import statistics
from contextlib import nullcontext

import torch
import torchaudio
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except ImportError:
    mp = None

MP_HAS_SOLUTIONS = bool(mp is not None and hasattr(mp, "solutions"))
MP_HAS_TASKS = False

if mp is not None:
    try:
        from mediapipe.tasks import python as mp_tasks_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore
        MP_HAS_TASKS = True
    except Exception:
        MP_HAS_TASKS = False


# ==========================
# CONFIG
# ==========================

INPUT_VIDEO = "input.webm"              # Your source video
SRT_FILE = "subs.srt"                  # Full-video subtitles
AUDIO_WAV = "audio.wav"                # Extracted audio

OUTPUT_DIR = "highlights"              # Folder for output clips
FINAL_OUTPUT = "final_highlight.mp4"   # Single best highlight

# Choose one: "coding", "fitness", "gaming"
CONTENT_TYPE = "coding"

# Video format for Shorts/Reels/TikTok
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
OUTPUT_FPS = 30

# Whisper model size: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_SIZE = "small"

# Sentiment model (text-based emotional intensity)
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Highlight selection
MIN_HIGHLIGHT_LEN = 15      # seconds
MAX_HIGHLIGHT_LEN = 45.0     # seconds
CONTEXT_BEFORE = 1.5         # seconds before a high-scoring segment
CONTEXT_AFTER = 1.5          # seconds after a high-scoring segment
NUM_HIGHLIGHTS = 3           # how many clips to export
MIN_GAP_BETWEEN_HIGHLIGHTS = 4.0  # seconds of separation between clips
LAST_WORD_PAD = 0.25         # small buffer to ensure last word finishes
FACE_MIN_CONFIDENCE = 0.45   # mediapipe detection confidence
FACE_ANALYSIS_FPS = 12.0     # sampling FPS for face tracking inside a clip
FACE_TRACK_DISTANCE = 0.12   # max normalized horizontal delta to keep same track
FACE_TRACK_MAX_GAP = 1.0     # seconds a track can go unseen before recycled
SPEAKER_ACTIVITY_THRESHOLD = 0.0035  # threshold for choosing active speaker
FACE_RECENTER_AFTER = 0.35    # seconds without detection before easing to center

# Speaker lock parameters - prevents jittery switching between people
SPEAKER_LOCK_MIN_DURATION = 3.0     # minimum seconds to stay with a speaker before switching
SPEAKER_SWITCH_THRESHOLD = 2.0      # other speaker must have 2x more cumulative lip activity to switch
SPEAKER_SMOOTHING_WINDOW = 0.5      # seconds of smoothing for crop position
SPEAKER_POSITION_SMOOTHING = 0.5    # blend factor for position smoothing (0=instant, 1=no change)

# Segment smoothing - eliminates jitter from micro-segments
MIN_SEGMENT_DURATION = 0.5          # minimum segment duration in seconds (shorter segments get merged)
SEGMENT_MERGE_THRESHOLD = 0.15      # merge segments with centers within this distance
SEGMENT_ABSORB_THRESHOLD = 0.3      # very short segments absorbed if neighbor is within this distance

# Lip movement detection - for identifying who is speaking
LIP_MOVEMENT_HISTORY_FRAMES = 5     # number of frames to track for lip movement delta
LIP_MOVEMENT_MIN_DELTA = 0.003      # minimum change in mouth opening to count as "speaking"
LIP_SPEAKING_THRESHOLD = 0.008      # cumulative lip movement to consider someone speaking

# Animated Caption Style Configuration
CAPTION_FONT_NAME = "Arial"         # Font family for captions
CAPTION_FONT_SIZE = 58              # Font size in pixels
CAPTION_PRIMARY_COLOR = "&H00FFFFFF"  # White (AABBGGRR format)
CAPTION_HIGHLIGHT_COLOR = "&H0000FF00"  # Bright green for current word
CAPTION_OUTLINE_COLOR = "&H00000000"  # Black outline
CAPTION_BACK_COLOR = "&H80000000"     # Semi-transparent black background
CAPTION_OUTLINE_WIDTH = 3           # Outline thickness
CAPTION_SHADOW_DEPTH = 2            # Shadow depth
CAPTION_MARGIN_V = 120              # Vertical margin from bottom
CAPTION_WORDS_PER_LINE = 4          # Max words per caption line
CAPTION_USE_WORD_HIGHLIGHT = True   # Enable word-by-word highlighting

# Auto-Hook Overlay Configuration
HOOK_ENABLED = False                # Controlled by --hook CLI flag
HOOK_SCAN_SECONDS = 5.0             # Seconds at start of clip to scan for hook
HOOK_MIN_WORDS = 3                  # Minimum words for a valid hook
HOOK_MAX_WORDS = 10                 # Maximum words to display
HOOK_FONT_NAME = "Arial"            # Font for hook overlay
HOOK_FONT_SIZE = 42                 # Font size for hook text
HOOK_PRIMARY_COLOR = "black"        # Hook text color (FFmpeg format)
HOOK_BG_COLOR = "white"             # White background like OpusClip
HOOK_BG_OPACITY = 0.95              # Background opacity (0-1)
HOOK_POSITION_Y = 120               # Vertical position from top
HOOK_FADE_IN_DURATION = 0.3         # Fade-in duration in seconds
HOOK_DISPLAY_DURATION = 3.0         # How long hook stays visible (after fade)
HOOK_FADE_OUT_DURATION = 0.3        # Fade-out duration in seconds
HOOK_BOX_PADDING = 20               # Padding around hook text
HOOK_BOX_BORDER_W = 24              # Horizontal padding for box
HOOK_SHADOW_COLOR = "black@0.3"     # Subtle shadow for depth

# Multi-Speaker Layout Configuration
LAYOUT_MODE = "auto"                # "auto", "single", "split", "wide"
LAYOUT_SPLIT_THRESHOLD = 0.35       # Min horizontal distance between faces to trigger split-screen
LAYOUT_WIDE_THRESHOLD = 0.25        # Max distance to use wide shot instead of split
LAYOUT_MIN_FACES_FOR_SPLIT = 2      # Minimum faces needed for split-screen
LAYOUT_SPLIT_GAP = 20               # Pixels between split panels
LAYOUT_BOTH_SPEAKING_WINDOW = 1.5   # Seconds window to detect both speakers active
LAYOUT_BOTH_SPEAKING_RATIO = 0.3    # If both have >30% of max activity, consider both active
LAYOUT_SEGMENT_MIN_DURATION = 2.0   # Minimum duration for a layout segment
LAYOUT_WIDE_ZOOM = 0.85             # Zoom factor for wide shot (0.85 = show 85% width)

FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detection/"
    "short_range/float16/1/face_detection_short_range.tflite"
)
FACE_DETECTOR_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "models",
    "face_detection_short_range.tflite"
)

# Face Landmarker model (for lip detection)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "models",
    "face_landmarker.task"
)

# Reuse existing intermediates when possible
FORCE_TRANSCRIBE = False
FORCE_AUDIO_EXTRACTION = False


# ==========================
# NICHE-SPECIFIC HOOKS & WEIGHTS
# ==========================

NICHE_CONFIG = {
    "coding": {
        "HOOK_KEYWORDS": [
            "here's the trick",
            "here is the trick",
            "the trick is",
            "the secret",
            "no one tells you",
            "nobody tells you",
            "you won't believe",
            "this is how you",
            "how to write",
            "how to build",
            "how to fix",
            "never do this",
            "biggest mistake",
            "huge mistake",
            "do this instead",
            "stop doing this",
            "in just one line",
            "in one line",
            "in 5 minutes",
            "in five minutes",
            "let me show you",
            "watch this",
        ],
        # Coding: clarity + hook language > raw hype
        "WEIGHTS": {
            "emotion": 1.8,
            "energy": 1.5,
            "hook": 2.7,
            "speech_rate": 0.9,
        },
    },
    "fitness": {
        "HOOK_KEYWORDS": [
            "do this every day",
            "do this daily",
            "every morning",
            "every night",
            "stop doing this",
            "never do this",
            "biggest mistake",
            "huge mistake",
            "this changed my body",
            "this changed my life",
            "in 30 days",
            "in 7 days",
            "in a week",
            "burn fat",
            "build muscle",
            "get abs",
            "lose weight fast",
            "you won't believe",
            "here's why",
            "this is why",
        ],
        # Fitness: energy + emotion + hooks are strong
        "WEIGHTS": {
            "emotion": 2.4,
            "energy": 2.5,
            "hook": 2.2,
            "speech_rate": 0.7,
        },
    },
    "gaming": {
        "HOOK_KEYWORDS": [
            "watch this",
            "no way this happened",
            "you won't believe this",
            "craziest thing",
            "insane",
            "this insane",
            "this clutch",
            "this play",
            "this clip",
            "i shouldn't have survived",
            "i should not have survived",
            "1 hp",
            "one hp",
            "no health",
            "speedrun",
            "world record",
            "new record",
            "broken build",
            "overpowered",
            "op build",
            "broken strat",
            "broken strategy",
            "you have to try this",
        ],
        # Gaming: hype + energy + hooks, less importance on speech rate
        "WEIGHTS": {
            "emotion": 2.6,
            "energy": 2.7,
            "hook": 2.4,
            "speech_rate": 0.5,
        },
    },
}


def get_niche_config(content_type: str):
    if content_type not in NICHE_CONFIG:
        raise ValueError(f"Unknown CONTENT_TYPE '{content_type}'. "
                         f"Valid: {list(NICHE_CONFIG.keys())}")
    return NICHE_CONFIG[content_type]


# ==========================
# UTILS
# ==========================

def format_timestamp(t: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    if t < 0:
        t = 0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int((t - math.floor(t)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.strip()
    main, millis = ts.split(",")
    hours, minutes, seconds = [int(part) for part in main.split(":")]
    return hours * 3600 + minutes * 60 + int(seconds) + int(millis) / 1000.0


def parse_srt_to_segments(srt_path: str) -> List[Dict]:
    """Parse an SRT file back into Whisper-like segments."""
    with open(srt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    segments: List[Dict] = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        if "-->" not in time_line:
            continue

        start_ts, end_ts = [part.strip() for part in time_line.split("-->")]
        text = " ".join(line.strip() for line in lines[2:]).strip()

        try:
            start_sec = parse_timestamp(start_ts)
            end_sec = parse_timestamp(end_ts)
        except ValueError:
            continue

        segments.append({
            "id": idx_line,
            "start": start_sec,
            "end": end_sec,
            "text": text,
        })

    return segments


def cache_is_fresh(path: str, reference_mtime: float) -> bool:
    return os.path.exists(path) and os.path.getmtime(path) >= reference_mtime


def run_ffmpeg_cmd(cmd: List[str]) -> None:
    print("[FFmpeg] Command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_clip_srt(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    output_path: str
) -> None:
    """Create a trimmed SRT where timestamps are relative to `clip_start`."""

    def trim_text_to_clip(text: str, seg_start: float, seg_end: float) -> str:
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

        # Try to end on sentence boundary to avoid dangling words.
        sentence_breaks = [truncated.rfind(marker) for marker in (".", "!", "?", ",")]
        best_break = max(sentence_breaks)
        if best_break >= max(len(truncated) * 0.35, 1):
            truncated = truncated[:best_break + 1]

        return truncated.strip()

    entries = []
    counter = 1

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        if seg_end < clip_start or seg_start > clip_end:
            continue

        adj_start = max(seg_start, clip_start) - clip_start
        adj_end = min(seg_end, clip_end) - clip_start

        if adj_end - adj_start < 1e-2:
            continue

        text = seg["text"].strip()
        if seg_end > clip_end + 1e-3:
            text = trim_text_to_clip(text, seg_start, seg_end)

        if not text:
            continue

        entries.append((counter, adj_start, adj_end, text))
        counter += 1

    if not entries:
        # Ensure ffmpeg still has something to burn to avoid runtime errors.
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
    cs = int((seconds % 1) * 100)  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_header(
    play_res_x: int = 1080,
    play_res_y: int = 1920
) -> str:
    """Generate ASS file header with styles for animated captions."""
    return f"""[Script Info]
Title: Animated Captions
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CAPTION_FONT_NAME},{CAPTION_FONT_SIZE},{CAPTION_PRIMARY_COLOR},{CAPTION_HIGHLIGHT_COLOR},{CAPTION_OUTLINE_COLOR},{CAPTION_BACK_COLOR},1,0,0,0,100,100,0,0,1,{CAPTION_OUTLINE_WIDTH},{CAPTION_SHADOW_DEPTH},2,40,40,{CAPTION_MARGIN_V},1
Style: Highlight,{CAPTION_FONT_NAME},{CAPTION_FONT_SIZE},{CAPTION_HIGHLIGHT_COLOR},{CAPTION_PRIMARY_COLOR},{CAPTION_OUTLINE_COLOR},{CAPTION_BACK_COLOR},1,0,0,0,100,100,0,0,1,{CAPTION_OUTLINE_WIDTH},{CAPTION_SHADOW_DEPTH},2,40,40,{CAPTION_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def extract_words_for_clip(
    segments: List[Dict],
    clip_start: float,
    clip_end: float
) -> List[Dict]:
    """
    Extract word-level timing data for a clip from Whisper segments.
    Returns list of words with start, end, and text.
    """
    words = []
    
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Skip segments outside clip range
        if seg_end < clip_start or seg_start > clip_end:
            continue
        
        # Check if segment has word-level timestamps
        seg_words = seg.get("words", [])
        
        if seg_words:
            for word_info in seg_words:
                word_start = word_info.get("start", seg_start)
                word_end = word_info.get("end", seg_end)
                word_text = word_info.get("word", "").strip()
                
                # Skip words outside clip range
                if word_end < clip_start or word_start > clip_end:
                    continue
                
                if word_text:
                    # Adjust to relative time
                    rel_start = max(word_start - clip_start, 0)
                    rel_end = min(word_end - clip_start, clip_end - clip_start)
                    
                    words.append({
                        "start": rel_start,
                        "end": rel_end,
                        "text": word_text
                    })
        else:
            # Fallback: split segment text into words with estimated timing
            text = seg["text"].strip()
            text_words = text.split()
            if not text_words:
                continue
                
            # Clamp segment to clip boundaries
            clamped_start = max(seg_start, clip_start)
            clamped_end = min(seg_end, clip_end)
            duration = clamped_end - clamped_start
            
            if duration <= 0:
                continue
                
            word_duration = duration / len(text_words)
            
            for i, word in enumerate(text_words):
                word_start = clamped_start + i * word_duration
                word_end = word_start + word_duration
                
                words.append({
                    "start": word_start - clip_start,
                    "end": word_end - clip_start,
                    "text": word
                })
    
    return words


def group_words_into_lines(
    words: List[Dict],
    words_per_line: int = 4
) -> List[Dict]:
    """
    Group words into caption lines for display.
    Each line contains multiple words shown together.
    """
    if not words:
        return []
    
    lines = []
    current_line_words = []
    
    for word in words:
        current_line_words.append(word)
        
        if len(current_line_words) >= words_per_line:
            # Create a line entry
            line_start = current_line_words[0]["start"]
            line_end = current_line_words[-1]["end"]
            
            lines.append({
                "start": line_start,
                "end": line_end,
                "words": current_line_words.copy()
            })
            current_line_words = []
    
    # Handle remaining words
    if current_line_words:
        line_start = current_line_words[0]["start"]
        line_end = current_line_words[-1]["end"]
        
        lines.append({
            "start": line_start,
            "end": line_end,
            "words": current_line_words
        })
    
    return lines


def write_clip_ass(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    output_path: str
) -> None:
    """
    Create an ASS subtitle file with word-by-word highlighting animation.
    Each word gets highlighted in green as it's spoken.
    """
    # Extract words for this clip
    words = extract_words_for_clip(segments, clip_start, clip_end)
    
    if not words:
        # Fallback to empty subtitle
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generate_ass_header())
        return
    
    # Group words into display lines
    lines = group_words_into_lines(words, CAPTION_WORDS_PER_LINE)
    
    # Generate ASS content
    content = generate_ass_header()
    
    if CAPTION_USE_WORD_HIGHLIGHT:
        # Generate events with word-by-word highlighting
        for line in lines:
            line_words = line["words"]
            line_start = line["start"]
            line_end = line["end"]
            
            # For each word timing within this line, create a dialogue event
            # showing only words spoken so far, with current word highlighted
            for word_idx, current_word in enumerate(line_words):
                word_start = current_word["start"]
                word_end = current_word["end"]
                
                # Build the text showing only words up to current (hide future words)
                text_parts = []
                for i, w in enumerate(line_words):
                    if i > word_idx:
                        # Future word - don't show it yet
                        continue
                    elif i == word_idx:
                        # Highlight current word with color override and slight scale
                        text_parts.append(
                            f"{{\\c{CAPTION_HIGHLIGHT_COLOR}\\fscx105\\fscy105}}"
                            f"{w['text'].upper()}"
                            f"{{\\c{CAPTION_PRIMARY_COLOR}\\fscx100\\fscy100}}"
                        )
                    else:
                        # Past word - show in normal color
                        text_parts.append(w["text"].upper())
                
                line_text = " ".join(text_parts)
                
                # Add dialogue event
                start_ts = format_ass_timestamp(word_start)
                end_ts = format_ass_timestamp(word_end)
                
                content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}\n"
    else:
        # Simple mode: show whole line without per-word highlighting
        for line in lines:
            line_text = " ".join(w["text"].upper() for w in line["words"])
            start_ts = format_ass_timestamp(line["start"])
            end_ts = format_ass_timestamp(line["end"])
            
            content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{line_text}\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


# ==========================
# AUTO-HOOK SYSTEM
# ==========================

def detect_hook_phrase(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    hook_keywords: List[str],
    scan_seconds: float = HOOK_SCAN_SECONDS,
    min_words: int = HOOK_MIN_WORDS,
    max_words: int = HOOK_MAX_WORDS
) -> Optional[Dict]:
    """
    Detect a catchy hook phrase from the first few seconds of a clip.
    
    Prioritizes:
    1. Segments containing hook keywords from niche config
    2. First complete sentence/phrase if no keywords found
    
    Returns dict with:
    - text: The hook phrase text
    - start: Start time relative to clip
    - end: End time relative to clip
    - has_keyword: Whether a hook keyword was matched
    """
    scan_end = clip_start + scan_seconds
    
    # Collect segments within scan window
    scan_segments = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Check if segment overlaps with scan window
        if seg_end >= clip_start and seg_start <= scan_end:
            text = seg.get("text", "").strip()
            if text:
                scan_segments.append({
                    "text": text,
                    "start": max(seg_start - clip_start, 0),
                    "end": min(seg_end - clip_start, clip_end - clip_start),
                    "words": seg.get("words", [])
                })
    
    if not scan_segments:
        return None
    
    # Priority 1: Find segment with hook keyword
    for seg in scan_segments:
        text_lower = seg["text"].lower()
        for keyword in hook_keywords:
            if keyword in text_lower:
                # Found a hook keyword - extract the phrase
                hook_text = _extract_hook_around_keyword(
                    seg["text"], 
                    keyword, 
                    max_words
                )
                if hook_text and len(hook_text.split()) >= min_words:
                    print(f"[Hook] Found keyword match: '{keyword}' -> '{hook_text}'")
                    return {
                        "text": hook_text,
                        "start": seg["start"],
                        "end": min(seg["end"], HOOK_DISPLAY_DURATION + HOOK_FADE_IN_DURATION),
                        "has_keyword": True
                    }
    
    # Priority 2: Use first sentence/phrase as hook
    combined_text = " ".join(seg["text"] for seg in scan_segments)
    words = combined_text.split()
    
    if len(words) < min_words:
        return None
    
    # Try to find a natural break point (sentence end)
    hook_words = []
    for i, word in enumerate(words[:max_words]):
        hook_words.append(word)
        # Check for sentence-ending punctuation
        if word.endswith(('.', '!', '?')) and len(hook_words) >= min_words:
            break
    
    # If no sentence break found, take up to max_words
    if len(hook_words) < min_words:
        hook_words = words[:max_words]
    
    hook_text = " ".join(hook_words)
    
    # Clean up trailing punctuation for display
    hook_text = hook_text.rstrip('.,;:')
    
    if len(hook_text.split()) >= min_words:
        print(f"[Hook] Using opening phrase: '{hook_text}'")
        return {
            "text": hook_text,
            "start": scan_segments[0]["start"],
            "end": min(scan_segments[0]["end"], HOOK_DISPLAY_DURATION + HOOK_FADE_IN_DURATION),
            "has_keyword": False
        }
    
    return None


def _extract_hook_around_keyword(
    text: str, 
    keyword: str, 
    max_words: int
) -> str:
    """
    Extract a phrase centered around the hook keyword.
    Tries to capture context around the keyword for a compelling hook.
    """
    text_lower = text.lower()
    keyword_pos = text_lower.find(keyword)
    
    if keyword_pos == -1:
        return text[:max_words * 10]  # Fallback: return beginning
    
    words = text.split()
    keyword_words = keyword.split()
    
    # Find which word index contains the keyword start
    char_count = 0
    keyword_start_idx = 0
    for i, word in enumerate(words):
        word_end = char_count + len(word)
        if char_count <= keyword_pos < word_end:
            keyword_start_idx = i
            break
        char_count = word_end + 1  # +1 for space
    
    # Calculate how many words to take before/after keyword
    keyword_len = len(keyword_words)
    words_before = max(0, (max_words - keyword_len) // 2)
    words_after = max_words - keyword_len - words_before
    
    start_idx = max(0, keyword_start_idx - words_before)
    end_idx = min(len(words), keyword_start_idx + keyword_len + words_after)
    
    result = " ".join(words[start_idx:end_idx])
    return result


def generate_hook_overlay_filter(
    hook_text: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT
) -> str:
    """
    Generate FFmpeg filter expression for animated hook overlay.
    
    Creates a centered text overlay at the top of the video with:
    - White background box (OpusClip style)
    - Black text for readability
    - Fade-in/out animation
    - Centered horizontally, compact width
    """
    # Escape special characters for FFmpeg drawtext
    escaped_text = hook_text.replace("'", "'\\''").replace(":", "\\:").replace("\\", "\\\\")
    
    # Calculate timing
    fade_in_end = HOOK_FADE_IN_DURATION
    visible_end = fade_in_end + HOOK_DISPLAY_DURATION
    fade_out_end = visible_end + HOOK_FADE_OUT_DURATION
    
    # Alpha expression: fade in -> hold -> fade out
    alpha_expr = (
        f"if(lt(t,{fade_in_end}),t/{HOOK_FADE_IN_DURATION},"
        f"if(lt(t,{visible_end}),1,"
        f"if(lt(t,{fade_out_end}),(({fade_out_end}-t)/{HOOK_FADE_OUT_DURATION}),0)))"
    )
    
    # Box color with opacity for the alpha animation
    box_color = f"{HOOK_BG_COLOR}@{HOOK_BG_OPACITY}"
    
    # Drawtext filter with white box background, centered
    # Using boxborderw for padding around text
    drawtext_filter = (
        f"drawtext=text='{escaped_text}':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize={HOOK_FONT_SIZE}:"
        f"fontcolor={HOOK_PRIMARY_COLOR}:"
        f"x=(w-text_w)/2:"
        f"y={HOOK_POSITION_Y}:"
        f"alpha='{alpha_expr}':"
        f"box=1:"
        f"boxcolor={box_color}:"
        f"boxborderw={HOOK_BOX_PADDING}:"
        f"shadowcolor={HOOK_SHADOW_COLOR}:"
        f"shadowx=2:shadowy=2"
    )
    
    return drawtext_filter


def apply_hook_overlay(
    input_video: str,
    output_video: str,
    hook_text: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT
) -> None:
    """
    Apply hook overlay to a video file.
    Creates a new video with the hook text animation at the top.
    """
    hook_filter = generate_hook_overlay_filter(hook_text, target_w, target_h)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", hook_filter,
        "-c:a", "copy",
        output_video
    ]
    
    print(f"[Hook] Applying hook overlay: '{hook_text}'")
    subprocess.run(cmd, check=True, capture_output=True)


# ==========================
# STEP 1: TRANSCRIBE WITH WHISPER (PYTORCH)
# ==========================

def transcribe_to_srt_and_segments(
    input_video: str,
    srt_path: str,
    model_size: str = "small"
) -> List[Dict]:
    """
    Use Whisper (PyTorch) to transcribe `input_video`, write an SRT file,
    and return the list of segments with word-level timestamps.
    """
    print(f"[Whisper] Loading model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"[Whisper] Transcribing '{input_video}' with word timestamps...")
    result = model.transcribe(input_video, task="transcribe", word_timestamps=True)

    segments = result["segments"]

    print(f"[Whisper] Writing SRT to '{srt_path}'...")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

    print("[Whisper] Done transcription.")
    return segments


# ==========================
# STEP 2: AUDIO EXTRACTION & ANALYSIS (TORCHAUDIO)
# ==========================

def extract_audio(input_video: str, audio_path: str) -> None:
    """
    Extract audio from the input video to a WAV file.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",                # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",       # 16kHz
        "-ac", "1",           # mono
        audio_path
    ]
    run_ffmpeg_cmd(cmd)
    print(f"[Audio] Extracted audio to '{audio_path}'.")


def compute_rms_per_segment(
    audio_path: str,
    segments: List[Dict]
) -> List[float]:
    """
    Compute RMS loudness for each speech segment using torchaudio.
    """
    print("[Audio] Loading audio with torchaudio...")
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono

    rms_values = []
    for seg in segments:
        start = max(seg["start"], 0)
        end = max(seg["end"], start + 1e-2)
        s = int(start * sr)
        e = int(end * sr)
        e = min(e, waveform.shape[1])

        if e <= s:
            rms = 0.0
        else:
            seg_wave = waveform[:, s:e]
            rms = torch.sqrt((seg_wave ** 2).mean()).item()

        rms_values.append(rms)

    print("[Audio] Computed RMS for segments.")
    return rms_values


# ==========================
# STEP 3: TEXT SENTIMENT / HOOK SCORING (TRANSFORMERS + PYTORCH)
# ==========================

def load_sentiment_model(model_name: str):
    print(f"[NLP] Loading sentiment model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def sentiment_intensity(
    text: str,
    tokenizer,
    model
) -> float:
    """
    Use a sentiment model to approximate 'emotional intensity':
    1 - P(neutral). Works well for 3-label sentiment models.
    """
    if not text.strip():
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    # For cardiffnlp/twitter-roberta-base-sentiment-latest:
    # labels: [negative, neutral, positive]
    neutral_prob = probs[1].item()
    intensity = 1.0 - neutral_prob  # higher = more emotional
    return float(intensity)


def hook_keyword_bonus(text: str, hook_keywords: List[str]) -> float:
    """
    Check if the text contains typical 'hook' phrases.
    """
    t = text.lower()
    for phrase in hook_keywords:
        if phrase in t:
            return 1.0
    return 0.0


# ==========================
# STEP 4: HIGHLIGHT SCORING & SELECTION
# ==========================

def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx - mn < 1e-8:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def score_segments_for_highlights(
    segments: List[Dict],
    rms_values: List[float],
    tokenizer,
    model,
    hook_keywords: List[str],
    weights: Dict[str, float]
) -> List[Dict]:
    """
    For each Whisper segment, compute:
    - text_emotion: sentiment intensity
    - audio_energy: RMS (normalized)
    - speech_rate: words / second (normalized)
    - hook_bonus: presence of hooky phrases

    Then compute an overall highlight_score, using niche-specific weights.
    """
    print("[Highlight] Scoring segments...")
    text_emotions = []
    speech_rates = []
    hook_bonuses = []

    # First pass: compute raw features
    for seg in segments:
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]
        duration = max(end - start, 1e-2)
        words = text.split()

        emo = sentiment_intensity(text, tokenizer, model)
        text_emotions.append(emo)

        speech_rate = len(words) / duration
        speech_rates.append(speech_rate)

        hook = hook_keyword_bonus(text, hook_keywords)
        hook_bonuses.append(hook)

    # Normalize audio energy & speech rates
    rms_norm = normalize(rms_values)
    speech_rates_norm = normalize(speech_rates)

    w_emo = weights["emotion"]
    w_energy = weights["energy"]
    w_hook = weights["hook"]
    w_rate = weights["speech_rate"]

    # Combine into a single score
    scored_segments = []
    for i, seg in enumerate(segments):
        emo = text_emotions[i]
        energy = rms_norm[i]
        rate = speech_rates_norm[i]
        hook = hook_bonuses[i]

        score = (
            w_emo * emo +
            w_energy * energy +
            w_hook * hook +
            w_rate * rate
        )

        new_seg = dict(seg)
        new_seg["emotion"] = emo
        new_seg["energy"] = energy
        new_seg["speech_rate"] = rate
        new_seg["hook_bonus"] = hook
        new_seg["score"] = score
        scored_segments.append(new_seg)

    print("[Highlight] Segments scored.")
    return scored_segments


def extend_interval_to_last_word(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    pad: float,
    video_duration: float
) -> float:
    """Extend `clip_end` so the final subtitle/audio finishes before cutting."""
    relevant_ends = [
        seg["end"]
        for seg in segments
        if seg["start"] < clip_end and seg["end"] > clip_start
    ]
    if not relevant_ends:
        return clip_end

    last_word_end = max(relevant_ends)
    extended_end = last_word_end + pad
    return min(max(clip_end, extended_end), video_duration)


def build_interval_for_segment(
    segment: Dict,
    segments: List[Dict],
    video_duration: float,
    min_len: float,
    max_len: float,
    ctx_before: float,
    ctx_after: float,
    last_word_pad: float
) -> Tuple[float, float]:
    start = max(segment["start"] - ctx_before, 0.0)
    end = min(segment["end"] + ctx_after, video_duration)
    duration = end - start

    if duration < min_len:
        deficit = min_len - duration
        start = max(start - deficit / 2.0, 0.0)
        end = min(start + min_len, video_duration)

    if end - start > max_len:
        end = min(start + max_len, video_duration)

    end = extend_interval_to_last_word(
        segments,
        start,
        end,
        last_word_pad,
        video_duration
    )
    return start, end


def intervals_conflict(
    start: float,
    end: float,
    selected: List[Dict],
    min_gap: float
) -> bool:
    for existing in selected:
        if end + min_gap <= existing["start"]:
            continue
        if start >= existing["end"] + min_gap:
            continue
        return True
    return False


def select_highlight_intervals(
    scored_segments: List[Dict],
    segments: List[Dict],
    num_highlights: int,
    min_len: float,
    max_len: float,
    ctx_before: float,
    ctx_after: float,
    min_gap: float,
    last_word_pad: float
) -> List[Dict]:
    if not scored_segments:
        return []

    video_duration = max((seg["end"] for seg in segments), default=0.0)
    sorted_segments = sorted(scored_segments, key=lambda s: s["score"], reverse=True)
    selected: List[Dict] = []

    for seg in sorted_segments:
        if len(selected) >= num_highlights:
            break

        start, end = build_interval_for_segment(
            seg,
            segments,
            video_duration,
            min_len,
            max_len,
            ctx_before,
            ctx_after,
            last_word_pad
        )

        if intervals_conflict(start, end, selected, min_gap):
            continue

        selected.append({
            "start": start,
            "end": end,
            "score": seg["score"],
            "text": seg["text"].strip()
        })

    if selected:
        print(f"[Highlight] Selected {len(selected)} interval(s):")
        for idx, hl in enumerate(selected, start=1):
            print(
                f"  #{idx}: {hl['start']:.2f}s -> {hl['end']:.2f}s "
                f"(len={hl['end'] - hl['start']:.2f}s, score={hl['score']:.3f})"
            )
    else:
        print("[Highlight] No non-overlapping intervals met the criteria.")

    return selected


# ==========================
# STEP 5A: SPEAKER FRAMING ANALYSIS (OPENCV + MEDIAPIPE)
# ==========================

def ensure_face_detector_model() -> Optional[str]:
    model_dir = os.path.dirname(FACE_DETECTOR_MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(FACE_DETECTOR_MODEL_PATH):
        return FACE_DETECTOR_MODEL_PATH

    try:
        print("[FaceTracking] Downloading face detector weights...")
        urllib.request.urlretrieve(
            FACE_DETECTOR_MODEL_URL,
            FACE_DETECTOR_MODEL_PATH
        )
        return FACE_DETECTOR_MODEL_PATH
    except Exception as exc:
        print(f"[FaceTracking] Failed to fetch detector model: {exc}")
        return None


def ensure_face_landmarker_model() -> Optional[str]:
    """Download Face Landmarker model for lip detection."""
    model_dir = os.path.dirname(FACE_LANDMARKER_MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        return FACE_LANDMARKER_MODEL_PATH

    try:
        print("[FaceTracking] Downloading face landmarker model for lip detection...")
        urllib.request.urlretrieve(
            FACE_LANDMARKER_MODEL_URL,
            FACE_LANDMARKER_MODEL_PATH
        )
        print("[FaceTracking] Face landmarker model downloaded successfully.")
        return FACE_LANDMARKER_MODEL_PATH
    except Exception as exc:
        print(f"[FaceTracking] Failed to fetch face landmarker model: {exc}")
        return None


def _mouth_open_metric(face_landmarks) -> float:
    """
    Measure mouth opening using multiple lip landmarks for accuracy.
    Uses inner lip landmarks for more precise speaking detection.
    
    FaceMesh inner lip landmarks:
    - 13: Upper inner lip center
    - 14: Lower inner lip center  
    - 78: Left inner lip corner
    - 308: Right inner lip corner
    - 82: Upper left inner lip
    - 312: Upper right inner lip
    - 87: Lower left inner lip
    - 317: Lower right inner lip
    """
    # Primary: vertical distance between inner lip centers
    upper_center = face_landmarks.landmark[13]
    lower_center = face_landmarks.landmark[14]
    vertical_opening = abs(lower_center.y - upper_center.y)
    
    # Secondary: horizontal lip stretch (lips widen when speaking)
    left_corner = face_landmarks.landmark[78]
    right_corner = face_landmarks.landmark[308]
    horizontal_stretch = abs(right_corner.x - left_corner.x)
    
    # Combine metrics - vertical opening is primary indicator
    # Normalize horizontal stretch relative to baseline (~0.15 for closed mouth)
    BASELINE_LIP_WIDTH = 0.12
    stretch_delta = max(0, horizontal_stretch - BASELINE_LIP_WIDTH)
    
    # Weight vertical opening more heavily (0.8) vs horizontal stretch (0.2)
    return vertical_opening * 0.8 + stretch_delta * 0.2


def _detect_faces_with_mesh(mesh, rgb_frame):
    results = mesh.process(rgb_frame)
    detections = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            xmin = max(min(xs), 0.0)
            xmax = min(max(xs), 1.0)
            ymin = max(min(ys), 0.0)
            ymax = min(max(ys), 1.0)
            width = max(xmax - xmin, 1e-3)
            height = max(ymax - ymin, 1e-3)
            center = (xmin + xmax) / 2.0
            
            # Get raw mouth opening value
            mouth_open = _mouth_open_metric(face_landmarks)
            
            # Normalize by face height to be scale-invariant
            mouth_open_normalized = mouth_open / height if height > 0.01 else mouth_open
            
            detections.append({
                "center": max(0.0, min(1.0, center)),
                "width": min(width, 1.0),
                "mouth_open": mouth_open_normalized,  # Raw mouth opening for delta tracking
                "activity": 0.0  # Will be computed from lip movement delta
            })
    return detections


def _detect_faces_with_landmarker(landmarker, rgb_frame):
    """
    Detect faces using MediaPipe FaceLandmarker (Tasks API).
    This provides face landmarks including lip positions for speaking detection.
    """
    detections = []
    if landmarker is None or mp is None:
        return detections

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)
    h, w, _ = rgb_frame.shape

    if not result.face_landmarks:
        return detections

    for face_landmarks in result.face_landmarks:
        # Get face bounding box from landmarks
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        xmin = max(min(xs), 0.0)
        xmax = min(max(xs), 1.0)
        ymin = max(min(ys), 0.0)
        ymax = min(max(ys), 1.0)
        width = max(xmax - xmin, 1e-3)
        height = max(ymax - ymin, 1e-3)
        center = (xmin + xmax) / 2.0

        # Measure mouth opening using lip landmarks
        # FaceLandmarker uses same landmark indices as FaceMesh:
        # 13: Upper inner lip center, 14: Lower inner lip center
        # 78: Left inner lip corner, 308: Right inner lip corner
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        left_corner = face_landmarks[78]
        right_corner = face_landmarks[308]

        vertical_opening = abs(lower_lip.y - upper_lip.y)
        horizontal_stretch = abs(right_corner.x - left_corner.x)
        
        BASELINE_LIP_WIDTH = 0.12
        stretch_delta = max(0, horizontal_stretch - BASELINE_LIP_WIDTH)
        mouth_open = vertical_opening * 0.8 + stretch_delta * 0.2

        # Normalize by face height to be scale-invariant
        mouth_open_normalized = mouth_open / height if height > 0.01 else mouth_open

        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": min(width, 1.0),
            "mouth_open": mouth_open_normalized,
            "activity": 0.0  # Will be computed from lip movement delta
        })

    return detections


def _detect_faces_with_tasks(detector, rgb_frame):
    detections = []
    if detector is None or mp is None:
        return detections

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    h, w, _ = rgb_frame.shape

    if not result.detections:
        return detections

    for det in result.detections:
        bbox = det.bounding_box
        center = (bbox.origin_x + bbox.width / 2.0) / max(w, 1)
        width = bbox.width / max(w, 1)
        score = det.categories[0].score if det.categories else 0.0
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": max(min(width, 1.0), 1e-3),
            "activity": score
        })

    return detections


def _detect_faces_with_cascade(cascade, frame_bgr):
    """
    Fallback face detection using Haar cascade.
    Note: Cannot detect lip movement, so activity is always 0.
    Speaker detection will fall back to "most recently visible" heuristic.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    detections = []
    for (x, y, fw, fh) in faces:
        center = (x + fw / 2) / max(w, 1)
        width = fw / max(w, 1)
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": max(min(width, 1.0), 1e-3),
            "mouth_open": 0.0,  # Cannot detect with Haar
            "activity": 0.0     # Will use position-based fallback
        })
    return detections


def _load_haar_cascade():
    cascade_path = getattr(cv2.data, "haarcascades", "")
    cascade_file = os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        return None
    return cascade


def _extract_frames_with_ffmpeg(
    input_video: str,
    clip_start: float,
    clip_end: float,
    analysis_fps: float,
    output_dir: str
) -> List[Tuple[float, str]]:
    """
    Use ffmpeg to extract frames from the video clip.
    Returns list of (timestamp, frame_path) tuples.
    This handles AV1 and other codecs that OpenCV struggles with.
    """
    duration = clip_end - clip_start
    # Extract frames at the specified FPS
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-vf", f"fps={analysis_fps}",
        "-q:v", "2",  # High quality JPEG
        output_pattern
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[FaceTracking] ffmpeg frame extraction failed: {e}")
        return []
    
    # Collect extracted frames with their timestamps
    frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
    frames_with_ts: List[Tuple[float, str]] = []
    
    for i, frame_path in enumerate(frame_files):
        # Calculate timestamp for each frame
        ts = clip_start + (i / analysis_fps)
        if ts <= clip_end:
            frames_with_ts.append((ts, frame_path))
    
    return frames_with_ts


def compute_speaker_samples(
    input_video: str,
    clip_start: float,
    clip_end: float,
    analysis_fps: float = FACE_ANALYSIS_FPS
) -> List[Dict]:
    if cv2 is None:
        return []

    actual_duration = max(clip_end - clip_start, 1e-3)

    detector_backend = "none"
    face_landmarker = None
    cascade = None

    # Priority 1: FaceLandmarker (provides lip landmarks for speaking detection)
    if MP_HAS_TASKS:
        model_path = ensure_face_landmarker_model()
        if model_path:
            try:
                base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
                landmarker_options = mp_tasks_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=4,
                    min_face_detection_confidence=FACE_MIN_CONFIDENCE,
                    min_face_presence_confidence=FACE_MIN_CONFIDENCE,
                    min_tracking_confidence=0.5
                )
                face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(landmarker_options)
                detector_backend = "landmarker"
            except Exception as exc:
                print(f"[FaceTracking] FaceLandmarker unavailable ({exc}); falling back.")

    # Priority 2: Haar cascade (no lip detection, uses position only)
    if detector_backend == "none":
        cascade = _load_haar_cascade()
        if cascade is None:
            print("[FaceTracking] Haar cascade not available; skipping speaker framing.")
            return []
        detector_backend = "haar"
        print("[FaceTracking] Warning: Using Haar cascade - lip detection disabled, speaker tracking by position only")

    print(f"[FaceTracking] Using '{detector_backend}' backend for clip {clip_start:.2f}-{clip_end:.2f}s")

    tracks: List[Dict] = []
    next_track_id = 0
    samples: List[Dict] = []
    # Defer initialization of last_center - will be set from first detection
    last_center: Optional[float] = None
    last_track_id: Optional[int] = None
    last_detection_time = clip_start
    
    # Speaker lock state - prevents jittery switching
    locked_track_id: Optional[int] = None
    lock_start_time: float = clip_start
    track_lip_activity: Dict[int, float] = {}  # cumulative LIP MOVEMENT (delta) per track
    track_mouth_history: Dict[int, List[float]] = {}  # recent mouth opening values per track
    smoothed_center: Optional[float] = None
    
    # Debug counters for lip detection
    lip_speaking_detections = 0
    total_multi_face_frames = 0

    # Use a temporary directory for frame extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames using ffmpeg (handles AV1 properly)
        frames_with_ts = _extract_frames_with_ffmpeg(
            input_video,
            clip_start,
            clip_end,
            analysis_fps,
            temp_dir
        )
        
        if not frames_with_ts:
            print(f"[FaceTracking] No frames extracted for clip {clip_start:.2f}-{clip_end:.2f}s")
            return []
        
        print(f"[FaceTracking] Extracted {len(frames_with_ts)} frames for analysis")
        
        for ts, frame_path in frames_with_ts:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            detections: List[Dict]
            if detector_backend == "landmarker" and face_landmarker is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = _detect_faces_with_landmarker(face_landmarker, rgb)
            else:
                detections = _detect_faces_with_cascade(cascade, frame)

            assigned_entries = []
            for det in detections:
                best_track = None
                best_dist = 1.0
                for track in tracks:
                    if ts - track["last_time"] > FACE_TRACK_MAX_GAP:
                        continue
                    dist = abs(det["center"] - track["center"])
                    if dist < FACE_TRACK_DISTANCE and dist < best_dist:
                        best_track = track
                        best_dist = dist

                if best_track is None:
                    best_track = {
                        "id": next_track_id,
                        "center": det["center"],
                        "last_time": ts
                    }
                    tracks.append(best_track)
                    next_track_id += 1

                best_track["center"] = det["center"]
                best_track["last_time"] = ts
                det["track_id"] = best_track["id"]
                
                # === LIP MOVEMENT DETECTION ===
                # Track mouth opening over time and compute DELTA (change) 
                # This detects actual speaking vs just having mouth open or moving around
                tid = best_track["id"]
                mouth_open = det.get("mouth_open", 0.0)
                
                # Initialize mouth history for this track if needed
                if tid not in track_mouth_history:
                    track_mouth_history[tid] = []
                
                history = track_mouth_history[tid]
                
                # Compute lip movement as the variance/change in mouth opening
                lip_movement = 0.0
                if len(history) >= 1:
                    # Calculate the absolute delta from previous frame
                    prev_mouth = history[-1]
                    lip_movement = abs(mouth_open - prev_mouth)
                    
                    # Only count as "speaking activity" if movement exceeds threshold
                    if lip_movement > LIP_MOVEMENT_MIN_DELTA:
                        track_lip_activity[tid] = track_lip_activity.get(tid, 0.0) + lip_movement
                
                # Update history (keep last N frames)
                history.append(mouth_open)
                if len(history) > LIP_MOVEMENT_HISTORY_FRAMES:
                    history.pop(0)
                
                # Store the lip movement score for speaker selection
                det["lip_movement"] = lip_movement
                det["activity"] = lip_movement  # Use lip movement as activity metric
                
                assigned_entries.append(det)

            # === SPEAKER LOCK LOGIC ===
            # Determine which speaker to follow based on LIP MOVEMENT (who is talking)
            # Uses hysteresis to prevent jitter
            active_entry = None
            
            if assigned_entries:
                if len(assigned_entries) == 1:
                    # Only one face - always use it
                    active_entry = assigned_entries[0]
                    if locked_track_id is None:
                        locked_track_id = active_entry.get("track_id")
                        lock_start_time = ts
                else:
                    # Multiple faces - detect who is SPEAKING based on lip movement
                    total_multi_face_frames += 1
                    
                    # Find the entry for the currently locked speaker
                    locked_entry = None
                    if locked_track_id is not None:
                        for entry in assigned_entries:
                            if entry.get("track_id") == locked_track_id:
                                locked_entry = entry
                                break
                    
                    # Check if we should switch speakers based on LIP ACTIVITY
                    time_with_current = ts - lock_start_time
                    should_switch = False
                    best_other_entry = None
                    
                    if locked_entry is None:
                        # Current speaker not visible - switch to whoever is speaking most
                        should_switch = True
                        # Find entry with most cumulative lip activity
                        best_other_score = -1.0
                        for entry in assigned_entries:
                            tid = entry.get("track_id")
                            score = track_lip_activity.get(tid, 0.0)
                            if score > best_other_score:
                                best_other_score = score
                                best_other_entry = entry
                    elif time_with_current >= SPEAKER_LOCK_MIN_DURATION:
                        # Enough time has passed - check if another speaker has more lip activity
                        current_lip_score = track_lip_activity.get(locked_track_id, 0.0)
                        
                        for entry in assigned_entries:
                            tid = entry.get("track_id")
                            if tid != locked_track_id:
                                other_lip_score = track_lip_activity.get(tid, 0.0)
                                # Only switch if other person has significantly more lip movement
                                # This means they've been speaking more
                                if other_lip_score > current_lip_score * SPEAKER_SWITCH_THRESHOLD:
                                    should_switch = True
                                    lip_speaking_detections += 1
                                    if best_other_entry is None or other_lip_score > track_lip_activity.get(best_other_entry.get("track_id"), 0.0):
                                        best_other_entry = entry
                    
                    if should_switch and best_other_entry is not None:
                        active_entry = best_other_entry
                        locked_track_id = active_entry.get("track_id")
                        lock_start_time = ts
                        # Reset lip activity scores when switching to give new speaker fair tracking
                        track_lip_activity = {locked_track_id: track_lip_activity.get(locked_track_id, 0.0)}
                    else:
                        # Stay with current speaker
                        active_entry = locked_entry if locked_entry else max(assigned_entries, key=lambda d: track_lip_activity.get(d.get("track_id"), 0.0))

            if active_entry is None:
                # No face detected this frame
                if last_center is None:
                    # Haven't seen any face yet; skip this sample
                    continue
                gap = ts - last_detection_time
                if gap > FACE_RECENTER_AFTER:
                    blend = min((gap - FACE_RECENTER_AFTER) / FACE_RECENTER_AFTER, 1.0)
                    last_center = last_center * (1.0 - blend) + 0.5 * blend
                # Apply position smoothing
                if smoothed_center is None:
                    smoothed_center = last_center
                else:
                    smoothed_center = smoothed_center * SPEAKER_POSITION_SMOOTHING + last_center * (1.0 - SPEAKER_POSITION_SMOOTHING)
                samples.append({
                    "time": ts,
                    "center": smoothed_center,
                    "track_id": last_track_id
                })
            else:
                last_center = active_entry["center"]
                last_track_id = active_entry.get("track_id")
                last_detection_time = ts
                # Apply position smoothing to prevent jarring jumps
                if smoothed_center is None:
                    smoothed_center = last_center
                else:
                    smoothed_center = smoothed_center * SPEAKER_POSITION_SMOOTHING + last_center * (1.0 - SPEAKER_POSITION_SMOOTHING)
                samples.append({
                    "time": ts,
                    "center": smoothed_center,
                    "track_id": last_track_id
                })

    # Clean up detectors
    if face_landmarker is not None and hasattr(face_landmarker, "close"):
        face_landmarker.close()

    # If no faces were detected at all, return a single centered sample
    if not samples:
        print(f"[FaceTracking] No faces detected in clip {clip_start:.2f}-{clip_end:.2f}s; using center crop.")
        samples = [
            {"time": clip_start, "center": 0.5, "track_id": None},
            {"time": clip_end, "center": 0.5, "track_id": None}
        ]
    else:
        # Log tracking summary with lip detection stats
        centers = [s["center"] for s in samples]
        avg_center = sum(centers) / len(centers)
        
        # Count unique track IDs to see how many speaker switches occurred
        track_changes = 0
        prev_tid = None
        for s in samples:
            if s.get("track_id") != prev_tid and prev_tid is not None:
                track_changes += 1
            prev_tid = s.get("track_id")
        
        # Log lip activity summary
        lip_summary = ", ".join([f"track{tid}={score:.4f}" for tid, score in sorted(track_lip_activity.items())])
        print(f"[FaceTracking] Clip {clip_start:.2f}-{clip_end:.2f}s: "
              f"{len(samples)} samples, avg center={avg_center:.3f}, "
              f"range=[{min(centers):.3f}, {max(centers):.3f}], "
              f"speaker switches={track_changes}")
        if detector_backend == "landmarker":
            print(f"[FaceTracking] Lip activity: {lip_summary}")
            print(f"[FaceTracking] Multi-face frames: {total_multi_face_frames}, lip-based switches: {lip_speaking_detections}")

    return samples


def build_speaker_segments(
    samples: List[Dict],
    clip_start: float,
    clip_end: float
) -> List[Dict]:
    if not samples:
        # Return a single centered segment for the entire clip
        return [{"start": clip_start, "end": clip_end, "center": 0.5}]

    samples = sorted(samples, key=lambda s: s["time"])
    
    # Backfill: if first sample is after clip_start, prepend a sample at clip_start
    # using the first detected center to avoid initial center=0.5 drift
    if samples[0]["time"] > clip_start + 0.01:
        samples.insert(0, {
            "time": clip_start,
            "center": samples[0]["center"],
            "track_id": samples[0].get("track_id")
        })

    current_start = clip_start
    current_centers = [samples[0]["center"]]
    current_track = samples[0].get("track_id")
    segments: List[Dict] = []

    for sample in samples[1:]:
        ts = min(max(sample["time"], clip_start), clip_end)
        if ts <= current_start:
            current_centers.append(sample["center"])
            continue

        sample_track = sample.get("track_id")
        if sample_track != current_track and current_centers:
            center_val = statistics.median(current_centers) if current_centers else 0.5
            segments.append({
                "start": current_start,
                "end": ts,
                "center": center_val
            })
            current_start = ts
            current_centers = [sample["center"]]
            current_track = sample_track
        else:
            current_centers.append(sample["center"])

    if current_start < clip_end:
        center_val = statistics.median(current_centers) if current_centers else 0.5
        segments.append({
            "start": current_start,
            "end": clip_end,
            "center": center_val
        })

    # === PHASE 1: Absorb very short segments into neighbors ===
    # Segments shorter than MIN_SEGMENT_DURATION get absorbed by the longer neighbor
    # if positions are similar enough
    while True:
        changed = False
        new_segments: List[Dict] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            seg_duration = seg["end"] - seg["start"]
            
            if seg_duration < MIN_SEGMENT_DURATION and len(segments) > 1:
                # This segment is too short - try to absorb it
                prev_seg = new_segments[-1] if new_segments else None
                next_seg = segments[i + 1] if i + 1 < len(segments) else None
                
                # Check which neighbor is closer in position
                prev_dist = abs(seg["center"] - prev_seg["center"]) if prev_seg else float('inf')
                next_dist = abs(seg["center"] - next_seg["center"]) if next_seg else float('inf')
                
                if prev_dist <= SEGMENT_ABSORB_THRESHOLD and prev_dist <= next_dist and prev_seg:
                    # Absorb into previous segment
                    prev_seg["end"] = seg["end"]
                    changed = True
                elif next_dist <= SEGMENT_ABSORB_THRESHOLD and next_seg:
                    # Absorb into next segment (extend next segment's start backwards)
                    next_seg["start"] = seg["start"]
                    # Blend center slightly toward absorbed segment
                    next_seg["center"] = (next_seg["center"] * 0.8 + seg["center"] * 0.2)
                    changed = True
                else:
                    # Can't absorb - keep the segment
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
            i += 1
        
        segments = new_segments
        if not changed:
            break
    
    # === PHASE 2: Merge adjacent segments with similar positions ===
    merged_segments: List[Dict] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        prev = merged_segments[-1]
        
        # Merge if positions are similar
        if abs(seg["center"] - prev["center"]) < SEGMENT_MERGE_THRESHOLD:
            # Weighted average of centers based on duration
            prev_dur = prev["end"] - prev["start"]
            seg_dur = seg["end"] - seg["start"]
            total_dur = prev_dur + seg_dur
            if total_dur > 0:
                prev["center"] = (prev["center"] * prev_dur + seg["center"] * seg_dur) / total_dur
            prev["end"] = seg["end"]
        else:
            merged_segments.append(seg)

    # === PHASE 3: Final cleanup - remove any remaining micro-segments ===
    final_segments: List[Dict] = []
    for seg in merged_segments:
        seg_duration = seg["end"] - seg["start"]
        if seg_duration < 0.1 and final_segments:
            # Very tiny segment - just extend previous
            final_segments[-1]["end"] = seg["end"]
        else:
            final_segments.append(seg)
    
    # Log segment summary
    if final_segments:
        print(f"[Segments] Built {len(final_segments)} segments from {len(samples)} samples")
        for i, seg in enumerate(final_segments):
            dur = seg["end"] - seg["start"]
            print(f"  Seg {i+1}: {seg['start']-clip_start:.2f}s-{seg['end']-clip_start:.2f}s (dur={dur:.2f}s, center={seg['center']:.3f})")

    return final_segments if final_segments else merged_segments


def crop_x_expression_for_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    target_w: int
) -> Optional[str]:
    if not segments:
        return None

    duration = clip_end - clip_start
    base_expr = f"(in_w-{target_w})/2"

    expr = base_expr
    for seg in reversed(segments):
        rel_start = max(seg["start"] - clip_start, 0.0)
        rel_end = min(seg["end"] - clip_start, duration)
        if rel_end <= rel_start:
            continue
        center_expr = (
            f"clip(({seg['center']:.4f})*in_w-{target_w}/2,0,in_w-{target_w})"
        )
        expr = (
            f"if(between(t,{rel_start:.3f},{rel_end:.3f}),"
            f"{center_expr},{expr})"
        )

    return expr


# ==========================
# STEP 5A-2: MULTI-SPEAKER LAYOUT DETECTION
# ==========================

def compute_multi_speaker_samples(
    input_video: str,
    clip_start: float,
    clip_end: float,
    analysis_fps: float = FACE_ANALYSIS_FPS
) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Enhanced face tracking that returns:
    1. samples: List of frame samples with all detected faces
    2. tracks: Dict mapping track_id to list of detections over time
    
    Each sample contains:
    - time: timestamp
    - faces: list of detected faces with center, width, activity
    - num_faces: count of faces detected
    """
    if cv2 is None:
        return [], {}

    detector_backend = "none"
    face_landmarker = None
    cascade = None

    # Priority 1: FaceLandmarker
    if MP_HAS_TASKS:
        model_path = ensure_face_landmarker_model()
        if model_path:
            try:
                base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
                landmarker_options = mp_tasks_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=4,
                    min_face_detection_confidence=FACE_MIN_CONFIDENCE,
                    min_face_presence_confidence=FACE_MIN_CONFIDENCE,
                    min_tracking_confidence=0.5
                )
                face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(landmarker_options)
                detector_backend = "landmarker"
            except Exception as exc:
                print(f"[MultiSpeaker] FaceLandmarker unavailable ({exc}); falling back.")

    # Priority 2: Haar cascade
    if detector_backend == "none":
        cascade = _load_haar_cascade()
        if cascade is None:
            print("[MultiSpeaker] No face detector available.")
            return [], {}
        detector_backend = "haar"

    print(f"[MultiSpeaker] Using '{detector_backend}' for layout detection")

    tracks: List[Dict] = []
    next_track_id = 0
    samples: List[Dict] = []
    track_detections: Dict[int, List[Dict]] = {}
    track_mouth_history: Dict[int, List[float]] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        frames_with_ts = _extract_frames_with_ffmpeg(
            input_video,
            clip_start,
            clip_end,
            analysis_fps,
            temp_dir
        )
        
        if not frames_with_ts:
            return [], {}
        
        for ts, frame_path in frames_with_ts:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if detector_backend == "landmarker" and face_landmarker is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = _detect_faces_with_landmarker(face_landmarker, rgb)
            else:
                detections = _detect_faces_with_cascade(cascade, frame)

            frame_faces = []
            for det in detections:
                # Match to existing track
                best_track = None
                best_dist = 1.0
                for track in tracks:
                    if ts - track["last_time"] > FACE_TRACK_MAX_GAP:
                        continue
                    dist = abs(det["center"] - track["center"])
                    if dist < FACE_TRACK_DISTANCE and dist < best_dist:
                        best_track = track
                        best_dist = dist

                if best_track is None:
                    best_track = {
                        "id": next_track_id,
                        "center": det["center"],
                        "last_time": ts
                    }
                    tracks.append(best_track)
                    track_detections[next_track_id] = []
                    next_track_id += 1

                best_track["center"] = det["center"]
                best_track["last_time"] = ts
                tid = best_track["id"]
                
                # Compute lip movement
                mouth_open = det.get("mouth_open", 0.0)
                if tid not in track_mouth_history:
                    track_mouth_history[tid] = []
                
                history = track_mouth_history[tid]
                lip_movement = 0.0
                if len(history) >= 1:
                    lip_movement = abs(mouth_open - history[-1])
                
                history.append(mouth_open)
                if len(history) > LIP_MOVEMENT_HISTORY_FRAMES:
                    history.pop(0)
                
                face_info = {
                    "track_id": tid,
                    "center": det["center"],
                    "width": det.get("width", 0.1),
                    "activity": lip_movement,
                    "time": ts
                }
                frame_faces.append(face_info)
                track_detections[tid].append(face_info)
            
            samples.append({
                "time": ts,
                "faces": frame_faces,
                "num_faces": len(frame_faces)
            })

    if face_landmarker is not None and hasattr(face_landmarker, "close"):
        face_landmarker.close()

    return samples, track_detections


def determine_layout_segments(
    samples: List[Dict],
    track_detections: Dict[int, List[Dict]],
    clip_start: float,
    clip_end: float
) -> List[Dict]:
    """
    Analyze multi-speaker samples to determine optimal layout for each time segment.
    
    Returns list of layout segments:
    - start, end: time range
    - layout: "single", "split", or "wide"
    - faces: list of face info (for split/wide modes)
    - active_track: track_id of active speaker (for single mode)
    """
    if not samples or LAYOUT_MODE == "single":
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": None}]
    
    if LAYOUT_MODE in ("split", "wide"):
        # Force specific layout
        return [{"start": clip_start, "end": clip_end, "layout": LAYOUT_MODE, "faces": []}]
    
    # Auto mode: analyze face positions and activity
    layout_samples = []
    
    # Compute cumulative activity per track
    track_activity: Dict[int, float] = {}
    for tid, detections in track_detections.items():
        track_activity[tid] = sum(d.get("activity", 0) for d in detections)
    
    # Find the two most active tracks (likely the main speakers)
    sorted_tracks = sorted(track_activity.items(), key=lambda x: x[1], reverse=True)
    main_tracks = [t[0] for t in sorted_tracks[:2]] if len(sorted_tracks) >= 2 else []
    
    if len(main_tracks) < 2:
        # Only one speaker detected - use single mode
        print(f"[Layout] Only {len(main_tracks)} speaker(s) detected, using single mode")
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": main_tracks[0] if main_tracks else None}]
    
    # Analyze each frame to determine layout
    for sample in samples:
        ts = sample["time"]
        faces = sample["faces"]
        
        # Find faces belonging to main tracks
        main_faces = [f for f in faces if f.get("track_id") in main_tracks]
        
        if len(main_faces) < 2:
            # One or no main speakers visible - single mode following most active visible
            visible_tracks = [f.get("track_id") for f in faces]
            active = main_tracks[0] if main_tracks[0] in visible_tracks else (main_tracks[1] if len(main_tracks) > 1 and main_tracks[1] in visible_tracks else None)
            layout_samples.append({
                "time": ts,
                "layout": "single",
                "active_track": active,
                "faces": faces
            })
        else:
            # Both main speakers visible - check distance
            face1 = next(f for f in main_faces if f.get("track_id") == main_tracks[0])
            face2 = next(f for f in main_faces if f.get("track_id") == main_tracks[1])
            
            distance = abs(face1["center"] - face2["center"])
            
            # Check activity levels
            act1 = face1.get("activity", 0)
            act2 = face2.get("activity", 0)
            max_act = max(act1, act2, 0.001)
            both_active = (act1 / max_act > LAYOUT_BOTH_SPEAKING_RATIO and 
                          act2 / max_act > LAYOUT_BOTH_SPEAKING_RATIO)
            
            if distance < LAYOUT_WIDE_THRESHOLD or both_active:
                # Close together or both speaking - wide shot
                layout_samples.append({
                    "time": ts,
                    "layout": "wide",
                    "faces": main_faces,
                    "center": (face1["center"] + face2["center"]) / 2
                })
            elif distance > LAYOUT_SPLIT_THRESHOLD:
                # Far apart - split screen
                layout_samples.append({
                    "time": ts,
                    "layout": "split",
                    "faces": sorted(main_faces, key=lambda f: f["center"]),  # Left to right
                    "left_center": min(face1["center"], face2["center"]),
                    "right_center": max(face1["center"], face2["center"])
                })
            else:
                # Medium distance - follow most active speaker
                active_track = main_tracks[0] if act1 >= act2 else main_tracks[1]
                layout_samples.append({
                    "time": ts,
                    "layout": "single",
                    "active_track": active_track,
                    "faces": faces
                })
    
    # Consolidate into segments with minimum duration
    if not layout_samples:
        return [{"start": clip_start, "end": clip_end, "layout": "single", "active_track": None}]
    
    segments = []
    current_layout = layout_samples[0]["layout"]
    current_start = clip_start
    current_data = layout_samples[0]
    
    for sample in layout_samples[1:]:
        if sample["layout"] != current_layout:
            # Layout change - check if current segment is long enough
            segment_duration = sample["time"] - current_start
            
            if segment_duration >= LAYOUT_SEGMENT_MIN_DURATION:
                # Save current segment
                seg = {
                    "start": current_start,
                    "end": sample["time"],
                    "layout": current_layout
                }
                if current_layout == "single":
                    seg["active_track"] = current_data.get("active_track")
                elif current_layout == "split":
                    seg["left_center"] = current_data.get("left_center", 0.25)
                    seg["right_center"] = current_data.get("right_center", 0.75)
                elif current_layout == "wide":
                    seg["center"] = current_data.get("center", 0.5)
                segments.append(seg)
                
                current_start = sample["time"]
                current_layout = sample["layout"]
                current_data = sample
            # Else: segment too short, keep current layout
    
    # Add final segment
    seg = {
        "start": current_start,
        "end": clip_end,
        "layout": current_layout
    }
    if current_layout == "single":
        seg["active_track"] = current_data.get("active_track")
    elif current_layout == "split":
        seg["left_center"] = current_data.get("left_center", 0.25)
        seg["right_center"] = current_data.get("right_center", 0.75)
    elif current_layout == "wide":
        seg["center"] = current_data.get("center", 0.5)
    segments.append(seg)
    
    # Log layout segments
    layout_counts = {"single": 0, "split": 0, "wide": 0}
    for seg in segments:
        layout_counts[seg["layout"]] = layout_counts.get(seg["layout"], 0) + 1
    print(f"[Layout] Detected {len(segments)} layout segments: "
          f"single={layout_counts['single']}, split={layout_counts['split']}, wide={layout_counts['wide']}")
    
    return segments


def analyze_best_layout(
    samples: List[Dict],
    track_detections: Dict[int, List[Dict]],
    clip_start: float,
    clip_end: float
) -> Dict:
    """
    Analyze multi-speaker samples and recommend the best layout for this clip.
    
    Returns a dict with:
    - recommended_layout: "single", "split", or "wide"
    - confidence: 0.0-1.0 score for the recommendation
    - metrics: detailed analysis metrics
    - reason: human-readable explanation
    """
    clip_duration = clip_end - clip_start
    
    if not samples or not track_detections:
        return {
            "recommended_layout": "single",
            "confidence": 1.0,
            "metrics": {},
            "reason": "No face data available - defaulting to single speaker mode"
        }
    
    # Compute metrics
    num_tracks = len(track_detections)
    
    # Count frames with 0, 1, 2+ faces
    frames_0_faces = sum(1 for s in samples if s["num_faces"] == 0)
    frames_1_face = sum(1 for s in samples if s["num_faces"] == 1)
    frames_2plus_faces = sum(1 for s in samples if s["num_faces"] >= 2)
    total_frames = len(samples)
    
    pct_2plus = frames_2plus_faces / max(total_frames, 1)
    pct_1_face = frames_1_face / max(total_frames, 1)
    pct_0_faces = frames_0_faces / max(total_frames, 1)
    
    # Compute average face distance when 2+ faces visible
    distances = []
    both_active_count = 0
    
    for sample in samples:
        faces = sample.get("faces", [])
        if len(faces) >= 2:
            # Get the two most prominent faces (by activity or position)
            sorted_faces = sorted(faces, key=lambda f: f.get("activity", 0), reverse=True)[:2]
            dist = abs(sorted_faces[0]["center"] - sorted_faces[1]["center"])
            distances.append(dist)
            
            # Check if both are speaking
            act1 = sorted_faces[0].get("activity", 0)
            act2 = sorted_faces[1].get("activity", 0)
            max_act = max(act1, act2, 0.001)
            if act1 / max_act > 0.3 and act2 / max_act > 0.3:
                both_active_count += 1
    
    avg_distance = sum(distances) / len(distances) if distances else 0
    pct_both_active = both_active_count / max(len(distances), 1) if distances else 0
    
    # Compute activity per track
    track_activities = {}
    for tid, detections in track_detections.items():
        total_activity = sum(d.get("activity", 0) for d in detections)
        track_activities[tid] = total_activity
    
    # Sort tracks by activity
    sorted_tracks = sorted(track_activities.items(), key=lambda x: x[1], reverse=True)
    
    # Activity ratio between top 2 speakers
    if len(sorted_tracks) >= 2:
        top_activity = sorted_tracks[0][1]
        second_activity = sorted_tracks[1][1]
        activity_ratio = second_activity / max(top_activity, 0.001)
    else:
        activity_ratio = 0
    
    # Build metrics dict
    metrics = {
        "num_tracks": num_tracks,
        "total_frames": total_frames,
        "pct_2plus_faces": round(pct_2plus * 100, 1),
        "pct_1_face": round(pct_1_face * 100, 1),
        "pct_0_faces": round(pct_0_faces * 100, 1),
        "avg_face_distance": round(avg_distance, 3),
        "pct_both_speaking": round(pct_both_active * 100, 1),
        "activity_ratio": round(activity_ratio, 2),
    }
    
    # Decision logic
    # Priority 1: If rarely 2+ faces, use single mode
    if pct_2plus < 0.3:
        return {
            "recommended_layout": "single",
            "confidence": 0.9,
            "metrics": metrics,
            "reason": f"Single speaker dominant ({pct_1_face*100:.0f}% frames with 1 face)"
        }
    
    # Priority 2: If 2+ faces often visible
    if pct_2plus >= 0.3:
        # Check distance to decide between split and wide
        if avg_distance > LAYOUT_SPLIT_THRESHOLD:
            # Faces are far apart - split screen is good
            # But if both are active often, might prefer wide
            if pct_both_active > 0.4:
                return {
                    "recommended_layout": "wide",
                    "confidence": 0.75,
                    "metrics": metrics,
                    "reason": f"Both speakers active ({pct_both_active*100:.0f}% of time), distance={avg_distance:.2f}"
                }
            else:
                return {
                    "recommended_layout": "split",
                    "confidence": 0.85,
                    "metrics": metrics,
                    "reason": f"Two speakers far apart (dist={avg_distance:.2f}), alternating speech"
                }
        
        elif avg_distance < LAYOUT_WIDE_THRESHOLD:
            # Faces are close together - wide shot
            return {
                "recommended_layout": "wide",
                "confidence": 0.8,
                "metrics": metrics,
                "reason": f"Speakers close together (dist={avg_distance:.2f}), wide shot shows both"
            }
        
        else:
            # Medium distance - depends on activity pattern
            if activity_ratio > 0.5 and pct_both_active > 0.3:
                return {
                    "recommended_layout": "wide",
                    "confidence": 0.65,
                    "metrics": metrics,
                    "reason": f"Balanced conversation (activity ratio={activity_ratio:.1f})"
                }
            else:
                return {
                    "recommended_layout": "single",
                    "confidence": 0.6,
                    "metrics": metrics,
                    "reason": f"One dominant speaker (activity ratio={activity_ratio:.1f})"
                }
    
    # Default fallback
    return {
        "recommended_layout": "single",
        "confidence": 0.5,
        "metrics": metrics,
        "reason": "Default fallback to single mode"
    }


# ==========================
# STEP 5: CREATE VERTICAL CLIP WITH SUBS (FFMPEG)
# ==========================

def create_vertical_with_subs(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30,
    crop_x_expr: Optional[str] = None
) -> None:
    """
    Use ffmpeg to:
    - seek to `start`, cut `duration`
    - scale to fit target height
    - center crop
    - burn subtitles (ASS or SRT)
    """
    cmd = ["ffmpeg", "-y"]

    # Seek before input for faster cutting
    cmd += ["-ss", f"{start:.3f}", "-i", input_video]
    cmd += ["-t", f"{duration:.3f}"]

    if crop_x_expr is None:
        x_expr = f"(in_w-{target_w})/2"
    else:
        # Escape single quotes in the expression for ffmpeg filter syntax
        x_expr = crop_x_expr.replace("'", "\\'")

    # Determine subtitle filter based on file extension
    # ASS files use 'ass' filter, SRT files use 'subtitles' filter
    sub_ext = os.path.splitext(sub_path)[1].lower()
    if sub_ext == ".ass":
        # Escape special characters in path for ffmpeg
        escaped_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"ass='{escaped_path}'"
    else:
        sub_filter = f"subtitles='{sub_path}'"

    # Use filter_complex with proper quoting to handle expressions with decimals.
    # The crop expression must be escaped to prevent ffmpeg from misinterpreting
    # decimal points (e.g., "0.000") as filter chain separators.
    vf = (
        f"scale=-2:{target_h},"
        f"crop={target_w}:{target_h}:'{x_expr}':0,"
        f"{sub_filter}"
    )

    cmd += [
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]

    run_ffmpeg_cmd(cmd)
    print(f"[Output] Wrote '{output_video}'.")


def create_split_screen_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    left_center: float,
    right_center: float,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30
) -> None:
    """
    Create a vertical split-screen clip showing two speakers stacked vertically.
    Each panel zooms in on the face of each speaker, like OpusClip.
    The faces are centered and cropped to show head/shoulders.
    """
    panel_h = (target_h - LAYOUT_SPLIT_GAP) // 2
    panel_w = target_w
    
    # For face-focused panels, we need to:
    # 1. Scale video so we can extract a square-ish region around each face
    # 2. Crop around each face's position to get a proper head/shoulders shot
    # The aspect ratio of each panel is 1080x~950, which is wider than 1:1
    # We want to zoom in more on the face, so use a smaller source crop
    
    cmd = ["ffmpeg", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-i", input_video]
    cmd += ["-t", f"{duration:.3f}"]
    
    # Escape subtitle path
    sub_ext = os.path.splitext(sub_path)[1].lower()
    escaped_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    if sub_ext == ".ass":
        sub_filter = f"ass='{escaped_path}'"
    else:
        sub_filter = f"subtitles='{sub_path}'"
    
    # For split screen with face focus:
    # - Scale source to a height where we can get good face crops
    # - Crop a region centered on each face
    # - The crop should be zoomed in enough to show head/shoulders
    # 
    # Each panel is 1080 x panel_h (~950px)
    # We want to crop from source in a way that:
    # 1. Centers on the face horizontally (using left_center, right_center)
    # 2. Crops from roughly upper-third of frame (where faces typically are)
    
    # Scale source to be wider than panel so we can pan to each face
    # Then crop a panel-sized region centered on each speaker
    scale_h = panel_h  # Scale to exact panel height
    
    # X position for each face (centered on their position)
    left_x = f"clip(({left_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    right_x = f"clip(({right_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    
    # Y position: crop from upper portion where faces are (about 10% from top)
    face_y_offset = 0.1  # Start crop 10% from top of frame
    
    # Complex filter:
    # 1. Create scaled version for top panel (left speaker)
    # 2. Create scaled version for bottom panel (right speaker)
    # 3. Crop each around their face position
    # 4. Stack with gap
    filter_complex = (
        f"[0:v]scale=-2:{scale_h*2}[src];"
        f"[src]crop={panel_w}:{panel_h}:'{left_x}':{int(scale_h*2*face_y_offset)}[top];"
        f"[src]crop={panel_w}:{panel_h}:'{right_x}':{int(scale_h*2*face_y_offset)}[bottom];"
        f"color=black:{target_w}x{LAYOUT_SPLIT_GAP}:d={duration}[gap];"
        f"[top][gap][bottom]vstack=inputs=3[stacked];"
        f"[stacked]{sub_filter}[out]"
    )
    
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]
    
    run_ffmpeg_cmd(cmd)
    print(f"[Output] Wrote split-screen clip '{output_video}'.")


def create_wide_shot_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    center: float,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30
) -> None:
    """
    Create a vertical clip with wide shot showing both speakers.
    Uses letterbox style: 16:9 video strip in middle with blurred background above/below.
    This matches the OpusClip wide shot style.
    """
    # Calculate the video strip dimensions (16:9 aspect ratio in the middle)
    video_strip_h = int(target_w * 9 / 16)  # 16:9 aspect ratio gives ~607px height for 1080 width
    top_padding = (target_h - video_strip_h) // 2
    bottom_padding = target_h - video_strip_h - top_padding
    
    cmd = ["ffmpeg", "-y"]
    cmd += ["-ss", f"{start:.3f}", "-i", input_video]
    cmd += ["-t", f"{duration:.3f}"]
    
    # Escape subtitle path
    sub_ext = os.path.splitext(sub_path)[1].lower()
    escaped_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    if sub_ext == ".ass":
        sub_filter = f"ass='{escaped_path}'"
    else:
        sub_filter = f"subtitles='{sub_path}'"
    
    # Complex filter:
    # 1. Create blurred background scaled to full height
    # 2. Create sharp video strip scaled to 16:9 centered
    # 3. Overlay the sharp strip on the blurred background
    # 4. Add subtitles
    # Calculate horizontal crop position for the wide shot
    # We crop from the original video to show both speakers
    filter_complex = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur=20:5[bg];"
        f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[strip];"
        f"[bg][strip]overlay=(W-w)/2:(H-h)/2[comp];"
        f"[comp]{sub_filter}[out]"
    )
    
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_video
    ]
    
    run_ffmpeg_cmd(cmd)
    print(f"[Output] Wrote wide-shot clip '{output_video}'.")


def create_multi_layout_clip(
    input_video: str,
    sub_path: str,
    output_video: str,
    start: float,
    duration: float,
    layout_segments: List[Dict],
    speaker_segments: List[Dict],
    clip_start: float,
    clip_end: float,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30
) -> None:
    """
    Create a clip with DYNAMIC layout switching based on layout_segments.
    Each segment is rendered with its own layout (single/split/wide),
    then concatenated together, with subtitles applied at the end.
    
    This mimics professional tools like OpusClip that switch layouts
    throughout a clip based on who is speaking.
    """
    if not layout_segments:
        # Fallback to single mode
        crop_expr = crop_x_expression_for_segments(speaker_segments, clip_start, clip_end, target_w)
        create_vertical_with_subs(
            input_video, sub_path, output_video, start, duration,
            target_w, target_h, fps, crop_expr
        )
        return
    
    # If only one segment, render it directly with subs
    if len(layout_segments) == 1:
        seg = layout_segments[0]
        seg_layout = seg["layout"]
        
        if seg_layout == "split":
            create_split_screen_clip(
                input_video, sub_path, output_video, start, duration,
                seg.get("left_center", 0.25), seg.get("right_center", 0.75),
                target_w, target_h, fps
            )
        elif seg_layout == "wide":
            create_wide_shot_clip(
                input_video, sub_path, output_video, start, duration,
                seg.get("center", 0.5), target_w, target_h, fps
            )
        else:
            crop_expr = crop_x_expression_for_segments(speaker_segments, clip_start, clip_end, target_w)
            create_vertical_with_subs(
                input_video, sub_path, output_video, start, duration,
                target_w, target_h, fps, crop_expr
            )
        return
    
    # Multiple segments - use dynamic switching
    print(f"[Layout] Creating dynamic layout clip with {len(layout_segments)} segments")
    for i, seg in enumerate(layout_segments):
        seg_dur = seg["end"] - seg["start"]
        print(f"  Segment {i+1}: {seg['layout'].upper()} ({seg_dur:.2f}s)")
    
    # Create temp directory for segment clips
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        for i, seg in enumerate(layout_segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_layout = seg["layout"]
            seg_duration = seg_end - seg_start
            
            temp_output = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            
            if seg_layout == "split":
                _render_split_segment(
                    input_video, temp_output, seg_start, seg_duration,
                    seg.get("left_center", 0.25), seg.get("right_center", 0.75),
                    target_w, target_h, fps
                )
            elif seg_layout == "wide":
                _render_wide_segment(
                    input_video, temp_output, seg_start, seg_duration,
                    seg.get("center", 0.5), target_w, target_h, fps
                )
            else:  # single
                # Get crop expression for this time range from speaker segments
                crop_expr = _get_crop_for_time_range(
                    speaker_segments, seg_start, seg_end, clip_start, target_w
                )
                _render_single_segment(
                    input_video, temp_output, seg_start, seg_duration,
                    crop_expr, target_w, target_h, fps
                )
            
            segment_files.append(temp_output)
        
        # Create concat list file
        concat_list = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")
        
        # Concatenate segments (without subtitles yet)
        temp_concat = os.path.join(temp_dir, "concat_nosubs.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            temp_concat
        ]
        print("[FFmpeg] Concatenating layout segments...")
        subprocess.run(concat_cmd, check=True, capture_output=True)
        
        # Add subtitles to final output
        sub_ext = os.path.splitext(sub_path)[1].lower()
        escaped_path = sub_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        if sub_ext == ".ass":
            sub_filter = f"ass='{escaped_path}'"
        else:
            sub_filter = f"subtitles='{sub_path}'"
        
        final_cmd = [
            "ffmpeg", "-y",
            "-i", temp_concat,
            "-vf", sub_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_video
        ]
        run_ffmpeg_cmd(final_cmd)
        print(f"[Output] Wrote dynamic layout clip '{output_video}'.")


def _render_single_segment(
    input_video: str,
    output_path: str,
    start: float,
    duration: float,
    crop_x_expr: Optional[str],
    target_w: int,
    target_h: int,
    fps: int
) -> None:
    """Render a single-speaker segment without subtitles."""
    if crop_x_expr is None:
        x_expr = f"(in_w-{target_w})/2"
    else:
        x_expr = crop_x_expr.replace("'", "\\'")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-vf", f"scale=-2:{target_h},crop={target_w}:{target_h}:'{x_expr}':0",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _render_split_segment(
    input_video: str,
    output_path: str,
    start: float,
    duration: float,
    left_center: float,
    right_center: float,
    target_w: int,
    target_h: int,
    fps: int
) -> None:
    """Render a split-screen segment without subtitles (face-focused panels)."""
    panel_h = (target_h - LAYOUT_SPLIT_GAP) // 2
    panel_w = target_w
    scale_h = panel_h
    
    # X position for each face (centered on their position)
    left_x = f"clip(({left_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    right_x = f"clip(({right_center:.4f})*iw-{panel_w}/2,0,iw-{panel_w})"
    
    # Y position: crop from upper portion where faces are
    face_y_offset = 0.1
    
    filter_complex = (
        f"[0:v]scale=-2:{scale_h*2}[src];"
        f"[src]crop={panel_w}:{panel_h}:'{left_x}':{int(scale_h*2*face_y_offset)}[top];"
        f"[src]crop={panel_w}:{panel_h}:'{right_x}':{int(scale_h*2*face_y_offset)}[bottom];"
        f"color=black:{target_w}x{LAYOUT_SPLIT_GAP}:d={duration}[gap];"
        f"[top][gap][bottom]vstack=inputs=3[out]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _render_wide_segment(
    input_video: str,
    output_path: str,
    start: float,
    duration: float,
    center: float,
    target_w: int,
    target_h: int,
    fps: int
) -> None:
    """Render a wide-shot segment without subtitles (letterbox style with blurred background)."""
    # Create letterbox: blurred background with sharp 16:9 video strip in middle
    filter_complex = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"boxblur=20:5[bg];"
        f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[strip];"
        f"[bg][strip]overlay=(W-w)/2:(H-h)/2[out]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-r", str(fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _get_crop_for_time_range(
    speaker_segments: List[Dict],
    seg_start: float,
    seg_end: float,
    clip_start: float,
    target_w: int
) -> Optional[str]:
    """
    Extract the crop expression for a specific time range within the clip.
    Filters speaker_segments to only include those overlapping with seg_start-seg_end.
    """
    if not speaker_segments:
        return None
    
    # Filter segments that overlap with our time range
    relevant_segments = []
    for spk_seg in speaker_segments:
        spk_start = spk_seg["start"]
        spk_end = spk_seg["end"]
        
        # Check for overlap
        if spk_end <= seg_start or spk_start >= seg_end:
            continue
        
        # Clip to our range
        clipped_start = max(spk_start, seg_start)
        clipped_end = min(spk_end, seg_end)
        
        relevant_segments.append({
            "start": clipped_start,
            "end": clipped_end,
            "center": spk_seg["center"]
        })
    
    if not relevant_segments:
        return None
    
    # Build crop expression for these segments
    # Note: times need to be relative to seg_start (which becomes t=0 in the segment)
    base_expr = f"(in_w-{target_w})/2"
    expr = base_expr
    
    for seg in reversed(relevant_segments):
        rel_start = max(seg["start"] - seg_start, 0.0)
        rel_end = min(seg["end"] - seg_start, seg_end - seg_start)
        
        if rel_end <= rel_start:
            continue
        
        center_expr = f"clip(({seg['center']:.4f})*in_w-{target_w}/2,0,in_w-{target_w})"
        expr = f"if(between(t,{rel_start:.3f},{rel_end:.3f}),{center_expr},{expr})"
    
    return expr


# ==========================
# MAIN PIPELINE
# ==========================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Highlights Generator - Create viral short-form clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_highlight.py                    # Basic highlight generation
  python video_highlight.py --hook             # Enable auto-hook overlay
  python video_highlight.py --hook --layout split  # Hook + forced split layout
        """
    )
    
    parser.add_argument(
        "--hook",
        action="store_true",
        help="Enable auto-hook overlay - adds attention-grabbing text at the top of each clip"
    )
    
    parser.add_argument(
        "--layout",
        choices=["auto", "single", "split", "wide"],
        default=None,
        help="Override layout mode (default: uses LAYOUT_MODE from config)"
    )
    
    parser.add_argument(
        "--content-type",
        choices=["coding", "fitness", "gaming"],
        default=None,
        help="Override content type for niche-specific scoring"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input video path"
    )
    
    parser.add_argument(
        "--num-clips",
        type=int,
        default=None,
        help="Number of highlight clips to generate"
    )
    
    return parser.parse_args()


def main():
    global LAYOUT_MODE, CONTENT_TYPE, INPUT_VIDEO, NUM_HIGHLIGHTS, HOOK_ENABLED
    
    args = parse_args()
    
    # Apply CLI overrides
    if args.hook:
        HOOK_ENABLED = True
        print("[Config] Auto-hook overlay ENABLED")
    
    if args.layout:
        LAYOUT_MODE = args.layout
        
    if args.content_type:
        CONTENT_TYPE = args.content_type
        
    if args.input:
        INPUT_VIDEO = args.input
        
    if args.num_clips:
        NUM_HIGHLIGHTS = args.num_clips
    
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    niche_cfg = get_niche_config(CONTENT_TYPE)
    hook_keywords = niche_cfg["HOOK_KEYWORDS"]
    weights = niche_cfg["WEIGHTS"]

    print(f"[Config] CONTENT_TYPE='{CONTENT_TYPE}'")
    print(f"[Config] LAYOUT_MODE='{LAYOUT_MODE}'")
    print(f"[Config] Using niche-specific hooks and weights.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_mtime = os.path.getmtime(INPUT_VIDEO)

    # 1) Transcribe to SRT + get Whisper segments (PyTorch)
    if not FORCE_TRANSCRIBE and cache_is_fresh(SRT_FILE, video_mtime):
        print(f"[Whisper] Reusing cached transcription '{SRT_FILE}'.")
        segments = parse_srt_to_segments(SRT_FILE)
        if not segments:
            print("[Whisper] Cached SRT could not be parsed, re-transcribing...")
            segments = transcribe_to_srt_and_segments(
                INPUT_VIDEO,
                SRT_FILE,
                WHISPER_MODEL_SIZE
            )
    else:
        segments = transcribe_to_srt_and_segments(
            INPUT_VIDEO,
            SRT_FILE,
            WHISPER_MODEL_SIZE
        )

    # 2) Extract audio once (for torchaudio analysis)
    if not FORCE_AUDIO_EXTRACTION and cache_is_fresh(AUDIO_WAV, video_mtime):
        print(f"[Audio] Reusing cached audio '{AUDIO_WAV}'.")
    else:
        extract_audio(INPUT_VIDEO, AUDIO_WAV)

    # 3) Compute audio RMS per segment
    rms_values = compute_rms_per_segment(AUDIO_WAV, segments)

    # 4) Load sentiment model and score segments with niche config
    tokenizer, sentiment_model = load_sentiment_model(SENTIMENT_MODEL_NAME)
    scored_segments = score_segments_for_highlights(
        segments,
        rms_values,
        tokenizer,
        sentiment_model,
        hook_keywords,
        weights
    )

    # 5) Pick up to NUM_HIGHLIGHTS viral-friendly intervals
    highlight_intervals = select_highlight_intervals(
        scored_segments,
        segments,
        num_highlights=NUM_HIGHLIGHTS,
        min_len=MIN_HIGHLIGHT_LEN,
        max_len=MAX_HIGHLIGHT_LEN,
        ctx_before=CONTEXT_BEFORE,
        ctx_after=CONTEXT_AFTER,
        min_gap=MIN_GAP_BETWEEN_HIGHLIGHTS,
        last_word_pad=LAST_WORD_PAD
    )

    if not highlight_intervals:
        raise RuntimeError("No highlight intervals could be selected.")

    generated_outputs = []
    for idx, interval in enumerate(highlight_intervals):
        clip_start = interval["start"]
        clip_end = interval["end"]
        duration = clip_end - clip_start

        if idx == 0:
            output_name = FINAL_OUTPUT
        else:
            output_name = f"highlight_{idx+1:02}.mp4"

        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Generate animated ASS subtitles with word-by-word highlighting
        clip_ass_name = f"{os.path.splitext(output_name)[0]}_subs.ass"
        clip_ass_path = os.path.join(OUTPUT_DIR, clip_ass_name)
        write_clip_ass(segments, clip_start, clip_end, clip_ass_path)

        speaker_segments = []
        layout_segments = []
        recommended_layout = None
        
        if cv2 is not None and mp is not None:
            # Compute multi-speaker layout analysis
            print(f"\n[Clip {idx+1}] Analyzing multi-speaker layout...")
            multi_samples, track_detections = compute_multi_speaker_samples(
                INPUT_VIDEO,
                clip_start,
                clip_end
            )
            
            # Get layout recommendation for this clip
            layout_analysis = analyze_best_layout(
                multi_samples,
                track_detections,
                clip_start,
                clip_end
            )
            recommended_layout = layout_analysis["recommended_layout"]
            
            print(f"[Clip {idx+1}] Layout Analysis:")
            print(f"  Recommended: {recommended_layout.upper()} (confidence: {layout_analysis['confidence']:.0%})")
            print(f"  Reason: {layout_analysis['reason']}")
            metrics = layout_analysis.get("metrics", {})
            if metrics:
                print(f"  Metrics: {metrics.get('num_tracks', 0)} tracks, "
                      f"{metrics.get('pct_2plus_faces', 0):.0f}% multi-face, "
                      f"avg dist={metrics.get('avg_face_distance', 0):.2f}, "
                      f"both speaking={metrics.get('pct_both_speaking', 0):.0f}%")
            
            # Use recommended layout if in auto mode, otherwise use configured mode
            effective_layout = recommended_layout if LAYOUT_MODE == "auto" else LAYOUT_MODE
            
            if effective_layout != "single":
                layout_segments = determine_layout_segments(
                    multi_samples,
                    track_detections,
                    clip_start,
                    clip_end
                )
            
            # Also compute single-speaker tracking for fallback
            samples = compute_speaker_samples(
                INPUT_VIDEO,
                clip_start,
                clip_end
            )
            speaker_segments = build_speaker_segments(samples, clip_start, clip_end)
        
        # Determine effective layout mode for this clip
        effective_layout = "single"
        if LAYOUT_MODE == "auto" and recommended_layout:
            effective_layout = recommended_layout
        elif LAYOUT_MODE != "single":
            effective_layout = LAYOUT_MODE
        
        # Create clip with appropriate layout
        if effective_layout != "single" and layout_segments:
            create_multi_layout_clip(
                INPUT_VIDEO,
                clip_ass_path,
                output_path,
                start=clip_start,
                duration=duration,
                layout_segments=layout_segments,
                speaker_segments=speaker_segments,
                clip_start=clip_start,
                clip_end=clip_end,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
                fps=OUTPUT_FPS
            )
        else:
            # Single speaker mode
            crop_expr = crop_x_expression_for_segments(
                speaker_segments,
                clip_start,
                clip_end,
                TARGET_WIDTH
            )
            create_vertical_with_subs(
                INPUT_VIDEO,
                clip_ass_path,
                output_path,
                start=clip_start,
                duration=duration,
                target_w=TARGET_WIDTH,
                target_h=TARGET_HEIGHT,
                fps=OUTPUT_FPS,
                crop_x_expr=crop_expr
            )

        # Apply auto-hook overlay if enabled
        if HOOK_ENABLED:
            hook_info = detect_hook_phrase(
                segments,
                clip_start,
                clip_end,
                hook_keywords
            )
            
            if hook_info:
                # Apply hook overlay to the generated clip
                hook_output_path = output_path.replace(".mp4", "_hooked.mp4")
                try:
                    apply_hook_overlay(
                        output_path,
                        hook_output_path,
                        hook_info["text"],
                        TARGET_WIDTH,
                        TARGET_HEIGHT
                    )
                    # Replace original with hooked version
                    os.replace(hook_output_path, output_path)
                    print(f"[Hook] Applied hook to clip {idx+1}: \"{hook_info['text'][:50]}...\"" 
                          if len(hook_info['text']) > 50 else f"[Hook] Applied hook to clip {idx+1}: \"{hook_info['text']}\"")
                except subprocess.CalledProcessError as e:
                    print(f"[Hook] Warning: Failed to apply hook overlay: {e}")
                    # Clean up temp file if it exists
                    if os.path.exists(hook_output_path):
                        os.remove(hook_output_path)
            else:
                print(f"[Hook] No suitable hook phrase found for clip {idx+1}")

        generated_outputs.append(output_path)

    print("\n[Done] Generated highlight videos:")
    for path in generated_outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()