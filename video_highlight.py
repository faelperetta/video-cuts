from __future__ import annotations

import os
import math
import subprocess
import urllib.request
import tempfile
import glob
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
MIN_HIGHLIGHT_LEN = 8.0      # seconds
MAX_HIGHLIGHT_LEN = 30.0     # seconds
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
    and return the list of segments.
    """
    print(f"[Whisper] Loading model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"[Whisper] Transcribing '{input_video}'...")
    result = model.transcribe(input_video, task="transcribe")

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
# STEP 5: CREATE VERTICAL CLIP WITH SUBS (FFMPEG)
# ==========================

def create_vertical_with_subs(
    input_video: str,
    srt_path: str,
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
    - burn subtitles
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

    # Use filter_complex with proper quoting to handle expressions with decimals.
    # The crop expression must be escaped to prevent ffmpeg from misinterpreting
    # decimal points (e.g., "0.000") as filter chain separators.
    vf = (
        f"scale=-2:{target_h},"
        f"crop={target_w}:{target_h}:'{x_expr}':0,"
        f"subtitles='{srt_path}'"
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


# ==========================
# MAIN PIPELINE
# ==========================

def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    niche_cfg = get_niche_config(CONTENT_TYPE)
    hook_keywords = niche_cfg["HOOK_KEYWORDS"]
    weights = niche_cfg["WEIGHTS"]

    print(f"[Config] CONTENT_TYPE='{CONTENT_TYPE}'")
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
        clip_srt_name = f"{os.path.splitext(output_name)[0]}_subs.srt"
        clip_srt_path = os.path.join(OUTPUT_DIR, clip_srt_name)

        write_clip_srt(segments, clip_start, clip_end, clip_srt_path)

        crop_expr = None
        if cv2 is not None and mp is not None:
            samples = compute_speaker_samples(
                INPUT_VIDEO,
                clip_start,
                clip_end
            )
            speaker_segments = build_speaker_segments(samples, clip_start, clip_end)
            crop_expr = crop_x_expression_for_segments(
                speaker_segments,
                clip_start,
                clip_end,
                TARGET_WIDTH
            )

        create_vertical_with_subs(
            INPUT_VIDEO,
            clip_srt_path,
            output_path,
            start=clip_start,
            duration=duration,
            target_w=TARGET_WIDTH,
            target_h=TARGET_HEIGHT,
            fps=OUTPUT_FPS,
            crop_x_expr=crop_expr
        )

        generated_outputs.append(output_path)

    print("\n[Done] Generated highlight videos:")
    for path in generated_outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()