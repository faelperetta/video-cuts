from __future__ import annotations

import os
import math
import subprocess
import urllib.request
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
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detection/"
    "short_range/float16/1/face_detection_short_range.tflite"
)
FACE_DETECTOR_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "models",
    "face_detection_short_range.tflite"
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


def _mouth_open_metric(face_landmarks) -> float:
    # FaceMesh landmark indices: 13 upper lip, 14 lower lip
    upper = face_landmarks.landmark[13]
    lower = face_landmarks.landmark[14]
    return abs(lower.y - upper.y)


def _detect_faces_with_mesh(mesh, rgb_frame):
    results = mesh.process(rgb_frame)
    detections = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            xmin = max(min(xs), 0.0)
            xmax = min(max(xs), 1.0)
            width = max(xmax - xmin, 1e-3)
            center = (xmin + xmax) / 2.0
            mouth_open = _mouth_open_metric(face_landmarks)
            detections.append({
                "center": max(0.0, min(1.0, center)),
                "width": min(width, 1.0),
                "activity": mouth_open
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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    detections = []
    for (x, y, fw, fh) in faces:
        center = (x + fw / 2) / max(w, 1)
        width = fw / max(w, 1)
        activity = fh / max(h, 1)
        detections.append({
            "center": max(0.0, min(1.0, center)),
            "width": max(min(width, 1.0), 1e-3),
            "activity": activity
        })
    return detections


def _load_haar_cascade():
    cascade_path = getattr(cv2.data, "haarcascades", "")
    cascade_file = os.path.join(cascade_path, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        return None
    return cascade


def compute_speaker_samples(
    input_video: str,
    clip_start: float,
    clip_end: float,
    analysis_fps: float = FACE_ANALYSIS_FPS
) -> List[Dict]:
    if cv2 is None:
        return []

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return []

    actual_duration = max(clip_end - clip_start, 1e-3)
    total_frames = max(int(actual_duration * analysis_fps), 1)
    sample_step = actual_duration / total_frames

    detector_backend = "none"
    tasks_detector = None
    mesh = None
    cascade = None

    if MP_HAS_TASKS:
        model_path = ensure_face_detector_model()
        if model_path:
            try:
                base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
                detector_options = mp_tasks_vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=FACE_MIN_CONFIDENCE
                )
                tasks_detector = mp_tasks_vision.FaceDetector.create_from_options(detector_options)
                detector_backend = "tasks"
            except Exception as exc:
                print(f"[FaceTracking] Tasks detector unavailable ({exc}); falling back.")

    if detector_backend == "none" and MP_HAS_SOLUTIONS and mp is not None:
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=FACE_MIN_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        detector_backend = "mesh"

    if detector_backend == "none":
        cascade = _load_haar_cascade()
        if cascade is None:
            print("[FaceTracking] Haar cascade not available; skipping speaker framing.")
            cap.release()
            return []
        detector_backend = "haar"

    tracks: List[Dict] = []
    next_track_id = 0
    samples: List[Dict] = []
    last_center = 0.5
    last_track_id: Optional[int] = None
    last_detection_time = clip_start

    with nullcontext():
        for i in range(total_frames + 1):
            rel_offset = min(i * sample_step, actual_duration)
            ts = clip_start + rel_offset
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            success, frame = cap.read()
            if not success or frame is None:
                continue

            detections: List[Dict]
            if detector_backend == "tasks":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = _detect_faces_with_tasks(tasks_detector, rgb)
            elif detector_backend == "mesh" and mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = _detect_faces_with_mesh(mesh, rgb)
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
                assigned_entries.append(det)

            active_entry = None
            if assigned_entries:
                active_entry = max(assigned_entries, key=lambda d: d["activity"])
                if (active_entry["activity"] < SPEAKER_ACTIVITY_THRESHOLD and
                        len(assigned_entries) > 1):
                    active_entry = None

            if active_entry is None:
                gap = ts - last_detection_time
                if gap > FACE_RECENTER_AFTER:
                    blend = min((gap - FACE_RECENTER_AFTER) / FACE_RECENTER_AFTER, 1.0)
                    last_center = last_center * (1.0 - blend) + 0.5 * blend
                samples.append({
                    "time": ts,
                    "center": last_center,
                    "track_id": last_track_id
                })
            else:
                last_center = active_entry["center"]
                last_track_id = active_entry.get("track_id")
                last_detection_time = ts
                samples.append({
                    "time": ts,
                    "center": last_center,
                    "track_id": last_track_id
                })

    cap.release()
    if mesh is not None and hasattr(mesh, "close"):
        mesh.close()
    if tasks_detector is not None and hasattr(tasks_detector, "close"):
        tasks_detector.close()
    return samples


def build_speaker_segments(
    samples: List[Dict],
    clip_start: float,
    clip_end: float
) -> List[Dict]:
    if not samples:
        return []

    samples = sorted(samples, key=lambda s: s["time"])
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

    merged_segments: List[Dict] = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        prev = merged_segments[-1]
        if seg["start"] - prev["end"] < 0.05 and abs(seg["center"] - prev["center"]) < 0.02:
            prev["end"] = seg["end"]
            prev["center"] = (prev["center"] + seg["center"]) / 2.0
        else:
            merged_segments.append(seg)

    return merged_segments


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
        x_expr = crop_x_expr

    vf = (
        f"scale=-2:{target_h},"
        f"crop={target_w}:{target_h}:{x_expr}:0,"
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