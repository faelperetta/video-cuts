import torch
import torchaudio
import warnings
import logging
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from videocuts.utils.system import run_ffmpeg_cmd
from videocuts.utils.device import TORCH_DEVICE

logger = logging.getLogger(__name__)

def extract_audio(input_video: str, audio_path: str) -> None:
    """Extract audio from the input video to an MP3 file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",                # no video
        "-acodec", "libmp3lame",
        "-b:a", "64k",        # 64kbps (sufficient for RMS analysis)
        "-ar", "16000",       # 16kHz
        "-ac", "1",           # mono
        audio_path
    ]
    run_ffmpeg_cmd(cmd)
    logger.info(f"Extracted audio to '{audio_path}'.")

def compute_rms_per_segment(
    audio_path: str,
    segments: List[Dict]
) -> List[float]:
    """Compute RMS loudness for each speech segment using torchaudio."""
    logger.info("Loading audio with torchaudio...")
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

    logger.info("Computed RMS for segments.")
    return rms_values

def load_sentiment_model(model_name: str):
    """Load the sentiment analysis model and tokenizer."""
    logger.info(f"Loading sentiment model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    try:
        model = model.to(TORCH_DEVICE)
    except Exception as e:
        warnings.warn(f"Could not move model to device {TORCH_DEVICE}: {e}")
    model.eval()
    return tokenizer, model

def sentiment_intensity(
    text: str,
    tokenizer,
    model
) -> float:
    """Use a sentiment model to approximate emotional intensity."""
    if not text.strip():
        return 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    for k in inputs:
        inputs[k] = inputs[k].to(TORCH_DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    # For cardiffnlp/twitter-roberta-base-sentiment-latest:
    # labels: [negative, neutral, positive]
    neutral_prob = probs[1].item()
    intensity = 1.0 - neutral_prob  # higher = more emotional
    return float(intensity)
