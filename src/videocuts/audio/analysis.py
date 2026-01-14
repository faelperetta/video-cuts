import torch
import torchaudio
import warnings
import logging
import hashlib
import subprocess
import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from videocuts.utils.system import run_ffmpeg_cmd
from videocuts.utils.device import TORCH_DEVICE

logger = logging.getLogger(__name__)


def extract_audio_normalized(
    input_video: str, 
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1
) -> dict:
    """
    Extract normalized audio for ASR/diarization.
    
    Args:
        input_video: Path to input video file
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default: 16000 Hz)
        channels: Number of channels (default: 1 for mono)
    
    Returns:
        Metadata dict: {sample_rate, channels, duration_s, sha256}
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_path
    ]
    run_ffmpeg_cmd(cmd)
    
    # Extract metadata
    metadata = _extract_audio_metadata(output_path)
    logger.info(f"Extracted normalized audio to '{output_path}' ({metadata['duration_s']:.2f}s)")
    return metadata


def extract_audio_hq(input_video: str, output_path: str) -> dict:
    """
    Extract high-quality audio (48kHz stereo) for final render pipeline.
    
    Args:
        input_video: Path to input video file
        output_path: Path for output WAV file
    
    Returns:
        Metadata dict: {sample_rate, channels, duration_s, sha256}
    """
    return extract_audio_normalized(
        input_video, 
        output_path, 
        sample_rate=48000, 
        channels=2
    )


def _extract_audio_metadata(audio_path: str) -> dict:
    """
    Extract metadata from audio file using ffprobe.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dict with sample_rate, channels, duration_s, sha256
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe_data = json.loads(result.stdout)
    
    stream = probe_data["streams"][0]
    
    # Calculate SHA256
    sha256_hash = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return {
        "sample_rate": int(stream.get("sample_rate", 16000)),
        "channels": int(stream.get("channels", 1)),
        "duration_s": float(stream.get("duration", 0)),
        "sha256": sha256_hash.hexdigest()
    }


def extract_audio(input_video: str, audio_path: str) -> None:
    """
    DEPRECATED: Use extract_audio_normalized() instead.
    
    Extract audio from input video to WAV file (backward compatible).
    """
    extract_audio_normalized(input_video, audio_path)
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
