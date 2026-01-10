from __future__ import annotations
import os
import whisper
import logging
from typing import List, Dict, Tuple, Optional
from videocuts.utils.system import format_timestamp, parse_timestamp
from videocuts.utils.device import TORCH_DEVICE, is_intel_accel_enabled

logger = logging.getLogger(__name__)

def transcribe_video(
    input_video: str,
    srt_path: str,
    model_size: str = "small",
    language: Optional[str] = None
) -> Tuple[List[Dict], str]:
    """
    Transcribe a video file using OpenAI's Whisper model (hybrid).
    Uses faster-whisper for Intel optimized path if XPU/Intel is selected 
    AND Intel acceleration is explicitly enabled.
    """
    use_intel_path = (is_intel_accel_enabled() and 
                      (TORCH_DEVICE.type == "xpu" or "intel" in str(TORCH_DEVICE).lower()))
    
    if use_intel_path:
        from faster_whisper import WhisperModel
        # Note: We use device="cpu" with compute_type="int8" because native XPU 
        # has SPIR-V symbol issues with Whisper kernels. int8 on Intel CPUs
        # is extremely fast and avoids the LLVM duplicate option crashes.
        logger.info(f"Loading faster-whisper model '{model_size}' (Intel optimized path)...")
        
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        logger.info(f"Transcribing '{input_video}' using faster-whisper...")
        segments_gen, info = model.transcribe(
            input_video,
            language=language,
            word_timestamps=True
        )
        
        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })
        detected_language = info.language
    else:
        import whisper
        logger.info(f"Loading standard whisper model '{model_size}' on device '{TORCH_DEVICE}'...")
        model = whisper.load_model(model_size, device=str(TORCH_DEVICE))

        if language:
            logger.info(f"Using specified language: '{language}'")
        else:
            logger.info("Auto-detecting language...")
        
        logger.info(f"Transcribing '{input_video}' with word timestamps...")
        result = model.transcribe(
            input_video, 
            task="transcribe", 
            word_timestamps=True, 
            language=language
        )

        segments = result["segments"]
        detected_language = result.get("language", language or "en")
    
    logger.info(f"Detected language: '{detected_language}'")

    logger.info(f"Writing SRT to '{srt_path}'...")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()

            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

    logger.info("Done transcription.")
    return segments, detected_language

def parse_srt_to_segments(srt_path: str) -> List[Dict]:
    """Parse an SRT file back into Whisper-like segments."""
    if not os.path.exists(srt_path): return []
    with open(srt_path, "r", encoding="utf-8") as f: raw = f.read()
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2 or "-->" not in lines[1]: continue
        start_ts, end_ts = [p.strip() for p in lines[1].split("-->")]
        try:
            segments.append({
                "id": lines[0].strip(),
                "start": parse_timestamp(start_ts),
                "end": parse_timestamp(end_ts),
                "text": " ".join(l.strip() for l in lines[2:]).strip()
            })
        except: continue
    return segments
