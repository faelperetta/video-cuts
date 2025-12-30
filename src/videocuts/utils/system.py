import os
import math
import subprocess
from typing import List

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

import logging

logger = logging.getLogger(__name__)

def run_ffmpeg_cmd(cmd: List[str]) -> None:
    """Execute an FFmpeg command via subprocess."""
    logger.debug("Executing FFmpeg command: " + " ".join(cmd))
    # Capture output to avoid cluttering unless there is an error
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed with exit code {e.returncode}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise

def cache_is_fresh(path: str, reference_mtime: float) -> bool:
    """Check if a file exists and is newer than the reference modification time."""
    return os.path.exists(path) and os.path.getmtime(path) >= reference_mtime
