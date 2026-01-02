import subprocess
import json
import logging
import numpy as np
from typing import Dict, Generator, Optional, Tuple

logger = logging.getLogger(__name__)

def get_video_info(input_video: str) -> Dict:
    """Get video metadata (width, height, fps) using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate",
        "-of", "json",
        input_video
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        
        # Parse fps usually "num/den"
        if "r_frame_rate" in stream:
            parts = stream["r_frame_rate"].split('/')
            if len(parts) == 2:
                num, den = map(int, parts)
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(parts[0])
        else:
            fps = 30.0
            
        return {"width": width, "height": height, "fps": fps}
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"width": 1920, "height": 1080, "fps": 30.0}

def frame_generator_from_ffmpeg(
    input_video: str,
    clip_start: float,
    clip_end: float,
    fps: float
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Yields (timestamp, frame_bgr) directly from ffmpeg pipe.
    """
    info = get_video_info(input_video)
    width = info["width"]
    height = info["height"]
    duration = clip_end - clip_start
    
    cmd = [
        "ffmpeg",
        "-ss", f"{clip_start:.3f}",
        "-i", input_video,
        "-t", f"{duration:.3f}",
        "-vf", f"fps={fps}",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.DEVNULL,
        bufsize=10**7
    )
    
    frame_size = width * height * 3
    frame_idx = 0
    
    while True:
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            break
            
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
        ts = clip_start + (frame_idx / fps)
        yield ts, frame
        frame_idx += 1
        
    process.stdout.close()
    process.wait()
