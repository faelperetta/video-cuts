"""YouTube video download service using yt-dlp with optimizations."""
import os
import shutil
import logging
from typing import Optional, Dict, Any
from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)


def get_ydl_opts(output_path: str, use_aria2: bool = True) -> Dict[str, Any]:
    """
    Build yt-dlp options with speed optimizations.
    Adapted from download.py with improvements for service usage.
    """
    opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_path,
        "merge_output_format": "webm",
        # Speed optimizations
        "concurrent_fragment_downloads": 8,
        "fragment_retries": 10,
        "retries": 10,
        "buffersize": 1024 * 64,  # 64KB buffer
        "http_chunk_size": 1024 * 1024 * 10,  # 10MB chunks
        # Quiet mode for service
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [],
    }
    
    # Use aria2c if available (much faster)
    if use_aria2 and shutil.which("aria2c"):
        logger.info("Using aria2c for faster downloads")
        opts["external_downloader"] = "aria2c"
        opts["external_downloader_args"] = {
            "aria2c": [
                "--min-split-size=1M",
                "--max-connection-per-server=16",
                "--max-concurrent-downloads=16",
                "--split=16",
            ]
        }
    
    return opts


def download_video(url: str, output_dir: str, filename: str = "source") -> Dict[str, Any]:
    """
    Download a video from YouTube or other supported platforms.
    
    Args:
        url: Video URL
        output_dir: Directory to save the video
        filename: Output filename without extension
        
    Returns:
        Dict with video metadata (title, duration, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.%(ext)s")
    
    opts = get_ydl_opts(output_path)
    
    with YoutubeDL(opts) as ydl:
        logger.info(f"Downloading video from: {url}")
        info = ydl.extract_info(url, download=True)
        
        # Find the actual downloaded file
        ext = info.get("ext", "webm")
        actual_path = os.path.join(output_dir, f"{filename}.{ext}")
        
        return {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
            "file_path": actual_path,
            "ext": ext,
        }


def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """
    Get video metadata without downloading.
    
    Args:
        url: Video URL
        
    Returns:
        Dict with video metadata or None if failed
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }
    
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
            }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None
