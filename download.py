import argparse
import shutil
from yt_dlp import YoutubeDL


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube and other platforms"
    )
    parser.add_argument("url", help="URL of the video to download")
    parser.add_argument(
        "-o", "--output",
        default="input",
        help="Output filename without extension (default: input)"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download audio only"
    )
    parser.add_argument(
        "--no-aria2",
        action="store_true",
        help="Disable aria2c external downloader even if available"
    )
    return parser.parse_args()


def get_ydl_opts(output_name: str, audio_only: bool, use_aria2: bool) -> dict:
    """Build yt-dlp options with speed optimizations."""
    
    if audio_only:
        format_spec = "bestaudio/best"
    else:
        format_spec = "bestvideo+bestaudio/best"
    
    opts = {
        "format": format_spec,
        "outtmpl": f"{output_name}.%(ext)s",
        # Speed optimizations
        "concurrent_fragment_downloads": 8,  # Download 8 fragments in parallel
        "fragment_retries": 10,
        "retries": 10,
        "buffersize": 1024 * 64,  # 64KB buffer
        "http_chunk_size": 1024 * 1024 * 10,  # 10MB chunks
    }
    
    # Use aria2c if available (much faster - supports multi-connection downloads)
    if use_aria2 and shutil.which("aria2c"):
        print("✓ Using aria2c for faster downloads (16 connections)")
        opts["external_downloader"] = "aria2c"
        opts["external_downloader_args"] = {
            "aria2c": [
                "--min-split-size=1M",
                "--max-connection-per-server=16",
                "--max-concurrent-downloads=16",
                "--split=16",
            ]
        }
    else:
        print("✓ Using concurrent fragment downloads (8 parallel)")
    
    return opts


def main():
    args = parse_args()
    
    use_aria2 = not args.no_aria2
    ydl_opts = get_ydl_opts(args.output, args.audio_only, use_aria2)
    
    print(f"Downloading: {args.url}")
    print(f"Output: {args.output}.<ext>")
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([args.url])
    
    print("✓ Download complete!")


if __name__ == "__main__":
    main()