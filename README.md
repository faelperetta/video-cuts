# Video Highlights Generator

This project extracts short, vertical highlight clips (TikTok, Shorts, Reels) from a long-form source video. The pipeline uses Whisper for transcription, torchaudio for loudness, and a sentiment model to score segments. Selected clips are exported with burned-in subtitles, context padding, and optional speaker-centric reframing.

## Features
- Downloads source videos (via `download.py`) and transcribes them with Whisper.
- Scores every subtitle segment using energy, sentiment, and hook keywords to find engaging moments.
- Generates multiple highlight intervals with context padding, trims subtitles, and burns clip-specific SRTs into the exported MP4s.
- Attempts to reframe the footage to keep the active speaker centered when exporting 9:16 videos.
- Caches intermediate artifacts (audio, SRT, sentiment tensors) to speed up re-runs.

## Requirements
- Python 3.13 with the provided `.venv` (activate via `source .venv/bin/activate`).
- ffmpeg installed on the system path.
- GPU-optional: AV1 decoding falls back to CPU when hardware support is missing.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place or download a source video as `input.webm` (use `python download.py` if configured).
2. Run the highlight pipeline:
   ```bash
   python video_highlight.py
   ```
3. Generated clips and per-clip subtitles appear in `highlights/`.

## Bug Backlog
- **Speaker framing still misaligns**: even after adding the Mediapipe Tasks detector and fallbacks, some exported clips keep the frame on the background instead of the speaker. Need to debug `compute_speaker_samples()` and the crop expression logic so the active person remains centered.
