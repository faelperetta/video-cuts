# Video Highlights Generator

This project extracts short, vertical highlight clips (TikTok, Shorts, Reels) from a long-form source video. The pipeline uses Whisper for transcription, torchaudio for loudness, and a sentiment model to score segments. Selected clips are exported with burned-in subtitles, context padding, and optional speaker-centric reframing.

## Features
- Downloads source videos (via `download.py`) and transcribes them with Whisper.
- Scores every subtitle segment using energy, sentiment, and hook keywords to find engaging moments.
- Generates multiple highlight intervals with context padding, trims subtitles, and burns clip-specific SRTs into the exported MP4s.
- **Dynamic Multi-Speaker Layouts**: Automatically switches between SINGLE, SPLIT, and WIDE layouts within clips based on speaker detection.
- **Animated Word-by-Word Captions**: ASS subtitles with green highlighting on the current word as it's spoken.
- **Auto-Hook Overlay** (optional): Adds attention-grabbing text at the top of clips using hook keywords from niche config.
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

### Command-Line Options

```bash
python video_highlight.py --help
```

| Flag | Description |
|------|-------------|
| `--hook` | Enable auto-hook overlay - adds attention-grabbing text at the top of each clip |
| `--layout {auto,single,split,wide}` | Override layout mode (default: uses LAYOUT_MODE from config) |
| `--content-type {coding,fitness,gaming}` | Override content type for niche-specific scoring |
| `--input INPUT` | Override input video path |
| `--num-clips N` | Number of highlight clips to generate |

### Examples

```bash
# Basic highlight generation
python video_highlight.py

# Enable auto-hook overlay
python video_highlight.py --hook

# Hook + forced split-screen layout
python video_highlight.py --hook --layout split

# Generate 5 clips with gaming niche settings
python video_highlight.py --content-type gaming --num-clips 5
```

## Bug Backlog
- **Speaker framing still misaligns**: even after adding the Mediapipe Tasks detector and fallbacks, some exported clips keep the frame on the background instead of the speaker. Need to debug `compute_speaker_samples()` and the crop expression logic so the active person remains centered.
