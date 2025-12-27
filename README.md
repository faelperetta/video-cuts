# Video Highlights Generator

This project extracts short, vertical highlight clips (TikTok, Shorts, Reels) from a long-form source video. The pipeline uses Whisper for transcription, torchaudio for loudness, and a sentiment model to score segments. Selected clips are exported with burned-in subtitles, context padding, and optional speaker-centric reframing.

## Features
- Downloads source videos (via `download.py`) and transcribes them with Whisper.
- **Automatic Language Detection**: Whisper auto-detects the video language and transcribes in that language (supports 99+ languages).
- **LLM-Powered Clip Selection**: Optionally use OpenAI GPT models for intelligent clip identification (auto-enabled when `OPENAI_API_KEY` is set).
- Scores every subtitle segment using energy, sentiment, and hook keywords to find engaging moments.
- Generates multiple highlight intervals with context padding, trims subtitles, and burns clip-specific SRTs into the exported MP4s.
- **Dynamic Multi-Speaker Layouts**: Automatically switches between SINGLE, SPLIT, and WIDE layouts within clips based on speaker detection.
- **Animated Word-by-Word Captions**: ASS subtitles with green highlighting on the current word as it's spoken.
- **Auto-Hook Overlay** (optional): Adds attention-grabbing text at the top of clips using hook keywords from niche config.
- **Fast Mode**: Development mode with ultrafast encoding and reduced analysis (5-10x faster).
- Caches intermediate artifacts (audio as compressed MP3, SRT, sentiment tensors) to speed up re-runs.

## Requirements
- Python 3.13 with the provided `.venv` (activate via `source .venv/bin/activate`).
- ffmpeg installed on the system path.
- GPU-optional: AV1 decoding falls back to CPU when hardware support is missing.
- (Optional) `OPENAI_API_KEY` environment variable for LLM-based clip selection.

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
3. Generated clips appear in `clips_output/`.

### Command-Line Options

```bash
python video_highlight.py --help
```

| Flag | Description |
|------|-------------|
| `--fast` | Fast mode for development - uses ultrafast encoding and reduced analysis (5-10x faster) |
| `--hook` | Enable auto-hook overlay - adds attention-grabbing text at the top of each clip |
| `--language LANG` | Override language detection (e.g., `pt` for Portuguese, `en` for English, `es` for Spanish) |
| `--use-llm` | Force LLM mode for clip identification (auto-enabled if `OPENAI_API_KEY` is set) |
| `--llm-model MODEL` | Override LLM model (default: `gpt-4o-mini`). Options: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| `--layout {auto,single,split,wide}` | Override layout mode (default: uses LAYOUT_MODE from config) |
| `--content-type {coding,fitness,gaming}` | Override content type for niche-specific scoring |
| `--input INPUT` | Override input video path |
| `--num-clips N` | Number of highlight clips to generate |

### Examples

```bash
# Basic highlight generation
python video_highlight.py

# Fast mode for quick preview iterations
python video_highlight.py --fast --num-clips 1

# Portuguese video with hook overlay
python video_highlight.py --language pt --hook

# Use LLM for smarter clip selection
python video_highlight.py --use-llm --num-clips 5

# Full production run: 3 clips with gaming niche, hooks, and LLM
python video_highlight.py --content-type gaming --num-clips 3 --hook --use-llm
```

## LLM Mode

When `OPENAI_API_KEY` is set in your environment, the script automatically uses GPT to analyze the transcript and identify the best viral clips. This provides:

- Smarter clip selection based on content understanding
- Hook phrase identification
- Language-aware analysis (prompt instructs LLM to match video language)

The LLM prompt template is stored in `prompt.md` and can be customized.

## Bug Backlog
- **Speaker framing still misaligns**: even after adding the Mediapipe Tasks detector and fallbacks, some exported clips keep the frame on the background instead of the speaker. Need to debug `compute_speaker_samples()` and the crop expression logic so the active person remains centered.
