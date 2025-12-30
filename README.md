# Video Highlights Generator (`videocuts`)

A professional, modularized video highlights generator that extracts short, vertical clips (TikTok, Shorts, Reels) from long-form source videos. This project uses AI for transcription, sentiment analysis, face tracking, and intelligent clip selection.

## 🚀 Key Features

- **Modular Architecture**: Professional Python package structure for high maintainability and scalability.
- **Automatic Language Detection**: Powered by Whisper (supports 99+ languages).
- **LLM-Powered Intelligence**: Optional OpenAI integration for viral clip identification and hook generation.
- **Dynamic Multi-Speaker Layouts**: Automatic switching between SINGLE, SPLIT, and WIDE layouts based on real-time face tracking.
- **Animated Captions**: Beautiful ASS subtitles with word-by-word highlighting.
- **Auto-Hook system**: Attention-grabbing visual overlays based on content analysis.
- **Fast Mode**: Development-optimized mode for rapid iteration (5-10x faster).
- **Web-Ready**: Architecture designed for easy integration into FastAPI or other web services.

## 📁 Project Structure

```text
.
├── src/videocuts/          # Main package
│   ├── audio/              # Transcription & sentiment analysis
│   ├── video/              # Tracking, frames, & processing
│   ├── caption/            # Subtitle & hook generation
│   ├── highlights/         # Heuristic clip selection
│   ├── llm/                # LLM-based clip selection
│   ├── utils/              # Shared utilities
│   ├── config.py           # Centralized configuration
│   ├── main.py             # Pipeline orchestrator
│   └── cli.py              # Command-line entry point
├── video_highlight.py      # Legacy wrapper script
└── pyproject.toml          # Package metadata & dependencies
```

## 🛠 Installation

Requires Python 3.10+ and FFmpeg.

1.  **Clone & Install Dependencies**:
    ```bash
    pip install -e .
    ```

2.  **External Requirements**:
    - `ffmpeg` must be in your system PATH.
    - `OPENAI_API_KEY` (Optional) for LLM features.

## 📖 Usage

### Command-Line interface

You can run the project using the wrapper script or the package CLI:

```bash
# Using the wrapper
python video_highlight.py --input video.mp4

# Using the package directly
python -m videocuts.cli --input video.mp4 --hook --use-llm
```

### Common Flags

| Flag | Description |
| :--- | :--- |
| `--input <path>` | Path to the source video. |
| `--hook` | Enable auto-hook text overlays. |
| `--use-llm` | Use AI for smart clip identification. |
| `--layout {auto,single,split,wide}` | Override the multi-speaker layout mode. |
| `--content-type {coding,fitness,gaming}`| Use niche-specific weights for scoring. |
| `--num-clips <N>` | Number of highlights to generate. |
| `--fast` | Accelerate processing for previews. |

### Examples

```bash
# Basic highlight generation
python video_highlight.py

# Full production run: 3 clips with hooks and LLM intelligence
python video_highlight.py --content-type gaming --num-clips 3 --hook --use-llm

# Fast mode for quick iterations
python video_highlight.py --fast --num-clips 1
```

## 🧠 Advanced Features

### LLM Mode
When `OPENAI_API_KEY` is present, the system uses GPT-4o-mini (default) to understand the transcript context, identify the most engaging moments, and generate catchy hooks.

### Multi-Speaker Detection
The system analyzes face positions and lip activity to determine which layout provides the best viewing experience for the current clip context.

---

## 📜 License
This project is provided as-is for educational and creative purposes.
