from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class VideoConfig:
    """Video format and output settings."""
    target_width: int = 1080
    target_height: int = 1920
    output_fps: int = 30
    ffmpeg_preset: str = "medium"    # ultrafast/superfast/veryfast/faster/fast/medium/slow/slower/veryslow
    ffmpeg_crf: int = 18             # Quality (lower = better, 18-23 recommended)


@dataclass
class HighlightConfig:
    """Highlight selection and timing parameters."""
    min_length: float = 15.0          # seconds
    max_length: float = 80.0       # seconds
    context_before: float = 1.5       # seconds before high-scoring segment
    context_after: float = 1.5        # seconds after high-scoring segment
    num_highlights: int = 8           # how many clips to export
    min_gap: float = 4.0              # seconds between clips
    last_word_pad: float = 0.25       # buffer to ensure last word finishes


@dataclass
class FaceTrackingConfig:
    """Face detection and tracking parameters."""
    min_confidence: float = 0.30      # mediapipe detection confidence: relaxed to catch side profiles
    analysis_fps: float = 12.0        # sampling FPS for face tracking
    track_distance: float = 0.12      # max normalized horizontal delta for same track
    track_max_gap: float = 1.0        # seconds before track is recycled
    activity_threshold: float = 0.0035  # threshold for choosing active speaker
    recenter_after: float = 1.5       # seconds without detection before easing to center
    min_face_width: float = 0.05      # minimum face width (normalized): relaxed for wide shots
    use_yolo: bool = True             # enable YOLO heavy detection
    yolo_model_path: str = "src/videocuts/models/yolov8m-face.pt"  # Medium model .pt (avoids OpenVINO LLVM conflict)
    yolo_device: str = "cpu"          # CPU inference (avoids LLVM conflict with transcription OpenVINO)
    use_openvino: bool = False         # enable OpenVINO acceleration on Intel Arc


@dataclass
class SpeakerLockConfig:
    """Speaker lock parameters to prevent jittery switching."""
    min_duration: float = 3.0         # minimum seconds before switching
    switch_threshold: float = 2.0     # other speaker needs 2x more activity to switch
    smoothing_window: float = 0.5     # seconds of smoothing for crop position
    position_smoothing: float = 0.9   # normal smoothing (slow/gentle)


@dataclass
class SegmentSmoothingConfig:
    """Segment smoothing to eliminate jitter from micro-segments."""
    min_duration: float = 0.5         # minimum segment duration in seconds
    merge_threshold: float = 0.15     # merge segments with centers within this distance
    absorb_threshold: float = 0.3     # absorb very short segments within this distance


@dataclass
class LipDetectionConfig:
    """Lip movement detection for identifying who is speaking."""
    history_frames: int = 5           # frames to track for lip movement delta
    min_delta: float = 0.003          # minimum change to count as "speaking"
    speaking_threshold: float = 0.008  # cumulative movement to consider speaking


@dataclass
class CaptionConfig:
    """Animated caption style configuration."""
    font_name: str = "Montserrat"
    font_path: str = ""                  # Path to font file (optional, auto-detected if empty)
    font_size: int = 75                  # Larger font for mobile visibility
    primary_color: str = "&H00FFFFFF"    # White (AABBGGRR format)
    highlight_color: str = "&H0000FF00"  # Bright green for current word
    outline_color: str = "&H00000000"    # Black outline
    back_color: str = "&H80000000"       # Semi-transparent black background
    outline_width: int = 4
    shadow_depth: int = 2
    margin_v: int = 380                  # Higher margin to clear YouTube Shorts UI
    words_per_line: int = 3              # Fewer words per line for larger font
    use_word_highlight: bool = True      # Enable word-by-word highlighting


@dataclass
class HookConfig:
    """Auto-hook overlay configuration."""
    enabled: bool = False             # Controlled by --hook CLI flag
    scan_seconds: float = 5.0         # Seconds at start to scan for hook
    min_words: int = 3
    max_words: int = 12               # Allow slightly longer hooks
    font_name: str = "Montserrat"
    font_path: str = ""               # Path to font file (auto-detected if empty)
    font_size: int = 52               # Larger font for better visibility
    primary_color: str = "black"      # Hook text color (FFmpeg format)
    bg_color: str = "white"           # White background for hook
    bg_opacity: float = 0.98          # Nearly opaque for better readability
    position_y: int = 850             # Centered position (approx middle of 1920p)
    fade_in: float = 0.4              # Slightly longer fade-in
    display_duration: float = 4.0     # Longer display time (4 seconds)
    fade_out: float = 0.5             # Slightly longer fade-out
    box_padding: int = 40             # Increased padding as requested
    box_border_w: int = 8             # (Unused in Pillow mode)
    corner_radius: int = 20           # Radius for rounded corners
    shadow_color: str = "#00000000"   # No shadow (fully transparent)


@dataclass
class ProfanityConfig:
    """Profanity filtering configuration."""
    enabled: bool = True
    custom_words: List[str] = field(default_factory=list)


@dataclass
class LayoutConfig:
    """Multi-speaker layout configuration."""
    mode: str = "auto"                # "auto", "single", "split", "wide"
    split_threshold: float = 0.35     # Min distance to trigger split-screen
    wide_threshold: float = 0.25      # Max distance for wide shot
    min_faces_for_split: int = 2
    split_gap: int = 20               # Pixels between split panels
    both_speaking_window: float = 1.5  # Seconds to detect both speakers active
    both_speaking_ratio: float = 0.3   # Both have >30% of max activity
    segment_min_duration: float = 2.0
    wide_zoom: float = 0.85           # Zoom factor for wide shot


@dataclass
class ModelConfig:
    """Model paths and settings."""
    whisper_size: str = "small"
    whisper_language: Optional[str] = None  # None = auto-detect, or ISO 639-1 code
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    face_detector_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_detection/"
        "short_range/float16/1/face_detection_short_range.tflite"
    )
    face_landmarker_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )


@dataclass
class LLMConfig:
    """External LLM configuration for clip identification."""
    provider: str = "openai"              # "openai"
    model: str = "gpt-4o-mini"             # Model to use
    enabled: bool = False                  # Auto-enabled if API key is set
    prompt_template_path: str = "prompt.md"  # Path to the prompt template
    max_tokens: int = 4000
    temperature: float = 0.7


@dataclass
class TranscriptionConfig:
    """Transcription provider configuration (Epic 2 - US-2.1)."""
    # Provider selection: "local" or "openai"
    provider: str = "local"
    
    # Local faster-whisper settings
    model: str = "small"                    # Whisper model size (small for testing, large-v3 for production)
    compute_type: str = "float16"           # float16, int8_float16, int8
    device: str = "auto"                    # auto, openvino, cuda, cpu
    vad_enabled: bool = True                # Voice Activity Detection
    vad_filter_threshold: float = 0.5       # VAD sensitivity
    
    # OpenAI API settings (only used when provider="openai")
    openai_model: str = "whisper-1"
    
    # Output settings
    output_word_timestamps: bool = True


@dataclass
class PathConfig:
    """File paths and directories."""
    input_video: str = "input.webm"
    project_name: Optional[str] = None
    
    @property
    def project_root(self) -> str:
        return self.project_name if self.project_name else "."

    @property
    def srt_file(self) -> str:
        return os.path.join(self.project_root, "subs.srt")

    @property
    def audio_16k_mono(self) -> str:
        """Primary audio file for ASR/diarization (16kHz mono WAV)."""
        return os.path.join(self.project_root, "audio_16k_mono.wav")
    
    @property
    def audio_48k_stereo(self) -> str:
        """High-quality audio for final render (48kHz stereo WAV)."""
        return os.path.join(self.project_root, "audio_48k_stereo.wav")
    
    @property
    def audio_metadata(self) -> str:
        """Audio metadata JSON file."""
        return os.path.join(self.project_root, "audio_metadata.json")
    
    @property
    def transcript_json(self) -> str:
        """JSON transcript with word-level timestamps."""
        return os.path.join(self.project_root, "transcript.json")
    
    @property
    def audio_wav(self) -> str:
        """DEPRECATED: Use audio_16k_mono instead."""
        return self.audio_16k_mono

    @property
    def output_dir(self) -> str:
        return os.path.join(self.project_root, "clips_output")

    @property
    def final_output(self) -> str:
        return os.path.join(self.project_root, "final_highlight.mp4")
    
    @property
    def face_detector_model(self) -> str:
        return os.path.join(self.output_dir, "models", "face_detection_short_range.tflite")
    
    @property
    def face_landmarker_model(self) -> str:
        return os.path.join(self.output_dir, "models", "face_landmarker.task")

    @property
    def openvino_face_model(self) -> str:
        # Default exported path from ultralytics
        return "src/videocuts/models/yolov8n-face_openvino_model/yolov8n-face.xml"


@dataclass
class Config:
    """Main configuration container combining all settings."""
    paths: PathConfig = field(default_factory=PathConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    face_tracking: FaceTrackingConfig = field(default_factory=FaceTrackingConfig)
    speaker_lock: SpeakerLockConfig = field(default_factory=SpeakerLockConfig)
    segment_smoothing: SegmentSmoothingConfig = field(default_factory=SegmentSmoothingConfig)
    lip_detection: LipDetectionConfig = field(default_factory=LipDetectionConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    hook: HookConfig = field(default_factory=HookConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    content_type: str = "coding"      # "coding", "fitness", "gaming"
    force_transcribe: bool = False
    force_audio_extraction: bool = False
    fast_mode: bool = False           # --fast flag for faster dev iterations
    debug: bool = False               # --debug flag for verbose logging
    cpu_limit: int = 0                # 0 = unlimited, >0 = max threads
    profanity: ProfanityConfig = field(default_factory=ProfanityConfig)
    
    def enable_fast_mode(self) -> None:
        """Enable fast mode for quicker development iterations."""
        self.fast_mode = True
        self.video.ffmpeg_preset = "ultrafast"
        self.video.ffmpeg_crf = 23
        self.face_tracking.analysis_fps = 6.0


