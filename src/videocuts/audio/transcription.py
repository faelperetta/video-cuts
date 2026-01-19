"""Audio transcription with configurable providers (Epic 2 - US-2.1)."""
from __future__ import annotations
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

from videocuts.config import TranscriptionConfig
from videocuts.models.transcript import Transcript, Segment, Word
from videocuts.utils.system import format_timestamp, parse_timestamp
from videocuts.utils.device import TORCH_DEVICE, is_intel_accel_enabled

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Interface
# =============================================================================

class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe audio file and return Transcript object."""
        pass


# =============================================================================
# Local faster-whisper Provider
# =============================================================================

class LocalFasterWhisperProvider(TranscriptionProvider):
    """Local transcription using faster-whisper with Intel optimizations."""
    
    def __init__(self, cfg: TranscriptionConfig):
        self.cfg = cfg
        self._model = None
    
    def _get_device_and_compute(self) -> Tuple[str, str]:
        """
        Determine optimal device and compute type.
        
        Per Epic 2 spec:
        - OpenVINO is recommended for Intel Arc
        - Fallback to int8_float16 if VRAM constrained
        - CUDA for NVIDIA, CPU int8 as last resort
        """
        if self.cfg.device != "auto":
            return self.cfg.device, self.cfg.compute_type
        
        # Auto-detect best configuration
        if is_intel_accel_enabled():
            # Intel Arc: Try OpenVINO first (Epic 2 recommended)
            # faster-whisper supports "auto" which will use OpenVINO if available
            try:
                # Check if OpenVINO is available
                import openvino  # noqa: F401
                logger.info("OpenVINO detected, using for Intel Arc acceleration")
                # OpenVINO doesn't support float16, use int8 (or int8_float16 per Epic spec)
                return "auto", "int8"
            except ImportError:
                logger.info("OpenVINO not available, using CPU with int8")
                return "cpu", "int8"
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", self.cfg.compute_type
        except ImportError:
            pass
        
        return "cpu", "int8"
    
    def _load_model(self):
        """Lazy-load the whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            
            device, compute_type = self._get_device_and_compute()
            logger.info(f"Loading faster-whisper model '{self.cfg.model}' "
                       f"(device={device}, compute_type={compute_type})")
            
            self._model = WhisperModel(
                self.cfg.model,
                device=device,
                compute_type=compute_type
            )
        return self._model
    
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe audio using faster-whisper."""
        model = self._load_model()
        
        logger.info(f"Transcribing '{audio_path}' with faster-whisper...")
        
        # Configure VAD if enabled
        vad_params = None
        if self.cfg.vad_enabled:
            vad_params = {"threshold": self.cfg.vad_filter_threshold}
        
        segments_gen, info = model.transcribe(
            audio_path,
            word_timestamps=self.cfg.output_word_timestamps,
            vad_filter=self.cfg.vad_enabled,
            vad_parameters=vad_params
        )
        
        segments = []
        for idx, seg in enumerate(segments_gen):
            words = []
            if hasattr(seg, 'words') and seg.words:
                words = [
                    Word(
                        start_s=w.start,
                        end_s=w.end,
                        word=w.word
                    )
                    for w in seg.words
                ]
            
            segments.append(Segment(
                id=idx,
                start_s=seg.start,
                end_s=seg.end,
                text=seg.text.strip(),
                avg_logprob=getattr(seg, 'avg_logprob', 0.0),
                no_speech_prob=getattr(seg, 'no_speech_prob', 0.0),
                words=words
            ))
        
        logger.info(f"Transcription complete: {len(segments)} segments, "
                   f"language='{info.language}'")
        
        return Transcript(
            provider="local_faster_whisper",
            model=self.cfg.model,
            language=info.language,
            duration_s=info.duration,
            segments=segments
        )


# =============================================================================
# OpenAI API Provider
# =============================================================================

class OpenAIProvider(TranscriptionProvider):
    """Transcription using OpenAI Whisper API."""
    
    def __init__(self, cfg: TranscriptionConfig):
        self.cfg = cfg
        self._client = None
    
    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client
    
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe audio using OpenAI Whisper API."""
        client = self._get_client()
        
        logger.info(f"Transcribing '{audio_path}' with OpenAI API...")
        
        with open(audio_path, "rb") as audio_file:
            # Request verbose JSON for word-level timestamps
            response = client.audio.transcriptions.create(
                model=self.cfg.openai_model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
        
        segments = []
        for idx, seg in enumerate(response.segments or []):
            words = []
            # OpenAI returns words at response level in verbose mode
            if hasattr(response, 'words') and response.words:
                # Filter words that fall within this segment
                for w in response.words:
                    if seg.start <= w.start < seg.end:
                        words.append(Word(
                            start_s=w.start,
                            end_s=w.end,
                            word=w.word
                        ))
            
            segments.append(Segment(
                id=idx,
                start_s=seg.start,
                end_s=seg.end,
                text=seg.text.strip(),
                avg_logprob=getattr(seg, 'avg_logprob', 0.0),
                no_speech_prob=getattr(seg, 'no_speech_prob', 0.0),
                words=words
            ))
        
        logger.info(f"OpenAI transcription complete: {len(segments)} segments")
        
        return Transcript(
            provider="openai",
            model=self.cfg.openai_model,
            language=getattr(response, 'language', 'en') or "en",
            duration_s=getattr(response, 'duration', 0.0) or 0.0,
            segments=segments
        )


# =============================================================================
# Process Isolated Provider
# =============================================================================

def _isolated_worker(
    cfg: TranscriptionConfig,
    audio_path: str,
    result_queue: "multiprocessing.Queue"
):
    """Worker function to run transcription in a separate process."""
    try:
        # Re-initialize logging in the new process
        logging.basicConfig(level=logging.INFO)
        # Create a local provider instance within this isolated process
        # This is where faster-whisper/OpenVINO gets loaded safely
        provider = LocalFasterWhisperProvider(cfg)
        transcript = provider.transcribe(audio_path)
        result_queue.put(("success", transcript))
    except Exception as e:
        logger.exception("Error in isolated transcription worker")
        result_queue.put(("error", str(e)))

class ProcessIsolatedWhisperProvider(TranscriptionProvider):
    """
    Runs transcription in a separate process to avoid LLVM/OpenMP conflicts.
    Useful when mixing OpenVINO (faster-whisper) and PyTorch XPU (YOLO).
    """
    
    def __init__(self, cfg: TranscriptionConfig):
        self.cfg = cfg

    def transcribe(self, audio_path: str) -> Transcript:
        import multiprocessing
        
        logger.info(f"Starting isolated transcription for '{audio_path}'")
        
        ctx = multiprocessing.get_context('spawn') # Use spawn to ensure clean slate
        queue = ctx.Queue()
        
        p = ctx.Process(
            target=_isolated_worker,
            args=(self.cfg, audio_path, queue)
        )
        
        p.start()
        
        # Wait for result with monitoring
        import time
        import queue as pyqueue
        
        start_time = time.time()
        result_payload = None
        
        while time.time() - start_time < self.cfg.isolated_worker_timeout_s:
            if not p.is_alive():
                # Process died check if we have data
                try:
                    result_payload = queue.get_nowait()
                    break
                except pyqueue.Empty:
                    exit_code = p.exitcode
                    raise RuntimeError(f"Isolated transcription process died unexpectedly (exit code: {exit_code}). See logs for details.")
            
            try:
                # Poll queue
                result_payload = queue.get(timeout=0.5)
                break
            except pyqueue.Empty:
                continue
                
        if result_payload is None:
            p.kill()
            raise RuntimeError(f"Isolated transcription timed out after {self.cfg.isolated_worker_timeout_s}s")

        p.join()
        status, result = result_payload
        
        if status == "error":
            raise RuntimeError(f"Isolated transcription failed: {result}")
            
        return result


# =============================================================================
# Provider Factory
# =============================================================================

def get_transcription_provider(cfg: TranscriptionConfig) -> TranscriptionProvider:
    """Get the appropriate transcription provider based on config."""
    if cfg.provider == "openai":
        # Verify API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, falling back to local provider")
            return LocalFasterWhisperProvider(cfg)
        return OpenAIProvider(cfg)

    # Process Isolation removed per user request
    # if cfg.provider == "isolated":
    #     return ProcessIsolatedWhisperProvider(cfg)
    
    return LocalFasterWhisperProvider(cfg)


# =============================================================================
# Main Entry Point
# =============================================================================

def transcribe_audio(
    audio_path: str,
    cfg: TranscriptionConfig,
    output_json_path: Optional[str] = None
) -> Transcript:
    """
    Transcribe audio file using configured provider.
    
    Args:
        audio_path: Path to audio file (should be 16kHz mono WAV)
        cfg: Transcription configuration
        output_json_path: Optional path to save transcript JSON
    
    Returns:
        Transcript object with segments and word-level timestamps
    """
    provider = get_transcription_provider(cfg)
    
    try:
        transcript = provider.transcribe(audio_path)
    except Exception as e:
        # Fallback to local if OpenAI fails
        if cfg.provider == "openai":
            logger.warning(f"OpenAI transcription failed: {e}. Falling back to local.")
            local_provider = LocalFasterWhisperProvider(cfg)
            transcript = local_provider.transcribe(audio_path)
        else:
            raise
    
    if output_json_path:
        transcript.save(output_json_path)
        logger.info(f"Saved transcript to '{output_json_path}'")
    
    return transcript


# =============================================================================
# Legacy Compatibility
# =============================================================================

def transcribe_video(
    input_video: str,
    srt_path: str,
    model_size: str = "small",
    language: Optional[str] = None
) -> Tuple[List[Dict], str]:
    """
    DEPRECATED: Legacy function for backward compatibility.
    
    Use transcribe_audio() with TranscriptionConfig instead.
    """
    logger.warning("transcribe_video() is deprecated. Use transcribe_audio() instead.")
    
    # Create legacy config
    cfg = TranscriptionConfig(
        provider="local",
        model=model_size
    )
    
    # Note: This function takes video, not audio
    # For backward compat, we pass video directly (faster-whisper handles extraction)
    provider = LocalFasterWhisperProvider(cfg)
    transcript = provider.transcribe(input_video)
    
    # Write SRT for compatibility
    _write_srt(transcript, srt_path)
    
    return transcript.to_legacy_segments(), transcript.language


def _write_srt(transcript: Transcript, srt_path: str) -> None:
    """Write transcript to SRT format."""
    os.makedirs(os.path.dirname(srt_path) or ".", exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for seg in transcript.segments:
            f.write(f"{seg.id + 1}\n")
            f.write(f"{format_timestamp(seg.start_s)} --> {format_timestamp(seg.end_s)}\n")
            f.write(f"{seg.text}\n\n")


def parse_srt_to_segments(srt_path: str) -> List[Dict]:
    """Parse an SRT file back into legacy segments format."""
    if not os.path.exists(srt_path):
        return []
    
    with open(srt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    segments = []
    
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        
        start_ts, end_ts = [p.strip() for p in lines[1].split("-->")]
        try:
            segments.append({
                "id": lines[0].strip(),
                "start": parse_timestamp(start_ts),
                "end": parse_timestamp(end_ts),
                "text": " ".join(l.strip() for l in lines[2:]).strip()
            })
        except Exception:
            continue
    
    return segments
