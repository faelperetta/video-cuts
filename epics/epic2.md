# EPIC 2 — Transcription & Speaker Segmentation

This epic covers the ingestion of audio, speech-to-text conversion, and speaker identification (diarization). It establishes the "ground truth" data (text + timestamps + speaker labels) that downstream components (clipping, cropping, captioning) rely on.

---

## US-2.1 — Transcribe audio (Configurable: Local vs OpenAI)

**As a** system  
**I want to** generate timestamped text from the audio  
**So that** we can build captions and determine clip boundaries

### Default Technical Design

The system must support a configuration switch to choose between **Local** (default) and **API** providers.

#### Option A: Local `faster-whisper` (Recommended for Intel Arc)
*   **Library:** `faster-whisper`
*   **Model:** `large-v3`
*   **Compute Type:** `float16` (fallback to `int8_float16` if VRAM is constrained)
*   **VAD (Voice Activity Detection):** Enabled
*   **Device:** `openvino` (if available) or `cuda` / `cpu`

#### Option B: OpenAI API
*   **Endpoint:** `https://api.openai.com/v1/audio/transcriptions`
*   **Model:** `whisper-1`

#### Output Format: `transcript.json`

The output must adhere to this JSON schema:

```json
{
  "provider": "local_faster_whisper",
  "model": "large-v3",
  "language": "en",
  "duration_s": 3723.52,
  "segments": [
    {
      "id": 0,
      "start_s": 12.34,
      "end_s": 18.90,
      "text": " So here's the thing...",
      "avg_logprob": -0.12,
      "no_speech_prob": 0.01,
      "words": [
        { "start_s": 12.34, "end_s": 12.60, "word": "So" },
        { "start_s": 12.60, "end_s": 12.92, "word": "here's" }
      ]
    }
  ]
}
```


---

## US-2.0 — Normalize audio for ASR + diarization

**As a** system  
**I want to** extract and normalize audio from the uploaded video  
**So that** transcription and diarization are stable and reproducible

### Default Technical Design
*   **Extraction Tool:** FFmpeg
*   **Output Format:**
    *   Container: `wav`
    *   Sample Rate: `16000 Hz`
    *   Channels: `mono` (1 channel)
    *   Sample Format: `s16` (16-bit PCM)
*   **Secondary Output (Optional):**
    *   Create a `48000 Hz` stereo wav for the final render pipeline to ensure high-quality audio in the output clip.

### Acceptance Criteria
*   [ ] Given an input video, the pipeline produces a file named `audio_16k_mono.wav`.
*   [ ] The duration of `audio_16k_mono.wav` matches the input video duration within ±100ms.
*   [ ] The process is deterministic (same input file + same FFmpeg version = identical output bytes).
*   [ ] Metadata is extracted and stored (Sample Rate, Channels, Duration, SHA256 hash).