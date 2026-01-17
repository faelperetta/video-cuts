# =============================================================================
# VideoCuts Docker Image with Intel Arc XPU Support
# =============================================================================
# Base: Intel Optimized PyTorch with XPU support
# Supports: Intel Arc A-Series (A770, A750, A580), B-Series (B580, B570)
# =============================================================================

FROM intel/intel-optimized-pytorch:2.3.110-xpu-pip-base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Install system dependencies for OpenCV, MediaPipe, and video processing
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Video processing
    ffmpeg \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Setup working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------
# Copy only dependency files first for better layer caching
COPY pyproject.toml ./

# Upgrade pip and install remaining dependencies
# Note: torch/torchaudio already installed in base image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    # Core dependencies
    fastapi \
    "uvicorn[standard]" \
    "sqlalchemy[asyncio]" \
    asyncpg \
    pydantic-settings \
    python-dotenv \
    # Video/Audio processing
    opencv-python-headless \
    mediapipe \
    pillow \
    # AI/ML (whisper, etc.)
    openai-whisper \
    faster-whisper \
    openai \
    transformers \
    numpy \
    # Video download
    yt-dlp \
    # Text processing
    better-profanity \
    # HTTP
    requests \
    httpx

# -----------------------------------------------------------------------------
# Copy application source code
# -----------------------------------------------------------------------------
COPY src/ ./src/

# Install the package itself
RUN pip install --no-cache-dir -e .

# -----------------------------------------------------------------------------
# Configure runtime environment
# -----------------------------------------------------------------------------
# Create storage directory
RUN mkdir -p /app/storage

# Environment defaults
ENV STORAGE_PATH=/app/storage
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# -----------------------------------------------------------------------------
# Default command: Run FastAPI server
# -----------------------------------------------------------------------------
CMD ["uvicorn", "videocuts.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
