import os
import urllib.request
import requests
import logging
from typing import Optional
from videocuts.video.detection import load_haar_cascade

logger = logging.getLogger(__name__)

def ensure_face_detector_model(model_url: str, model_path: str) -> Optional[str]:
    """Download Google Face Detector model (TFLite) if not present."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(model_path):
        return model_path

    logger.info("Downloading face detector model...")
    try:
        resp = requests.get(model_url, timeout=30)
        resp.raise_for_status()
        with open(model_path, "wb") as f: f.write(resp.content)
        logger.info("Face detector model downloaded successfully.")
        return model_path
    except Exception as exc:
        logger.error(f"Failed to fetch detector model: {exc}")
        return None

def ensure_face_landmarker_model(model_url: str, model_path: str) -> Optional[str]:
    """Download Face Landmarker model for lip detection."""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(model_path):
        return model_path

    logger.info("Downloading face landmarker model for lip detection...")
    try:
        resp = requests.get(model_url, timeout=60)
        resp.raise_for_status()
        with open(model_path, "wb") as f: f.write(resp.content)
        logger.info("Face landmarker model downloaded successfully.")
        return model_path
    except Exception as exc:
        logger.error(f"Failed to fetch face landmarker model: {exc}")
        return None
