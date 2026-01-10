import os
import logging

# Capture whether the user explicitly set the device selector.
# We MUST have a value set BEFORE importing torch to avoid the 
# "CommandLine Error: Option 'simd-mode' registered more than once!" crash.
_USER_SELECTOR = os.environ.get("ONEAPI_DEVICE_SELECTOR")
if _USER_SELECTOR is None:
    # Safe default to prevent driver discovery-time LLVM flag collision.
    # This value allows the app to start but we won't use XPU unless requested.
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

import torch

logger = logging.getLogger(__name__)

def is_intel_accel_enabled() -> bool:
    """Check if Intel acceleration (XPU/OpenVINO) is explicitly enabled by the user."""
    return _USER_SELECTOR is not None

def get_device() -> torch.device:
    """Choose the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Only return XPU if the user explicitly opted-in via ONEAPI_DEVICE_SELECTOR
    if is_intel_accel_enabled():
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        
    return torch.device("cpu")

def print_device_info(device: torch.device) -> None:
    """Log information about the selected device."""
    if device.type == "cuda":
        logger.info("CUDA is available. Using GPU for processing.")
    elif device.type == "xpu":
        device_name = torch.xpu.get_device_name(device.index if device.index is not None else 0)
        logger.info(f"XPU is available ({device_name}). Using XPU for processing.")
    else:
        logger.info("No CUDA or XPU available. Using CPU for processing.")

TORCH_DEVICE = get_device()
