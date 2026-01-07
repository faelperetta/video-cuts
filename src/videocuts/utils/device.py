import torch
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Choose the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for XPU (Intel GPU)
    # Note: On some systems, ONEAPI_DEVICE_SELECTOR=level_zero:0 might be needed 
    # to avoid LLVM duplicate option conflicts.
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
