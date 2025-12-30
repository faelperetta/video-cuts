import torch
import logging

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Choose the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def print_device_info(device: torch.device) -> None:
    """Log information about the selected device."""
    if device.type == "cuda":
        logger.info("CUDA is available. Using GPU for processing.")
    else:
        logger.info("CUDA not available. Using CPU for processing.")

TORCH_DEVICE = get_device()
