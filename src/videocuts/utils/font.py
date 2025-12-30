import os
import subprocess
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default fallback font paths (system fonts that are commonly available)
FALLBACK_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
]

def find_font_path(font_name: str, bold: bool = True) -> Optional[str]:
    """Search for a font file on the system by font name."""
    pattern = font_name
    if bold:
        pattern += ":weight=bold"
    
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            font_path = result.stdout.strip()
            if os.path.isfile(font_path):
                return font_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    font_dirs = [
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/TTF/",
        "/usr/local/share/fonts/",
        os.path.expanduser("~/.fonts/"),
        os.path.expanduser("~/.local/share/fonts/"),
    ]
    
    suffix = "-Bold" if bold else "-Regular"
    possible_names = [
        f"{font_name}{suffix}.ttf",
        f"{font_name}{suffix}.otf",
        f"{font_name.replace(' ', '')}{suffix}.ttf",
        f"{font_name.replace(' ', '')}{suffix}.otf",
        f"{font_name.lower()}{suffix.lower()}.ttf",
        f"{font_name.lower()}{suffix.lower()}.otf",
    ]
    
    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _, files in os.walk(font_dir):
            for filename in files:
                if filename in possible_names or filename.lower() in [n.lower() for n in possible_names]:
                    return os.path.join(root, filename)
    
    return None

def get_font_path_with_fallback(font_name: str, font_path: str = "", bold: bool = True) -> str:
    """Get a usable font path, with automatic detection and fallback."""
    if font_path and os.path.isfile(font_path):
        return font_path
    
    found_path = find_font_path(font_name, bold)
    if found_path:
        return found_path
    
    for fallback in FALLBACK_FONT_PATHS:
        if os.path.isfile(fallback):
            logger.warning(f"'{font_name}' not found, using fallback: {fallback}")
            return fallback
    
    logger.warning(f"No font file found for '{font_name}', FFmpeg may fail")
    return font_name

def verify_font_available(font_name: str, font_path: str = "") -> bool:
    """Verify that a font is available for rendering."""
    if font_path and os.path.isfile(font_path):
        return True
    
    try:
        result = subprocess.run(
            ["fc-list", f":family={font_name}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return find_font_path(font_name) is not None

def print_font_installation_guide(font_name: str) -> None:
    """Log a guide on how to install the missing font."""
    guide = f"""
'{font_name}' font not found on your system.
To install Montserrat font:
  Ubuntu/Debian: sudo apt install fonts-montserrat
  Fedora: sudo dnf install google-montserrat-fonts
  Arch: sudo pacman -S ttf-montserrat
  Manual: Download from https://fonts.google.com/specimen/Montserrat
          Extract and copy .ttf files to ~/.local/share/fonts/
          Run: fc-cache -f -v
"""
    logger.info(guide)
