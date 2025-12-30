#!/usr/bin/env python3
"""
Legacy entry point for Video Highlights Generator.
This script now acts as a wrapper for the modular 'videocuts' package.
"""
import sys
import os

# Add src to sys.path to allow imports from videocuts package without installation
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from videocuts.cli import main
except ImportError as e:
    print(f"Error: Could not import 'videocuts': {e}")
    print("Make sure you are running this script from the project root directory.")
    sys.exit(1)

if __name__ == "__main__":
    main()