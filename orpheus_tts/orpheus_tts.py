"""
Orpheus TTS - Main application module

A Python package for converting text to speech using Orpheus models.
This is the main entry point file that users can run directly.
"""

import sys
from .cli import main


if __name__ == "__main__":
    main() 