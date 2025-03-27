"""
Orpheus TTS - A Python package for text-to-speech using Orpheus models
"""

# API functions
from .api import format_prompt, generate_tokens_from_api, estimate_tokens

# Audio processing functions
from .audio import convert_to_audio, tokens_decoder_sync

# Synthesizer functions
from .synthesizer import (
    generate_speech, 
    generate_audio, 
    list_available_voices, 
    split_text_into_chunks,
    combine_audio_segments
)

# Benchmark functionality
from .benchmark_runner import main as benchmark_main

# Grid search functionality
from .grid_search import main as grid_search_main

# Best voices functionality
from .best_voices import main as best_voices_main

__version__ = "0.1.0" 