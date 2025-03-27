"""
Command-line interface for Orpheus TTS
"""

import os
import time
import argparse
from .config import (
    DEFAULT_VOICE,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY,
    MAX_CHUNK_SIZE,
)
from .synthesizer import generate_speech, list_available_voices


def main():
    """
    Main CLI entry point for Orpheus TTS.
    """
    parser = argparse.ArgumentParser(
        description="Orpheus Text-to-Speech - With automatic text chunking for long texts"
    )
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--file", type=str, help="Read input text from file")
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Voice to use (default: {DEFAULT_VOICE})",
    )
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p", type=float, default=TOP_P, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=REPETITION_PENALTY,
        help="Repetition penalty (>=1.1 required for stable generation)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated tokens and decoder debug info",
    )

    args = parser.parse_args()

    if args.list_voices:
        list_available_voices()
        return

    prompt = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.text:
        prompt = args.text

    if not prompt:
        prompt = input("Enter text to synthesize: ")
        if not prompt:
            prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."

    output_file = args.output
    if not output_file:
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")

    if len(prompt) > MAX_CHUNK_SIZE:
        print(f"Long text detected ({len(prompt)} characters). Will automatically split into chunks of ~{MAX_CHUNK_SIZE} characters for processing.")

    start_time = time.time()
    generate_speech(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file,
        verbose=args.verbose,
    )
    end_time = time.time()

    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}") 