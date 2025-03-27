"""
Orpheus TTS Benchmark Package

This package contains tools to benchmark and optimize the Orpheus TTS system.
"""

import argparse
import os
from datetime import datetime
import sys
import platform

from ..config import MAX_TOKENS
from .utils import initialize_metadata
from .text_length import benchmark_text_length
from .parameters import (
    benchmark_temperature,
    benchmark_top_p,
    benchmark_repetition_penalty
)
from .reporting import save_metadata_json, generate_report


def main():
    """
    Main function to run benchmark tests.
    """
    parser = argparse.ArgumentParser(description='Benchmark Orpheus TTS hyperparameters')
    parser.add_argument('--input-file', type=str, default='texts/fairies-of-the-waterfall.txt',
                        help='Path to input text file')
    parser.add_argument('--voice', type=str, default='tara',
                        help='Voice to use for benchmarking')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this benchmark run (default: auto-generated)')
    parser.add_argument('--max-length', type=int, default=3000,
                        help='Maximum text length to test')
    parser.add_argument('--length-step', type=int, default=250,
                        help='Step size for text length testing')
    parser.add_argument('--test-length', type=int, default=1000,
                        help='Fixed text length to use for parameter testing')
    parser.add_argument('--skip-length', action='store_true',
                        help='Skip text length benchmark')
    parser.add_argument('--skip-temperature', action='store_true',
                        help='Skip temperature benchmark')
    parser.add_argument('--skip-top-p', action='store_true',
                        help='Skip top-p benchmark')
    parser.add_argument('--skip-rep-penalty', action='store_true',
                        help='Skip repetition penalty benchmark')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create run name if not provided
    run_name = args.run_name or f"{args.voice}_{timestamp}"
    
    # Create output directory structure
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metadata dictionary
    metadata = initialize_metadata(
        timestamp=timestamp,
        run_name=run_name,
        voice=args.voice,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.platform(),
        max_tokens=MAX_TOKENS,
        parameters={
            "input_file": args.input_file,
            "max_length": args.max_length,
            "length_step": args.length_step,
            "test_length": args.test_length,
            "skip_length": args.skip_length,
            "skip_temperature": args.skip_temperature,
            "skip_top_p": args.skip_top_p,
            "skip_rep_penalty": args.skip_rep_penalty,
            "verbose": args.verbose
        }
    )
    
    print(f"Starting Orpheus TTS benchmark with voice: {args.voice}")
    print(f"Run name: {run_name}")
    print(f"Results will be saved to: {output_dir}")
    
    # Run benchmarks
    if not args.skip_length:
        benchmark_text_length(args.input_file, args.voice, output_dir, 
                              max_chars=args.max_length, step=args.length_step,
                              verbose=args.verbose, metadata=metadata)
    
    if not args.skip_temperature:
        benchmark_temperature(args.input_file, args.voice, output_dir,
                             text_length=args.test_length, verbose=args.verbose,
                             metadata=metadata)
    
    if not args.skip_top_p:
        benchmark_top_p(args.input_file, args.voice, output_dir,
                       text_length=args.test_length, verbose=args.verbose,
                       metadata=metadata)
    
    if not args.skip_rep_penalty:
        benchmark_repetition_penalty(args.input_file, args.voice, output_dir,
                                    text_length=args.test_length, verbose=args.verbose,
                                    metadata=metadata)
    
    # Save metadata
    save_metadata_json(output_dir, metadata)
    
    # Generate summary report
    report_file = generate_report(output_dir, args.voice, metadata)
    print(f"Benchmark complete! Full report available at: {report_file}") 