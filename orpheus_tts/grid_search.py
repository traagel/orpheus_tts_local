"""
Grid Search for Orpheus TTS

This script performs a comprehensive grid search across all available voices
and parameter combinations. The results are saved in a structured folder with
descriptive filenames.
"""

import os
import time
import argparse
import itertools
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np

from .config import AVAILABLE_VOICES
from .synthesizer import generate_audio, list_available_voices

def estimate_runtime(voices, temperatures, top_ps, rep_penalties, test_text, avg_time_per_run=50.0):
    """
    Estimate the total runtime for the grid search.
    
    Args:
        voices (list): List of voices to test
        temperatures (list): List of temperature values to test
        top_ps (list): List of top_p values to test
        rep_penalties (list): List of repetition penalty values to test
        test_text (str): The text to use for testing
        avg_time_per_run (float): Average time in seconds per run (from benchmarks)
        
    Returns:
        tuple: (total_combinations, estimated_time_seconds)
    """
    total_combinations = len(voices) * len(temperatures) * len(top_ps) * len(rep_penalties)
    # Calculate time based on average time from benchmark results
    estimated_time_seconds = total_combinations * avg_time_per_run
    return total_combinations, estimated_time_seconds

def format_time(seconds):
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds (float): Number of seconds
        
    Returns:
        str: Formatted time string (days, hours, minutes, seconds)
    """
    return str(timedelta(seconds=int(seconds)))

def create_output_dir(base_dir="grid_search_results"):
    """
    Create the output directory structure for the grid search results.
    
    Args:
        base_dir (str): Base directory name
        
    Returns:
        str: Path to the created timestamp directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    return timestamp_dir

def run_grid_search(
    voices,
    temperatures,
    top_ps,
    rep_penalties,
    test_text,
    output_dir,
    sample_duration=3.0,
    max_tokens=8192,
):
    """
    Run the grid search across all parameter combinations.
    
    Args:
        voices (list): List of voices to test
        temperatures (list): List of temperature values to test
        top_ps (list): List of top_p values to test
        rep_penalties (list): List of repetition penalty values to test
        test_text (str): The text to use for testing
        output_dir (str): Directory to save results
        sample_duration (float): Desired duration of each sample in seconds
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        dict: Summary of the grid search results
    """
    results = {}
    total_runs = len(voices) * len(temperatures) * len(top_ps) * len(rep_penalties)
    run_count = 0
    
    # Create a progress bar
    progress_bar = tqdm(total=total_runs, desc="Grid Search Progress")
    
    start_time = time.time()
    
    # First, adjust text length based on sample_duration if needed
    # Assuming about 150 chars = 5-7 seconds of audio
    approximate_chars = int(sample_duration * 25)
    if len(test_text) > approximate_chars:
        test_text = test_text[:approximate_chars].rsplit('.', 1)[0] + '.'
    
    for voice in voices:
        # Create voice directory
        voice_dir = os.path.join(output_dir, voice)
        os.makedirs(voice_dir, exist_ok=True)
        
        if voice not in results:
            results[voice] = {}
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(temperatures, top_ps, rep_penalties))
        
        for temp, top_p, rep_penalty in param_combinations:
            run_count += 1
            run_start = time.time()
            
            # Define output filename
            filename = f"{voice}_temp_{temp:.1f}_top_p_{top_p:.2f}_rep_penalty_{rep_penalty:.1f}.wav"
            output_path = os.path.join(voice_dir, filename)
            
            try:
                # Generate audio with a maximum token limit to prevent excessive generation
                audio_bytes = generate_audio(
                    text=test_text,
                    voice=voice,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    max_tokens=max_tokens,
                    verbose=False
                )
                
                # Save to file
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Record successful run
                run_time = time.time() - run_start
                if temp not in results[voice]:
                    results[voice][temp] = {}
                if top_p not in results[voice][temp]:
                    results[voice][temp][top_p] = {}
                
                results[voice][temp][top_p][rep_penalty] = {
                    'success': True,
                    'time': run_time,
                    'file': output_path
                }
                
                # Update progress bar
                elapsed = time.time() - start_time
                estimated_total = (elapsed / run_count) * total_runs
                remaining = estimated_total - elapsed
                
                progress_bar.set_postfix({
                    'voice': voice, 
                    'temp': f"{temp:.1f}", 
                    'top_p': f"{top_p:.2f}", 
                    'rep_penalty': f"{rep_penalty:.1f}",
                    'time': f"{run_time:.2f}s",
                    'ETA': format_time(remaining)
                })
                
            except Exception as e:
                # Record failed run
                if temp not in results[voice]:
                    results[voice][temp] = {}
                if top_p not in results[voice][temp]:
                    results[voice][temp][top_p] = {}
                
                results[voice][temp][top_p][rep_penalty] = {
                    'success': False,
                    'error': str(e)
                }
                
                print(f"\nError with {voice}, temp={temp}, top_p={top_p}, rep_penalty={rep_penalty}: {e}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    total_time = time.time() - start_time
    
    # Generate summary file
    summary_path = os.path.join(output_dir, 'grid_search_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Orpheus TTS Grid Search\n")
        f.write(f"======================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runs: {total_runs}\n")
        f.write(f"Total Time: {format_time(total_time)}\n\n")
        
        f.write(f"Parameter Ranges:\n")
        f.write(f"- Voices: {', '.join(voices)}\n")
        f.write(f"- Temperatures: {', '.join([str(t) for t in temperatures])}\n")
        f.write(f"- Top-p values: {', '.join([str(p) for p in top_ps])}\n")
        f.write(f"- Repetition penalties: {', '.join([str(r) for r in rep_penalties])}\n")
        f.write(f"- Max tokens: {max_tokens}\n\n")
        
        f.write(f"Test text: \"{test_text}\"\n\n")
        
        f.write(f"Results Summary:\n")
        for voice in results:
            successful_runs = sum(
                results[voice][t][p][r]['success'] 
                for t in results[voice] 
                for p in results[voice][t] 
                for r in results[voice][t][p]
            )
            total_voice_runs = len(temperatures) * len(top_ps) * len(rep_penalties)
            f.write(f"- {voice}: {successful_runs}/{total_voice_runs} successful\n")
        
        f.write("\nBest combinations for each voice (fastest successful run):\n")
        for voice in results:
            best_time = float('inf')
            best_params = None
            
            for t in results[voice]:
                for p in results[voice][t]:
                    for r in results[voice][t][p]:
                        if results[voice][t][p][r]['success'] and results[voice][t][p][r]['time'] < best_time:
                            best_time = results[voice][t][p][r]['time']
                            best_params = (t, p, r)
            
            if best_params:
                t, p, r = best_params
                f.write(f"- {voice}: temp={t}, top_p={p}, rep_penalty={r}, time={best_time:.2f}s\n")
                f.write(f"  File: {results[voice][t][p][r]['file']}\n")
            else:
                f.write(f"- {voice}: No successful runs\n")
    
    print(f"\nGrid search complete! Results saved to {output_dir}")
    print(f"Summary file: {summary_path}")
    
    return results

def main():
    """
    Main function to run the grid search script.
    """
    parser = argparse.ArgumentParser(description="Orpheus TTS Grid Search")
    parser.add_argument("--text", type=str, help="Text to convert to speech for testing")
    parser.add_argument("--file", type=str, help="Read input text from file")
    parser.add_argument("--output-dir", type=str, default="grid_search_results", 
                        help="Base directory for output files")
    parser.add_argument("--voices", type=str, nargs='+', 
                        help="Specific voices to test (default: all voices)")
    parser.add_argument("--temps", type=float, nargs='+', 
                        default=[0.3, 0.6, 0.9, 1.2],
                        help="Temperature values to test")
    parser.add_argument("--top-ps", type=float, nargs='+', 
                        default=[0.3, 0.6, 0.8, 0.95],
                        help="Top-p values to test")
    parser.add_argument("--rep-penalties", type=float, nargs='+', 
                        default=[1.1, 1.3, 1.5, 1.8],
                        help="Repetition penalty values to test")
    parser.add_argument("--list-voices", action="store_true", 
                        help="List available voices")
    parser.add_argument("--sample-duration", type=float, default=3.0,
                        help="Target duration in seconds for audio samples (will truncate text)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum number of tokens to generate for each sample")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Get the test text
    test_text = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            test_text = f.read().strip()
    elif args.text:
        test_text = args.text
    
    if not test_text:
        test_text = "This is a test of the Orpheus text-to-speech system with different parameters. How does it sound to you?"
    
    # Choose which voices to test
    voices = args.voices if args.voices else AVAILABLE_VOICES
    
    # Make sure all requested voices are valid
    invalid_voices = [v for v in voices if v not in AVAILABLE_VOICES]
    if invalid_voices:
        print(f"Warning: The following voices are not available: {', '.join(invalid_voices)}")
        voices = [v for v in voices if v in AVAILABLE_VOICES]
        if not voices:
            print("No valid voices specified. Exiting.")
            return
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Estimate runtime
    total_combinations, estimated_seconds = estimate_runtime(
        voices, args.temps, args.top_ps, args.rep_penalties, test_text
    )
    
    print(f"Orpheus TTS Grid Search")
    print(f"======================")
    print(f"Voices: {', '.join(voices)}")
    print(f"Temperatures: {args.temps}")
    print(f"Top-p values: {args.top_ps}")
    print(f"Repetition penalties: {args.rep_penalties}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Total combinations: {total_combinations}")
    print(f"Estimated runtime: {format_time(estimated_seconds)}")
    print(f"Output directory: {output_dir}")
    print(f"Test text: \"{test_text[:80]}{'...' if len(test_text) > 80 else ''}\"")
    
    proceed = input("Proceed with grid search? (y/n): ")
    if proceed.lower() not in ["y", "yes"]:
        print("Grid search canceled.")
        return
    
    # Run the grid search
    run_grid_search(
        voices=voices,
        temperatures=args.temps,
        top_ps=args.top_ps,
        rep_penalties=args.rep_penalties,
        test_text=test_text,
        output_dir=output_dir,
        sample_duration=args.sample_duration,
        max_tokens=args.max_tokens
    )

if __name__ == "__main__":
    main() 