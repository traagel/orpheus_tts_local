"""
Best Voices Generator for Orpheus TTS

This script generates audio using all available voices with their optimal parameters
as identified through grid search testing.
"""

import os
import time
import argparse
from datetime import datetime
import pathlib
from tqdm import tqdm

from .config import AVAILABLE_VOICES
from .synthesizer import generate_speech, split_text_into_chunks, list_available_voices

# Optimal parameters for each voice based on grid search results
OPTIMAL_VOICE_PARAMS = {
    "tara": {"temp": 1.2, "top_p": 0.8, "rep_penalty": 1.5},
    "leah": {"temp": 0.3, "top_p": 0.6, "rep_penalty": 1.8},
    "jess": {"temp": 1.2, "top_p": 0.8, "rep_penalty": 1.5},
    "leo": {"temp": 0.6, "top_p": 0.8, "rep_penalty": 1.8},
    "dan": {"temp": 0.3, "top_p": 0.6, "rep_penalty": 1.8},
    "mia": {"temp": 0.9, "top_p": 0.3, "rep_penalty": 1.1},
    "zac": {"temp": 1.2, "top_p": 0.95, "rep_penalty": 1.3},
    "zoe": {"temp": 0.6, "top_p": 0.95, "rep_penalty": 1.8}
}

def get_voice_categories():
    """
    Group voices by their general characteristics based on optimal parameters.
    
    Returns:
        dict: Voice categories with lists of voices in each category
    """
    categories = {
        "expressive": [],  # High temperature (0.9-1.2), high top_p
        "precise": [],     # Low temperature (0.3-0.6), high rep_penalty
        "balanced": [],    # Mid temperature with balanced parameters
        "unique": []       # Unusual parameter combinations
    }
    
    for voice, params in OPTIMAL_VOICE_PARAMS.items():
        if params["temp"] >= 0.9 and params["top_p"] >= 0.8:
            categories["expressive"].append(voice)
        elif params["temp"] <= 0.6 and params["rep_penalty"] >= 1.5:
            categories["precise"].append(voice)
        elif 0.6 <= params["temp"] < 0.9 and 0.6 <= params["top_p"] < 0.95:
            categories["balanced"].append(voice)
        else:
            categories["unique"].append(voice)
    
    return categories

def create_output_dir(input_file, base_dir="outputs/all"):
    """
    Create the output directory structure.
    
    Args:
        input_file (str): Path to the input file
        base_dir (str): Base directory for outputs
        
    Returns:
        str: Path to the created output directory
    """
    # Extract just the filename without extension from the input path
    input_name = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{input_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_all_voices(input_file, output_dir, selected_voices=None, max_tokens=8192, verbose=False):
    """
    Generate audio for the input text using all voices with their optimal parameters.
    
    Args:
        input_file (str): Path to the input text file
        output_dir (str): Directory to save the generated audio files
        selected_voices (list, optional): List of specific voices to use
        max_tokens (int): Maximum tokens to generate
        verbose (bool): Whether to show verbose output
        
    Returns:
        dict: Results summary with generation times and file paths
    """
    # Read the input text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Determine which voices to use
    voices_to_use = selected_voices if selected_voices else AVAILABLE_VOICES
    voices_to_use = [v for v in voices_to_use if v in OPTIMAL_VOICE_PARAMS]
    
    # Get voice categories for the report
    categories = get_voice_categories()
    
    results = {}
    total_voices = len(voices_to_use)
    
    # Create a progress bar
    progress_bar = tqdm(total=total_voices, desc="Generating Voices")
    
    print(f"\nGenerating audio for {total_voices} voices using optimal parameters")
    print(f"Input text: {input_file} ({len(text)} characters)")
    
    # Check if the text is long enough to require chunking
    needs_chunking = len(text) > 750
    if needs_chunking:
        chunks = split_text_into_chunks(text)
        print(f"Text will be split into {len(chunks)} chunks for processing")
    
    start_time = time.time()
    
    for voice in voices_to_use:
        voice_start = time.time()
        
        # Get optimal parameters for this voice
        params = OPTIMAL_VOICE_PARAMS[voice]
        
        # Define output filename
        output_file = os.path.join(output_dir, f"{voice}.wav")
        
        try:
            # Generate speech with optimal parameters
            generate_speech(
                prompt=text,
                voice=voice,
                temperature=params["temp"],
                top_p=params["top_p"],
                repetition_penalty=params["rep_penalty"],
                max_tokens=max_tokens,
                output_file=output_file,
                verbose=verbose
            )
            
            voice_time = time.time() - voice_start
            
            # Record results
            results[voice] = {
                "success": True,
                "time": voice_time,
                "file": output_file,
                "params": params
            }
            
            # Update progress bar with voice info
            progress_bar.set_postfix({
                'voice': voice,
                'time': f"{voice_time:.2f}s"
            })
            
        except Exception as e:
            results[voice] = {
                "success": False,
                "error": str(e)
            }
            print(f"\nError with {voice}: {e}")
        
        progress_bar.update(1)
    
    progress_bar.close()
    total_time = time.time() - start_time
    
    # Generate summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("Orpheus TTS - Best Voices Generation\n")
        f.write("===================================\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Text length: {len(text)} characters\n")
        f.write(f"Total time: {total_time:.2f} seconds\n\n")
        
        f.write("Voice Categories:\n")
        for category, voices in categories.items():
            f.write(f"- {category.capitalize()}: {', '.join(voices)}\n")
        f.write("\n")
        
        f.write("Voice Results:\n")
        for voice, data in results.items():
            if data["success"]:
                params = data["params"]
                f.write(f"- {voice}: Generated in {data['time']:.2f}s with temp={params['temp']}, ")
                f.write(f"top_p={params['top_p']}, rep_penalty={params['rep_penalty']}\n")
                f.write(f"  File: {data['file']}\n")
            else:
                f.write(f"- {voice}: Failed - {data['error']}\n")
    
    print(f"\nAll voices generated in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    print(f"Summary file: {summary_file}")
    
    return results

def main():
    """
    Main function to run the best voices generator script.
    """
    parser = argparse.ArgumentParser(description="Orpheus TTS - Generate Audio with Optimal Voice Parameters")
    parser.add_argument("--file", type=str, required=True, 
                        help="Input text file to convert to speech")
    parser.add_argument("--output-dir", type=str, default="outputs/all", 
                        help="Base directory for output files")
    parser.add_argument("--voices", type=str, nargs='+', 
                        help="Specific voices to use (default: all voices)")
    parser.add_argument("--categories", type=str, nargs='+', choices=["expressive", "precise", "balanced", "unique"],
                        help="Generate only voices from specific categories")
    parser.add_argument("--list-voices", action="store_true", 
                        help="List available voices and their optimal parameters")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum number of tokens to generate for each sample")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output during generation")
    
    args = parser.parse_args()
    
    if args.list_voices:
        print("Available voices with optimal parameters:")
        categories = get_voice_categories()
        
        for category, voices in categories.items():
            print(f"\n{category.capitalize()} voices:")
            for voice in voices:
                params = OPTIMAL_VOICE_PARAMS[voice]
                print(f"- {voice}: temp={params['temp']}, top_p={params['top_p']}, rep_penalty={params['rep_penalty']}")
        return
    
    if not os.path.exists(args.file):
        print(f"Error: Input file {args.file} not found")
        return
    
    # Select voices based on categories if specified
    selected_voices = args.voices
    if args.categories and not args.voices:
        categories = get_voice_categories()
        selected_voices = []
        for category in args.categories:
            selected_voices.extend(categories[category])
    
    # Create output directory based on input filename
    output_dir = create_output_dir(args.file, args.output_dir)
    
    # Generate audio for all selected voices
    generate_all_voices(
        input_file=args.file,
        output_dir=output_dir,
        selected_voices=selected_voices,
        max_tokens=args.max_tokens,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main() 