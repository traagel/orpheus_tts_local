"""
Text length benchmarking module for Orpheus TTS.
"""

import os
import time
import csv
from typing import Dict, Any, List, Optional, Tuple

from ..api import estimate_tokens
from ..synthesizer import generate_audio
from .utils import read_text_sample, save_csv, count_tokens, time_function


def benchmark_text_length(
    input_file: str,
    voice: str,
    output_dir: str,
    max_chars: int = 3000,
    step: int = 250,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    verbose: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Benchmark text generation with increasing text lengths.
    
    Args:
        input_file: Path to the input text file
        voice: Voice to use for benchmarking
        output_dir: Directory to save benchmark results
        max_chars: Maximum text length to test
        step: Step size for text length testing
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        repetition_penalty: Repetition penalty parameter for generation
        verbose: Whether to print verbose output
        metadata: Metadata dictionary to update with results
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Text Length ===")
    
    # Store test parameters in metadata
    if metadata:
        metadata["tests"]["text_length"] = {
            "max_chars": max_chars,
            "step": step,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
    
    # Get input file details
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    total_chars = len(full_text)
    total_tokens = count_tokens(full_text)
    
    if metadata:
        metadata["input_file_details"] = {
            "path": input_file,
            "total_chars": total_chars,
            "total_tokens": total_tokens
        }
    
    # Create output directory for CSV and audio files
    length_dir = os.path.join(output_dir, 'length_test')
    os.makedirs(length_dir, exist_ok=True)
    
    # Prepare result data for CSV
    headers = ['Length (chars)', 'Tokens', 'Success', 'Time (s)', 'Audio File']
    rows = []
    results = []
    
    # Test with increasing text lengths
    lengths = list(range(step, min(max_chars + step, total_chars), step))
    
    for length in lengths:
        text, actual_length = read_text_sample(input_file, length)
        tokens = count_tokens(text)
        
        if verbose:
            print(f"\nTesting length: {actual_length} chars ({tokens} tokens)")
        else:
            print(f"Testing length: {actual_length} chars ({tokens} tokens)...", end='', flush=True)
        
        try:
            # Time the generation
            result, generation_time = time_function(
                generate_audio,
                text,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Save the audio file
            audio_filename = f"length_{actual_length}.wav"
            audio_path = os.path.join(length_dir, audio_filename)
            with open(audio_path, 'wb') as f:
                f.write(result)
            
            success = True
            if not verbose:
                print(f" Success in {generation_time:.2f}s")
            
        except Exception as e:
            if verbose:
                print(f"Failed: {str(e)}")
            else:
                print(f" Failed: {str(e)}")
            generation_time = 0
            audio_filename = "failed"
            success = False
        
        # Add to results
        row_data = [actual_length, tokens, success, generation_time, audio_filename]
        rows.append(row_data)
        
        result_data = {
            "length": actual_length,
            "tokens": tokens,
            "success": success,
            "time": generation_time,
            "audio_file": audio_filename if success else None
        }
        results.append(result_data)
        
        if metadata:
            metadata["results"]["text_length"].append(result_data)
        
        # If we've failed, no need to test larger sizes
        if not success:
            break
    
    # Save results to CSV
    csv_path = save_csv(length_dir, 'length_results.csv', headers, rows)
    print(f"Saved length test results to {csv_path}")
    
    # Determine maximum successful length
    max_successful_length = 0
    max_successful_tokens = 0
    for result in results:
        if result["success"] and result["length"] > max_successful_length:
            max_successful_length = result["length"]
            max_successful_tokens = result["tokens"]
    
    # Calculate recommended max length (90% of max successful for safety)
    recommended_length = int(max_successful_length * 0.9)
    
    # Add recommendation to metadata
    if metadata:
        metadata["recommended_settings"]["max_length"] = recommended_length
        metadata["recommended_settings"]["max_tokens"] = max_successful_tokens
    
    print(f"Maximum successful text length: {max_successful_length} characters ({max_successful_tokens} tokens)")
    print(f"Recommended maximum length for safety: {recommended_length} characters")
    
    return {
        "results": results,
        "max_successful_length": max_successful_length,
        "max_successful_tokens": max_successful_tokens,
        "recommended_length": recommended_length
    } 