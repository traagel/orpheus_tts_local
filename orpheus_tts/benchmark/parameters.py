"""
Parameter benchmarking module for Orpheus TTS.

This module contains functions to benchmark different hyperparameters:
- Temperature
- Top-p (nucleus sampling)
- Repetition penalty
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple

from ..synthesizer import generate_audio
from .utils import read_text_sample, save_csv, count_tokens, time_function


def benchmark_temperature(
    input_file: str,
    voice: str,
    output_dir: str,
    text_length: int = 1000,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    verbose: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Benchmark different temperature values.
    
    Args:
        input_file: Path to the input text file
        voice: Voice to use for benchmarking
        output_dir: Directory to save benchmark results
        text_length: Length of text to use for testing
        top_p: Top-p parameter for generation
        repetition_penalty: Repetition penalty parameter for generation
        verbose: Whether to print verbose output
        metadata: Metadata dictionary to update with results
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Temperature ===")
    
    # Temperature values to test
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    
    # Store test parameters in metadata
    if metadata:
        metadata["tests"]["temperature"] = {
            "values": temperatures,
            "text_length": text_length,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        }
    
    # Read the text sample
    text, actual_length = read_text_sample(input_file, text_length)
    tokens = count_tokens(text)
    
    if verbose:
        print(f"Using text sample of {actual_length} characters ({tokens} tokens)")
    
    # Create output directory for temperature test
    temp_dir = os.path.join(output_dir, 'temperature_test')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Prepare result data for CSV
    headers = ['Temperature', 'Success', 'Time (s)', 'Audio File']
    rows = []
    results = []
    
    # Test each temperature
    for temp in temperatures:
        if verbose:
            print(f"\nTesting temperature: {temp}")
        else:
            print(f"Testing temperature: {temp}...", end='', flush=True)
        
        try:
            # Time the generation
            result, generation_time = time_function(
                generate_audio,
                text,
                voice=voice,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            # Save the audio file
            audio_filename = f"temp_{temp}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
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
        row_data = [temp, success, generation_time, audio_filename]
        rows.append(row_data)
        
        result_data = {
            "temperature": temp,
            "success": success,
            "time": generation_time,
            "audio_file": audio_filename if success else None
        }
        results.append(result_data)
        
        if metadata:
            metadata["results"]["temperature"].append(result_data)
    
    # Save results to CSV
    csv_path = save_csv(temp_dir, 'temperature_results.csv', headers, rows)
    print(f"Saved temperature test results to {csv_path}")
    
    # Find fastest successful temperature
    fastest_temp = None
    min_time = float('inf')
    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            fastest_temp = result["temperature"]
    
    # Add recommendation to metadata
    if metadata and fastest_temp is not None:
        metadata["recommended_settings"]["temperature"] = fastest_temp
    
    if fastest_temp is not None:
        print(f"Recommended temperature for speed: {fastest_temp}")
    else:
        print("No successful temperature found")
    
    return {
        "results": results,
        "recommended_temperature": fastest_temp
    }


def benchmark_top_p(
    input_file: str,
    voice: str,
    output_dir: str,
    text_length: int = 1000,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    verbose: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Benchmark different top-p values.
    
    Args:
        input_file: Path to the input text file
        voice: Voice to use for benchmarking
        output_dir: Directory to save benchmark results
        text_length: Length of text to use for testing
        temperature: Temperature parameter for generation
        repetition_penalty: Repetition penalty parameter for generation
        verbose: Whether to print verbose output
        metadata: Metadata dictionary to update with results
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Top-p ===")
    
    # Top-p values to test
    top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    
    # Store test parameters in metadata
    if metadata:
        metadata["tests"]["top_p"] = {
            "values": top_p_values,
            "text_length": text_length,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }
    
    # Read the text sample
    text, actual_length = read_text_sample(input_file, text_length)
    tokens = count_tokens(text)
    
    if verbose:
        print(f"Using text sample of {actual_length} characters ({tokens} tokens)")
    
    # Create output directory for top-p test
    top_p_dir = os.path.join(output_dir, 'top_p_test')
    os.makedirs(top_p_dir, exist_ok=True)
    
    # Prepare result data for CSV
    headers = ['Top-p', 'Success', 'Time (s)', 'Audio File']
    rows = []
    results = []
    
    # Test each top-p value
    for top_p_val in top_p_values:
        if verbose:
            print(f"\nTesting top-p: {top_p_val}")
        else:
            print(f"Testing top-p: {top_p_val}...", end='', flush=True)
        
        try:
            # Time the generation
            result, generation_time = time_function(
                generate_audio,
                text,
                voice=voice,
                temperature=temperature,
                top_p=top_p_val,
                repetition_penalty=repetition_penalty
            )
            
            # Save the audio file
            audio_filename = f"top_p_{top_p_val}.wav"
            audio_path = os.path.join(top_p_dir, audio_filename)
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
        row_data = [top_p_val, success, generation_time, audio_filename]
        rows.append(row_data)
        
        result_data = {
            "top_p": top_p_val,
            "success": success,
            "time": generation_time,
            "audio_file": audio_filename if success else None
        }
        results.append(result_data)
        
        if metadata:
            metadata["results"]["top_p"].append(result_data)
    
    # Save results to CSV
    csv_path = save_csv(top_p_dir, 'top_p_results.csv', headers, rows)
    print(f"Saved top-p test results to {csv_path}")
    
    # Find fastest successful top-p
    fastest_top_p = None
    min_time = float('inf')
    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            fastest_top_p = result["top_p"]
    
    # Add recommendation to metadata
    if metadata and fastest_top_p is not None:
        metadata["recommended_settings"]["top_p"] = fastest_top_p
    
    if fastest_top_p is not None:
        print(f"Recommended top-p for speed: {fastest_top_p}")
    else:
        print("No successful top-p found")
    
    return {
        "results": results,
        "recommended_top_p": fastest_top_p
    }


def benchmark_repetition_penalty(
    input_file: str,
    voice: str,
    output_dir: str,
    text_length: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Benchmark different repetition penalty values.
    
    Args:
        input_file: Path to the input text file
        voice: Voice to use for benchmarking
        output_dir: Directory to save benchmark results
        text_length: Length of text to use for testing
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        verbose: Whether to print verbose output
        metadata: Metadata dictionary to update with results
        
    Returns:
        Dictionary with benchmark results
    """
    print("\n=== Benchmarking Repetition Penalty ===")
    
    # Repetition penalty values to test
    rep_penalties = [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
    
    # Store test parameters in metadata
    if metadata:
        metadata["tests"]["repetition_penalty"] = {
            "values": rep_penalties,
            "text_length": text_length,
            "temperature": temperature,
            "top_p": top_p
        }
    
    # Read the text sample
    text, actual_length = read_text_sample(input_file, text_length)
    tokens = count_tokens(text)
    
    if verbose:
        print(f"Using text sample of {actual_length} characters ({tokens} tokens)")
    
    # Create output directory for repetition penalty test
    rep_dir = os.path.join(output_dir, 'repetition_penalty_test')
    os.makedirs(rep_dir, exist_ok=True)
    
    # Prepare result data for CSV
    headers = ['Repetition Penalty', 'Success', 'Time (s)', 'Audio File']
    rows = []
    results = []
    
    # Test each repetition penalty
    for rep_penalty in rep_penalties:
        if verbose:
            print(f"\nTesting repetition penalty: {rep_penalty}")
        else:
            print(f"Testing repetition penalty: {rep_penalty}...", end='', flush=True)
        
        try:
            # Time the generation
            result, generation_time = time_function(
                generate_audio,
                text,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty
            )
            
            # Save the audio file
            audio_filename = f"rep_penalty_{rep_penalty}.wav"
            audio_path = os.path.join(rep_dir, audio_filename)
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
        row_data = [rep_penalty, success, generation_time, audio_filename]
        rows.append(row_data)
        
        result_data = {
            "repetition_penalty": rep_penalty,
            "success": success,
            "time": generation_time,
            "audio_file": audio_filename if success else None
        }
        results.append(result_data)
        
        if metadata:
            metadata["results"]["repetition_penalty"].append(result_data)
    
    # Save results to CSV
    csv_path = save_csv(rep_dir, 'repetition_penalty_results.csv', headers, rows)
    print(f"Saved repetition penalty test results to {csv_path}")
    
    # Find fastest successful repetition penalty
    fastest_rep_penalty = None
    min_time = float('inf')
    for result in results:
        if result["success"] and result["time"] < min_time:
            min_time = result["time"]
            fastest_rep_penalty = result["repetition_penalty"]
    
    # Add recommendation to metadata
    if metadata and fastest_rep_penalty is not None:
        metadata["recommended_settings"]["repetition_penalty"] = fastest_rep_penalty
    
    if fastest_rep_penalty is not None:
        print(f"Recommended repetition penalty for speed: {fastest_rep_penalty}")
    else:
        print("No successful repetition penalty found")
    
    return {
        "results": results,
        "recommended_repetition_penalty": fastest_rep_penalty
    } 