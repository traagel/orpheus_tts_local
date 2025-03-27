"""
Utility functions for the Orpheus TTS benchmarking system.
"""

import os
import time
import json
from datetime import datetime
import csv
from typing import Dict, Any, List, Tuple, Optional

from ..api import estimate_tokens


def initialize_metadata(timestamp: str, run_name: str, voice: str, 
                        python_version: str, platform: str, max_tokens: int,
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the metadata dictionary for a benchmark run.
    
    Args:
        timestamp: Timestamp for the benchmark run
        run_name: Name of the benchmark run
        voice: Voice used for the benchmark
        python_version: Python version used
        platform: Platform information
        max_tokens: Maximum token limit
        parameters: Benchmark parameters
        
    Returns:
        Initialized metadata dictionary
    """
    return {
        "timestamp": timestamp,
        "run_name": run_name,
        "voice": voice,
        "system_info": {
            "python_version": python_version,
            "platform": platform,
            "max_tokens": max_tokens
        },
        "parameters": parameters,
        "tests": {
            "text_length": {},
            "temperature": {},
            "top_p": {},
            "repetition_penalty": {}
        },
        "results": {
            "text_length": [],
            "temperature": [],
            "top_p": [],
            "repetition_penalty": []
        },
        "recommended_settings": {}
    }


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    return estimate_tokens(text)


def read_text_sample(input_file: str, length: int = None) -> Tuple[str, int]:
    """
    Read a sample of specified length from a text file.
    
    Args:
        input_file: Path to the input text file
        length: Maximum length of text to read (in characters)
        
    Returns:
        Tuple of (text_sample, actual_length)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get all text or just a portion based on length
    if length is not None and length < len(text):
        text = text[:length]
    
    return text, len(text)


def save_csv(output_dir: str, filename: str, headers: List[str], rows: List[List[Any]]) -> str:
    """
    Save data to a CSV file.
    
    Args:
        output_dir: Directory to save the file
        filename: Name of the CSV file
        headers: Column headers
        rows: Data rows
        
    Returns:
        Path to the saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    return csv_path


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (function_result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours" 