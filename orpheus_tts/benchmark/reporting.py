"""
Reporting module for Orpheus TTS benchmarking.

This module contains functions for generating reports and saving metadata.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


def save_metadata_json(output_dir: str, metadata: Dict[str, Any]) -> str:
    """
    Save metadata to a JSON file.
    
    Args:
        output_dir: Directory to save the file
        metadata: Metadata dictionary
        
    Returns:
        Path to the saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    return metadata_path


def generate_report(output_dir: str, voice: str, metadata: Dict[str, Any]) -> str:
    """
    Generate a summary report of the benchmark results.
    
    Args:
        output_dir: Directory to save the report
        voice: Voice used for benchmarking
        metadata: Metadata dictionary
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'benchmark_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Report header
        f.write(f"==================================\n")
        f.write(f"ORPHEUS TTS BENCHMARK REPORT\n")
        f.write(f"==================================\n")
        f.write(f"Voice: {voice}\n")
        f.write(f"Date: {metadata.get('timestamp', 'Unknown')}\n")
        f.write(f"Run Name: {metadata.get('run_name', 'Unknown')}\n\n")
        
        # System information
        sys_info = metadata.get('system_info', {})
        f.write(f"SYSTEM INFORMATION\n")
        f.write(f"----------------------------------\n")
        f.write(f"Python Version: {sys_info.get('python_version', 'Unknown')}\n")
        f.write(f"Platform: {sys_info.get('platform', 'Unknown')}\n")
        f.write(f"Max Tokens: {sys_info.get('max_tokens', 'Unknown')}\n\n")
        
        # Input file details
        input_file_details = metadata.get('input_file_details', {})
        f.write(f"INPUT FILE\n")
        f.write(f"----------------------------------\n")
        f.write(f"Path: {input_file_details.get('path', 'Unknown')}\n")
        f.write(f"Total Characters: {input_file_details.get('total_chars', 'Unknown')}\n")
        f.write(f"Total Tokens: {input_file_details.get('total_tokens', 'Unknown')}\n\n")
        
        # Recommended settings
        recommended = metadata.get('recommended_settings', {})
        f.write(f"RECOMMENDED SETTINGS\n")
        f.write(f"----------------------------------\n")
        f.write(f"Maximum Text Length: {recommended.get('max_length', 'Unknown')} characters\n")
        f.write(f"Maximum Tokens: {recommended.get('max_tokens', 'Unknown')}\n")
        if 'temperature' in recommended:
            f.write(f"Temperature: {recommended.get('temperature')}\n")
        if 'top_p' in recommended:
            f.write(f"Top-p: {recommended.get('top_p')}\n")
        if 'repetition_penalty' in recommended:
            f.write(f"Repetition Penalty: {recommended.get('repetition_penalty')}\n")
        f.write("\n")
        
        # Text length test results
        f.write(f"TEXT LENGTH TEST RESULTS\n")
        f.write(f"----------------------------------\n")
        text_length_results = metadata.get('results', {}).get('text_length', [])
        if text_length_results:
            max_successful = 0
            max_tokens = 0
            for result in text_length_results:
                if result.get('success', False):
                    max_successful = max(max_successful, result.get('length', 0))
                    max_tokens = max(max_tokens, result.get('tokens', 0))
            
            f.write(f"Maximum Successful Length: {max_successful} characters\n")
            f.write(f"Maximum Successful Tokens: {max_tokens}\n")
            f.write(f"Results by Length:\n")
            for result in text_length_results:
                success_str = "✓" if result.get('success', False) else "✗"
                time_str = f"{result.get('time', 0):.2f}s" if result.get('success', False) else "N/A"
                f.write(f"  - {result.get('length', 0)} chars ({result.get('tokens', 0)} tokens): {success_str} {time_str}\n")
        else:
            f.write("No text length tests performed.\n")
        f.write("\n")
        
        # Temperature test results
        f.write(f"TEMPERATURE TEST RESULTS\n")
        f.write(f"----------------------------------\n")
        temp_results = metadata.get('results', {}).get('temperature', [])
        if temp_results:
            fastest_temp = None
            min_time = float('inf')
            for result in temp_results:
                if result.get('success', False) and result.get('time', float('inf')) < min_time:
                    min_time = result.get('time', float('inf'))
                    fastest_temp = result.get('temperature')
            
            if fastest_temp is not None:
                f.write(f"Fastest Successful Temperature: {fastest_temp} ({min_time:.2f}s)\n")
            f.write(f"Results by Temperature:\n")
            for result in temp_results:
                success_str = "✓" if result.get('success', False) else "✗"
                time_str = f"{result.get('time', 0):.2f}s" if result.get('success', False) else "N/A"
                f.write(f"  - Temperature {result.get('temperature', 0)}: {success_str} {time_str}\n")
        else:
            f.write("No temperature tests performed.\n")
        f.write("\n")
        
        # Top-p test results
        f.write(f"TOP-P TEST RESULTS\n")
        f.write(f"----------------------------------\n")
        top_p_results = metadata.get('results', {}).get('top_p', [])
        if top_p_results:
            fastest_top_p = None
            min_time = float('inf')
            for result in top_p_results:
                if result.get('success', False) and result.get('time', float('inf')) < min_time:
                    min_time = result.get('time', float('inf'))
                    fastest_top_p = result.get('top_p')
            
            if fastest_top_p is not None:
                f.write(f"Fastest Successful Top-p: {fastest_top_p} ({min_time:.2f}s)\n")
            f.write(f"Results by Top-p:\n")
            for result in top_p_results:
                success_str = "✓" if result.get('success', False) else "✗"
                time_str = f"{result.get('time', 0):.2f}s" if result.get('success', False) else "N/A"
                f.write(f"  - Top-p {result.get('top_p', 0)}: {success_str} {time_str}\n")
        else:
            f.write("No top-p tests performed.\n")
        f.write("\n")
        
        # Repetition penalty test results
        f.write(f"REPETITION PENALTY TEST RESULTS\n")
        f.write(f"----------------------------------\n")
        rep_penalty_results = metadata.get('results', {}).get('repetition_penalty', [])
        if rep_penalty_results:
            fastest_rep_penalty = None
            min_time = float('inf')
            for result in rep_penalty_results:
                if result.get('success', False) and result.get('time', float('inf')) < min_time:
                    min_time = result.get('time', float('inf'))
                    fastest_rep_penalty = result.get('repetition_penalty')
            
            if fastest_rep_penalty is not None:
                f.write(f"Fastest Successful Repetition Penalty: {fastest_rep_penalty} ({min_time:.2f}s)\n")
            f.write(f"Results by Repetition Penalty:\n")
            for result in rep_penalty_results:
                success_str = "✓" if result.get('success', False) else "✗"
                time_str = f"{result.get('time', 0):.2f}s" if result.get('success', False) else "N/A"
                f.write(f"  - Repetition Penalty {result.get('repetition_penalty', 0)}: {success_str} {time_str}\n")
        else:
            f.write("No repetition penalty tests performed.\n")
        f.write("\n")
        
        # Conclusion
        f.write(f"CONCLUSION\n")
        f.write(f"----------------------------------\n")
        f.write(f"For the {voice} voice, the recommended settings are:\n")
        f.write(f"- Maximum text length: {recommended.get('max_length', 'Unknown')} characters\n")
        f.write(f"- Maximum tokens: {recommended.get('max_tokens', 'Unknown')}\n")
        
        if 'temperature' in recommended:
            f.write(f"- Temperature: {recommended.get('temperature')}\n")
        if 'top_p' in recommended:
            f.write(f"- Top-p: {recommended.get('top_p')}\n")
        if 'repetition_penalty' in recommended:
            f.write(f"- Repetition Penalty: {recommended.get('repetition_penalty')}\n")
        
        f.write("\nDetailed results and audio samples are available in the benchmark directory.\n")
        f.write(f"Complete metadata is available in: metadata.json\n")
        
    return report_path 