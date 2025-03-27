"""
Main synthesizer module for Orpheus TTS
"""

import time
import re
import io
import wave
from .config import (
    DEFAULT_VOICE,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPETITION_PENALTY,
    AVAILABLE_VOICES,
    MAX_CHUNK_SIZE,
    SAMPLE_RATE,
)
from .api import generate_tokens_from_api
from .audio import tokens_decoder_sync


def split_text_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE):
    """
    Split text into manageable chunks for processing.
    
    The function splits text at sentence boundaries (., !, ?) to ensure
    natural-sounding speech when processed in chunks. It attempts to keep
    chunks around the specified max_chunk_size while respecting sentence
    boundaries.
    
    Args:
        text (str): Text to split into chunks
        max_chunk_size (int): Target maximum size for each chunk
        
    Returns:
        list: List of text chunks
    """
    # If text is already small enough, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    # Pattern for sentence endings: ., !, ? followed by space or end of string
    sentence_pattern = r'(?<=[.!?])\s+'
    
    # Split by sentence boundaries
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is longer than max_chunk_size, we need to split it
        if len(sentence) > max_chunk_size:
            # If there's content in the current chunk, add it first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Split long sentence by comma or natural phrase breaks
            comma_splits = re.split(r'(?<=,|\;)\s+', sentence)
            
            # If comma splitting doesn't work well, fall back to word boundaries
            if max(len(s) for s in comma_splits) > max_chunk_size:
                # Split by words while respecting the max_chunk_size
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                        if temp_chunk:
                            temp_chunk += " " + word
                        else:
                            temp_chunk = word
                    else:
                        chunks.append(temp_chunk)
                        temp_chunk = word
                
                if temp_chunk:
                    chunks.append(temp_chunk)
            else:
                # Use comma-based splitting
                temp_chunk = ""
                for split in comma_splits:
                    if len(temp_chunk) + len(split) + 2 <= max_chunk_size:  # +2 for comma and space
                        if temp_chunk:
                            temp_chunk += ", " + split
                        else:
                            temp_chunk = split
                    else:
                        chunks.append(temp_chunk)
                        temp_chunk = split
                
                if temp_chunk:
                    chunks.append(temp_chunk)
        else:
            # Check if adding this sentence exceeds the chunk size
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:  # +1 for space
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Add the current chunk to chunks and start a new one
                chunks.append(current_chunk)
                current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def combine_audio_segments(segments, output_file=None):
    """
    Combine multiple audio segments into a single WAV file or byte stream.
    
    Args:
        segments (list): List of audio segment bytes
        output_file (str, optional): Path to save the combined WAV file
        
    Returns:
        bytes: Combined WAV audio data if no output_file is provided
    """
    all_audio = b''
    
    # If we need to write to a file
    if output_file:
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        
        for segment in segments:
            if isinstance(segment, list):
                # If segment is itself a list of audio chunks
                for chunk in segment:
                    wav_file.writeframes(chunk)
            else:
                wav_file.writeframes(segment)
        
        wav_file.close()
        return None
    else:
        # Create an in-memory WAV file
        memory_file = io.BytesIO()
        wav_file = wave.open(memory_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        
        for segment in segments:
            if isinstance(segment, list):
                # If segment is itself a list of audio chunks
                for chunk in segment:
                    wav_file.writeframes(chunk)
            else:
                wav_file.writeframes(segment)
        
        wav_file.close()
        return memory_file.getvalue()


def generate_speech(
    prompt,
    voice=DEFAULT_VOICE,
    output_file=None,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    repetition_penalty=REPETITION_PENALTY,
    verbose=False,
):
    """
    Generate speech from text.
    
    This function automatically splits long text into chunks for reliable
    generation, processes each chunk separately, and combines the results.
    
    Args:
        prompt (str): The text to convert to speech
        voice (str): The voice to use
        output_file (str, optional): Path to save WAV file
        temperature (float): Temperature parameter for generation
        top_p (float): Top-p sampling parameter
        max_tokens (int): Maximum number of tokens to generate
        repetition_penalty (float): Repetition penalty
        verbose (bool): Whether to print debug information
        
    Returns:
        list or bytes: List of audio segments if output_file is provided, otherwise WAV audio data as bytes
    """
    # Split the prompt into chunks if it's too long
    chunks = split_text_into_chunks(prompt)
    
    if len(chunks) == 1:
        # If there's only one chunk, process it directly
        return tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                verbose=verbose,
            ),
            output_file=output_file,
            verbose=verbose,
        )
    else:
        # Process each chunk and collect the audio
        if verbose:
            print(f"Text split into {len(chunks)} chunks for processing")
        
        all_segments = []
        
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            
            # Process this chunk (don't save to file yet)
            chunk_audio = tokens_decoder_sync(
                generate_tokens_from_api(
                    prompt=chunk,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    verbose=verbose,
                ),
                output_file=None,  # Don't save individual chunks
                verbose=verbose,
            )
            
            all_segments.append(chunk_audio)
        
        # Combine all audio segments
        if output_file:
            combine_audio_segments(all_segments, output_file)
            return all_segments
        else:
            return combine_audio_segments(all_segments)


def generate_audio(
    text,
    voice=DEFAULT_VOICE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    max_tokens=MAX_TOKENS,
    verbose=False
):
    """
    Generate audio for text using the specified voice and parameters.
    This is a simplified function for use in benchmarking.
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use
        temperature (float): Temperature parameter for generation
        top_p (float): Top-p sampling parameter
        repetition_penalty (float): Repetition penalty
        max_tokens (int): Maximum number of tokens to generate
        verbose (bool): Whether to print verbose information
        
    Returns:
        bytes: WAV audio data
    """
    return generate_speech(
        prompt=text, 
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        output_file=None,  # Return the bytes instead of saving to a file
        verbose=verbose
    )


def list_available_voices():
    """
    Print the list of available voices and emotion tags.
    """
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>") 