"""
Audio processing module for Orpheus TTS
"""

import os
import wave
import threading
import queue
import asyncio
import io
from .config import SAMPLE_RATE


def convert_to_audio(multiframe, count):
    """
    Convert token frames to audio samples.
    
    Args:
        multiframe (list): List of token IDs
        count (int): Token count
        
    Returns:
        numpy.ndarray: Audio samples
    """
    from decoder import convert_to_audio as orpheus_convert_to_audio
    return orpheus_convert_to_audio(multiframe, count)


def tokens_decoder_sync(syn_token_gen, output_file=None, verbose=False):
    """
    Synchronous wrapper for the asynchronous token decoder.
    
    Args:
        syn_token_gen: Generator yielding tokens
        output_file (str, optional): Path to save WAV file
        verbose (bool): Whether to print debug information
        
    Returns:
        bytes or list: WAV audio data as bytes if no output_file, otherwise list of audio segments
    """
    from .tokenizer import tokens_decoder
    
    audio_queue = queue.Queue()
    audio_segments = []
    token_count = 0

    wav_file = None
    memory_file = None
    
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    else:
        # Create an in-memory file-like object for WAV data
        memory_file = io.BytesIO()
        wav_file = wave.open(memory_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

    async def async_token_gen():
        nonlocal token_count
        for token in syn_token_gen:
            token_count += 1
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen(), verbose=verbose):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        audio_segments.append(audio)
        if wav_file:
            wav_file.writeframes(audio)

    if wav_file:
        wav_file.close()

    thread.join()

    duration = sum([len(segment) // 2 for segment in audio_segments]) / SAMPLE_RATE

    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")
    print(f"Total tokens processed: {token_count}")

    # Return bytes if no output file was provided
    if memory_file:
        return memory_file.getvalue()
    
    return audio_segments 