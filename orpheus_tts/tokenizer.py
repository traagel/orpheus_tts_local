"""
Tokenizer module for Orpheus TTS
"""

import asyncio
from .config import CUSTOM_TOKEN_PREFIX


def turn_token_into_id(token_string, index):
    """
    Convert token string to ID.
    
    Args:
        token_string (str): The token string
        index (int): The token index
        
    Returns:
        int or None: The token ID, or None if conversion fails
    """
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    if last_token_start == -1:
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    else:
        return None


async def tokens_decoder(token_gen, verbose=False):
    """
    Decode tokens into audio.
    
    Args:
        token_gen: Generator yielding tokens
        verbose (bool): Whether to print debug information
        
    Yields:
        numpy.ndarray: Audio samples
    """
    from .audio import convert_to_audio
    
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if verbose:
                print(f"[ID]: {token}")
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples 