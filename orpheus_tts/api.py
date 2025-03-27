"""
API module for interacting with LM Studio API
"""

import requests
import json
from .config import (
    API_URL,
    HEADERS,
    DEFAULT_VOICE,
    AVAILABLE_VOICES,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    REPETITION_PENALTY,
)


def format_prompt(prompt, voice=DEFAULT_VOICE):
    """
    Format a text prompt with the chosen voice for the Orpheus model.
    
    Args:
        prompt (str): The text to convert to speech
        voice (str): The voice to use
        
    Returns:
        str: Formatted prompt for the model
    """
    if voice not in AVAILABLE_VOICES:
        print(
            f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead."
        )
        voice = DEFAULT_VOICE
    formatted_prompt = f"{voice}: {prompt}"
    return f"<|audio|>{formatted_prompt}<|eot_id|>"


def estimate_tokens(text, voice=DEFAULT_VOICE):
    """
    Estimate the number of tokens in the text.
    This is a rough estimate based on whitespace tokenization.
    
    Args:
        text (str): The text to count tokens for
        voice (str): The voice to use
        
    Returns:
        int: Estimated token count
    """
    formatted_text = format_prompt(text, voice)
    # Simple whitespace-based tokenization for estimation
    return len(formatted_text.split())


def generate_tokens_from_api(
    prompt,
    voice=DEFAULT_VOICE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    repetition_penalty=REPETITION_PENALTY,
    verbose=False,
):
    """
    Generate tokens from the LM Studio API using the Orpheus model.
    
    Args:
        prompt (str): The text to convert to speech
        voice (str): The voice to use
        temperature (float): Temperature parameter for generation
        top_p (float): Top-p sampling parameter
        max_tokens (int): Maximum number of tokens to generate
        repetition_penalty (float): Repetition penalty
        verbose (bool): Whether to print debug information
        
    Yields:
        str: Generated tokens
    """
    formatted_prompt = format_prompt(prompt, voice)
    if verbose:
        print(f"[PROMPT]: {formatted_prompt}")

    payload = {
        "prompt": formatted_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stop": None,
        "stream": True,
        "repeat_penalty": repetition_penalty,
    }

    with requests.post(API_URL, headers=HEADERS, json=payload, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Error from llama-server: {response.status_code}, {response.text}"
            )

        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                json_data = line[len("data: ") :]
                if json_data.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(json_data)
                    token_text = data["choices"][0]["text"]
                    if verbose:
                        print(f"[TOKEN]: {token_text.strip()}")
                    yield token_text
                except Exception as e:
                    print(f"Error parsing streamed response: {e}") 