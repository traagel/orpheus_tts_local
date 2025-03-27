"""
Configuration constants for Orpheus TTS
"""

# LM Studio API settings
API_URL = "http://127.0.0.1:8080/v1/completions"
HEADERS = {"Content-Type": "application/json"}

# Model parameters
MAX_TOKENS = 4096 * 5
TEMPERATURE = 0.9
TOP_P = 1.0
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Text chunking for stable generation
MAX_CHUNK_SIZE = 750  # Maximum characters per chunk for reliable generation

