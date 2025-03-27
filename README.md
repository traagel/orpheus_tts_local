# Orpheus TTS

A Python package for text-to-speech using Orpheus models.

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# Generate speech from text
orpheus-tts --text "Hello, this is a test of the Orpheus text-to-speech system."

# Use a specific voice
orpheus-tts --text "Hello, this is a test." --voice leo

# List available voices
orpheus-tts --list-voices

# Read text from a file
orpheus-tts --file input.txt

# Specify output file
orpheus-tts --text "Hello world" --output output.wav

# Adjust generation parameters
orpheus-tts --text "Hello world" --temperature 0.7 --top_p 0.95 --repetition_penalty 1.2
```

### Python API

```python
from orpheus_tts.synthesizer import generate_speech, list_available_voices

# List available voices
list_available_voices()

# Generate speech
audio_segments = generate_speech(
    prompt="Hello, this is a test of the Orpheus text-to-speech system.",
    voice="tara",
    output_file="output.wav",
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.1,
    verbose=False
)
```

## Available Voices

- tara (default)
- leah
- jess
- leo
- dan
- mia
- zac
- zoe

## Emotion Tags

You can add emotion tags to your text to add emotional expression:

- `<laugh>`
- `<chuckle>`
- `<sigh>`
- `<cough>`
- `<sniffle>`
- `<groan>`
- `<yawn>`
- `<gasp>`

Example: "I'm really excited about this project `<laugh>`. It's going to be great!"

## Benchmarking

The package includes a comprehensive benchmarking tool to optimize hyperparameters and determine limitations:

```bash
# Run all benchmarks with default settings
./orpheus-benchmark

# Test with a specific voice
./orpheus-benchmark --voice leo

# Name your benchmark run for easier reference
./orpheus-benchmark --run-name my_special_benchmark

# Benchmark only specific parameters
./orpheus-benchmark --skip-length --skip-top-p

# Control the testing range (default max-length is 3000, step is 250)
./orpheus-benchmark --max-length 2000 --length-step 200 --test-length 500

# Get verbose output
./orpheus-benchmark --verbose
```

The benchmark tool will test:
- Maximum text length before failure
- Optimal temperature values
- Optimal top_p values
- Optimal repetition_penalty values

Results are saved in a subfolder with your benchmark run name (or auto-generated name based on voice and timestamp). Each benchmark run includes:
- WAV audio samples for each test
- CSV files with detailed results for each parameter test
- A comprehensive summary report with recommended settings
- A detailed JSON metadata file with all test parameters and results

With this structure, you can easily compare different benchmark runs and find optimal settings for your specific use case.

### For Developers

The benchmark functionality has been refactored into a modular structure under the `orpheus_tts/benchmark/` directory, making it easier to maintain and extend:

- `__init__.py` - Main entry point and argument parsing
- `text_length.py` - Text length benchmarking functions
- `parameters.py` - Parameter optimization (temperature, top_p, repetition_penalty)
- `reporting.py` - Report generation and metadata handling
- `utils.py` - Common utilities for benchmarking

If you want to add new benchmark tests, you can create a new module in the benchmark package and integrate it with the main entry point.

## Requirements

- Python 3.7+
- llama-cpp-python
- requests
- numpy
- sounddevice
- An Orpheus model loaded in LM Studio (server running on port 8080)

## License

Apache 2.0

