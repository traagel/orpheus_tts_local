"""
Benchmark Runner for Orpheus TTS

This module serves as an entry point to the modular benchmark package.
It provides backward compatibility and a simpler interface for the
comprehensive benchmarking functionality in the benchmark package.
"""

from .benchmark import main as benchmark_main

def main():
    """
    Main function that serves as the entry point to the benchmark package.
    
    This function is called by the orpheus-benchmark script and passes
    control to the modular benchmark package.
    
    Returns:
        The return value from the benchmark package's main function
    """
    print("Orpheus TTS Benchmark Runner")
    print("Running benchmarks using the modular benchmark system...")
    return benchmark_main()


if __name__ == "__main__":
    main() 