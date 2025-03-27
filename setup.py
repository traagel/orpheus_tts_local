"""
Setup file for Orpheus TTS
"""

from setuptools import setup, find_packages

setup(
    name="orpheus_tts",
    version="0.1.0",
    description="A Python package for text-to-speech using Orpheus models",
    author="Orpheus Contributors",
    author_email="user@example.com",
    packages=find_packages(),
    install_requires=[
        "llama-cpp-python",
        "requests",
        "numpy",
        "sounddevice",
    ],
    entry_points={
        'console_scripts': [
            'orpheus-tts=orpheus_tts.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 