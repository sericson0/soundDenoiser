"""
Sound Denoiser - Audio denoising tool for removing hiss from older recordings.

This package provides adaptive spectral denoising using librosa,
with controls for preserving audio fidelity and transients.
"""

__version__ = "0.1.0"

from .denoiser import AudioDenoiser
from .audio_player import AudioPlayer

__all__ = ["AudioDenoiser", "AudioPlayer", "__version__"]
