"""
Sound Denoiser - Audio denoising tool for removing hiss from older recordings.

This package provides adaptive spectral denoising using librosa,
with controls for preserving audio fidelity and transients.
Includes noise profile learning similar to iZotope RX.
"""

__version__ = "0.1.0"

from .denoiser import AudioDenoiser, NoiseProfile
from .audio_player import AudioPlayer

__all__ = ["AudioDenoiser", "NoiseProfile", "AudioPlayer", "__version__"]
