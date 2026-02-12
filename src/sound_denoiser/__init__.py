"""
Sound Denoiser - Audio denoising tool for removing hiss from older recordings.

This package provides adaptive spectral denoising using librosa,
with controls for preserving audio fidelity and transients.
Includes noise profile learning similar to iZotope RX.
Multiple denoising methods are available:
- spectral gating (best with learned noise profile)
- spectral subtraction
- adaptive blend of subtraction and gating
"""

__version__ = "0.1.0"

from .denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
from .audio_player import AudioPlayer
from .waveform_display import WaveformDisplay
from .ui_components import SeekBar, ParameterSlider, NoiseProfilePanel

__all__ = [
	"AudioDenoiser",
	"NoiseProfile",
	"DenoiseMethod",
	"AudioPlayer",
	"WaveformDisplay",
	"SeekBar",
	"ParameterSlider",
	"NoiseProfilePanel",
	"__version__",
]
