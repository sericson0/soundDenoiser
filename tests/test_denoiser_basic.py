import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure local src/ is importable when running tests directly from the repo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sound_denoiser import AudioDenoiser


def _sine_with_noise(duration=0.5, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    noise = 0.02 * np.random.randn(len(t))
    return tone + noise, sr


def test_process_changes_audio(tmp_path):
    """Processing should modify the signal while keeping shape consistent."""
    audio, sr = _sine_with_noise()
    wav_path = tmp_path / "input.wav"
    sf.write(wav_path, audio, sr)

    denoiser = AudioDenoiser(max_db_reduction=12.0, blend_original=0.0, noise_reduction_strength=0.9)
    original, loaded_sr = denoiser.load_audio(str(wav_path))
    assert loaded_sr == sr

    processed = denoiser.process()

    assert processed.shape == original.shape
    assert not np.allclose(processed, original)


def test_auto_learn_noise_profile_runs(tmp_path):
    """Auto noise profiling should return a profile and region without raising."""
    # Use low-amplitude noise to allow detection
    audio = 0.01 * np.random.randn(22050)
    sr = 22050
    wav_path = tmp_path / "noise.wav"
    sf.write(wav_path, audio, sr)

    denoiser = AudioDenoiser()
    denoiser.load_audio(str(wav_path))

    profile, region = denoiser.auto_learn_noise_profile(min_duration=0.05)
    assert profile is not None
    assert isinstance(region, tuple)
    assert region[0] < region[1]


def test_package_imports():
    """Top-level package exports remain available for GUI imports."""
    from sound_denoiser import AudioPlayer, NoiseProfile, DenoiseMethod  # noqa: F401

    assert AudioDenoiser is not None
