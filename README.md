# 🎵 Sound Denoiser

A Python application for removing hiss and noise from vintage/older audio recordings using adaptive spectral denoising.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Noise Profile Learning** - iZotope RX-style spectral denoising from a learned noise sample
- **Auto-Detect Noise Regions** - Automatically finds the quietest sections for noise sampling
- **Manual Region Selection** - Click and drag on waveform to select noise-only sections
- **Adaptive Spectral Denoising** - Uses librosa for intelligent noise reduction
- **Preserve Audio Fidelity** - Configurable maximum dB reduction to avoid over-cleaning
- **Transient Protection** - Maintains punch and attack of drums and percussion
- **Original Signal Blending** - Mix back original character to preserve warmth
- **Real-time Preview** - Listen to before/after comparison before saving
- **Modern GUI** - Clean, dark-themed interface with waveform visualization
- **Non-destructive** - Always saves to a new file, never overwrites originals

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/soundDenoiser.git
cd soundDenoiser

# Install with UV
uv sync

# Run the application
uv run sound-denoiser
```

### Using pip

```bash
# Clone and navigate to the directory
git clone https://github.com/yourusername/soundDenoiser.git
cd soundDenoiser

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Run the application
sound-denoiser
```

## Usage

### GUI Application

1. **Launch the application:**
   ```bash
   uv run sound-denoiser
   # or
   sound-denoiser
   ```

2. **Load an audio file** by clicking "📂 Load Audio"

3. **Learn a noise profile** (recommended for best results):
   - **Auto Detect**: Click "🔍 Auto Detect" to automatically find a quiet section
   - **Manual Selection**: Click and drag on the original waveform to select a region with only noise (no music)
   - Click "📝 Learn from Selection" to learn the noise profile
   - Toggle "Use learned profile" to switch between learned and adaptive modes

4. **Adjust parameters** as needed:
   - **Max dB Reduction** (default: 4dB) - Limits how much noise can be removed
   - **Blend Original** (default: 12%) - Mix back original signal for warmth
   - **Noise Reduction Strength** - Overall intensity of noise removal
   - **Transient Protection** - Preserve attack/punch of drums
   - **High Frequency Rolloff** - More aggressive reduction at high frequencies
   - **Temporal Smoothing** - Reduce musical noise artifacts

5. **Preview** the processed audio using the play controls

6. **Save** when satisfied with the result

### Programmatic Usage

```python
from sound_denoiser import AudioDenoiser

# Initialize with custom parameters
denoiser = AudioDenoiser(
    max_db_reduction=4.0,      # Maximum 4dB of noise reduction
    blend_original=0.12,       # Blend 12% of original signal
    noise_reduction_strength=0.7,
    transient_protection=0.5,
)

# Load audio
audio, sr = denoiser.load_audio("path/to/noisy_audio.wav")

# Option 1: Auto-detect noise region and learn profile
profile, (start, end) = denoiser.auto_learn_noise_profile(min_duration=0.5)
print(f"Noise region detected: {start:.2f}s - {end:.2f}s")

# Option 2: Manually specify a noise region (e.g., first 0.5 seconds)
# profile = denoiser.learn_noise_profile(start_time=0.0, end_time=0.5)

# Process with learned profile
processed = denoiser.process()

# Save the result
denoiser.save("path/to/cleaned_audio.wav")
```

### Noise Profile Learning

The noise profile learning feature works similar to iZotope RX:

```python
# Learn from a specific region known to contain only noise
denoiser.learn_noise_profile(start_time=0.0, end_time=0.5)

# Or auto-detect the quietest region
profile, region = denoiser.auto_learn_noise_profile()

# Toggle between learned profile and adaptive estimation
denoiser.set_use_learned_profile(True)   # Use learned profile
denoiser.set_use_learned_profile(False)  # Use adaptive estimation

# Clear learned profile
denoiser.clear_noise_profile()
```

## Parameters Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_db_reduction` | 8.0 dB | Maximum amount of noise reduction. Lower = gentler, more natural |
| `blend_original` | 8% | Amount of original signal mixed back in. Higher = more original character |
| `noise_reduction_strength` | 75% | Overall strength of the noise reduction algorithm |
| `transient_protection` | 40% | How much to protect transients (drums, attacks) from reduction |
| `high_freq_rolloff` | 60% | Apply more aggressive reduction at high frequencies where hiss lives |
| `smoothing_factor` | 15% | Temporal smoothing to reduce musical noise artifacts |

## Algorithm

The denoiser uses an iZotope RX-inspired spectral subtraction approach:

### With Learned Noise Profile (Recommended)
1. **Noise Profile Learning** - Analyzes a noise-only region to create spectral fingerprint
2. **Wiener Filtering** - Uses the learned profile for optimal noise estimation
3. **Over-subtraction** - Applies configurable over-subtraction for cleaner results
4. **Soft Thresholding** - Smooth transitions based on signal-to-noise ratio
5. **Transient Protection** - Preserves attack transients during reduction
6. **Maximum Reduction Limiting** - Ensures noise reduction never exceeds the specified maximum
7. **Original Blending** - Mixes back original signal to preserve character

### Without Noise Profile (Adaptive Mode)
1. **Noise Floor Estimation** - Uses percentile-based analysis to estimate the noise floor
2. **Transient Detection** - Identifies transient regions to protect from over-processing
3. **Spectral Gating** - Creates a soft mask based on SNR with smooth transitions
4. **Maximum Reduction Limiting** - Ensures noise reduction never exceeds the specified maximum
5. **Original Blending** - Mixes back original signal to preserve character

## Supported Formats

- WAV (recommended for output)
- MP3
- FLAC
- AIFF
- M4A/AAC
- OGG

## Example Files

The `example_tracks/` folder contains sample vintage recordings for testing:
- Various formats (FLAC, AIF, M4A)
- Different types of hiss and noise
- Multiple eras of recordings (1930s-1940s)

## Requirements

- Python 3.10+
- librosa
- numpy
- scipy
- soundfile
- customtkinter
- matplotlib
- sounddevice

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
