"""
Core audio denoising module using multiple techniques for hiss removal.

Implements:
- Spectral gating with learned noise profiles
- Spectral subtraction for broadband noise
- Multi-band adaptive noise reduction
- Wiener filtering for optimal noise estimation
- High-frequency focused processing where hiss typically lives
- Noise profile learning for targeted removal
- Perceptual weighting to preserve audio quality
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from typing import Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Metadata handling
try:
    from mutagen import File as MutagenFile
    from mutagen.flac import FLAC, Picture
    from mutagen.oggvorbis import OggVorbis
    from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, TDRC, TCON, TRCK, COMM
    from mutagen.wave import WAVE
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


class DenoiseMethod(Enum):
    """Available denoising methods."""
    SPECTRAL_SUBTRACTION = "spectral"    # Classic spectral subtraction (good for shellac/vinyl)
    WIENER = "wiener"                    # Wiener filtering (good for broadband noise)
    MULTIBAND = "multiband"              # Multi-band adaptive reduction
    COMBINED = "combined"                # Combine multiple methods
    SHELLAC = "shellac"                  # Optimized for 78rpm shellac records (hiss + groove)
    SPECTRAL_GATING = "gating"           # Pure spectral gating using learned noise profile


@dataclass
class NoiseProfile:
    """Learned noise profile from a sample region."""
    # Raw noise audio sample
    noise_clip: np.ndarray
    # Mean spectral magnitude of noise
    spectral_mean: np.ndarray
    # Standard deviation of noise spectrum
    spectral_std: np.ndarray
    # Sample rate used for learning
    sample_rate: int
    # Duration of noise sample in seconds
    duration: float
    # Start and end time of the sample region
    start_time: float
    end_time: float


class AudioDenoiser:
    """
    Multi-technique denoiser for removing hiss from audio recordings.

    Features:
    - Multiple denoising algorithms (spectral gating, spectral subtraction, Wiener, multiband)
    - Spectral gating with learned noise profiles
    - High-frequency emphasis for hiss targeting (2kHz-20kHz)
    - Transient preservation
    - Original signal blending
    - Perceptual weighting to preserve audio quality
    """

    # STFT parameters
    N_FFT = 2048
    HOP_LENGTH = 512

    # Frequency bands for multi-band processing (Hz)
    BAND_EDGES = [0, 300, 1000, 3000, 8000, 20000]

    # Shellac-specific frequency bands (Hz) - optimized for 78rpm records
    SHELLAC_BANDS = [0, 100, 300, 800, 2000, 4000, 8000, 16000]

    def __init__(
        self,
        max_db_reduction: float = 8.0,
        blend_original: float = 0.08,
        noise_reduction_strength: float = 0.85,
        transient_protection: float = 0.3,
        high_freq_emphasis: float = 1.5,
        method: DenoiseMethod = DenoiseMethod.SPECTRAL_SUBTRACTION,
        # New fine-tuning parameters
        hiss_start_freq: float = 3000.0,
        hiss_peak_freq: float = 8000.0,
        spectral_floor: float = 0.05,
        noise_threshold: float = 1.2,
    ):
        """
        Initialize the denoiser with parameters.

        Args:
            max_db_reduction: Maximum dB of noise reduction (default: 12.0)
            blend_original: Amount of original signal to blend back (0-1, default: 0.05)
            noise_reduction_strength: Overall strength of noise reduction (0-1, default: 0.85)
            transient_protection: How much to protect transients (0-1, default: 0.3)
            high_freq_emphasis: Extra reduction for high frequencies where hiss lives (default: 1.5)
            method: Denoising method to use (default: SPECTRAL_SUBTRACTION)
            hiss_start_freq: Frequency where hiss reduction begins (Hz, default: 2000)
            hiss_peak_freq: Frequency where hiss reduction is maximum (Hz, default: 6000)
            spectral_floor: Minimum signal to retain, prevents artifacts (0-1, default: 0.05)
            noise_threshold: Multiplier for noise estimate boundary (0.5-3.0, default: 1.0)
                Higher values = more aggressive (treats more as noise)
                Lower values = more conservative (preserves more signal)
        """
        self.max_db_reduction = max_db_reduction
        self.blend_original = blend_original
        self.noise_reduction_strength = noise_reduction_strength
        self.transient_protection = transient_protection
        self.high_freq_emphasis = high_freq_emphasis
        self.method = method

        # Fine-tuning parameters
        self.hiss_start_freq = hiss_start_freq
        self.hiss_peak_freq = hiss_peak_freq
        self.spectral_floor = spectral_floor
        self.noise_threshold = noise_threshold

        # Internal state
        self._audio: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._processed: Optional[np.ndarray] = None
        self._file_path: Optional[Path] = None

        # Noise profile (learned from sample region)
        self._noise_profile: Optional[NoiseProfile] = None
        self._use_learned_profile: bool = False

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa."""
        self._file_path = Path(file_path)
        self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)

        if self._audio.ndim == 1:
            self._audio = self._audio.reshape(1, -1)

        self._noise_profile = None
        self._use_learned_profile = False

        return self._audio, self._sr

    def get_duration(self) -> float:
        """Get duration of loaded audio in seconds."""
        if self._audio is None or self._sr is None:
            return 0.0
        return self._audio.shape[1] / self._sr

    def learn_noise_profile(self, start_time: float, end_time: float) -> NoiseProfile:
        """
        Learn noise profile from a specified region of the audio.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            NoiseProfile containing learned noise characteristics
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded. Call load_audio() first.")

        start_sample = int(start_time * self._sr)
        end_sample = int(end_time * self._sr)
        start_sample = max(0, start_sample)
        end_sample = min(self._audio.shape[1], end_sample)

        if end_sample <= start_sample:
            raise ValueError("Invalid time range for noise profile.")

        # Extract noise region (use first channel for profile)
        noise_clip = self._audio[0, start_sample:end_sample].copy()

        # Compute spectral statistics
        stft = librosa.stft(noise_clip, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)

        spectral_mean = np.mean(stft_mag, axis=1)
        spectral_std = np.std(stft_mag, axis=1)

        self._noise_profile = NoiseProfile(
            noise_clip=noise_clip,
            spectral_mean=spectral_mean,
            spectral_std=spectral_std,
            sample_rate=self._sr,
            duration=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
        )

        self._use_learned_profile = True
        return self._noise_profile

    def learn_noise_profile_from_regions(self, regions: List[Tuple[float, float]]) -> NoiseProfile:
        """
        Learn noise profile from multiple regions of the audio.

        Combines spectral statistics from all regions for a more robust noise estimate.

        Args:
            regions: List of (start_time, end_time) tuples in seconds

        Returns:
            NoiseProfile containing learned noise characteristics
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded. Call load_audio() first.")

        if not regions:
            raise ValueError("No regions provided.")

        all_noise_clips = []
        all_stft_mags = []
        total_duration = 0.0

        for start_time, end_time in regions:
            start_sample = int(start_time * self._sr)
            end_sample = int(end_time * self._sr)
            start_sample = max(0, start_sample)
            end_sample = min(self._audio.shape[1], end_sample)

            if end_sample <= start_sample:
                continue

            # Extract noise region (use first channel for profile)
            noise_clip = self._audio[0, start_sample:end_sample].copy()
            all_noise_clips.append(noise_clip)

            # Compute STFT for this region
            stft = librosa.stft(noise_clip, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
            stft_mag = np.abs(stft)
            all_stft_mags.append(stft_mag)

            total_duration += end_time - start_time

        if not all_stft_mags:
            raise ValueError("No valid audio data in the selected regions.")

        # Combine all STFT magnitudes horizontally (along time axis)
        combined_stft_mag = np.concatenate(all_stft_mags, axis=1)

        # Compute combined spectral statistics
        spectral_mean = np.mean(combined_stft_mag, axis=1)
        spectral_std = np.std(combined_stft_mag, axis=1)

        # Combine noise clips for the profile
        combined_noise_clip = np.concatenate(all_noise_clips)

        # Use first region's times for reference
        first_start = regions[0][0]
        last_end = regions[-1][1]

        self._noise_profile = NoiseProfile(
            noise_clip=combined_noise_clip,
            spectral_mean=spectral_mean,
            spectral_std=spectral_std,
            sample_rate=self._sr,
            duration=total_duration,
            start_time=first_start,
            end_time=last_end,
        )

        self._use_learned_profile = True
        return self._noise_profile

    def auto_detect_noise_region(self, min_duration: float = 0.5) -> Tuple[float, float]:
        """
        Automatically detect a region with noise but without music or silence.

        Uses spectral flatness to distinguish between:
        - Noise (high spectral flatness, moderate energy)
        - Music (low spectral flatness, varying energy)
        - Silence (very low energy)

        This is particularly effective for finding hiss/groove noise sections
        in old shellac recordings.
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded.")

        audio = self._audio[0]
        window_samples = int(min_duration * self._sr)
        hop_samples = window_samples // 4

        candidates = []

        for start in range(0, len(audio) - window_samples, hop_samples):
            end = start + window_samples
            window = audio[start:end]

            # Skip if too quiet (silence)
            rms = np.sqrt(np.mean(window ** 2))
            peak = np.max(np.abs(window))

            if peak < 0.002:  # Too quiet - likely silence
                continue

            if rms < 0.0005:  # Very low RMS - likely digital silence
                continue

            # Compute spectral flatness (noise is spectrally flat, music is not)
            stft = librosa.stft(window, n_fft=1024, hop_length=256)
            mag = np.abs(stft)

            # Geometric mean / arithmetic mean = spectral flatness
            # Higher values indicate more noise-like signal
            geometric_mean = np.exp(np.mean(np.log(mag + 1e-10), axis=0))
            arithmetic_mean = np.mean(mag, axis=0)
            flatness = np.mean(geometric_mean / (arithmetic_mean + 1e-10))

            # Check for low spectral variation (noise is consistent)
            spectral_var = np.var(np.mean(mag, axis=0))

            # Good noise regions have:
            # - High spectral flatness (noise-like)
            # - Low spectral variation over time (consistent)
            # - Moderate energy (not silence)

            # Score combining these factors
            # Higher score = more likely to be pure noise
            score = flatness * 100 - spectral_var * 1000 - rms * 10

            # Penalize very low or high energy
            if rms < 0.005 or rms > 0.1:
                score -= 50

            candidates.append((score, start, end, rms, flatness))

        if not candidates:
            # Fallback: use quietest region that's not silence
            return self._fallback_noise_detection(min_duration)

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return the best candidate
        _, start_sample, end_sample, _, _ = candidates[0]
        return (start_sample / self._sr, end_sample / self._sr)

    def _fallback_noise_detection(self, min_duration: float) -> Tuple[float, float]:
        """Fallback noise detection using simple RMS-based approach."""
        audio = self._audio[0]
        window_samples = int(min_duration * self._sr)
        hop_samples = window_samples // 4

        rms_values = []
        positions = []

        for start in range(0, len(audio) - window_samples, hop_samples):
            end = start + window_samples
            window = audio[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            peak = np.max(np.abs(window))

            # Skip silence
            if peak < 0.001:
                continue

            rms_values.append(rms)
            positions.append(start)

        if not rms_values:
            # Absolute fallback - use the beginning
            return (0.0, min_duration)

        rms_values = np.array(rms_values)
        positions = np.array(positions)

        # Find the quietest non-silent region
        sorted_indices = np.argsort(rms_values)
        idx = sorted_indices[0]
        start_sample = positions[idx]

        return (start_sample / self._sr, (start_sample + window_samples) / self._sr)

    def auto_learn_noise_profile(self, min_duration: float = 0.5) -> Tuple[NoiseProfile, Tuple[float, float]]:
        """Auto-detect quiet region and learn noise profile."""
        start_time, end_time = self.auto_detect_noise_region(min_duration)
        profile = self.learn_noise_profile(start_time, end_time)
        return profile, (start_time, end_time)

    def get_noise_profile(self) -> Optional[NoiseProfile]:
        return self._noise_profile

    def clear_noise_profile(self):
        self._noise_profile = None
        self._use_learned_profile = False

    def set_use_learned_profile(self, use: bool):
        self._use_learned_profile = use and self._noise_profile is not None

    def _apply_high_frequency_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply additional reduction to high frequencies where hiss typically lives.
        Uses configurable hiss_start_freq and hiss_peak_freq parameters.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Create frequency-dependent gain
        freq_bins = stft_mag.shape[0]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.N_FFT)

        # Use configurable hiss frequency range
        hiss_start = self.hiss_start_freq
        hiss_peak = self.hiss_peak_freq

        # Create smooth reduction curve
        gain = np.ones(freq_bins)
        for i, freq in enumerate(freqs):
            if freq > hiss_start:
                # Gradually increase reduction for higher frequencies
                reduction_amount = min(1.0, (freq - hiss_start) / (hiss_peak - hiss_start + 1))
                reduction_amount *= self.high_freq_emphasis - 1.0
                # Use spectral_floor as minimum gain to prevent artifacts
                min_gain = max(0.1, self.spectral_floor)
                gain[i] = max(min_gain, 1.0 - reduction_amount * self.noise_reduction_strength)

        # Apply gain
        stft_reduced = stft_mag * gain[:, np.newaxis] * np.exp(1j * stft_phase)

        return librosa.istft(stft_reduced, hop_length=self.HOP_LENGTH, length=len(audio))

    def _detect_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect transients for protection."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_env = onset_env / (np.max(onset_env) + 1e-8)
        transient_mask = np.clip(onset_env * 2, 0, 1)
        return median_filter(transient_mask, size=3)

    def _estimate_noise_spectrum(self, stft_mag: np.ndarray) -> np.ndarray:
        """
        Estimate noise spectrum using minimum statistics.

        Uses the assumption that the minimum energy in each frequency bin
        over time represents the noise floor.
        """
        # Use minimum statistics with smoothing
        # Take the 10th percentile across time as noise estimate
        noise_estimate = np.percentile(stft_mag, 10, axis=1)

        # Smooth the estimate
        noise_estimate = uniform_filter1d(noise_estimate, size=5)

        return noise_estimate

    def _spectral_gating(self, audio: np.ndarray) -> np.ndarray:
        """
        Pure spectral gating using the learned noise profile.

        This method requires a learned noise profile. It applies a soft gate
        that attenuates frequency bins where the signal is below or near the
        learned noise floor. More aggressive than spectral subtraction but
        very effective when you have a good noise sample.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get noise estimate - prefer learned profile
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
            noise_std = self._noise_profile.spectral_std
        else:
            # Fall back to automatic estimation if no profile learned
            noise_estimate = self._estimate_noise_spectrum(stft_mag)
            noise_std = noise_estimate * 0.5  # Approximate std

        # Apply noise threshold multiplier
        gate_threshold = noise_estimate * self.noise_threshold

        # Expand to match STFT shape
        gate_threshold_2d = gate_threshold[:, np.newaxis]
        noise_std_2d = noise_std[:, np.newaxis] if noise_std is not None else gate_threshold_2d * 0.3

        # Soft gating: compute gain based on how far above threshold the signal is
        # Gain = 1 when signal >> threshold, Gain -> reduction_amount when signal <= threshold
        margin = stft_mag - gate_threshold_2d

        # Sigmoid-like soft gate centered around the threshold
        # Use noise_std to control the transition width
        transition_width = np.maximum(noise_std_2d * 2, 1e-10)
        gate_gain = 1.0 / (1.0 + np.exp(-margin / transition_width * 4))

        # Apply reduction strength - controls how much attenuation below threshold
        min_gain = 1.0 - self.noise_reduction_strength
        min_gain = max(min_gain, self.spectral_floor)
        gate_gain = min_gain + gate_gain * (1.0 - min_gain)

        # Smooth gain temporally to reduce musical noise
        gate_gain = median_filter(gate_gain, size=(1, 3))

        # Apply gate
        stft_gated = stft_mag * gate_gain * np.exp(1j * stft_phase)

        return librosa.istft(stft_gated, hop_length=self.HOP_LENGTH, length=len(audio))

    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Classic spectral subtraction for noise removal.

        Subtracts estimated noise spectrum from the signal spectrum.
        Uses over-subtraction factor and spectral floor to reduce musical noise.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get noise estimate
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
        else:
            noise_estimate = self._estimate_noise_spectrum(stft_mag)

        # Apply noise threshold - multiplies the noise estimate to define the boundary
        # Higher threshold = more aggressive (treats more as noise)
        # Lower threshold = more conservative (preserves more signal)
        noise_estimate = noise_estimate * self.noise_threshold

        # Over-subtraction factor (reduces musical noise)
        alpha = 1.0 + self.noise_reduction_strength * 2.0

        # Spectral floor (prevents negative values and musical noise artifacts)
        # Uses the configurable spectral_floor parameter
        beta = self.spectral_floor + (1 - self.noise_reduction_strength) * 0.05

        # Perform spectral subtraction
        noise_estimate_2d = noise_estimate[:, np.newaxis]
        subtracted = stft_mag ** 2 - alpha * (noise_estimate_2d ** 2)

        # Apply spectral floor
        spectral_floor = beta * (noise_estimate_2d ** 2)
        subtracted = np.maximum(subtracted, spectral_floor)

        # Take square root to get magnitude
        subtracted_mag = np.sqrt(subtracted)

        # Reconstruct with original phase
        stft_clean = subtracted_mag * np.exp(1j * stft_phase)

        return librosa.istft(stft_clean, hop_length=self.HOP_LENGTH, length=len(audio))

    def _wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Wiener filtering for optimal noise reduction.

        Estimates the optimal filter that minimizes mean square error
        between the clean signal and the filtered noisy signal.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get noise estimate
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
        else:
            noise_estimate = self._estimate_noise_spectrum(stft_mag)

        # Apply noise threshold - multiplies the noise estimate to define the boundary
        noise_estimate = noise_estimate * self.noise_threshold
        noise_power = noise_estimate ** 2

        # Signal power estimate
        signal_power = stft_mag ** 2

        # Wiener filter gain (with regularization)
        noise_power_2d = noise_power[:, np.newaxis]
        snr_prior = np.maximum(signal_power - noise_power_2d, 0) / (noise_power_2d + 1e-10)

        # Apply strength parameter
        snr_prior = snr_prior * self.noise_reduction_strength

        # Wiener gain
        wiener_gain = snr_prior / (snr_prior + 1)

        # Smooth gain to reduce musical noise
        wiener_gain = median_filter(wiener_gain, size=(3, 3))

        # Apply gain
        stft_clean = stft_mag * wiener_gain * np.exp(1j * stft_phase)

        return librosa.istft(stft_clean, hop_length=self.HOP_LENGTH, length=len(audio))

    def _multiband_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Multi-band adaptive noise reduction.

        Splits the audio into frequency bands and applies different
        reduction levels based on the noise characteristics in each band.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT)

        # Get noise estimate per band
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
        else:
            noise_estimate = self._estimate_noise_spectrum(stft_mag)

        # Apply noise threshold - multiplies the noise estimate to define the boundary
        noise_estimate = noise_estimate * self.noise_threshold

        # Create gain matrix
        gain = np.ones_like(stft_mag)

        # Process each band
        for i in range(len(self.BAND_EDGES) - 1):
            low_freq = self.BAND_EDGES[i]
            high_freq = self.BAND_EDGES[i + 1]

            # Find frequency bin indices for this band
            band_mask = (freqs >= low_freq) & (freqs < high_freq)

            if not np.any(band_mask):
                continue

            # Estimate noise level in this band
            band_noise = np.mean(noise_estimate[band_mask])

            # Higher reduction for higher frequency bands (where hiss lives)
            freq_factor = 1.0 + (i / (len(self.BAND_EDGES) - 1)) * (self.high_freq_emphasis - 1.0)

            # Calculate band-specific reduction
            for j, is_in_band in enumerate(band_mask):
                if is_in_band:
                    # Calculate SNR for each frame
                    snr = stft_mag[j, :] / (band_noise + 1e-10)

                    # Adaptive gain based on SNR
                    band_gain = np.clip(1 - (freq_factor * self.noise_reduction_strength) / (snr + 1), 0.1, 1.0)
                    gain[j, :] = band_gain

        # Smooth gain temporally
        gain = uniform_filter1d(gain, size=5, axis=1)

        # Apply gain
        stft_clean = stft_mag * gain * np.exp(1j * stft_phase)

        return librosa.istft(stft_clean, hop_length=self.HOP_LENGTH, length=len(audio))

    def _combined_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Combined approach using multiple methods.

        Applies spectral subtraction first, then Wiener filtering,
        and finally multi-band processing for optimal results.
        """
        # Stage 1: Gentle spectral subtraction
        temp_strength = self.noise_reduction_strength
        self.noise_reduction_strength = temp_strength * 0.5
        denoised = self._spectral_subtraction(audio)

        # Stage 2: Wiener filtering
        self.noise_reduction_strength = temp_strength * 0.7
        denoised = self._wiener_filter(denoised)

        # Restore strength
        self.noise_reduction_strength = temp_strength

        # Stage 3: Final high-frequency cleanup
        if self.high_freq_emphasis > 1.0:
            denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        return denoised
    def _shellac_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Specialized denoising for shellac 78rpm records.

        Addresses the specific noise characteristics of shellac recordings:
        - Groove noise (low-frequency rumble from worn grooves)
        - Surface hiss (broadband high-frequency noise)
        - Crackle (impulsive noise - addressed separately)

        Uses a multi-stage approach:
        1. Low-frequency rumble reduction (below 80Hz)
        2. Broadband surface noise reduction with frequency-dependent strength
        3. High-frequency hiss targeting (2kHz-12kHz)
        """
        # Compute STFT with good frequency resolution for low-freq work
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT)

        # Get noise estimate
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
        else:
            noise_estimate = self._estimate_noise_spectrum(stft_mag)

        # Apply noise threshold - multiplies the noise estimate to define the boundary
        noise_estimate = noise_estimate * self.noise_threshold

        # Create frequency-dependent gain matrix
        gain = np.ones_like(stft_mag)

        # Define shellac-specific reduction strengths per frequency band
        # (low_freq, high_freq, reduction_multiplier)
        shellac_bands = [
            (0, 60, 1.8),       # Sub-bass rumble - strong reduction
            (60, 150, 1.4),    # Bass rumble - moderate reduction
            (150, 500, 0.8),   # Low-mids - gentle (preserve warmth)
            (500, 2000, 0.9),  # Mids - gentle (preserve vocals)
            (2000, 4000, 1.3), # Upper-mids - moderate hiss
            (4000, 8000, 1.6), # High - strong hiss reduction
            (8000, 16000, 2.0), # Very high - aggressive hiss reduction
        ]

        for low_freq, high_freq, reduction_mult in shellac_bands:
            band_mask = (freqs >= low_freq) & (freqs < high_freq)

            if not np.any(band_mask):
                continue

            # Calculate band-specific gains
            for j, (is_in_band, freq) in enumerate(zip(band_mask, freqs)):
                if is_in_band:
                    band_noise_level = noise_estimate[j]

                    # Calculate SNR
                    snr = stft_mag[j, :] / (band_noise_level + 1e-10)

                    # Strength adjusted by band multiplier and user setting
                    effective_strength = self.noise_reduction_strength * reduction_mult

                    # Adaptive gain - more reduction for lower SNR
                    band_gain = np.clip(1 - effective_strength / (snr + 1), 0.05, 1.0)
                    gain[j, :] = band_gain

        # Smooth gain to reduce musical noise artifacts
        gain = median_filter(gain, size=(3, 5))
        gain = uniform_filter1d(gain, size=3, axis=1)

        # Apply gain
        stft_clean = stft_mag * gain * np.exp(1j * stft_phase)

        # Inverse STFT
        denoised = librosa.istft(stft_clean, hop_length=self.HOP_LENGTH, length=len(audio))

        # Optional: Apply gentle high-frequency shelf for additional hiss control
        if self.high_freq_emphasis > 1.0:
            denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        return denoised

    def _process_channel(self, audio_channel: np.ndarray) -> np.ndarray:
        """Process a single audio channel with the selected denoising method."""

        # Apply the selected denoising method
        if self.method == DenoiseMethod.SPECTRAL_GATING:
            # Pure spectral gating using learned noise profile
            denoised = self._spectral_gating(audio_channel)
            if self.high_freq_emphasis > 1.0:
                denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        elif self.method == DenoiseMethod.SPECTRAL_SUBTRACTION:
            denoised = self._spectral_subtraction(audio_channel)
            if self.high_freq_emphasis > 1.0:
                denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        elif self.method == DenoiseMethod.WIENER:
            denoised = self._wiener_filter(audio_channel)
            if self.high_freq_emphasis > 1.0:
                denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        elif self.method == DenoiseMethod.MULTIBAND:
            denoised = self._multiband_reduction(audio_channel)

        elif self.method == DenoiseMethod.COMBINED:
            denoised = self._combined_reduction(audio_channel)

        elif self.method == DenoiseMethod.SHELLAC:
            denoised = self._shellac_reduction(audio_channel)

        else:
            # Default to spectral subtraction (better for old recordings)
            denoised = self._spectral_subtraction(audio_channel)
            if self.high_freq_emphasis > 1.0:
                denoised = self._apply_high_frequency_reduction(denoised, self._sr)

        # Apply maximum dB reduction limit (prevents over-processing)
        max_reduction_linear = 10 ** (-self.max_db_reduction / 20)
        diff = audio_channel - denoised
        max_allowed_diff = np.abs(audio_channel) * (1 - max_reduction_linear)
        diff_limited = np.clip(diff, -max_allowed_diff, max_allowed_diff)
        denoised = audio_channel - diff_limited

        # Transient protection - blend back original at transients
        if self.transient_protection > 0:
            transient_mask = self._detect_transients(audio_channel, self._sr)
            transient_mask_full = np.interp(
                np.linspace(0, 1, len(audio_channel)),
                np.linspace(0, 1, len(transient_mask)),
                transient_mask
            )
            blend_factor = transient_mask_full * self.transient_protection
            denoised = denoised * (1 - blend_factor) + audio_channel * blend_factor

        # Blend in original signal
        if self.blend_original > 0:
            denoised = denoised * (1 - self.blend_original) + audio_channel * self.blend_original

        return denoised

    def process(self, audio: Optional[np.ndarray] = None, sr: Optional[int] = None) -> np.ndarray:
        """Process audio to remove hiss/noise."""
        if audio is not None:
            self._audio = audio if audio.ndim == 2 else audio.reshape(1, -1)
        if sr is not None:
            self._sr = sr

        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded.")

        processed_channels = []
        for channel in self._audio:
            processed_channel = self._process_channel(channel)
            processed_channels.append(processed_channel)

        self._processed = np.array(processed_channels)

        if self._processed.shape[0] == 1:
            return self._processed.squeeze()

        return self._processed

    def get_original(self) -> Optional[np.ndarray]:
        if self._audio is None:
            return None
        if self._audio.shape[0] == 1:
            return self._audio.squeeze()
        return self._audio

    def get_processed(self) -> Optional[np.ndarray]:
        if self._processed is None:
            return None
        if self._processed.shape[0] == 1:
            return self._processed.squeeze()
        return self._processed

    def get_sample_rate(self) -> Optional[int]:
        return self._sr

    def _copy_metadata(self, source_path: Path, dest_path: Path) -> bool:
        """
        Copy metadata (including album art) from source to destination audio file.

        Supports copying between: MP3, FLAC, OGG, WAV, M4A/AAC formats.

        Args:
            source_path: Path to the original file with metadata
            dest_path: Path to the newly saved file

        Returns:
            True if metadata was copied successfully, False otherwise
        """
        if not MUTAGEN_AVAILABLE:
            return False

        try:
            # Open source file to read metadata
            source_audio = MutagenFile(str(source_path))
            if source_audio is None:
                return False

            dest_ext = dest_path.suffix.lower()

            # Extract common metadata fields from source
            metadata = {}
            pictures = []

            # Handle different source formats
            if hasattr(source_audio, 'tags') and source_audio.tags:
                tags = source_audio.tags

                # FLAC and OGG Vorbis use Vorbis comments
                if hasattr(tags, 'get'):
                    # Try common Vorbis comment fields
                    for field, key in [
                        ('title', 'TITLE'), ('artist', 'ARTIST'), ('album', 'ALBUM'),
                        ('date', 'DATE'), ('genre', 'GENRE'), ('tracknumber', 'TRACKNUMBER'),
                        ('comment', 'COMMENT'), ('albumartist', 'ALBUMARTIST'),
                        ('composer', 'COMPOSER'), ('discnumber', 'DISCNUMBER')
                    ]:
                        value = tags.get(key) or tags.get(key.lower())
                        if value:
                            metadata[field] = value[0] if isinstance(value, list) else value

                # Extract ID3 tags (MP3)
                if hasattr(tags, 'getall'):
                    for frame in ['TIT2', 'TPE1', 'TALB', 'TDRC', 'TCON', 'TRCK', 'COMM']:
                        try:
                            frames = tags.getall(frame)
                            if frames:
                                if frame == 'TIT2':
                                    metadata['title'] = str(frames[0])
                                elif frame == 'TPE1':
                                    metadata['artist'] = str(frames[0])
                                elif frame == 'TALB':
                                    metadata['album'] = str(frames[0])
                                elif frame == 'TDRC':
                                    metadata['date'] = str(frames[0])
                                elif frame == 'TCON':
                                    metadata['genre'] = str(frames[0])
                                elif frame == 'TRCK':
                                    metadata['tracknumber'] = str(frames[0])
                        except:
                            pass

                    # Extract album art from ID3
                    try:
                        apic_frames = tags.getall('APIC')
                        for apic in apic_frames:
                            pictures.append({
                                'data': apic.data,
                                'mime': apic.mime,
                                'type': apic.type,
                                'desc': apic.desc
                            })
                    except:
                        pass

            # Extract pictures from FLAC
            if hasattr(source_audio, 'pictures'):
                for pic in source_audio.pictures:
                    pictures.append({
                        'data': pic.data,
                        'mime': pic.mime,
                        'type': pic.type,
                        'desc': pic.desc if hasattr(pic, 'desc') else ''
                    })

            # Extract from MP4/M4A
            if hasattr(source_audio, 'tags') and source_audio.tags:
                tags = source_audio.tags
                # MP4 atom names
                mp4_map = {
                    '\xa9nam': 'title', '\xa9ART': 'artist', '\xa9alb': 'album',
                    '\xa9day': 'date', '\xa9gen': 'genre', 'trkn': 'tracknumber'
                }
                for atom, field in mp4_map.items():
                    if atom in tags:
                        val = tags[atom]
                        if isinstance(val, list) and val:
                            val = val[0]
                            if isinstance(val, tuple):
                                val = str(val[0])  # Track number tuple
                            metadata[field] = str(val)

                # MP4 cover art
                if 'covr' in tags:
                    for cover in tags['covr']:
                        pictures.append({
                            'data': bytes(cover),
                            'mime': 'image/jpeg' if cover.imageformat == 13 else 'image/png',
                            'type': 3,  # Front cover
                            'desc': ''
                        })

            # Now write metadata to destination
            if dest_ext == '.flac':
                dest_audio = FLAC(str(dest_path))

                # Set text metadata
                for field, value in metadata.items():
                    dest_audio[field.upper()] = value

                # Add pictures
                dest_audio.clear_pictures()
                for pic_data in pictures:
                    pic = Picture()
                    pic.data = pic_data['data']
                    pic.mime = pic_data['mime']
                    pic.type = pic_data.get('type', 3)
                    pic.desc = pic_data.get('desc', '')
                    dest_audio.add_picture(pic)

                dest_audio.save()

            elif dest_ext == '.ogg':
                dest_audio = OggVorbis(str(dest_path))

                # Set text metadata
                for field, value in metadata.items():
                    dest_audio[field.upper()] = value

                # OGG can embed pictures as base64 in METADATA_BLOCK_PICTURE
                import base64
                for pic_data in pictures:
                    pic = Picture()
                    pic.data = pic_data['data']
                    pic.mime = pic_data['mime']
                    pic.type = pic_data.get('type', 3)
                    pic.desc = pic_data.get('desc', '')
                    pic_encoded = base64.b64encode(pic.write()).decode('ascii')
                    dest_audio['METADATA_BLOCK_PICTURE'] = [pic_encoded]

                dest_audio.save()

            elif dest_ext == '.wav':
                # WAV files can have ID3 tags
                try:
                    dest_audio = WAVE(str(dest_path))
                    if dest_audio.tags is None:
                        dest_audio.add_tags()

                    # Add ID3 tags
                    if 'title' in metadata:
                        dest_audio.tags.add(TIT2(encoding=3, text=metadata['title']))
                    if 'artist' in metadata:
                        dest_audio.tags.add(TPE1(encoding=3, text=metadata['artist']))
                    if 'album' in metadata:
                        dest_audio.tags.add(TALB(encoding=3, text=metadata['album']))
                    if 'date' in metadata:
                        dest_audio.tags.add(TDRC(encoding=3, text=metadata['date']))
                    if 'genre' in metadata:
                        dest_audio.tags.add(TCON(encoding=3, text=metadata['genre']))
                    if 'tracknumber' in metadata:
                        dest_audio.tags.add(TRCK(encoding=3, text=metadata['tracknumber']))

                    # Add pictures
                    for pic_data in pictures:
                        dest_audio.tags.add(APIC(
                            encoding=3,
                            mime=pic_data['mime'],
                            type=pic_data.get('type', 3),
                            desc=pic_data.get('desc', ''),
                            data=pic_data['data']
                        ))

                    dest_audio.save()
                except Exception:
                    # WAV ID3 support can be flaky
                    pass

            return True

        except Exception as e:
            # Log but don't fail - metadata is nice to have but not critical
            print(f"Warning: Could not copy metadata: {e}")
            return False

    def save(self, output_path: str, audio: Optional[np.ndarray] = None, format: str = "FLAC") -> str:
        """
        Save processed audio to file, preserving metadata from the original file.

        Args:
            output_path: Path to save the output file
            audio: Optional audio data (uses processed audio if not provided)
            format: Output format - 'FLAC', 'WAV', 'OGG' (default: FLAC)
        """
        if audio is None:
            audio = self._processed

        if audio is None:
            raise ValueError("No processed audio to save.")

        output_path = Path(output_path)

        if audio.ndim == 2:
            audio = audio.T

        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.99

        # Determine subtype based on format
        format_upper = format.upper()
        if format_upper == 'FLAC':
            subtype = 'PCM_24'
        elif format_upper == 'WAV':
            subtype = 'PCM_24'
        elif format_upper == 'OGG':
            subtype = 'VORBIS'
        else:
            subtype = None

        sf.write(str(output_path), audio, self._sr, subtype=subtype)

        # Copy metadata from original file if available
        if self._file_path and self._file_path.exists():
            self._copy_metadata(self._file_path, output_path)

        return str(output_path)

    def update_parameters(
        self,
        max_db_reduction: Optional[float] = None,
        blend_original: Optional[float] = None,
        noise_reduction_strength: Optional[float] = None,
        transient_protection: Optional[float] = None,
        high_freq_emphasis: Optional[float] = None,
        method: Optional[DenoiseMethod] = None,
        hiss_start_freq: Optional[float] = None,
        hiss_peak_freq: Optional[float] = None,
        spectral_floor: Optional[float] = None,
        noise_threshold: Optional[float] = None,
    ):
        """Update denoiser parameters."""
        if max_db_reduction is not None:
            self.max_db_reduction = max_db_reduction
        if blend_original is not None:
            self.blend_original = blend_original
        if noise_reduction_strength is not None:
            self.noise_reduction_strength = noise_reduction_strength
        if transient_protection is not None:
            self.transient_protection = transient_protection
        if high_freq_emphasis is not None:
            self.high_freq_emphasis = high_freq_emphasis
        if method is not None:
            self.method = method
        if hiss_start_freq is not None:
            self.hiss_start_freq = hiss_start_freq
        if hiss_peak_freq is not None:
            self.hiss_peak_freq = hiss_peak_freq
        if spectral_floor is not None:
            self.spectral_floor = spectral_floor
        if noise_threshold is not None:
            self.noise_threshold = noise_threshold

    def set_method(self, method: DenoiseMethod):
        """Set the denoising method."""
        self.method = method

    def get_method(self) -> DenoiseMethod:
        """Get the current denoising method."""
        return self.method

    @staticmethod
    def get_available_methods() -> List[DenoiseMethod]:
        """Get list of available denoising methods."""
        return list(DenoiseMethod)