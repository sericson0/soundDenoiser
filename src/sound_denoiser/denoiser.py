"""
Core audio denoising module using multiple techniques for hiss removal.

Implements:
- Adaptive blend of spectral subtraction and gating
- Adaptive 2D gain smoothing (time + frequency) for musical noise suppression
- Multiresolution STFT processing for better transient handling
- High-frequency synthesis for reconstruction of signal detail buried in noise
- Noise profile learning for targeted removal
- Transient protection and original signal blending
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
    ADAPTIVE_BLEND = "adaptive"          # Intelligent blend of subtraction and gating


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
    - Multiple denoising algorithms (spectral gating, spectral subtraction, adaptive blend)
    - Spectral gating with learned noise profiles
    - Transient preservation
    - Original signal blending
    - Perceptual weighting to preserve audio quality
    """

    # STFT parameters
    N_FFT = 2048
    HOP_LENGTH = 512

    def __init__(
        self,
        blend_original: float = 0.05,
        reduction_db: float = 18.0,
        transient_protection: float = 0.15,
        method: DenoiseMethod = DenoiseMethod.ADAPTIVE_BLEND,
        spectral_floor: float = 0.025,
        noise_threshold_db: float = 3.5,
        artifact_control: float = 0.7,
        adaptive_blend: bool = True,
    ):
        """
        Initialize the denoiser with parameters.

        Args:
            blend_original: Amount of original signal to blend back (0-1, default: 0.05)
            reduction_db: Maximum noise reduction in decibels (0-40, default: 18.0)
                6 dB = noise halved, 12 dB = noise quartered, 20 dB = noise at 10%
            transient_protection: How much to protect transients (0-1, default: 0.15)
            method: Denoising method to use (default: ADAPTIVE_BLEND)
            spectral_floor: Minimum signal to retain, prevents artifacts (0-1, default: 0.025)
            noise_threshold_db: Noise boundary above estimate in dB (0-10, default: 3.5)
                Higher = more aggressive (treats more as noise)
                0 dB = boundary at estimated noise level
                3.5 dB ~ 1.5x multiplier, 6 dB ~ 2x multiplier
            artifact_control: Balance between subtraction and gating (0-1, default: 0.7)
                0 = pure spectral subtraction (may cause musical noise/chirpy artifacts)
                1 = pure spectral gating (may cause noise bursts/pumping)
            adaptive_blend: When True, varies the blend based on signal characteristics
        """
        self.blend_original = blend_original
        self.reduction_db = reduction_db
        self.transient_protection = transient_protection
        self.method = method

        # Fine-tuning parameters
        self.spectral_floor = spectral_floor
        self.noise_threshold_db = noise_threshold_db
        self.artifact_control = artifact_control
        self.adaptive_blend = adaptive_blend

        # Internal state
        self._audio: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._processed: Optional[np.ndarray] = None
        self._file_path: Optional[Path] = None

        # Noise profile (learned from sample region)
        self._noise_profile: Optional[NoiseProfile] = None
        self._use_learned_profile: bool = False

    @property
    def _noise_threshold_mult(self) -> float:
        """Convert noise threshold from dB to linear multiplier."""
        return 10 ** (self.noise_threshold_db / 20)

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

    def _find_noise_from_edge(
        self, audio: np.ndarray, step_samples: int,
        from_start: bool = True, silence_peak: float = 0.002,
        max_scan_seconds: float = 15.0
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find a noise region at one edge of the audio by scanning inward.
        
        Two-phase approach that handles both stable and fading noise:
        
        Phase 1 (initial calibration):
            Walk inward from the edge, skip silence. Collect the first 3
            non-silent windows unconditionally — these are at the very edge
            and are the most reliably noise-only audio.
            
        Phase 2 (expanding calibration):
            Continue walking inward. Accept windows whose RMS is within 5x
            of the initial floor (handles gradual fade-in/out of hiss).
            Track the running max RMS as the true noise floor.  Stop
            expansion when a window exceeds 5x initial floor.
            
        Phase 3 (final walk):
            Use the expanded noise floor. Accept windows < noise_floor * 3.
            Stop at the first window that exceeds this threshold (music).
        
        Args:
            audio: Audio array (1D, full track)
            step_samples: Window size in samples (e.g. 0.1s worth)
            from_start: If True, scan from sample 0 forward.
                        If False, scan from the end backward.
            silence_peak: Peak amplitude below which a window is silence
            max_scan_seconds: Max seconds from the edge to scan.
        
        Returns:
            (start_sample, end_sample, noise_floor_rms) or None.
        """
        sr = self._sr
        total = len(audio)
        max_scan_samples = int(max_scan_seconds * sr)
        
        # Determine the scan zone
        if from_start:
            zone_start = 0
            zone_end = min(total, max_scan_samples)
        else:
            zone_start = max(0, total - max_scan_samples)
            zone_end = total
        
        zone = audio[zone_start:zone_end]
        n = len(zone)
        if n < step_samples:
            return None
        
        # ── Compute RMS and peak for every window in the zone ──
        window_starts = list(range(0, n - step_samples + 1, step_samples))
        if not window_starts:
            return None
        
        rms_arr = np.empty(len(window_starts))
        peak_arr = np.empty(len(window_starts))
        for i, ws in enumerate(window_starts):
            chunk = zone[ws:ws + step_samples]
            rms_arr[i] = np.sqrt(np.mean(chunk ** 2))
            peak_arr[i] = np.max(np.abs(chunk))
        
        is_silence = (peak_arr < silence_peak) | (rms_arr < 0.0002)
        
        # ── Walk inward from the edge ──
        if from_start:
            indices = list(range(len(window_starts)))
        else:
            indices = list(range(len(window_starts) - 1, -1, -1))
        
        # Phase 1: skip silence, collect first 3 non-silent as calibration
        calibration_count = 3
        calibration_rms = []
        noise_accepted = []  # list of window indices accepted as noise
        phase1_done = False
        walk_start_idx = 0  # where to resume after phase 1
        
        for pos, i in enumerate(indices):
            if is_silence[i]:
                if not calibration_rms:
                    continue  # Still in leading silence
                else:
                    # Silence gap after some noise — tolerate short gaps
                    noise_accepted.append(i)
                    continue
            
            calibration_rms.append(rms_arr[i])
            noise_accepted.append(i)
            
            if len(calibration_rms) >= calibration_count:
                walk_start_idx = pos + 1
                phase1_done = True
                break
        
        if not phase1_done or not calibration_rms:
            return None
        
        initial_floor = max(calibration_rms)
        if initial_floor < 1e-6:
            return None
        
        # Phase 2: expand calibration — accept windows < initial_floor * 5
        expansion_limit = initial_floor * 5.0
        noise_floor = initial_floor  # running max of accepted noise RMS
        
        for pos in range(walk_start_idx, len(indices)):
            i = indices[pos]
            
            if is_silence[i]:
                # Tolerate isolated silence windows within noise
                noise_accepted.append(i)
                continue
            
            if rms_arr[i] <= expansion_limit:
                noise_accepted.append(i)
                if rms_arr[i] > noise_floor:
                    noise_floor = rms_arr[i]
            else:
                # Exceeded expansion limit — switch to phase 3
                walk_start_idx = pos
                break
        else:
            # All windows accepted (rare — short file with only noise)
            walk_start_idx = len(indices)
        
        # Phase 3: final walk with tighter threshold
        final_threshold = noise_floor * 3.0
        
        for pos in range(walk_start_idx, len(indices)):
            i = indices[pos]
            
            if is_silence[i]:
                noise_accepted.append(i)
                continue
            
            if rms_arr[i] <= final_threshold:
                noise_accepted.append(i)
            else:
                break  # Hit music — stop
        
        # Trim trailing silence from accepted windows
        while noise_accepted:
            if is_silence[noise_accepted[-1]]:
                noise_accepted.pop()
            else:
                break
        
        if not noise_accepted:
            return None
        
        # Convert to sample positions
        first_idx = min(noise_accepted)
        last_idx = max(noise_accepted)
        region_start = zone_start + window_starts[first_idx]
        region_end = zone_start + window_starts[last_idx] + step_samples
        region_end = min(region_end, total)
        
        return (region_start, region_end, noise_floor)
    
    def auto_detect_noise_regions(self, min_duration: float = 0.5) -> List[Tuple[float, float]]:
        """
        Automatically detect noise regions at the very start and very end
        of the file only.
        
        Scans inward from each edge: skips silence, collects noise, stops
        the moment music begins.  Never looks into the middle of the track.
        
        Returns:
            List of (start_time, end_time) tuples for detected noise regions.
            May be empty if no suitable regions found.
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded.")
        
        audio = self._audio[0]
        step_samples = int(0.1 * self._sr)  # 0.1s analysis windows
        
        regions = []
        
        # ── Scan from the START of the file (lead-in groove) ──
        start_result = self._find_noise_from_edge(
            audio, step_samples, from_start=True
        )
        if start_result:
            s, e, _ = start_result
            dur = (e - s) / self._sr
            if dur >= min_duration:
                regions.append((s / self._sr, e / self._sr))
        
        # ── Scan from the END of the file (run-out groove) ──
        end_result = self._find_noise_from_edge(
            audio, step_samples, from_start=False
        )
        if end_result:
            s, e, _ = end_result
            dur = (e - s) / self._sr
            if dur >= min_duration:
                regions.append((s / self._sr, e / self._sr))
        
        return regions
    
    def auto_learn_noise_profile(self, min_duration: float = 0.5) -> Tuple[NoiseProfile, List[Tuple[float, float]]]:
        """
        Auto-detect noise regions at beginning/end of file and learn profile.
        
        Searches the lead-in and run-out areas for clean noise samples,
        uses both if available, and raises an error if neither is usable.
        
        Returns:
            Tuple of (NoiseProfile, list_of_regions)
            
        Raises:
            ValueError: If no suitable noise regions could be found.
        """
        regions = self.auto_detect_noise_regions(min_duration)
        
        if not regions:
            raise ValueError(
                "Could not find a sufficient noise-only sample.\n\n"
                "The beginning and end of the recording do not contain "
                "clean noise (hiss/groove noise without music).\n\n"
                "Please use 'Make Selection' to manually select a region "
                "that contains only noise."
            )
        
        profile = self.learn_noise_profile_from_regions(regions)
        return profile, regions

    def get_noise_profile(self) -> Optional[NoiseProfile]:
        return self._noise_profile

    def clear_noise_profile(self):
        self._noise_profile = None
        self._use_learned_profile = False

    def set_use_learned_profile(self, use: bool):
        self._use_learned_profile = use and self._noise_profile is not None

    # (Removed _apply_high_frequency_reduction and all related code)

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

    def _compute_gain_matrix(
        self, stft_mag: np.ndarray, noise_estimate: np.ndarray,
        noise_std: np.ndarray, min_gain: float, strength: float
    ) -> np.ndarray:
        """
        Compute the blended subtraction/gating gain matrix for a single STFT resolution.

        Returns a gain matrix (same shape as stft_mag) in [min_gain, 1.0].
        """
        noise_estimate_scaled = noise_estimate * self._noise_threshold_mult

        # === SPECTRAL SUBTRACTION GAIN ===
        alpha = 1.0 + strength * 2.0
        beta = self.spectral_floor + (1 - strength) * 0.05

        noise_2d = noise_estimate_scaled[:, np.newaxis]
        subtracted = stft_mag ** 2 - alpha * (noise_2d ** 2)
        spectral_floor_val = beta * (noise_2d ** 2)
        subtracted = np.maximum(subtracted, spectral_floor_val)
        subtraction_mag = np.sqrt(subtracted)

        # === SPECTRAL GATING GAIN ===
        gate_threshold = noise_estimate_scaled[:, np.newaxis]
        noise_std_2d = noise_std[:, np.newaxis] if noise_std is not None else gate_threshold * 0.3

        margin = stft_mag - gate_threshold
        transition_width = np.maximum(noise_std_2d * 2, 1e-10)
        gate_gain = 1.0 / (1.0 + np.exp(-margin / transition_width * 4))

        gate_min_gain = max(min_gain, self.spectral_floor)
        gate_gain = gate_min_gain + gate_gain * (1.0 - gate_min_gain)
        gating_mag = stft_mag * gate_gain

        # === COMPUTE BLEND WEIGHTS ===
        base_gating_weight = self.artifact_control

        if self.adaptive_blend:
            # Derive n_fft from stft shape: freq bins = n_fft/2 + 1
            n_fft_actual = (stft_mag.shape[0] - 1) * 2
            freqs = librosa.fft_frequencies(sr=self._sr, n_fft=n_fft_actual)
            freq_weights = np.clip((freqs - 500) / 4000, 0, 1)
            freq_gating_bias = 0.3 * (1 - freq_weights)
            freq_gating_bias = freq_gating_bias[:, np.newaxis]

            snr = stft_mag / (noise_2d + 1e-10)
            snr_normalized = np.clip((snr - 1) / 5, 0, 1)
            snr_gating_bias = 0.2 * (1 - snr_normalized)

            frame_energy = np.mean(stft_mag ** 2, axis=0)
            energy_diff = np.diff(frame_energy, prepend=frame_energy[0])
            transient_mask = np.clip(energy_diff / (np.mean(frame_energy) + 1e-10), 0, 1)
            transient_mask = uniform_filter1d(transient_mask, size=3)
            transient_gating_bias = 0.2 * transient_mask

            gating_weight = base_gating_weight + freq_gating_bias + snr_gating_bias + transient_gating_bias
            gating_weight = np.clip(gating_weight, 0, 1)
        else:
            gating_weight = base_gating_weight

        # === BLEND ===
        subtraction_weight = 1.0 - gating_weight
        blended_mag = subtraction_weight * subtraction_mag + gating_weight * gating_mag

        # Convert to gain
        gain = blended_mag / (stft_mag + 1e-10)
        gain = np.clip(gain, min_gain, 1.0)

        return gain

    def _adaptive_2d_smooth(self, gain: np.ndarray, stft_mag: np.ndarray,
                            noise_2d: np.ndarray) -> np.ndarray:
        """
        Adaptive 2D gain smoothing (RX Algorithm B equivalent).

        Smooths the gain curve in both time and frequency, with the amount of
        smoothing adapted to the local SNR. Low-SNR regions (mostly noise) get
        more smoothing to suppress musical noise artifacts. High-SNR regions
        (mostly signal) get less smoothing to preserve detail.
        """
        # Compute local SNR for each time-frequency bin
        snr = stft_mag / (noise_2d + 1e-10)

        # Adaptive smoothing width: more smoothing in low-SNR regions
        # SNR < 2 -> heavy smoothing, SNR > 8 -> minimal smoothing
        smooth_factor = np.clip(1.0 - (snr - 2) / 6, 0.1, 1.0)

        # Frequency smoothing (along axis 0) - adaptive kernel
        gain_freq_smooth = uniform_filter1d(gain, size=5, axis=0)
        # Time smoothing (along axis 1) - adaptive kernel
        gain_time_smooth = uniform_filter1d(gain, size=5, axis=1)
        # Combined 2D smooth
        gain_2d_smooth = (gain_freq_smooth + gain_time_smooth) / 2.0

        # Blend between original gain and smoothed gain based on local SNR
        # Low SNR -> use smoothed gain (suppresses musical noise)
        # High SNR -> use original gain (preserves signal detail)
        gain_out = gain * (1.0 - smooth_factor) + gain_2d_smooth * smooth_factor

        return np.clip(gain_out, np.min(gain), 1.0)

    def _multiresolution_denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Multiresolution denoising (RX Algorithm C equivalent).

        Uses two STFT resolutions:
        - Long window (4096): good frequency resolution for tonal/sustained content
        - Short window (1024): good time resolution for transients

        The two are blended based on transient detection, so transients stay
        sharp while sustained noise is removed with fine frequency precision.
        """
        # Get noise estimates
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
            noise_std = self._noise_profile.spectral_std
        else:
            stft_temp = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
            noise_estimate = self._estimate_noise_spectrum(np.abs(stft_temp))
            noise_std = noise_estimate * 0.5

        min_gain = 10 ** (-self.reduction_db / 20)
        strength = 1.0 - min_gain

        # === LONG WINDOW (fine frequency resolution, coarse time) ===
        n_fft_long = 4096
        hop_long = 1024
        stft_long = librosa.stft(audio, n_fft=n_fft_long, hop_length=hop_long)
        stft_long_mag = np.abs(stft_long)
        stft_long_phase = np.angle(stft_long)

        # Resample noise estimate to match long FFT bins
        noise_est_long = np.interp(
            librosa.fft_frequencies(sr=self._sr, n_fft=n_fft_long),
            librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT),
            noise_estimate
        )
        noise_std_long = np.interp(
            librosa.fft_frequencies(sr=self._sr, n_fft=n_fft_long),
            librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT),
            noise_std
        )

        gain_long = self._compute_gain_matrix(
            stft_long_mag, noise_est_long, noise_std_long, min_gain, strength
        )
        # Apply adaptive 2D smoothing to long-window gain
        noise_2d_long = (noise_est_long * self._noise_threshold_mult)[:, np.newaxis]
        gain_long = self._adaptive_2d_smooth(gain_long, stft_long_mag, noise_2d_long)

        denoised_long = librosa.istft(
            stft_long_mag * gain_long * np.exp(1j * stft_long_phase),
            hop_length=hop_long, length=len(audio)
        )

        # === SHORT WINDOW (fine time resolution, coarse frequency) ===
        n_fft_short = 1024
        hop_short = 256
        stft_short = librosa.stft(audio, n_fft=n_fft_short, hop_length=hop_short)
        stft_short_mag = np.abs(stft_short)
        stft_short_phase = np.angle(stft_short)

        # Resample noise estimate to match short FFT bins
        noise_est_short = np.interp(
            librosa.fft_frequencies(sr=self._sr, n_fft=n_fft_short),
            librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT),
            noise_estimate
        )
        noise_std_short = np.interp(
            librosa.fft_frequencies(sr=self._sr, n_fft=n_fft_short),
            librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT),
            noise_std
        )

        gain_short = self._compute_gain_matrix(
            stft_short_mag, noise_est_short, noise_std_short, min_gain, strength
        )
        # Light smoothing for short window (preserve transients)
        gain_short = median_filter(gain_short, size=(1, 3))

        denoised_short = librosa.istft(
            stft_short_mag * gain_short * np.exp(1j * stft_short_phase),
            hop_length=hop_short, length=len(audio)
        )

        # === BLEND BASED ON TRANSIENT DETECTION ===
        # Use onset strength to detect transients
        onset_env = librosa.onset.onset_strength(y=audio, sr=self._sr,
                                                  hop_length=hop_short)
        onset_env = onset_env / (np.max(onset_env) + 1e-8)

        # Upsample transient mask to sample level
        transient_weight = np.interp(
            np.linspace(0, 1, len(audio)),
            np.linspace(0, 1, len(onset_env)),
            onset_env
        )
        # Smooth transition
        transient_weight = uniform_filter1d(transient_weight, size=max(1, int(self._sr * 0.01)))
        transient_weight = np.clip(transient_weight * 3, 0, 1)

        # During transients, use short-window result; otherwise use long-window
        denoised = denoised_long * (1 - transient_weight) + denoised_short * transient_weight

        return denoised

    def _hf_synthesis(self, denoised: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        High-frequency synthesis (RX Algorithm D equivalent).

        Reconstructs high-frequency signal detail that may have been buried
        in noise and removed during denoising. Uses spectral envelope matching
        to synthesize plausible high-frequency content from the denoised mid-band.

        This helps avoid the "dull" or "muffled" quality that aggressive denoising
        can produce in the upper frequencies.
        """
        stft_orig = librosa.stft(original, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_den = librosa.stft(denoised, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        orig_mag = np.abs(stft_orig)
        den_mag = np.abs(stft_den)
        den_phase = np.angle(stft_den)

        freqs = librosa.fft_frequencies(sr=self._sr, n_fft=self.N_FFT)

        # Get noise estimate for determining what was removed
        if self._use_learned_profile and self._noise_profile is not None:
            noise_estimate = self._noise_profile.spectral_mean
        else:
            noise_estimate = self._estimate_noise_spectrum(orig_mag)
        noise_2d = (noise_estimate * self._noise_threshold_mult)[:, np.newaxis]

        # Define HF region (above 4kHz)
        hf_start = 4000.0
        hf_mask = freqs >= hf_start

        if not np.any(hf_mask):
            return denoised

        # For each HF bin, estimate how much signal was lost vs noise removed
        # If original was significantly above noise floor, some signal was likely lost
        orig_snr = orig_mag / (noise_2d + 1e-10)
        den_snr = den_mag / (noise_2d + 1e-10)

        # More sensitive signal loss detection:
        # - Lowered orig_snr threshold from 1.5 to 1.2 (detect signal closer to noise)
        # - Wider detection range (orig_snr 1.2 to 6x)
        # - More sensitive den_snr check (was /2, now /1.5)
        signal_loss = (np.clip((orig_snr - 1.2) / 4.8, 0, 1) *
                       np.clip(1.0 - den_snr / 1.5, 0, 1))

        # Only apply in HF region
        hf_signal_loss = np.zeros_like(signal_loss)
        hf_signal_loss[hf_mask, :] = signal_loss[hf_mask, :]

        # Taper in gradually from hf_start
        taper = np.zeros(len(freqs))
        taper[hf_mask] = np.clip((freqs[hf_mask] - hf_start) / 1500, 0, 1)
        hf_signal_loss *= taper[:, np.newaxis]

        # Synthesize: add back a fraction of the removed energy
        removed = orig_mag - den_mag
        removed = np.maximum(removed, 0)

        # Calibrated synthesis strength:
        # - Base strength scales with reduction_db (more reduction = more synthesis)
        # - Max ~0.45 for aggressive but safe synthesis
        synthesis_strength = np.clip(self.reduction_db / 30, 0, 0.6) * 0.45
        synthesis_strength = min(synthesis_strength, 0.45)

        synthesis = removed * hf_signal_loss * synthesis_strength

        # Smooth the synthesis to avoid introducing new artifacts
        synthesis = uniform_filter1d(synthesis, size=3, axis=0)
        synthesis = uniform_filter1d(synthesis, size=3, axis=1)

        # Add synthesized HF content
        result_mag = den_mag + synthesis
        stft_result = result_mag * np.exp(1j * den_phase)

        return librosa.istft(stft_result, hop_length=self.HOP_LENGTH, length=len(denoised))

    def _process_channel(self, audio_channel: np.ndarray) -> np.ndarray:
        """
        Process a single audio channel.

        Pipeline:
        1. Multiresolution denoising (blends long-window for frequency precision
           and short-window for transient preservation, with adaptive 2D smoothing)
        2. High-frequency synthesis (reconstructs HF detail buried in noise)
        3. Transient protection (blends back original at detected onsets)
        4. Original signal blending (dry/wet mix)
        """
        # Core denoising with multiresolution + adaptive 2D smoothing
        denoised = self._multiresolution_denoise(audio_channel)

        # Reconstruct high-frequency detail
        denoised = self._hf_synthesis(denoised, audio_channel)

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
        blend_original: Optional[float] = None,
        reduction_db: Optional[float] = None,
        transient_protection: Optional[float] = None,
        method: Optional[DenoiseMethod] = None,
        spectral_floor: Optional[float] = None,
        noise_threshold_db: Optional[float] = None,
        artifact_control: Optional[float] = None,
        adaptive_blend: Optional[bool] = None,
    ):
        """Update denoiser parameters."""
        if blend_original is not None:
            self.blend_original = blend_original
        if reduction_db is not None:
            self.reduction_db = reduction_db
        if transient_protection is not None:
            self.transient_protection = transient_protection
        if method is not None:
            self.method = method
        if spectral_floor is not None:
            self.spectral_floor = spectral_floor
        if noise_threshold_db is not None:
            self.noise_threshold_db = noise_threshold_db
        if artifact_control is not None:
            self.artifact_control = artifact_control
        if adaptive_blend is not None:
            self.adaptive_blend = adaptive_blend

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