"""
Core audio denoising module using adaptive spectral techniques.

Implements spectral gating and noise reduction while preserving
transients and audio fidelity. Includes noise profile learning
similar to iZotope RX's spectral denoiser.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from typing import Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class NoiseProfile:
    """Learned noise profile from a sample region."""
    # Mean spectral magnitude of noise
    spectral_mean: np.ndarray
    # Standard deviation of noise spectrum
    spectral_std: np.ndarray
    # Minimum observed values (noise floor)
    spectral_min: np.ndarray
    # Maximum observed values (noise ceiling)
    spectral_max: np.ndarray
    # Sample rate used for learning
    sample_rate: int
    # Duration of noise sample in seconds
    duration: float
    # Start and end time of the sample region
    start_time: float
    end_time: float
    
    def get_threshold(self, sensitivity: float = 1.0) -> np.ndarray:
        """
        Get adaptive threshold based on noise statistics.
        
        Args:
            sensitivity: Multiplier for threshold (higher = more aggressive)
            
        Returns:
            Threshold array per frequency bin
        """
        # Use mean + scaled standard deviation as threshold
        # This is similar to iZotope RX's approach
        return self.spectral_mean + sensitivity * self.spectral_std


class AudioDenoiser:
    """
    Adaptive spectral denoiser for removing hiss from audio recordings.
    
    Features:
    - Noise profile learning from selected regions
    - Auto-detection of quiet regions for noise sampling
    - Spectral subtraction with learned noise profile (iZotope RX-style)
    - Adaptive noise floor estimation
    - Spectral gating with soft masking
    - Transient preservation
    - Original signal blending
    - Maximum dB reduction limiting
    """
    
    # STFT parameters (shared across methods)
    N_FFT = 2048
    HOP_LENGTH = 512
    
    def __init__(
        self,
        max_db_reduction: float = 4.0,
        blend_original: float = 0.12,
        noise_reduction_strength: float = 0.7,
        transient_protection: float = 0.5,
        high_freq_rolloff: float = 0.8,
        smoothing_factor: float = 0.1,
    ):
        """
        Initialize the denoiser with parameters.
        
        Args:
            max_db_reduction: Maximum dB of noise reduction to apply (default: 4.0)
            blend_original: Amount of original signal to blend back (0-1, default: 0.12)
            noise_reduction_strength: Overall strength of noise reduction (0-1, default: 0.7)
            transient_protection: How much to protect transients (0-1, default: 0.5)
            high_freq_rolloff: Reduce high frequency noise more aggressively (0-1, default: 0.8)
            smoothing_factor: Temporal smoothing for noise estimate (0-1, default: 0.1)
        """
        self.max_db_reduction = max_db_reduction
        self.blend_original = blend_original
        self.noise_reduction_strength = noise_reduction_strength
        self.transient_protection = transient_protection
        self.high_freq_rolloff = high_freq_rolloff
        self.smoothing_factor = smoothing_factor
        
        # Internal state
        self._audio: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._processed: Optional[np.ndarray] = None
        self._file_path: Optional[Path] = None
        
        # Noise profile (learned from sample region)
        self._noise_profile: Optional[NoiseProfile] = None
        self._use_learned_profile: bool = False
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio data, sample rate)
        """
        self._file_path = Path(file_path)
        # Load with original sample rate to preserve quality
        self._audio, self._sr = librosa.load(file_path, sr=None, mono=False)
        
        # Handle mono files
        if self._audio.ndim == 1:
            self._audio = self._audio.reshape(1, -1)
        
        # Reset noise profile when loading new audio
        self._noise_profile = None
        self._use_learned_profile = False
            
        return self._audio, self._sr
    
    def get_duration(self) -> float:
        """Get duration of loaded audio in seconds."""
        if self._audio is None or self._sr is None:
            return 0.0
        return self._audio.shape[1] / self._sr
    
    def learn_noise_profile(
        self,
        start_time: float,
        end_time: float,
    ) -> NoiseProfile:
        """
        Learn noise profile from a specified region of the audio.
        
        This is similar to iZotope RX's "Learn" function - it analyzes
        a section that contains only noise (no desired signal) to create
        a spectral fingerprint of the noise.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            NoiseProfile containing learned noise characteristics
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded. Call load_audio() first.")
        
        # Convert times to samples
        start_sample = int(start_time * self._sr)
        end_sample = int(end_time * self._sr)
        
        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(self._audio.shape[1], end_sample)
        
        if end_sample <= start_sample:
            raise ValueError("Invalid time range for noise profile.")
        
        # Extract noise region (use first channel for profile, apply to all)
        noise_region = self._audio[0, start_sample:end_sample]
        
        # Compute STFT of noise region
        stft = librosa.stft(noise_region, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        
        # Compute statistics across time frames
        # These statistics form our noise "fingerprint"
        spectral_mean = np.mean(stft_mag, axis=1)
        spectral_std = np.std(stft_mag, axis=1)
        spectral_min = np.min(stft_mag, axis=1)
        spectral_max = np.max(stft_mag, axis=1)
        
        # Apply smoothing across frequency bins to reduce variance
        spectral_mean = median_filter(spectral_mean, size=5)
        spectral_std = median_filter(spectral_std, size=5)
        
        # Create and store noise profile
        self._noise_profile = NoiseProfile(
            spectral_mean=spectral_mean,
            spectral_std=spectral_std,
            spectral_min=spectral_min,
            spectral_max=spectral_max,
            sample_rate=self._sr,
            duration=end_time - start_time,
            start_time=start_time,
            end_time=end_time,
        )
        
        self._use_learned_profile = True
        
        return self._noise_profile
    
    def auto_detect_noise_region(
        self,
        min_duration: float = 0.5,
        num_candidates: int = 5,
    ) -> Tuple[float, float]:
        """
        Automatically detect the quietest region of audio for noise sampling.
        
        Analyzes the audio to find sections with consistently low energy
        that likely contain only noise (no musical content).
        
        Args:
            min_duration: Minimum duration in seconds for noise region
            num_candidates: Number of candidate regions to evaluate
            
        Returns:
            Tuple of (start_time, end_time) for the detected noise region
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded. Call load_audio() first.")
        
        # Use first channel for analysis
        audio = self._audio[0]
        
        # Calculate RMS energy in overlapping windows
        window_samples = int(min_duration * self._sr)
        hop_samples = window_samples // 4
        
        # Compute RMS for each window
        rms_values = []
        positions = []
        
        for start in range(0, len(audio) - window_samples, hop_samples):
            end = start + window_samples
            window = audio[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
            positions.append(start)
        
        rms_values = np.array(rms_values)
        positions = np.array(positions)
        
        # Find the quietest regions
        # Sort by RMS and take the quietest candidates
        sorted_indices = np.argsort(rms_values)
        
        # Evaluate candidates based on multiple criteria
        best_score = float('inf')
        best_region = (0.0, min_duration)
        
        for idx in sorted_indices[:num_candidates * 3]:
            start_sample = positions[idx]
            end_sample = start_sample + window_samples
            
            # Extract candidate region
            region = audio[start_sample:end_sample]
            
            # Compute STFT for spectral analysis
            stft = librosa.stft(region, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
            stft_mag = np.abs(stft)
            
            # Score based on:
            # 1. Low overall energy (RMS)
            rms_score = rms_values[idx]
            
            # 2. Consistent energy (low variance) - noise is typically steady
            variance_score = np.std(np.sum(stft_mag, axis=0))
            
            # 3. Smooth spectral shape (hiss has characteristic smooth spectrum)
            spectral_smoothness = np.mean(np.abs(np.diff(np.mean(stft_mag, axis=1))))
            
            # 4. Absence of transients (check for sudden changes)
            energy_envelope = np.sum(stft_mag, axis=0)
            transient_score = np.max(np.abs(np.diff(energy_envelope))) / (np.mean(energy_envelope) + 1e-8)
            
            # Combined score (lower is better)
            # Weights tuned for typical hiss characteristics
            combined_score = (
                rms_score * 1.0 +
                variance_score * 0.5 +
                spectral_smoothness * 0.3 +
                transient_score * 2.0  # Heavily penalize transients
            )
            
            if combined_score < best_score:
                best_score = combined_score
                best_region = (start_sample / self._sr, end_sample / self._sr)
        
        return best_region
    
    def auto_learn_noise_profile(
        self,
        min_duration: float = 0.5,
    ) -> Tuple[NoiseProfile, Tuple[float, float]]:
        """
        Automatically detect quiet region and learn noise profile from it.
        
        Combines auto_detect_noise_region and learn_noise_profile into
        a single convenience method.
        
        Args:
            min_duration: Minimum duration in seconds for noise region
            
        Returns:
            Tuple of (NoiseProfile, (start_time, end_time))
        """
        start_time, end_time = self.auto_detect_noise_region(min_duration)
        profile = self.learn_noise_profile(start_time, end_time)
        return profile, (start_time, end_time)
    
    def get_noise_profile(self) -> Optional[NoiseProfile]:
        """Get the current learned noise profile."""
        return self._noise_profile
    
    def clear_noise_profile(self):
        """Clear the learned noise profile and use adaptive estimation."""
        self._noise_profile = None
        self._use_learned_profile = False
    
    def set_use_learned_profile(self, use: bool):
        """Enable or disable use of learned noise profile."""
        self._use_learned_profile = use and self._noise_profile is not None
    
    def _estimate_noise_floor(self, stft_mag: np.ndarray) -> np.ndarray:
        """
        Estimate the noise floor using adaptive percentile method.
        
        Uses a combination of:
        - Minimum statistics across time
        - Median filtering for stability
        - Percentile-based estimation for robustness
        
        Args:
            stft_mag: Magnitude spectrogram
            
        Returns:
            Estimated noise floor per frequency bin
        """
        # Use lower percentile to estimate noise floor
        # This captures the quietest moments which represent noise
        noise_floor = np.percentile(stft_mag, 10, axis=1)
        
        # Apply median filter to smooth the estimate across frequency
        noise_floor = median_filter(noise_floor, size=5)
        
        # Boost estimate slightly to be conservative
        noise_floor *= 1.2
        
        return noise_floor
    
    def _detect_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect transients in the audio signal.
        
        Uses onset strength envelope to identify transient regions
        that should be protected from noise reduction.
        
        Args:
            audio: Audio signal (mono)
            sr: Sample rate
            
        Returns:
            Transient mask (same length as STFT time frames)
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Normalize
        onset_env = onset_env / (np.max(onset_env) + 1e-8)
        
        # Create soft transient mask
        transient_mask = np.clip(onset_env * 2, 0, 1)
        
        # Apply some smoothing
        transient_mask = median_filter(transient_mask, size=3)
        
        return transient_mask
    
    def _create_spectral_gate_with_profile(
        self,
        stft_mag: np.ndarray,
        transient_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create spectral gate mask using learned noise profile.
        
        This implements iZotope RX-style spectral subtraction:
        - Uses learned noise profile as baseline
        - Applies over-subtraction for cleaner results
        - Includes spectral floor to prevent artifacts
        
        Args:
            stft_mag: Magnitude spectrogram
            transient_mask: Transient detection mask
            
        Returns:
            Soft mask for noise reduction
        """
        if self._noise_profile is None:
            raise ValueError("No noise profile learned.")
        
        # Convert max dB reduction to linear scale
        max_reduction = 10 ** (-self.max_db_reduction / 20)
        
        # Get noise threshold from profile
        # sensitivity scales with noise reduction strength
        sensitivity = 1.0 + self.noise_reduction_strength
        noise_threshold = self._noise_profile.get_threshold(sensitivity)
        
        # Over-subtraction factor (iZotope RX uses this for cleaner results)
        # Higher values = more aggressive but may cause artifacts
        over_subtraction = 1.0 + self.noise_reduction_strength * 0.5
        
        # Spectral floor to prevent musical noise
        spectral_floor = self._noise_profile.spectral_min * 0.1
        
        # Expand to match spectrogram shape
        noise_threshold = noise_threshold[:, np.newaxis]
        spectral_floor = spectral_floor[:, np.newaxis]
        noise_mean = self._noise_profile.spectral_mean[:, np.newaxis]
        
        # Compute gain using Wiener-like filter
        # G = max((|X|^2 - alpha * |N|^2) / |X|^2, floor)
        signal_power = stft_mag ** 2
        noise_power = (noise_mean * over_subtraction) ** 2
        
        # Wiener gain
        gain_squared = (signal_power - noise_power) / (signal_power + 1e-10)
        gain_squared = np.maximum(gain_squared, 0)
        
        # Convert to magnitude domain
        mask = np.sqrt(gain_squared)
        
        # Apply soft thresholding based on SNR
        snr = stft_mag / (noise_threshold + 1e-8)
        
        # Sigmoid transition for smoothness
        softness = 3.0
        threshold_factor = 1.2
        soft_mask = 1 / (1 + np.exp(-softness * (snr - threshold_factor)))
        
        # Combine Wiener gain with soft threshold
        mask = mask * soft_mask + (1 - soft_mask) * max_reduction
        
        # Apply high frequency rolloff
        freq_bins = mask.shape[0]
        freq_weight = np.linspace(1.0, self.high_freq_rolloff, freq_bins)
        mask = mask * freq_weight[:, np.newaxis] + (1 - freq_weight[:, np.newaxis]) * max_reduction
        
        # Protect transients
        transient_protection_mask = transient_mask * self.transient_protection
        mask = mask + transient_protection_mask * (1 - mask)
        
        # Ensure minimum gain (max dB reduction limit)
        mask = np.maximum(mask, max_reduction)
        mask = np.minimum(mask, 1.0)
        
        # Temporal smoothing to reduce musical noise
        if self.smoothing_factor > 0:
            kernel_size = max(3, int(self.smoothing_factor * 10))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = median_filter(mask, size=(1, kernel_size))
        
        # Additional frequency smoothing to reduce artifacts
        mask = median_filter(mask, size=(3, 1))
        
        return mask
    
    def _create_spectral_gate(
        self,
        stft_mag: np.ndarray,
        noise_floor: np.ndarray,
        transient_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create an adaptive spectral gate mask (fallback when no profile learned).
        
        Args:
            stft_mag: Magnitude spectrogram
            noise_floor: Estimated noise floor
            transient_mask: Transient detection mask
            
        Returns:
            Soft mask for noise reduction
        """
        # Convert max dB reduction to linear scale
        max_reduction = 10 ** (-self.max_db_reduction / 20)
        
        # Expand noise floor to match spectrogram shape
        noise_floor_expanded = noise_floor[:, np.newaxis]
        
        # Calculate signal-to-noise ratio per bin
        snr = stft_mag / (noise_floor_expanded + 1e-8)
        
        # Create soft mask using sigmoid-like function
        # This provides smooth transitions instead of hard gating
        threshold = 1.5  # SNR threshold for gating
        softness = 2.0   # Softness of the transition
        
        mask = 1 / (1 + np.exp(-softness * (snr - threshold)))
        
        # Apply noise reduction strength
        mask = 1 - self.noise_reduction_strength * (1 - mask)
        
        # Apply high frequency rolloff (more aggressive reduction at high frequencies)
        freq_bins = mask.shape[0]
        freq_weight = np.linspace(1.0, self.high_freq_rolloff, freq_bins)
        mask = mask * freq_weight[:, np.newaxis] + (1 - freq_weight[:, np.newaxis]) * max_reduction
        
        # Protect transients
        transient_protection_mask = transient_mask * self.transient_protection
        mask = mask + transient_protection_mask * (1 - mask)
        
        # Ensure minimum gain (max dB reduction limit)
        mask = np.maximum(mask, max_reduction)
        
        # Temporal smoothing to reduce musical noise
        if self.smoothing_factor > 0:
            kernel_size = max(3, int(self.smoothing_factor * 10))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = median_filter(mask, size=(1, kernel_size))
        
        return mask
    
    def _process_channel(self, audio_channel: np.ndarray) -> np.ndarray:
        """
        Process a single audio channel.
        
        Args:
            audio_channel: Single channel audio data
            
        Returns:
            Denoised audio channel
        """
        # Compute STFT
        stft = librosa.stft(audio_channel, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)
        
        # Detect transients
        transient_mask = self._detect_transients(audio_channel, self._sr)
        
        # Ensure transient mask matches STFT time frames
        if len(transient_mask) != stft_mag.shape[1]:
            transient_mask = np.interp(
                np.linspace(0, 1, stft_mag.shape[1]),
                np.linspace(0, 1, len(transient_mask)),
                transient_mask
            )
        
        # Create spectral gate based on whether we have a learned profile
        if self._use_learned_profile and self._noise_profile is not None:
            mask = self._create_spectral_gate_with_profile(stft_mag, transient_mask)
        else:
            noise_floor = self._estimate_noise_floor(stft_mag)
            mask = self._create_spectral_gate(stft_mag, noise_floor, transient_mask)
        
        # Apply mask
        stft_denoised_mag = stft_mag * mask
        
        # Reconstruct complex STFT
        stft_denoised = stft_denoised_mag * np.exp(1j * stft_phase)
        
        # Inverse STFT
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.HOP_LENGTH, length=len(audio_channel))
        
        # Blend with original signal
        audio_denoised = (1 - self.blend_original) * audio_denoised + self.blend_original * audio_channel
        
        return audio_denoised
    
    def process(self, audio: Optional[np.ndarray] = None, sr: Optional[int] = None) -> np.ndarray:
        """
        Process audio to remove hiss/noise.
        
        Args:
            audio: Optional audio data (uses loaded audio if not provided)
            sr: Optional sample rate (uses loaded sample rate if not provided)
            
        Returns:
            Denoised audio
        """
        if audio is not None:
            self._audio = audio if audio.ndim == 2 else audio.reshape(1, -1)
        if sr is not None:
            self._sr = sr
            
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded. Call load_audio() first or provide audio data.")
        
        # Process each channel
        processed_channels = []
        for channel in self._audio:
            processed_channel = self._process_channel(channel)
            processed_channels.append(processed_channel)
        
        self._processed = np.array(processed_channels)
        
        # Squeeze if mono
        if self._processed.shape[0] == 1:
            return self._processed.squeeze()
        
        return self._processed
    
    def get_original(self) -> Optional[np.ndarray]:
        """Get the original loaded audio."""
        if self._audio is None:
            return None
        if self._audio.shape[0] == 1:
            return self._audio.squeeze()
        return self._audio
    
    def get_processed(self) -> Optional[np.ndarray]:
        """Get the processed audio."""
        if self._processed is None:
            return None
        if self._processed.shape[0] == 1:
            return self._processed.squeeze()
        return self._processed
    
    def get_sample_rate(self) -> Optional[int]:
        """Get the sample rate of loaded audio."""
        return self._sr
    
    def save(self, output_path: str, audio: Optional[np.ndarray] = None) -> str:
        """
        Save processed audio to file.
        
        Args:
            output_path: Path to save the output file
            audio: Optional audio data (uses processed audio if not provided)
            
        Returns:
            Path to saved file
        """
        if audio is None:
            audio = self._processed
            
        if audio is None:
            raise ValueError("No processed audio to save. Call process() first.")
        
        output_path = Path(output_path)
        
        # Ensure audio is in correct format for soundfile
        if audio.ndim == 2:
            audio = audio.T  # soundfile expects (samples, channels)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.99
        
        sf.write(str(output_path), audio, self._sr)
        
        return str(output_path)
    
    def update_parameters(
        self,
        max_db_reduction: Optional[float] = None,
        blend_original: Optional[float] = None,
        noise_reduction_strength: Optional[float] = None,
        transient_protection: Optional[float] = None,
        high_freq_rolloff: Optional[float] = None,
        smoothing_factor: Optional[float] = None,
    ):
        """
        Update denoiser parameters.
        
        Args:
            max_db_reduction: Maximum dB of noise reduction
            blend_original: Amount of original signal to blend back
            noise_reduction_strength: Overall strength of noise reduction
            transient_protection: How much to protect transients
            high_freq_rolloff: High frequency noise reduction aggressiveness
            smoothing_factor: Temporal smoothing factor
        """
        if max_db_reduction is not None:
            self.max_db_reduction = max_db_reduction
        if blend_original is not None:
            self.blend_original = blend_original
        if noise_reduction_strength is not None:
            self.noise_reduction_strength = noise_reduction_strength
        if transient_protection is not None:
            self.transient_protection = transient_protection
        if high_freq_rolloff is not None:
            self.high_freq_rolloff = high_freq_rolloff
        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor
