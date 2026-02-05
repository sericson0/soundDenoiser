"""
Core audio denoising module using adaptive spectral techniques.

Implements spectral gating and noise reduction while preserving
transients and audio fidelity.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional
from pathlib import Path


class AudioDenoiser:
    """
    Adaptive spectral denoiser for removing hiss from audio recordings.
    
    Features:
    - Adaptive noise floor estimation
    - Spectral gating with soft masking
    - Transient preservation
    - Original signal blending
    - Maximum dB reduction limiting
    """
    
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
            
        return self._audio, self._sr
    
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
    
    def _create_spectral_gate(
        self,
        stft_mag: np.ndarray,
        noise_floor: np.ndarray,
        transient_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create an adaptive spectral gate mask.
        
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
        # STFT parameters optimized for hiss removal
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        stft = librosa.stft(audio_channel, n_fft=n_fft, hop_length=hop_length)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)
        
        # Estimate noise floor
        noise_floor = self._estimate_noise_floor(stft_mag)
        
        # Detect transients
        transient_mask = self._detect_transients(audio_channel, self._sr)
        
        # Ensure transient mask matches STFT time frames
        if len(transient_mask) != stft_mag.shape[1]:
            transient_mask = np.interp(
                np.linspace(0, 1, stft_mag.shape[1]),
                np.linspace(0, 1, len(transient_mask)),
                transient_mask
            )
        
        # Create spectral gate
        mask = self._create_spectral_gate(stft_mag, noise_floor, transient_mask)
        
        # Apply mask
        stft_denoised_mag = stft_mag * mask
        
        # Reconstruct complex STFT
        stft_denoised = stft_denoised_mag * np.exp(1j * stft_phase)
        
        # Inverse STFT
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length, length=len(audio_channel))
        
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
