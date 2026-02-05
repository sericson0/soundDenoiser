"""
Core audio denoising module using multiple techniques for hiss removal.

Implements:
- noisereduce library for stationary noise reduction
- Custom spectral gating for additional hiss targeting
- High-frequency focused processing where hiss typically lives
- Noise profile learning for targeted removal
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from typing import Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
import noisereduce as nr


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
    - noisereduce library for robust stationary noise removal
    - Spectral gating with learned noise profiles
    - High-frequency emphasis for hiss targeting (2kHz-20kHz)
    - Transient preservation
    - Original signal blending
    """
    
    # STFT parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    
    def __init__(
        self,
        max_db_reduction: float = 12.0,
        blend_original: float = 0.05,
        noise_reduction_strength: float = 0.85,
        transient_protection: float = 0.3,
        high_freq_emphasis: float = 1.5,
        smoothing_factor: float = 0.2,
    ):
        """
        Initialize the denoiser with parameters.
        
        Args:
            max_db_reduction: Maximum dB of noise reduction (default: 12.0)
            blend_original: Amount of original signal to blend back (0-1, default: 0.05)
            noise_reduction_strength: Overall strength of noise reduction (0-1, default: 0.85)
            transient_protection: How much to protect transients (0-1, default: 0.3)
            high_freq_emphasis: Extra reduction for high frequencies where hiss lives (default: 1.5)
            smoothing_factor: Temporal smoothing for noise estimate (0-1, default: 0.2)
        """
        self.max_db_reduction = max_db_reduction
        self.blend_original = blend_original
        self.noise_reduction_strength = noise_reduction_strength
        self.transient_protection = transient_protection
        self.high_freq_emphasis = high_freq_emphasis
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
    
    def auto_detect_noise_region(self, min_duration: float = 0.5) -> Tuple[float, float]:
        """
        Automatically detect the quietest region of audio for noise sampling.
        """
        if self._audio is None or self._sr is None:
            raise ValueError("No audio loaded.")
        
        audio = self._audio[0]
        window_samples = int(min_duration * self._sr)
        hop_samples = window_samples // 4
        
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
        
        # Find quietest non-silent region
        sorted_indices = np.argsort(rms_values)
        
        for idx in sorted_indices[:20]:
            start_sample = positions[idx]
            end_sample = start_sample + window_samples
            region = audio[start_sample:end_sample]
            
            # Skip if too quiet (likely silence)
            if np.max(np.abs(region)) < 0.001:
                continue
                
            return (start_sample / self._sr, end_sample / self._sr)
        
        # Fallback to first quiet region
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
        Hiss is usually in the 2kHz-16kHz range.
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)
        
        # Create frequency-dependent gain
        freq_bins = stft_mag.shape[0]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.N_FFT)
        
        # Hiss typically lives above 2kHz
        hiss_start_freq = 2000
        hiss_peak_freq = 6000
        
        # Create smooth reduction curve
        gain = np.ones(freq_bins)
        for i, freq in enumerate(freqs):
            if freq > hiss_start_freq:
                # Gradually increase reduction for higher frequencies
                reduction_amount = min(1.0, (freq - hiss_start_freq) / (hiss_peak_freq - hiss_start_freq))
                reduction_amount *= self.high_freq_emphasis - 1.0
                gain[i] = max(0.3, 1.0 - reduction_amount * self.noise_reduction_strength)
        
        # Apply gain
        stft_reduced = stft_mag * gain[:, np.newaxis] * np.exp(1j * stft_phase)
        
        return librosa.istft(stft_reduced, hop_length=self.HOP_LENGTH, length=len(audio))
    
    def _detect_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect transients for protection."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_env = onset_env / (np.max(onset_env) + 1e-8)
        transient_mask = np.clip(onset_env * 2, 0, 1)
        return median_filter(transient_mask, size=3)
    
    def _process_channel(self, audio_channel: np.ndarray) -> np.ndarray:
        """Process a single audio channel with multiple denoising techniques."""
        
        # Technique 1: Use noisereduce library (very effective for stationary noise like hiss)
        if self._use_learned_profile and self._noise_profile is not None:
            # Use learned noise clip for better targeting
            denoised = nr.reduce_noise(
                y=audio_channel,
                sr=self._sr,
                y_noise=self._noise_profile.noise_clip,
                prop_decrease=self.noise_reduction_strength,
                stationary=True,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
            )
        else:
            # Stationary noise reduction without noise sample
            denoised = nr.reduce_noise(
                y=audio_channel,
                sr=self._sr,
                prop_decrease=self.noise_reduction_strength,
                stationary=True,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
            )
        
        # Technique 2: Additional high-frequency reduction for remaining hiss
        if self.high_freq_emphasis > 1.0:
            denoised = self._apply_high_frequency_reduction(denoised, self._sr)
        
        # Technique 3: Apply maximum dB reduction limit
        # This prevents over-processing
        max_reduction_linear = 10 ** (-self.max_db_reduction / 20)
        
        # Compare original and denoised, limit the reduction
        diff = audio_channel - denoised
        max_allowed_diff = np.abs(audio_channel) * (1 - max_reduction_linear)
        diff_limited = np.clip(diff, -max_allowed_diff, max_allowed_diff)
        denoised = audio_channel - diff_limited
        
        # Technique 4: Transient protection - blend back original at transients
        if self.transient_protection > 0:
            transient_mask = self._detect_transients(audio_channel, self._sr)
            # Interpolate transient mask to match audio length
            transient_mask_full = np.interp(
                np.linspace(0, 1, len(audio_channel)),
                np.linspace(0, 1, len(transient_mask)),
                transient_mask
            )
            # Blend: more original at transients
            blend_factor = transient_mask_full * self.transient_protection
            denoised = denoised * (1 - blend_factor) + audio_channel * blend_factor
        
        # Technique 5: Blend in original signal
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
    
    def save(self, output_path: str, audio: Optional[np.ndarray] = None, format: str = "FLAC") -> str:
        """
        Save processed audio to file.
        
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
        
        return str(output_path)
    
    def update_parameters(
        self,
        max_db_reduction: Optional[float] = None,
        blend_original: Optional[float] = None,
        noise_reduction_strength: Optional[float] = None,
        transient_protection: Optional[float] = None,
        high_freq_emphasis: Optional[float] = None,
        smoothing_factor: Optional[float] = None,
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
        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor
