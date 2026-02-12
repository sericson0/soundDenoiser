"""
Audio playback module for previewing denoised audio.

Provides play, pause, stop, and seek functionality
using sounddevice for low-latency playback.
"""

import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable
from enum import Enum


class PlaybackState(Enum):
    """Audio playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class AudioPlayer:
    """
    Audio player for previewing original and processed audio.
    
    Features:
    - Play/pause/stop controls
    - Seek functionality
    - A/B comparison between original and processed
    - Progress callbacks for UI updates
    """
    
    def __init__(self):
        """Initialize the audio player."""
        self._audio: Optional[np.ndarray] = None
        self._sr: int = 44100
        self._state: PlaybackState = PlaybackState.STOPPED
        self._position: int = 0
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._completion_callback: Optional[Callable[[], None]] = None
        self._loop: bool = False
        
    def load(self, audio: np.ndarray, sample_rate: int):
        """
        Load audio data for playback.
        
        Args:
            audio: Audio data (1D for mono, 2D for stereo with shape (channels, samples))
            sample_rate: Sample rate of the audio
        """
        self.stop()
        
        with self._lock:
            # Ensure proper shape for playback
            if audio.ndim == 1:
                self._audio = audio.reshape(-1, 1)
            elif audio.ndim == 2:
                # Convert from (channels, samples) to (samples, channels)
                if audio.shape[0] <= 2:
                    self._audio = audio.T
                else:
                    self._audio = audio
            else:
                raise ValueError("Audio must be 1D or 2D array")
            
            self._sr = sample_rate
            self._position = 0
            
    def set_progress_callback(self, callback: Optional[Callable[[float], None]]):
        """
        Set callback for progress updates.
        
        Args:
            callback: Function that receives progress as float (0-1)
        """
        self._progress_callback = callback
        
    def set_completion_callback(self, callback: Optional[Callable[[], None]]):
        """
        Set callback for when playback completes.
        
        Args:
            callback: Function called when playback finishes
        """
        self._completion_callback = callback
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """
        Callback for audio output stream.
        
        Args:
            outdata: Output buffer to fill
            frames: Number of frames requested
            time_info: Time information from PortAudio
            status: Status flags
        """
        with self._lock:
            if self._audio is None or self._state != PlaybackState.PLAYING:
                outdata.fill(0)
                return
            
            # Calculate chunk to play
            remaining = frames
            out_offset = 0
            
            while remaining > 0:
                start = self._position
                end = min(start + remaining, len(self._audio))
                chunk_size = end - start
                
                if chunk_size <= 0:
                    if self._loop:
                        # Loop back to beginning
                        self._position = 0
                        continue
                    else:
                        outdata[out_offset:].fill(0)
                        self._state = PlaybackState.STOPPED
                        if self._completion_callback:
                            threading.Thread(target=self._completion_callback).start()
                        return
                
                outdata[out_offset:out_offset + chunk_size] = self._audio[start:end]
                self._position = end
                out_offset += chunk_size
                remaining -= chunk_size
                
                # If we reached the end and looping, wrap around
                if self._position >= len(self._audio) and self._loop:
                    self._position = 0
            
            # Report progress
            if self._progress_callback and len(self._audio) > 0:
                progress = self._position / len(self._audio)
                threading.Thread(target=self._progress_callback, args=(progress,)).start()
    
    def play(self):
        """Start or resume playback."""
        if self._audio is None:
            return
            
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                return
            
            # Reset position if at end
            if self._position >= len(self._audio):
                self._position = 0
            
            self._state = PlaybackState.PLAYING
        
        # Create and start stream if needed
        if self._stream is None or not self._stream.active:
            channels = self._audio.shape[1] if self._audio.ndim == 2 else 1
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=channels,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
    
    def pause(self):
        """Pause playback."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED
    
    def set_loop(self, loop: bool):
        """Enable or disable loop playback."""
        self._loop = loop

    def stop(self):
        """Stop playback and reset position."""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._position = 0
            self._loop = False
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            
        if self._progress_callback:
            self._progress_callback(0.0)
    
    def seek(self, position: float):
        """
        Seek to position in audio.
        
        Args:
            position: Position as fraction (0-1)
        """
        if self._audio is None:
            return
            
        with self._lock:
            self._position = int(position * len(self._audio))
            self._position = max(0, min(self._position, len(self._audio) - 1))
    
    def get_position(self) -> float:
        """
        Get current playback position.
        
        Returns:
            Position as fraction (0-1)
        """
        if self._audio is None or len(self._audio) == 0:
            return 0.0
        
        with self._lock:
            return self._position / len(self._audio)
    
    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state
    
    def get_duration(self) -> float:
        """
        Get duration of loaded audio in seconds.
        
        Returns:
            Duration in seconds
        """
        if self._audio is None:
            return 0.0
        return len(self._audio) / self._sr
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._state == PlaybackState.PLAYING
    
    def is_loaded(self) -> bool:
        """Check if audio is loaded."""
        return self._audio is not None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self._audio = None
