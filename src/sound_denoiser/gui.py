"""
Modern GUI for the Sound Denoiser application.

Provides an intuitive interface for loading audio, adjusting
denoising parameters, previewing results, and saving output.
Includes noise profile learning with visual region selection.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
import librosa
import librosa.display
import threading
from pathlib import Path
from typing import Optional, Tuple
import queue

# Try to import windnd for drag and drop support (Windows)
WINDND_AVAILABLE = False
try:
    import windnd
    WINDND_AVAILABLE = True
except ImportError:
    pass

# Handle imports for both module and direct execution
try:
    from .denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
    from .audio_player import AudioPlayer, PlaybackState
except ImportError:
    from denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
    from audio_player import AudioPlayer, PlaybackState


# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class WaveformDisplay(ctk.CTkFrame):
    """Widget for displaying audio waveforms/spectrograms with click-to-seek and noise region selection."""
    
    def __init__(self, master, on_region_select=None, on_seek=None, waveform_color='#00d9ff', **kwargs):
        super().__init__(master, **kwargs)
        
        self.on_region_select = on_region_select
        self.on_seek = on_seek  # Callback for click-to-seek
        self._duration = 0.0
        self._selection_enabled = False
        self._selected_region: Optional[Tuple[float, float]] = None
        self._waveform_color = waveform_color
        self._current_position = 0.0
        self._view_mode = "waveform"  # "waveform" or "spectrogram"
        
        # Store waveform data for redrawing
        self._audio_data = None
        self._sample_rate = 44100
        self._title = "Waveform"
        
        # View toggle frame
        toggle_frame = ctk.CTkFrame(self, fg_color="transparent", height=28)
        toggle_frame.pack(fill="x", padx=5, pady=(5, 0))
        
        # View mode segmented button
        self.view_toggle = ctk.CTkSegmentedButton(
            toggle_frame,
            values=["Waveform", "Spectrogram"],
            command=self._on_view_change,
            font=ctk.CTkFont(size=11),
            height=24,
            fg_color="#252535",
            selected_color="#6c3483",
            selected_hover_color="#8e44ad",
            unselected_color="#1a1a2e",
            unselected_hover_color="#252535"
        )
        self.view_toggle.set("Waveform")
        self.view_toggle.pack(side="left", padx=2)
        
        # Separator
        sep1 = ctk.CTkFrame(toggle_frame, width=1, height=20, fg_color="#444444")
        sep1.pack(side="left", padx=8)
        
        # Selection mode toggle
        self._selection_mode_enabled = False
        self.selection_mode_btn = ctk.CTkButton(
            toggle_frame,
            text="[+] Select",
            command=self._toggle_selection_mode,
            font=ctk.CTkFont(size=10),
            width=70,
            height=24,
            fg_color="#1a1a2e",
            hover_color="#252535",
            border_width=1,
            border_color="#444444"
        )
        self.selection_mode_btn.pack(side="left", padx=2)
        
        # Separator
        sep2 = ctk.CTkFrame(toggle_frame, width=1, height=20, fg_color="#444444")
        sep2.pack(side="left", padx=8)
        
        # Zoom controls
        self._zoom_level = 1.0
        self._zoom_offset = 0.0  # Start position as fraction of audio
        
        self.zoom_out_btn = ctk.CTkButton(
            toggle_frame,
            text="-",
            command=self._zoom_out,
            font=ctk.CTkFont(size=14, weight="bold"),
            width=28,
            height=24,
            fg_color="#1a1a2e",
            hover_color="#252535"
        )
        self.zoom_out_btn.pack(side="left", padx=1)
        
        self.zoom_label = ctk.CTkLabel(
            toggle_frame,
            text="1x",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=30
        )
        self.zoom_label.pack(side="left", padx=1)
        
        self.zoom_in_btn = ctk.CTkButton(
            toggle_frame,
            text="+",
            command=self._zoom_in,
            font=ctk.CTkFont(size=14, weight="bold"),
            width=28,
            height=24,
            fg_color="#1a1a2e",
            hover_color="#252535"
        )
        self.zoom_in_btn.pack(side="left", padx=1)
        
        self.zoom_reset_btn = ctk.CTkButton(
            toggle_frame,
            text="R",
            command=self._zoom_reset,
            font=ctk.CTkFont(size=12),
            width=28,
            height=24,
            fg_color="#1a1a2e",
            hover_color="#252535"
        )
        self.zoom_reset_btn.pack(side="left", padx=1)
        
        # Time display label
        self.time_label = ctk.CTkLabel(
            toggle_frame,
            text="0:00 / 0:00",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.time_label.pack(side="right", padx=5)
        
        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 2.2), dpi=100, facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a2e')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.18)
        
        # Style the axes
        self.ax.tick_params(colors='#888888', labelsize=8)
        self.ax.spines['bottom'].set_color('#444444')
        self.ax.spines['left'].set_color('#444444')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bind click events for seeking
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_drag)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._is_dragging = False
        self._drag_seeking = False
        
        # Playback update throttling for performance
        self._last_playhead_update = 0
        
        # Playhead line
        self.playhead = None
        
        # Played area shading
        self.played_area = None
        
        # Selection rectangle for noise region
        self.selection_rect = None
        
        # Span selector for region selection
        self.span_selector = None
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS."""
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    def _update_time_label(self, position: float = None):
        """Update the time display label."""
        if position is None:
            position = self._current_position
        current_time = position * self._duration
        self.time_label.configure(
            text=f"{self._format_time(current_time)} / {self._format_time(self._duration)}"
        )
    
    def _on_view_change(self, value: str):
        """Handle view mode change."""
        new_mode = "waveform" if value == "Waveform" else "spectrogram"
        if new_mode != self._view_mode:
            self._view_mode = new_mode
            self._redraw_display()
    
    def _toggle_selection_mode(self):
        """Toggle between seek mode and selection mode."""
        self._selection_mode_enabled = not self._selection_mode_enabled
        self.enable_selection(self._selection_mode_enabled)
        
        if self._selection_mode_enabled:
            self.selection_mode_btn.configure(
                fg_color="#ff6b6b",
                hover_color="#ff8f8f",
                text="[*] Select"
            )
        else:
            self.selection_mode_btn.configure(
                fg_color="#1a1a2e",
                hover_color="#252535",
                text="[+] Select"
            )
    
    def _zoom_in(self):
        """Zoom in on the waveform."""
        if self._zoom_level < 16.0:  # Max 16x zoom
            self._zoom_level *= 2.0
            self._update_zoom()
    
    def _zoom_out(self):
        """Zoom out on the waveform."""
        if self._zoom_level > 1.0:
            self._zoom_level /= 2.0
            # Adjust offset to keep center in view
            self._zoom_offset = max(0, min(self._zoom_offset, 1.0 - 1.0/self._zoom_level))
            self._update_zoom()
    
    def _zoom_reset(self):
        """Reset zoom to show full waveform."""
        self._zoom_level = 1.0
        self._zoom_offset = 0.0
        self._update_zoom()
    
    def _update_zoom(self):
        """Update the zoom level display and redraw."""
        self.zoom_label.configure(text=f"{self._zoom_level:.0f}x")
        self._redraw_display()
    
    def get_visible_time_range(self) -> tuple:
        """Get the currently visible time range based on zoom."""
        visible_duration = self._duration / self._zoom_level
        start_time = self._zoom_offset * self._duration
        end_time = start_time + visible_duration
        return (start_time, end_time)
    
    def _redraw_display(self):
        """Redraw the current display (waveform or spectrogram)."""
        if self._audio_data is None:
            return
        
        if self._view_mode == "waveform":
            self._plot_waveform_internal()
        else:
            self._plot_spectrogram_internal()
    
    def _plot_waveform_internal(self):
        """Internal method to plot waveform with zoom support."""
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        self.selection_rect = None
        self.playhead = None
        self.played_area = None
        
        audio = self._audio_data
        sr = self._sample_rate
        
        # Calculate visible time range based on zoom
        visible_start, visible_end = self.get_visible_time_range()
        start_sample = int(visible_start * sr)
        end_sample = int(visible_end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        # Extract visible audio data
        visible_audio = audio[start_sample:end_sample]
        visible_times = np.linspace(visible_start, visible_end, len(visible_audio))
        
        # Downsample for faster plotting
        if len(visible_audio) > 10000:
            step = len(visible_audio) // 10000
            times_display = visible_times[::step]
            audio_display = visible_audio[::step]
        else:
            times_display = visible_times
            audio_display = visible_audio
        
        self.ax.fill_between(times_display, audio_display, alpha=0.6, color=self._waveform_color)
        self.ax.plot(times_display, audio_display, color=self._waveform_color, linewidth=0.5, alpha=0.8)
        self.ax.set_xlim(visible_start, visible_end)
        self.ax.set_ylim(-1, 1)
        
        # Restore selection if any and it's visible
        if self._selected_region:
            sel_start, sel_end = self._selected_region
            if sel_end > visible_start and sel_start < visible_end:
                self._draw_selection_rect(max(sel_start, visible_start), min(sel_end, visible_end))
        
        # Show zoom indicator in title
        zoom_text = f" [{self._zoom_level:.0f}x zoom]" if self._zoom_level > 1.0 else ""
        self.ax.set_title(self._title + zoom_text, color='#ffffff', fontsize=10, fontweight='bold')
        self.ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        self.ax.tick_params(colors='#888888', labelsize=8)
        
        self.canvas.draw()
        
        # Restore playhead
        if self._current_position > 0:
            self.update_playhead(self._current_position)
    
    def _plot_spectrogram_internal(self):
        """Internal method to plot spectrogram with zoom support and aggressive performance optimization."""
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        self.selection_rect = None
        self.playhead = None
        self.played_area = None
        
        audio = self._audio_data
        sr = self._sample_rate
        
        # Calculate visible time range based on zoom
        visible_start, visible_end = self.get_visible_time_range()
        visible_duration = visible_end - visible_start
        start_sample = int(visible_start * sr)
        end_sample = int(visible_end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        # Extract visible audio with some padding for smoother edges
        pad_samples = 512
        padded_start = max(0, start_sample - pad_samples)
        padded_end = min(len(audio), end_sample + pad_samples)
        visible_audio = audio[padded_start:padded_end]
        
        # AGGRESSIVE performance optimization - much larger hop lengths
        # Target approximately 500-800 time frames max for smooth display
        target_frames = 600
        audio_len = len(visible_audio)
        
        # Calculate hop length to achieve target frames
        ideal_hop = max(512, audio_len // target_frames)
        
        # Round to power of 2 for efficiency
        hop_length = 512
        for h in [512, 1024, 2048, 4096]:
            if h >= ideal_hop:
                hop_length = h
                break
        else:
            hop_length = 4096
        
        # FFT size - use smaller for better performance
        n_fft = min(2048, hop_length * 4)
        
        # Downsample audio for very long segments (additional speedup)
        # BUT limit downsampling to ensure we can still display up to 12kHz
        # (need effective_sr >= 24000 Hz for 12kHz Nyquist limit)
        max_downsample = max(1, sr // 24000)  # Ensure effective_sr >= 24kHz
        
        if audio_len > 500000:  # > ~11 seconds at 44.1kHz
            downsample_factor = min(max_downsample, audio_len // 200000)
            if downsample_factor > 1:
                visible_audio = visible_audio[::downsample_factor]
                effective_sr = sr // downsample_factor
                hop_length = max(256, hop_length // downsample_factor)
                n_fft = min(1024, hop_length * 4)
            else:
                effective_sr = sr
        else:
            effective_sr = sr
        
        # Compute spectrogram for visible region only
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(visible_audio, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        # Calculate time offset for correct positioning
        time_offset = padded_start / sr
        time_end = padded_end / sr
        
        # Plot mel spectrogram - y_axis='mel' uses mel scale (non-linear)
        img = librosa.display.specshow(
            D, sr=effective_sr, hop_length=hop_length, x_axis='time', y_axis='mel',
            ax=self.ax, cmap='magma', x_coords=np.linspace(time_offset, time_end, D.shape[1]),
            fmax=max_freq
        )
        
        # Set axis limits - X is zoom-dependent
        self.ax.set_xlim(visible_start, visible_end)
        
        # Restore selection if any and it's visible
        if self._selected_region:
            sel_start, sel_end = self._selected_region
            if sel_end > visible_start and sel_start < visible_end:
                self._draw_selection_rect(max(sel_start, visible_start), min(sel_end, visible_end))
        
        # Show zoom indicator in title
        zoom_text = f" [{self._zoom_level:.0f}x zoom]" if self._zoom_level > 1.0 else ""
        self.ax.set_title(self._title.replace("Waveform", "Spectrogram") + zoom_text, color='#ffffff', fontsize=10, fontweight='bold')
        self.ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        self.ax.set_ylabel('Frequency (Hz)', color='#888888', fontsize=8)
        self.ax.tick_params(colors='#888888', labelsize=8)
        
        self.canvas.draw()
        
        # Restore playhead
        if self._current_position > 0:
            self.update_playhead(self._current_position)
    
    def _on_canvas_click(self, event):
        """Handle mouse click on canvas for seeking."""
        if event.inaxes != self.ax or self._duration <= 0:
            return
        
        # Check if we're in selection mode
        if self._selection_enabled:
            return  # Let span selector handle it
        
        # Convert x position to time fraction
        x_pos = event.xdata
        if x_pos is not None:
            position = max(0, min(1, x_pos / self._duration))
            self._drag_seeking = True
            self._current_position = position
            self.update_playhead(position)
            self._update_time_label(position)
            
            if self.on_seek:
                self.on_seek(position)
    
    def _on_canvas_drag(self, event):
        """Handle mouse drag on canvas for seeking."""
        if not self._drag_seeking or event.inaxes != self.ax or self._duration <= 0:
            return
        
        x_pos = event.xdata
        if x_pos is not None:
            position = max(0, min(1, x_pos / self._duration))
            self._current_position = position
            self.update_playhead(position)
            self._update_time_label(position)
            
            if self.on_seek:
                self.on_seek(position)
    
    def _on_canvas_release(self, event):
        """Handle mouse release after seeking."""
        self._drag_seeking = False
    
    def _on_scroll(self, event):
        """Handle scroll wheel for panning when zoomed."""
        if self._zoom_level <= 1.0:
            return  # No panning needed at 1x zoom
        
        # Calculate pan step (10% of visible area)
        pan_step = 0.1 / self._zoom_level
        
        if event.button == 'up':
            # Pan left
            self._zoom_offset = max(0, self._zoom_offset - pan_step)
        elif event.button == 'down':
            # Pan right
            max_offset = 1.0 - 1.0 / self._zoom_level
            self._zoom_offset = min(max_offset, self._zoom_offset + pan_step)
        
        self._redraw_display()
        
    def enable_selection(self, enable: bool = True):
        """Enable or disable region selection mode."""
        self._selection_enabled = enable
        
        if enable and self._duration > 0:
            # Create span selector
            self.span_selector = SpanSelector(
                self.ax,
                self._on_span_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='#ff6b6b'),
                interactive=True,
                drag_from_anywhere=True,
            )
        elif self.span_selector:
            self.span_selector.set_active(False)
            self.span_selector = None
            
    def _on_span_select(self, xmin, xmax):
        """Handle span selection."""
        if not self._selection_enabled:
            return
            
        # Clamp to valid range
        xmin = max(0, xmin)
        xmax = min(self._duration, xmax)
        
        if xmax - xmin < 0.1:  # Minimum 100ms selection
            return
            
        self._selected_region = (xmin, xmax)
        self._draw_selection_rect(xmin, xmax)
        
        if self.on_region_select:
            self.on_region_select(xmin, xmax)
            
    def _draw_selection_rect(self, start: float, end: float):
        """Draw selection rectangle on waveform."""
        # Remove existing rectangle
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            
        # Draw new rectangle
        self.selection_rect = self.ax.axvspan(
            start, end,
            alpha=0.3,
            color='#ff6b6b',
            label='Noise Region'
        )
        self.canvas.draw_idle()
        
    def set_noise_region(self, start: float, end: float):
        """Set and display a noise region."""
        self._selected_region = (start, end)
        self._draw_selection_rect(start, end)
        
    def clear_noise_region(self):
        """Clear the noise region selection."""
        self._selected_region = None
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw_idle()
        
    def plot_waveform(
        self,
        audio: Optional[np.ndarray],
        sr: int,
        title: str = "Waveform",
        color: str = '#00d9ff'
    ):
        """
        Plot audio waveform or spectrogram based on current view mode.
        
        Args:
            audio: Audio data
            sr: Sample rate
            title: Plot title
            color: Waveform color
        """
        self._waveform_color = color
        self._title = title
        self._sample_rate = sr
        self._current_position = 0.0
        
        if audio is not None:
            # Downsample for display if needed
            if len(audio.shape) > 1:
                audio = audio[0] if audio.shape[0] <= 2 else audio[:, 0]
            
            # Store for later use
            self._audio_data = audio.copy()
            self._duration = len(audio) / sr
        else:
            self._duration = 0.0
            self._audio_data = None
        
        # Update time label
        self._update_time_label(0)
        
        # Draw based on current view mode
        self._redraw_display()
    
    def update_playhead(self, position: float, skip_draw: bool = False):
        """
        Update playhead position and played area shading.
        
        Args:
            position: Position as fraction (0-1)
            skip_draw: If True, only update time label without redrawing canvas
        """
        if self._duration <= 0:
            return
            
        time_pos = position * self._duration
        self._current_position = position
        
        # Update time label (always do this)
        self._update_time_label(position)
        
        if skip_draw:
            return
        
        # Get visible time range for zoom-aware rendering
        visible_start, visible_end = self.get_visible_time_range()
        
        # Auto-scroll when zoomed and playhead reaches end of visible range
        if self._zoom_level > 1.0 and time_pos >= visible_end - 0.1:
            # Scroll so playhead is at left edge
            total_duration = self._duration
            visible_duration = visible_end - visible_start
            new_offset = min(time_pos / total_duration, 1.0 - (visible_duration / total_duration))
            new_offset = max(0.0, new_offset)
            
            if abs(new_offset - self._zoom_offset) > 0.01:
                self._zoom_offset = new_offset
                # Redraw the plot with new offset
                if self._view_mode == "spectrogram":
                    self._plot_spectrogram_internal()
                else:
                    self._plot_waveform_internal()
                # Recalculate visible range after redraw
                visible_start, visible_end = self.get_visible_time_range()
        
        # Remove old playhead
        if self.playhead is not None:
            try:
                self.playhead.remove()
            except:
                pass
            self.playhead = None
        
        # Remove old played area (only for waveform mode to reduce flicker in spectrogram)
        if self.played_area is not None:
            try:
                self.played_area.remove()
            except:
                pass
            self.played_area = None
        
        # Only draw playhead if it's in the visible range
        playhead_visible = visible_start <= time_pos <= visible_end
        
        # Draw played area (grayed out) - clip to visible range (waveform only)
        if time_pos > visible_start and self._view_mode != "spectrogram":
            played_end = min(time_pos, visible_end)
            played_start = visible_start
            self.played_area = self.ax.axvspan(
                played_start, played_end,
                alpha=0.3,
                color='#333333',
                zorder=1
            )
        
        # Draw playhead line if visible
        if position > 0 and playhead_visible:
            self.playhead = self.ax.axvline(
                x=time_pos,
                color='#ffffff',
                linewidth=2,
                alpha=0.9,
                zorder=10
            )
        
        self.canvas.draw_idle()
    
    def reset_playhead(self):
        """Reset playhead to start."""
        self._current_position = 0.0
        self._update_time_label(0)
        if self.playhead is not None:
            try:
                self.playhead.remove()
            except:
                pass
            self.playhead = None
        if self.played_area is not None:
            try:
                self.played_area.remove()
            except:
                pass
            self.played_area = None
        self.canvas.draw_idle()


class SeekBar(ctk.CTkFrame):
    """Seekable progress bar for audio playback with time display."""
    
    def __init__(
        self,
        master,
        color: str = "#00d9ff",
        on_seek=None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.on_seek = on_seek
        self._duration = 0.0
        self._is_seeking = False
        
        # Time label (current)
        self.time_current = ctk.CTkLabel(
            self,
            text="0:00",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=40
        )
        self.time_current.pack(side="left", padx=(0, 5))
        
        # Seek slider
        self.slider = ctk.CTkSlider(
            self,
            from_=0,
            to=1,
            number_of_steps=1000,
            command=self._on_slider_change,
            progress_color=color,
            button_color=color,
            button_hover_color=color,
            height=12
        )
        self.slider.set(0)
        self.slider.pack(side="left", fill="x", expand=True)
        
        # Bind mouse events for seeking
        self.slider.bind("<ButtonPress-1>", self._on_seek_start)
        self.slider.bind("<ButtonRelease-1>", self._on_seek_end)
        
        # Time label (duration)
        self.time_duration = ctk.CTkLabel(
            self,
            text="0:00",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=40
        )
        self.time_duration.pack(side="left", padx=(5, 0))
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    
    def _on_slider_change(self, value):
        """Handle slider value change."""
        if self._is_seeking and self.on_seek:
            self.on_seek(value)
        # Update current time display
        current_time = value * self._duration
        self.time_current.configure(text=self._format_time(current_time))
            
    def _on_seek_start(self, event):
        """Handle seek start."""
        self._is_seeking = True
        
    def _on_seek_end(self, event):
        """Handle seek end."""
        if self._is_seeking and self.on_seek:
            self.on_seek(self.slider.get())
        self._is_seeking = False
        
    def set_duration(self, duration: float):
        """Set the total duration."""
        self._duration = duration
        self.time_duration.configure(text=self._format_time(duration))
        
    def set_position(self, position: float):
        """Set position (0-1) without triggering seek callback."""
        if not self._is_seeking:
            self.slider.set(position)
            current_time = position * self._duration
            self.time_current.configure(text=self._format_time(current_time))
            
    def get_position(self) -> float:
        """Get current position (0-1)."""
        return self.slider.get()


class ParameterSlider(ctk.CTkFrame):
    """Custom parameter slider with label and value display."""
    
    def __init__(
        self,
        master,
        label: str,
        from_: float,
        to: float,
        default: float,
        unit: str = "",
        command=None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.unit = unit
        self.command = command
        
        # Label
        self.label = ctk.CTkLabel(
            self,
            text=label,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cccccc"
        )
        self.label.pack(anchor="w")
        
        # Slider frame
        slider_frame = ctk.CTkFrame(self, fg_color="transparent")
        slider_frame.pack(fill="x", pady=(2, 0))
        
        # Value label
        self.value_label = ctk.CTkLabel(
            slider_frame,
            text=f"{default:.1f}{unit}",
            font=ctk.CTkFont(size=11),
            text_color="#00d9ff",
            width=60
        )
        self.value_label.pack(side="right", padx=(5, 0))
        
        # Slider
        self.slider = ctk.CTkSlider(
            slider_frame,
            from_=from_,
            to=to,
            number_of_steps=100,
            command=self._on_change,
            progress_color="#00d9ff",
            button_color="#00d9ff",
            button_hover_color="#00b8d4"
        )
        self.slider.set(default)
        self.slider.pack(side="left", fill="x", expand=True)
        
    def _on_change(self, value):
        """Handle slider change."""
        self.value_label.configure(text=f"{value:.1f}{self.unit}")
        if self.command:
            self.command(value)
            
    def get(self) -> float:
        """Get current value."""
        return self.slider.get()
    
    def set(self, value: float):
        """Set slider value."""
        self.slider.set(value)
        self.value_label.configure(text=f"{value:.1f}{self.unit}")


class NoiseProfilePanel(ctk.CTkFrame):
    """Panel for noise profile learning controls."""
    
    def __init__(
        self,
        master,
        on_learn_manual,
        on_learn_auto,
        on_clear,
        on_toggle_use,
        **kwargs
    ):
        super().__init__(master, fg_color="#151525", corner_radius=10, **kwargs)
        
        self.on_learn_manual = on_learn_manual
        self.on_learn_auto = on_learn_auto
        self.on_clear = on_clear
        self.on_toggle_use = on_toggle_use
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="🎯 Noise Profile Learning",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ff6b6b"
        )
        title.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Description
        desc = ctk.CTkLabel(
            self,
            text="Select a region with only noise (hiss) for better results",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            wraplength=200
        )
        desc.pack(padx=10, anchor="w")
        
        # Separator
        sep = ctk.CTkFrame(self, height=1, fg_color="#333333")
        sep.pack(fill="x", padx=10, pady=10)
        
        # Status indicator
        self.status_frame = ctk.CTkFrame(self, fg_color="#1a2a3a", corner_radius=6)
        self.status_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="⚪ No noise profile learned",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.status_label.pack(pady=8, padx=10, anchor="w")
        
        self.region_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#666666"
        )
        self.region_label.pack(pady=(0, 8), padx=10, anchor="w")
        
        # Use profile toggle
        self.use_profile_var = ctk.BooleanVar(value=False)
        self.use_profile_switch = ctk.CTkSwitch(
            self,
            text="Use learned profile",
            variable=self.use_profile_var,
            command=self._on_toggle,
            font=ctk.CTkFont(size=11),
            progress_color="#ff6b6b",
            button_color="#ff6b6b",
            button_hover_color="#ff8f8f",
            state="disabled"
        )
        self.use_profile_switch.pack(pady=(0, 10), padx=10, anchor="w")
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Auto-detect button
        self.auto_btn = ctk.CTkButton(
            btn_frame,
            text="🔍 Auto Detect",
            command=self.on_learn_auto,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#2d5a27",
            hover_color="#3d7a37",
            height=32,
            corner_radius=6,
            state="disabled"
        )
        self.auto_btn.pack(fill="x", pady=(0, 5))
        
        # Manual selection instruction
        self.select_label = ctk.CTkLabel(
            btn_frame,
            text="Or click & drag on waveform to select",
            font=ctk.CTkFont(size=10),
            text_color="#666666"
        )
        self.select_label.pack(pady=(5, 5))
        
        # Learn from selection button
        self.learn_btn = ctk.CTkButton(
            btn_frame,
            text="📝 Learn from Selection",
            command=self.on_learn_manual,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1a5276",
            hover_color="#2471a3",
            height=32,
            corner_radius=6,
            state="disabled"
        )
        self.learn_btn.pack(fill="x", pady=(0, 5))
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="✖ Clear Profile",
            command=self._on_clear,
            font=ctk.CTkFont(size=11),
            fg_color="#555555",
            hover_color="#777777",
            height=28,
            corner_radius=6,
            state="disabled"
        )
        self.clear_btn.pack(fill="x")
        
    def _on_toggle(self):
        """Handle use profile toggle."""
        self.on_toggle_use(self.use_profile_var.get())
        
    def _on_clear(self):
        """Handle clear button."""
        self.on_clear()
        self.update_status(None)
        
    def enable_controls(self, enable: bool = True):
        """Enable or disable controls."""
        state = "normal" if enable else "disabled"
        self.auto_btn.configure(state=state)
        
    def enable_learn_button(self, enable: bool = True):
        """Enable or disable the learn button."""
        state = "normal" if enable else "disabled"
        self.learn_btn.configure(state=state)
        
    def update_status(self, profile: Optional[NoiseProfile], region: Optional[Tuple[float, float]] = None):
        """Update the status display."""
        if profile is not None:
            self.status_label.configure(
                text="🟢 Noise profile learned",
                text_color="#4ade80"
            )
            if region:
                self.region_label.configure(
                    text=f"Region: {region[0]:.2f}s - {region[1]:.2f}s ({region[1]-region[0]:.2f}s)"
                )
            self.use_profile_switch.configure(state="normal")
            self.use_profile_var.set(True)
            self.clear_btn.configure(state="normal")
            self._on_toggle()  # Trigger callback
        else:
            self.status_label.configure(
                text="⚪ No noise profile learned",
                text_color="#888888"
            )
            self.region_label.configure(text="")
            self.use_profile_switch.configure(state="disabled")
            self.use_profile_var.set(False)
            self.clear_btn.configure(state="disabled")


class SoundDenoiserApp(ctk.CTk):
    """Main application window for Sound Denoiser."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Sound Denoiser - Hiss Removal")
        self.geometry("1280x850")
        self.minsize(1100, 750)
        
        # Initialize components
        self.denoiser = AudioDenoiser()
        self.player_original = AudioPlayer()
        self.player_processed = AudioPlayer()
        self.current_player: Optional[AudioPlayer] = None
        
        # State
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.is_processing = False
        self.processing_queue = queue.Queue()
        self.selected_noise_region: Optional[Tuple[float, float]] = None
        
        # Build UI
        self._create_ui()
        
        # Setup drag and drop if available
        self._setup_drag_and_drop()
        
        # Start update loop
        self._update_ui()
    
    def _setup_drag_and_drop(self):
        """Setup drag and drop file loading using windnd."""
        # Initialize the dropped file holder (raw data from windnd)
        self._pending_drop_files = None
        
        if WINDND_AVAILABLE:
            try:
                # Hook drag and drop to the main window
                windnd.hook_dropfiles(self, func=self._on_drop_files)
            except Exception as e:
                print(f"Drag and drop setup failed: {e}")
    
    def _on_drop_files(self, file_list):
        """Handle dropped files from windnd (called from a different thread).
        
        This runs in a non-GIL context from Windows COM, so we store raw data
        and let the main thread do all processing.
        """
        # Store the raw file list - main thread will process it
        # Keep this as minimal as possible to avoid GIL issues
        self._pending_drop_files = file_list
    
    def _process_dropped_files(self, file_list):
        """Process dropped files (runs in main thread)."""
        if not file_list:
            return
        
        audio_extensions = {'.wav', '.mp3', '.flac', '.aif', '.aiff', '.m4a', '.ogg'}
        
        for file_path in file_list:
            try:
                # Decode bytes to string if necessary
                if isinstance(file_path, bytes):
                    path_str = file_path.decode('utf-8', errors='replace')
                else:
                    path_str = str(file_path)
                
                path = Path(path_str)
                
                if path.suffix.lower() in audio_extensions and path.exists():
                    self._load_audio_file(str(path))
                    return  # Only load the first valid audio file
            except Exception as e:
                print(f"Error processing dropped file: {e}")
                continue
        
        # No valid audio file found
        messagebox.showwarning(
            "Invalid File",
            f"Please drop an audio file.\nSupported formats: {', '.join(audio_extensions)}"
        )
        
    def _create_ui(self):
        """Create the user interface."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self._create_header()
        
        # Main content
        self._create_main_content()
        
        # Status bar
        self._create_status_bar()
        
    def _create_header(self):
        """Create header with title and file controls."""
        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_columnconfigure(1, weight=1)
        
        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        title = ctk.CTkLabel(
            title_frame,
            text="🎵 Sound Denoiser",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#00d9ff"
        )
        title.pack(side="left")
        
        subtitle = ctk.CTkLabel(
            title_frame,
            text="  |  Hiss Removal for Vintage Recordings",
            font=ctk.CTkFont(size=14),
            text_color="#888888"
        )
        subtitle.pack(side="left", padx=(10, 0))
        
        # File controls
        file_frame = ctk.CTkFrame(header, fg_color="transparent")
        file_frame.grid(row=0, column=1, padx=20, pady=15, sticky="e")
        
        self.load_btn = ctk.CTkButton(
            file_frame,
            text="📂 Load Audio",
            command=self._load_audio,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d5a27",
            hover_color="#3d7a37",
            height=36,
            corner_radius=8
        )
        self.load_btn.pack(side="left", padx=5)
        
        self.save_btn = ctk.CTkButton(
            file_frame,
            text="💾 Save Output",
            command=self._save_audio,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1a5276",
            hover_color="#2471a3",
            height=36,
            corner_radius=8,
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)
        
    def _create_main_content(self):
        """Create main content area."""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Waveforms and playback
        self._create_waveform_panel(main_frame)
        
        # Right panel - Parameters
        self._create_parameter_panel(main_frame)
        
    def _create_waveform_panel(self, parent):
        """Create waveform display and playback controls."""
        left_panel = ctk.CTkFrame(parent, fg_color="#0f0f1a", corner_radius=12)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Original waveform
        original_frame = ctk.CTkFrame(left_panel, fg_color="#151525", corner_radius=10)
        original_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        original_frame.grid_rowconfigure(0, weight=1)
        original_frame.grid_columnconfigure(0, weight=1)
        
        self.waveform_original = WaveformDisplay(
            original_frame,
            on_region_select=self._on_noise_region_selected,
            on_seek=lambda pos: self._on_seek("original", pos)
        )
        self.waveform_original.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.waveform_original.plot_waveform(None, 44100, "Original Audio (Click to seek, drag to select noise)", "#ff9f43")
        
        # Original playback controls (compact row)
        orig_controls = ctk.CTkFrame(original_frame, fg_color="transparent")
        orig_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        self.play_orig_btn = ctk.CTkButton(
            orig_controls,
            text="▶",
            width=50,
            height=28,
            command=lambda: self._toggle_play("original"),
            fg_color="#ff9f43",
            hover_color="#f39c12",
            text_color="#000000",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.play_orig_btn.pack(side="left", padx=2)
        
        self.stop_orig_btn = ctk.CTkButton(
            orig_controls,
            text="⏹",
            width=50,
            height=28,
            command=lambda: self._stop_play("original"),
            fg_color="#555555",
            hover_color="#777777",
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.stop_orig_btn.pack(side="left", padx=2)
        
        # Label hint
        hint_label = ctk.CTkLabel(
            orig_controls,
            text="Click on waveform to seek",
            font=ctk.CTkFont(size=10),
            text_color="#666666"
        )
        hint_label.pack(side="right", padx=5)
        
        # Processed waveform
        processed_frame = ctk.CTkFrame(left_panel, fg_color="#151525", corner_radius=10)
        processed_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        processed_frame.grid_rowconfigure(0, weight=1)
        processed_frame.grid_columnconfigure(0, weight=1)
        
        self.waveform_processed = WaveformDisplay(
            processed_frame,
            on_seek=lambda pos: self._on_seek("processed", pos)
        )
        self.waveform_processed.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.waveform_processed.plot_waveform(None, 44100, "Processed Audio (Denoised)", "#00d9ff")
        
        # Processed playback controls (compact row)
        proc_controls = ctk.CTkFrame(processed_frame, fg_color="transparent")
        proc_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        self.play_proc_btn = ctk.CTkButton(
            proc_controls,
            text="▶",
            width=50,
            height=28,
            command=lambda: self._toggle_play("processed"),
            fg_color="#00d9ff",
            hover_color="#00b8d4",
            text_color="#000000",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.play_proc_btn.pack(side="left", padx=2)
        
        self.stop_proc_btn = ctk.CTkButton(
            proc_controls,
            text="⏹",
            width=50,
            height=28,
            command=lambda: self._stop_play("processed"),
            fg_color="#555555",
            hover_color="#777777",
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.stop_proc_btn.pack(side="left", padx=2)
        
        # Sync button - syncs processed position to original position
        self.sync_btn = ctk.CTkButton(
            proc_controls,
            text="🔗 Sync",
            width=70,
            height=28,
            command=self._sync_to_original,
            fg_color="#6c3483",
            hover_color="#8e44ad",
            font=ctk.CTkFont(size=12),
            state="disabled"
        )
        self.sync_btn.pack(side="left", padx=5)
        
    def _create_parameter_panel(self, parent):
        """Create parameter controls panel."""
        right_panel = ctk.CTkFrame(parent, fg_color="#0f0f1a", corner_radius=12)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # Create scrollable content
        scroll_frame = ctk.CTkScrollableFrame(
            right_panel,
            fg_color="transparent",
            scrollbar_button_color="#333333"
        )
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Noise Profile Section
        self.noise_profile_panel = NoiseProfilePanel(
            scroll_frame,
            on_learn_manual=self._learn_noise_profile_manual,
            on_learn_auto=self._learn_noise_profile_auto,
            on_clear=self._clear_noise_profile,
            on_toggle_use=self._toggle_use_profile,
        )
        self.noise_profile_panel.pack(fill="x", padx=5, pady=(5, 10))
        
        # Parameters Section
        params_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=10)
        params_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        # Title
        params_title = ctk.CTkLabel(
            params_frame,
            text="⚙️ Denoising Parameters",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#ffffff"
        )
        params_title.pack(pady=(10, 5), padx=10, anchor="w")
        
        # Separator
        sep = ctk.CTkFrame(params_frame, height=1, fg_color="#333333")
        sep.pack(fill="x", padx=10, pady=(0, 10))
        
        # Parameters
        params_inner = ctk.CTkFrame(params_frame, fg_color="transparent")
        params_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        # Denoising Method Selector
        method_frame = ctk.CTkFrame(params_inner, fg_color="transparent")
        method_frame.pack(fill="x", pady=(0, 12))
        
        method_label = ctk.CTkLabel(
            method_frame,
            text="Denoising Method",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cccccc"
        )
        method_label.pack(anchor="w")
        
        # Method names for display
        self._method_names = {
            "Spectral Subtraction": DenoiseMethod.SPECTRAL_SUBTRACTION,
            "Wiener Filter": DenoiseMethod.WIENER,
            "Multi-Band Adaptive": DenoiseMethod.MULTIBAND,
            "Combined (All Methods)": DenoiseMethod.COMBINED,
            "Shellac/78rpm (Hiss+Groove)": DenoiseMethod.SHELLAC,
            "NoiseReduce (Legacy)": DenoiseMethod.NOISEREDUCE,
        }
        
        self.method_dropdown = ctk.CTkOptionMenu(
            method_frame,
            values=list(self._method_names.keys()),
            command=self._on_method_change,
            fg_color="#252535",
            button_color="#6c3483",
            button_hover_color="#8e44ad",
            dropdown_fg_color="#1a1a2e",
            dropdown_hover_color="#252535",
            font=ctk.CTkFont(size=11),
            width=200
        )
        self.method_dropdown.set("Spectral Subtraction")
        self.method_dropdown.pack(anchor="w", pady=(5, 0))
        
        # Method description - shows which parameters are most effective
        self._method_descriptions = {
            "Spectral Subtraction": "Best for: General hiss. Key params: Strength, Noise Threshold, HF Reduction",
            "Wiener Filter": "Best for: Broadband noise. Key params: Strength, Noise Threshold",
            "Multi-Band Adaptive": "Best for: Complex noise. Key params: Strength, Noise Threshold",
            "Combined (All Methods)": "Best for: Heavy noise. Uses Spectral + Wiener + Threshold",
            "Shellac/78rpm (Hiss+Groove)": "Best for: Old 78s. Key params: Strength, Noise Threshold",
            "NoiseReduce (Legacy)": "ML-based. May not work well on vintage recordings",
        }
        
        self.method_desc_label = ctk.CTkLabel(
            method_frame,
            text=self._method_descriptions["Spectral Subtraction"],
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            wraplength=190
        )
        self.method_desc_label.pack(anchor="w", pady=(3, 0))
        
        # Max dB Reduction
        self.max_db_slider = ParameterSlider(
            params_inner,
            label="Max dB Reduction",
            from_=1.0,
            to=30.0,
            default=12.0,
            unit=" dB",
            command=self._on_parameter_change
        )
        self.max_db_slider.pack(fill="x", pady=(0, 12))
        
        # Blend Original
        self.blend_slider = ParameterSlider(
            params_inner,
            label="Blend Original Signal",
            from_=0.0,
            to=50.0,
            default=5.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.blend_slider.pack(fill="x", pady=(0, 12))
        
        # Noise Reduction Strength
        self.strength_slider = ParameterSlider(
            params_inner,
            label="Noise Reduction Strength",
            from_=0.0,
            to=100.0,
            default=85.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.strength_slider.pack(fill="x", pady=(0, 12))
        
        # Transient Protection
        self.transient_slider = ParameterSlider(
            params_inner,
            label="Transient Protection",
            from_=0.0,
            to=100.0,
            default=30.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.transient_slider.pack(fill="x", pady=(0, 12))
        
        # High Frequency Emphasis (extra reduction for hiss frequencies)
        self.hf_emphasis_slider = ParameterSlider(
            params_inner,
            label="High Freq Hiss Reduction",
            from_=1.0,
            to=3.0,
            default=1.5,
            unit="x",
            command=self._on_parameter_change
        )
        self.hf_emphasis_slider.pack(fill="x", pady=(0, 12))
        
        # Smoothing Factor
        self.smoothing_slider = ParameterSlider(
            params_inner,
            label="Temporal Smoothing",
            from_=0.0,
            to=50.0,
            default=20.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.smoothing_slider.pack(fill="x", pady=(0, 5))
        
        # Fine-Tuning Section
        fine_tune_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=10)
        fine_tune_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        fine_tune_header = ctk.CTkFrame(fine_tune_frame, fg_color="transparent")
        fine_tune_header.pack(fill="x", padx=10, pady=(10, 5))
        
        fine_tune_label = ctk.CTkLabel(
            fine_tune_header,
            text="Fine-Tuning",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#bb8fce"
        )
        fine_tune_label.pack(side="left")
        
        fine_tune_inner = ctk.CTkFrame(fine_tune_frame, fg_color="transparent")
        fine_tune_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        # Hiss Start Frequency
        self.hiss_start_slider = ParameterSlider(
            fine_tune_inner,
            label="Hiss Start Freq",
            from_=500.0,
            to=5000.0,
            default=2000.0,
            unit="Hz",
            command=self._on_parameter_change
        )
        self.hiss_start_slider.pack(fill="x", pady=(0, 12))
        
        # Hiss Peak Frequency
        self.hiss_peak_slider = ParameterSlider(
            fine_tune_inner,
            label="Hiss Peak Freq",
            from_=3000.0,
            to=12000.0,
            default=6000.0,
            unit="Hz",
            command=self._on_parameter_change
        )
        self.hiss_peak_slider.pack(fill="x", pady=(0, 12))
        
        # Spectral Floor (artifact prevention)
        self.spectral_floor_slider = ParameterSlider(
            fine_tune_inner,
            label="Spectral Floor",
            from_=0.0,
            to=20.0,
            default=5.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.spectral_floor_slider.pack(fill="x", pady=(0, 12))
        
        # Low Cut Frequency (rumble removal)
        self.low_cut_slider = ParameterSlider(
            fine_tune_inner,
            label="Low Cut (Rumble)",
            from_=0.0,
            to=200.0,
            default=0.0,
            unit="Hz",
            command=self._on_parameter_change
        )
        self.low_cut_slider.pack(fill="x", pady=(0, 12))
        
        # Noise Threshold (noise/signal boundary)
        self.noise_threshold_slider = ParameterSlider(
            fine_tune_inner,
            label="Noise Threshold",
            from_=0.5,
            to=3.0,
            default=1.0,
            unit="x",
            command=self._on_parameter_change
        )
        self.noise_threshold_slider.pack(fill="x", pady=(0, 5))
        
        # Buttons Section
        buttons_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=10)
        buttons_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        buttons_inner = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        buttons_inner.pack(fill="x", padx=10, pady=10)
        
        # Process button
        self.process_btn = ctk.CTkButton(
            buttons_inner,
            text="🔄 Apply Denoising",
            command=self._process_audio,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            height=45,
            corner_radius=10,
            state="disabled"
        )
        self.process_btn.pack(fill="x", pady=(0, 8))
        
        # Reset button
        self.reset_btn = ctk.CTkButton(
            buttons_inner,
            text="↺ Reset to Defaults",
            command=self._reset_parameters,
            font=ctk.CTkFont(size=12),
            fg_color="#555555",
            hover_color="#777777",
            height=35,
            corner_radius=8
        )
        self.reset_btn.pack(fill="x", pady=(0, 10))
        
        # Output Format Section
        format_frame = ctk.CTkFrame(buttons_inner, fg_color="transparent")
        format_frame.pack(fill="x", pady=(5, 0))
        
        format_label = ctk.CTkLabel(
            format_frame,
            text="Output Format:",
            font=ctk.CTkFont(size=12),
            text_color="#cccccc"
        )
        format_label.pack(side="left")
        
        self.output_format = ctk.CTkOptionMenu(
            format_frame,
            values=["FLAC", "WAV", "OGG"],
            font=ctk.CTkFont(size=12),
            fg_color="#2d5a27",
            button_color="#2d5a27",
            button_hover_color="#3d7a37",
            dropdown_fg_color="#1a1a2e",
            dropdown_hover_color="#333333",
            width=100
        )
        self.output_format.set("FLAC")
        self.output_format.pack(side="right")
        
        # Info box
        info_frame = ctk.CTkFrame(scroll_frame, fg_color="#1a2a3a", corner_radius=8)
        info_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        info_text = ctk.CTkLabel(
            info_frame,
            text="💡 Tips:\n"
                 "• Use Auto Detect to find quiet sections\n"
                 "• Or click & drag on waveform for manual selection\n"
                 "• Learning a noise profile improves results\n"
                 "• Lower dB reduction = gentler denoising",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
            justify="left"
        )
        info_text.pack(padx=10, pady=10, anchor="w")
        
    def _create_status_bar(self):
        """Create status bar."""
        status_frame = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=0, height=30)
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready - Load an audio file to begin",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.status_label.pack(side="left", padx=15, pady=5)
        
        self.file_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#00d9ff"
        )
        self.file_label.pack(side="right", padx=15, pady=5)
        
    def _on_noise_region_selected(self, start: float, end: float):
        """Handle noise region selection from waveform."""
        self.selected_noise_region = (start, end)
        self.noise_profile_panel.enable_learn_button(True)
        self._set_status(f"Noise region selected: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        
    def _learn_noise_profile_manual(self):
        """Learn noise profile from manually selected region."""
        if self.selected_noise_region is None:
            messagebox.showwarning(
                "No Selection",
                "Please select a noise region on the waveform first.\n\n"
                "Click and drag on the original waveform to select a region "
                "that contains only noise (no music/vocals)."
            )
            return
            
        start, end = self.selected_noise_region
        
        try:
            self._set_status("Learning noise profile...")
            profile = self.denoiser.learn_noise_profile(start, end)
            self.noise_profile_panel.update_status(profile, (start, end))
            self._set_status(f"Noise profile learned from {start:.2f}s - {end:.2f}s")
            
            # Auto-reprocess
            if self.denoiser.get_original() is not None:
                self._process_audio()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to learn noise profile:\n{str(e)}")
            
    def _learn_noise_profile_auto(self):
        """Auto-detect quiet region and learn noise profile."""
        if self.denoiser.get_original() is None:
            return
            
        self._set_status("Auto-detecting quiet region...")
        
        def auto_learn_thread():
            try:
                profile, region = self.denoiser.auto_learn_noise_profile(min_duration=0.5)
                self.processing_queue.put(("noise_profile", profile, region))
            except Exception as e:
                self.processing_queue.put(("error", str(e)))
        
        threading.Thread(target=auto_learn_thread, daemon=True).start()
        
    def _clear_noise_profile(self):
        """Clear the learned noise profile."""
        self.denoiser.clear_noise_profile()
        self.waveform_original.clear_noise_region()
        self.selected_noise_region = None
        self.noise_profile_panel.enable_learn_button(False)
        self._set_status("Noise profile cleared - using adaptive estimation")
        
    def _toggle_use_profile(self, use: bool):
        """Toggle use of learned noise profile."""
        self.denoiser.set_use_learned_profile(use)
        
        if use:
            self._set_status("Using learned noise profile for denoising")
        else:
            self._set_status("Using adaptive noise estimation")
        
    def _load_audio(self):
        """Open file dialog to load audio file."""
        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.flac *.aif *.aiff *.m4a *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("FLAC Files", "*.flac"),
            ("AIFF Files", "*.aif *.aiff"),
            ("M4A Files", "*.m4a"),
            ("All Files", "*.*")
        ]
        
        # Try to start in example_tracks folder
        initial_dir = Path(__file__).parent.parent.parent.parent / "example_tracks"
        if not initial_dir.exists():
            initial_dir = Path.home()
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        if file_path:
            self._load_audio_file(file_path)
    
    def _load_audio_file(self, file_path: str):
        """Load audio from a file path (used by both file dialog and drag-drop)."""
        self.input_path = Path(file_path)
        self._set_status(f"Loading: {self.input_path.name}...")
        
        # Clear previous noise profile
        self._clear_noise_profile()
        
        # Reset waveform playheads
        self.waveform_original.reset_playhead()
        self.waveform_processed.reset_playhead()
        
        # Load in background thread
        def load_thread():
            try:
                audio, sr = self.denoiser.load_audio(str(self.input_path))
                self.processing_queue.put(("loaded", audio, sr))
            except Exception as e:
                self.processing_queue.put(("error", str(e)))
        
        threading.Thread(target=load_thread, daemon=True).start()
        
    def _save_audio(self):
        """Save processed audio."""
        if self.denoiser.get_processed() is None:
            messagebox.showwarning("No Processed Audio", "Please process the audio first.")
            return
        
        # Get selected output format
        output_format = self.output_format.get()
        ext = output_format.lower()
        
        # Default output filename with selected format
        default_name = f"{self.input_path.stem}_denoised.{ext}"
        
        # File type options based on format
        if output_format == "FLAC":
            filetypes = [("FLAC Files", "*.flac"), ("All Files", "*.*")]
            default_ext = ".flac"
        elif output_format == "WAV":
            filetypes = [("WAV Files", "*.wav"), ("All Files", "*.*")]
            default_ext = ".wav"
        else:  # OGG
            filetypes = [("OGG Files", "*.ogg"), ("All Files", "*.*")]
            default_ext = ".ogg"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=default_ext,
            initialfile=default_name,
            filetypes=filetypes
        )
        
        if not file_path:
            return
            
        try:
            self.denoiser.save(file_path, format=output_format)
            self._set_status(f"Saved: {Path(file_path).name}")
            messagebox.showinfo("Success", f"Audio saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save audio:\n{str(e)}")
            
    def _process_audio(self):
        """Process audio with current parameters."""
        if self.denoiser.get_original() is None:
            return
            
        self.is_processing = True
        self.process_btn.configure(state="disabled", text="⏳ Processing...")
        self._set_status("Processing audio...")
        
        # Stop playback
        self.player_original.stop()
        self.player_processed.stop()
        
        # Update parameters
        self.denoiser.update_parameters(
            max_db_reduction=self.max_db_slider.get(),
            blend_original=self.blend_slider.get() / 100.0,
            noise_reduction_strength=self.strength_slider.get() / 100.0,
            transient_protection=self.transient_slider.get() / 100.0,
            high_freq_emphasis=self.hf_emphasis_slider.get(),
            smoothing_factor=self.smoothing_slider.get() / 100.0,
            hiss_start_freq=self.hiss_start_slider.get(),
            hiss_peak_freq=self.hiss_peak_slider.get(),
            spectral_floor=self.spectral_floor_slider.get() / 100.0,
            low_cut_freq=self.low_cut_slider.get(),
            noise_threshold=self.noise_threshold_slider.get(),
        )
        
        # Process in background thread
        def process_thread():
            try:
                processed = self.denoiser.process()
                self.processing_queue.put(("processed", processed))
            except Exception as e:
                self.processing_queue.put(("error", str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
        
    def _on_parameter_change(self, value):
        """Handle parameter change - enable reprocessing hint."""
        if self.denoiser.get_original() is not None and not self.is_processing:
            self.process_btn.configure(fg_color="#884499")
    
    def _on_method_change(self, method_name: str):
        """Handle denoising method change."""
        method = self._method_names.get(method_name, DenoiseMethod.SPECTRAL_SUBTRACTION)
        self.denoiser.set_method(method)
        
        # Update method description
        desc = self._method_descriptions.get(method_name, "")
        self.method_desc_label.configure(text=desc)
        
        # Define which sliders are relevant for each method
        dim_color = "#666666"
        active_color = "#cccccc"
        
        # HF slider is less relevant for Multiband (has own bands) and Shellac (auto-tuned)
        hf_relevant = method_name not in ["Multi-Band Adaptive", "Shellac/78rpm (Hiss+Groove)"]
        self.hf_emphasis_slider.label.configure(text_color=active_color if hf_relevant else dim_color)
        
        # Hiss frequency sliders - relevant for Spectral, Wiener, Combined, NoiseReduce
        hiss_freq_relevant = method_name in ["Spectral Subtraction", "Wiener Filter", "Combined (Best)", "NoiseReduce (Legacy)"]
        self.hiss_start_slider.label.configure(text_color=active_color if hiss_freq_relevant else dim_color)
        self.hiss_peak_slider.label.configure(text_color=active_color if hiss_freq_relevant else dim_color)
        
        # Spectral floor - relevant for Spectral and Wiener
        floor_relevant = method_name in ["Spectral Subtraction", "Wiener Filter", "Combined (Best)"]
        self.spectral_floor_slider.label.configure(text_color=active_color if floor_relevant else dim_color)
        
        # Low cut - relevant for all methods (always applied post-processing)
        self.low_cut_slider.label.configure(text_color=active_color)
        
        # Noise threshold - relevant for all spectral methods (not NoiseReduce which uses own logic)
        threshold_relevant = method_name != "NoiseReduce (Legacy)"
        self.noise_threshold_slider.label.configure(text_color=active_color if threshold_relevant else dim_color)
        
        self._set_status(f"Method changed to: {method_name}")
        # Highlight process button to indicate reprocessing needed
        if self.denoiser.get_original() is not None and not self.is_processing:
            self.process_btn.configure(fg_color="#884499")
            
    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.max_db_slider.set(12.0)
        self.blend_slider.set(5.0)
        self.strength_slider.set(85.0)
        self.transient_slider.set(30.0)
        self.hf_emphasis_slider.set(1.5)
        self.smoothing_slider.set(20.0)
        # Fine-tuning defaults
        self.hiss_start_slider.set(2000.0)
        self.hiss_peak_slider.set(6000.0)
        self.spectral_floor_slider.set(5.0)
        self.low_cut_slider.set(0.0)
        self.noise_threshold_slider.set(1.0)
        # Method default
        self.method_dropdown.set("Spectral Subtraction")
        self.denoiser.set_method(DenoiseMethod.SPECTRAL_SUBTRACTION)
        
    def _toggle_play(self, which: str):
        """Toggle play/pause for original or processed audio."""
        if which == "original":
            player = self.player_original
            btn = self.play_orig_btn
            other_player = self.player_processed
            other_btn = self.play_proc_btn
        else:
            player = self.player_processed
            btn = self.play_proc_btn
            other_player = self.player_original
            other_btn = self.play_orig_btn
            
        # Pause other player (not stop - preserve position)
        if other_player.is_playing():
            other_player.pause()
            other_btn.configure(text="▶")
        
        if player.is_playing():
            player.pause()
            btn.configure(text="▶")
        else:
            player.play()
            btn.configure(text="⏸")
            self.current_player = player
            
    def _stop_play(self, which: str):
        """Stop playback."""
        if which == "original":
            self.player_original.stop()
            self.play_orig_btn.configure(text="▶")
            self.waveform_original.reset_playhead()
        else:
            self.player_processed.stop()
            self.play_proc_btn.configure(text="▶")
            self.waveform_processed.reset_playhead()
            
    def _on_seek(self, which: str, position: float):
        """Handle seek bar change."""
        if which == "original":
            self.player_original.seek(position)
            self.waveform_original.update_playhead(position)
        else:
            self.player_processed.seek(position)
            self.waveform_processed.update_playhead(position)
            
    def _sync_to_original(self):
        """Sync processed player position to original player position."""
        # Get the original player's position
        orig_position = self.player_original.get_position()
        
        # Set the processed player to the same position
        self.player_processed.seek(orig_position)
        self.waveform_processed.update_playhead(orig_position)
        
        # If original is playing, also play processed (and stop original for A/B comparison)
        if self.player_original.is_playing():
            self.player_original.pause()
            self.play_orig_btn.configure(text="▶")
            self.player_processed.play()
            self.play_proc_btn.configure(text="⏸")
        
        # Format the time for status
        duration = self.waveform_original._duration
        current_time = orig_position * duration
        self._set_status(f"Synced to position: {self.waveform_original._format_time(current_time)}")
            
    def _set_status(self, message: str):
        """Update status bar message."""
        self.status_label.configure(text=message)
        
    def _update_ui(self):
        """Periodic UI update loop."""
        # Check for processing results
        try:
            while True:
                msg = self.processing_queue.get_nowait()
                
                if msg[0] == "loaded":
                    _, audio, sr = msg
                    self._on_audio_loaded(audio, sr)
                    
                elif msg[0] == "processed":
                    _, processed = msg
                    self._on_audio_processed(processed)
                    
                elif msg[0] == "noise_profile":
                    _, profile, region = msg
                    self._on_noise_profile_learned(profile, region)
                    
                elif msg[0] == "error":
                    _, error = msg
                    messagebox.showerror("Error", f"An error occurred:\n{error}")
                    self._set_status("Error occurred")
                    self.is_processing = False
                    self.process_btn.configure(state="normal", text="🔄 Apply Denoising")
                    
        except queue.Empty:
            pass
        
        # Check for drag-and-drop files (polled from instance variable for thread safety)
        if self._pending_drop_files is not None:
            file_list = self._pending_drop_files
            self._pending_drop_files = None
            self._process_dropped_files(file_list)
        
        # Update waveform playheads during playback
        # Note: update_playhead automatically skips canvas redraws in spectrogram mode
        if self.player_original.is_playing():
            pos = self.player_original.get_position()
            self.waveform_original.update_playhead(pos)
                
        if self.player_processed.is_playing():
            pos = self.player_processed.get_position()
            self.waveform_processed.update_playhead(pos)
            
        # Update play buttons on completion
        if self.player_original.get_state() == PlaybackState.STOPPED:
            self.play_orig_btn.configure(text="▶")
        if self.player_processed.get_state() == PlaybackState.STOPPED:
            self.play_proc_btn.configure(text="▶")
        
        # Schedule next update
        self.after(50, self._update_ui)
        
    def _on_audio_loaded(self, audio: np.ndarray, sr: int):
        """Handle audio loaded event."""
        # Update waveform display
        display_audio = audio[0] if audio.ndim == 2 else audio
        self.waveform_original.plot_waveform(
            display_audio, sr,
            "Original Audio (Click to seek, drag to select noise)", "#ff9f43"
        )
        self.waveform_processed.plot_waveform(None, sr, "Processed Audio (Denoised)", "#00d9ff")
        
        # Enable noise region selection
        self.waveform_original.enable_selection(True)
        
        # Load into player
        self.player_original.load(audio, sr)
        
        # Enable controls
        self.play_orig_btn.configure(state="normal")
        self.stop_orig_btn.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.noise_profile_panel.enable_controls(True)
        
        # Disable processed controls until processing
        self.play_proc_btn.configure(state="disabled")
        self.stop_proc_btn.configure(state="disabled")
        self.sync_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        # Update status
        duration = self.waveform_original._duration
        self._set_status(f"Loaded: {duration:.1f}s @ {sr}Hz - Select noise region or use Auto Detect")
        self.file_label.configure(text=self.input_path.name)
        
        # Auto-process
        self._process_audio()
        
    def _on_audio_processed(self, processed: np.ndarray):
        """Handle audio processed event."""
        self.is_processing = False
        
        # Update waveform display
        sr = self.denoiser.get_sample_rate()
        display_audio = processed[0] if processed.ndim == 2 else processed
        self.waveform_processed.plot_waveform(display_audio, sr, "Processed Audio (Denoised)", "#00d9ff")
        
        # Load into player
        self.player_processed.load(processed, sr)
        
        # Enable controls
        self.play_proc_btn.configure(state="normal")
        self.stop_proc_btn.configure(state="normal")
        self.sync_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.process_btn.configure(state="normal", text="🔄 Apply Denoising", fg_color="#6c3483")
        
        # Update status
        profile_status = "with learned profile" if self.denoiser._use_learned_profile else "with adaptive estimation"
        self._set_status(f"Processing complete {profile_status} - Preview and adjust as needed")
        
    def _on_noise_profile_learned(self, profile: NoiseProfile, region: Tuple[float, float]):
        """Handle noise profile learned from auto-detect."""
        self.selected_noise_region = region
        self.waveform_original.set_noise_region(*region)
        self.noise_profile_panel.update_status(profile, region)
        self.noise_profile_panel.enable_learn_button(True)
        self._set_status(f"Auto-detected noise region: {region[0]:.2f}s - {region[1]:.2f}s")
        
        # Auto-reprocess
        if self.denoiser.get_original() is not None:
            self._process_audio()
        
    def on_closing(self):
        """Clean up on window close."""
        self.player_original.cleanup()
        self.player_processed.cleanup()
        self.destroy()


def main():
    """Main entry point for the application."""
    app = SoundDenoiserApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
