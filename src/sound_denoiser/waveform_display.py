"""Waveform, spectrogram, and spectrum display with interactive controls."""

from typing import Optional, Tuple

import customtkinter as ctk
import librosa
import librosa.display
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class WaveformDisplay(ctk.CTkFrame):
    """Widget for displaying audio waveforms, spectrograms, and spectra."""

    def __init__(
        self,
        master,
        on_region_select=None,
        on_seek=None,
        waveform_color: str = "#00d9ff",
        **kwargs,
    ):
        super().__init__(master, **kwargs)

        self.on_region_select = on_region_select
        self.on_seek = on_seek
        self._duration = 0.0
        self._selection_enabled = False
        self._selected_region: Optional[Tuple[float, float]] = None
        self._waveform_color = waveform_color
        self._current_position = 0.0
        self._view_mode = "waveform"

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
            values=["Waveform", "Spectrogram", "Frequency"],
            command=self._on_view_change,
            font=ctk.CTkFont(size=11),
            height=24,
            fg_color="#252535",
            selected_color="#6c3483",
            selected_hover_color="#8e44ad",
            unselected_color="#1a1a2e",
            unselected_hover_color="#252535",
        )
        self.view_toggle.set("Waveform")
        self.view_toggle.pack(side="left", padx=2)

        # Selection mode state (controlled externally via enable_selection)
        self._selection_mode_enabled = False

        self.noise_floor_freqs: Optional[np.ndarray] = None
        self.noise_floor_levels: Optional[np.ndarray] = None
        self.noise_floor_plot = None
        self.noise_threshold_mult: float = 1.0
        self.noise_threshold_plot = None
        self.analyzer_img = None
        self.analyzer_line = None
        self.analyzer_fill = None
        self._analyzer_freqs: Optional[np.ndarray] = None
        self._analyzer_levels: Optional[np.ndarray] = None
        self._analyzer_peak_levels: Optional[np.ndarray] = None  # Peak hold levels
        self._analyzer_floor_db = -60.0
        self._analyzer_ceiling_db = 40.0
        self._analyzer_nfft = 2048
        self._analyzer_filter = None
        self._analyzer_filter_sr = None

        # Comparison audio for overlay (e.g., denoised vs original)
        self._comparison_audio: Optional[np.ndarray] = None
        self._comparison_sr: int = 44100
        self._comparison_levels: Optional[np.ndarray] = None
        self._comparison_peak_levels: Optional[np.ndarray] = None
        self.comparison_line = None
        self.comparison_fill = None

        # Separator before zoom controls
        sep1 = ctk.CTkFrame(toggle_frame, width=1, height=20, fg_color="#444444")
        sep1.pack(side="left", padx=8)

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
            hover_color="#252535",
        )
        self.zoom_out_btn.pack(side="left", padx=1)

        self.zoom_label = ctk.CTkLabel(
            toggle_frame,
            text="1x",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=30,
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
            hover_color="#252535",
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
            hover_color="#252535",
        )
        self.zoom_reset_btn.pack(side="left", padx=1)

        # Time display label
        self.time_label = ctk.CTkLabel(
            toggle_frame,
            text="0:00 / 0:00",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        self.time_label.pack(side="right", padx=5)

        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 2.2), dpi=100, facecolor="#1a1a2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1a1a2e")
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)

        # Style the axes
        self.ax.tick_params(colors="#888888", labelsize=8)
        self.ax.spines["bottom"].set_color("#444444")
        self.ax.spines["left"].set_color("#444444")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bind click events for seeking and spectrum drag
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_drag)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
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

    def set_noise_floor_curve(self, freqs: Optional[np.ndarray], levels: Optional[np.ndarray]):
        """Set or clear the learned noise floor trace used in spectrum view."""
        if freqs is None or levels is None:
            self.noise_floor_freqs = None
            self.noise_floor_levels = None
        else:
            self.noise_floor_freqs = np.asarray(freqs, dtype=float)
            self.noise_floor_levels = np.asarray(levels, dtype=float)
            order = np.argsort(self.noise_floor_freqs)
            self.noise_floor_freqs = self.noise_floor_freqs[order]
            self.noise_floor_levels = self.noise_floor_levels[order]

        if self._view_mode == "spectrum":
            self._plot_spectrum_internal()

    def set_noise_threshold_multiplier(self, mult: float):
        """Set current noise threshold multiplier for the shifted noise floor line."""
        try:
            mult = float(mult)
        except Exception:
            return
        self.noise_threshold_mult = max(0.01, mult)
        if self._view_mode == "spectrum":
            self._refresh_threshold_artists()

    def _refresh_threshold_artists(self):
        """Update learned noise floor trace on the spectrum view."""
        if not hasattr(self, "ax"):
            return

        if self.analyzer_fill is not None:
            try:
                self.analyzer_fill.remove()
            except Exception:
                pass
            self.analyzer_fill = None

        floor_freqs = self.noise_floor_freqs
        floor_levels = self.noise_floor_levels

        if floor_freqs is not None and floor_levels is not None:
            if self.noise_floor_plot is None:
                (self.noise_floor_plot,) = self.ax.plot(
                    floor_freqs,
                    floor_levels,
                    color="#999999",
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                )
            else:
                self.noise_floor_plot.set_data(floor_freqs, floor_levels)

            # Purple line: noise floor shifted by the threshold multiplier
            offset_db = 20 * np.log10(max(self.noise_threshold_mult, 1e-3))
            shifted_levels = floor_levels + offset_db

            if self.noise_threshold_plot is None:
                (self.noise_threshold_plot,) = self.ax.plot(
                    floor_freqs,
                    shifted_levels,
                    color="#b187ff",
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                )
            else:
                self.noise_threshold_plot.set_data(floor_freqs, shifted_levels)
        else:
            if self.noise_floor_plot is not None:
                self.noise_floor_plot.set_data([], [])
            if self.noise_threshold_plot is not None:
                self.noise_threshold_plot.set_data([], [])

        if self.analyzer_line is not None:
            self.analyzer_line.set_zorder(2)

        self.canvas.draw_idle()

    def _build_analyzer_filter(self, sr: int):
        """Prepare mel filter bank for the analyzer based on current sample rate."""
        nyquist = max(2000.0, sr / 2)
        n_mels = 256
        fmin = 40.0
        fmax = min(nyquist, 20000.0)
        self._analyzer_filter = librosa.filters.mel(
            sr=sr,
            n_fft=self._analyzer_nfft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=True,
        )
        self._analyzer_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
        self._analyzer_filter_sr = sr

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
        self.time_label.configure(text=f"{self._format_time(current_time)} / {self._format_time(self._duration)}")

    def _on_view_change(self, value: str):
        """Handle view mode change."""
        if value == "Waveform":
            new_mode = "waveform"
        elif value == "Spectrogram":
            new_mode = "spectrogram"
        else:
            new_mode = "spectrum"

        if new_mode != self._view_mode:
            self._view_mode = new_mode
            self._redraw_display()

    def set_view_mode(self, mode: str):
        """Programmatically set view mode and redraw."""
        normalized = mode.lower()
        if normalized not in ("waveform", "spectrogram", "spectrum"):
            return
        self._view_mode = normalized
        try:
            if normalized == "waveform":
                self.view_toggle.set("Waveform")
            elif normalized == "spectrogram":
                self.view_toggle.set("Spectrogram")
            else:
                self.view_toggle.set("Frequency")
        except Exception:
            pass
        self._redraw_display()

    def get_view_mode(self) -> str:
        """Return current view mode."""
        return self._view_mode

    def _zoom_in(self):
        """Zoom in on the waveform."""
        if self._zoom_level < 16.0:
            self._zoom_level *= 2.0
            self._update_zoom()

    def _zoom_out(self):
        """Zoom out on the waveform."""
        if self._zoom_level > 1.0:
            self._zoom_level /= 2.0
            self._zoom_offset = max(0, min(self._zoom_offset, 1.0 - 1.0 / self._zoom_level))
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

    def get_zoom_state(self) -> tuple:
        """Return current zoom level and offset."""
        return (self._zoom_level, self._zoom_offset)

    def set_zoom_state(self, level: float, offset: float):
        """Apply zoom level/offset with clamping and redraw."""
        if level < 1.0:
            level = 1.0
        level = min(level, 16.0)
        max_offset = max(0.0, 1.0 - 1.0 / level)
        offset = max(0.0, min(offset, max_offset))
        self._zoom_level = level
        self._zoom_offset = offset
        self._update_zoom()

    def get_visible_time_range(self) -> tuple:
        """Get the currently visible time range based on zoom."""
        visible_duration = self._duration / self._zoom_level
        start_time = self._zoom_offset * self._duration
        end_time = start_time + visible_duration
        return (start_time, end_time)

    def _redraw_display(self):
        """Redraw the current display (waveform, spectrogram, or spectrum)."""
        if self._audio_data is None:
            return

        if self._view_mode == "waveform":
            self._plot_waveform_internal()
        elif self._view_mode == "spectrogram":
            self._plot_spectrogram_internal()
        else:
            self._plot_spectrum_internal()

    def _plot_waveform_internal(self):
        """Internal method to plot waveform with zoom support."""
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.selection_rect = None
        self.playhead = None
        self.played_area = None
        self.noise_floor_plot = None
        self.noise_threshold_plot = None
        self.analyzer_img = None

        audio = self._audio_data
        sr = self._sample_rate

        visible_start, visible_end = self.get_visible_time_range()
        start_sample = int(visible_start * sr)
        end_sample = int(visible_end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        visible_audio = audio[start_sample:end_sample]
        visible_times = np.linspace(visible_start, visible_end, len(visible_audio))

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

        self.ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        self.ax.ticklabel_format(axis="x", style="plain", useOffset=False)

        if self._selected_region:
            sel_start, sel_end = self._selected_region
            if sel_end > visible_start and sel_start < visible_end:
                self._draw_selection_rect(max(sel_start, visible_start), min(sel_end, visible_end))

        zoom_text = f" [{self._zoom_level:.0f}x zoom]" if self._zoom_level > 1.0 else ""
        self.ax.set_title(self._title + zoom_text, color="#ffffff", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Time (s)", color="#888888", fontsize=8)
        self.ax.tick_params(colors="#888888", labelsize=8)

        self.canvas.draw()

        if self._current_position > 0:
            self.update_playhead(self._current_position)

    def _plot_spectrogram_internal(self):
        """Internal method to plot spectrogram with zoom support and performance optimizations."""
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.selection_rect = None
        self.playhead = None
        self.played_area = None

        audio = self._audio_data
        sr = self._sample_rate

        visible_start, visible_end = self.get_visible_time_range()
        visible_duration = visible_end - visible_start
        start_sample = int(visible_start * sr)
        end_sample = int(visible_end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        pad_samples = 512
        padded_start = max(0, start_sample - pad_samples)
        padded_end = min(len(audio), end_sample + pad_samples)
        visible_audio = audio[padded_start:padded_end]

        zoom_factor = max(1.0, self._zoom_level)
        target_frames = min(1200, int(320 * zoom_factor))
        audio_len = len(visible_audio)

        ideal_hop = max(512, audio_len // target_frames)

        hop_length = 1024
        for h in [256, 512, 1024, 2048, 4096, 8192]:
            if h >= ideal_hop:
                hop_length = h
                break
        else:
            hop_length = 8192

        n_fft = min(4096, hop_length * 4)

        max_downsample = max(1, sr // 16000)

        if audio_len > 200000:
            downsample_factor = min(max_downsample, max(2, audio_len // 150000))
            visible_audio = visible_audio[::downsample_factor]
            effective_sr = sr // downsample_factor
            hop_length = max(512, hop_length // downsample_factor)
            n_fft = min(4096, hop_length * 4)
        else:
            effective_sr = sr

        max_freq = min(12000, effective_sr / 2)
        mel_spec = librosa.feature.melspectrogram(
            y=visible_audio,
            sr=effective_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=256,
            fmax=max_freq,
            power=2.0,
        )
        D = librosa.power_to_db(mel_spec, ref=np.max)

        time_offset = padded_start / sr
        time_end = padded_end / sr

        librosa.display.specshow(
            D,
            sr=effective_sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=self.ax,
            cmap="magma",
            x_coords=np.linspace(time_offset, time_end, D.shape[1]),
            fmax=max_freq,
        )

        self.ax.set_xlim(visible_start, visible_end)

        self.ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        self.ax.ticklabel_format(axis="x", style="plain", useOffset=False)

        if self._selected_region:
            sel_start, sel_end = self._selected_region
            if sel_end > visible_start and sel_start < visible_end:
                self._draw_selection_rect(max(sel_start, visible_start), min(sel_end, visible_end))

        zoom_text = f" [{self._zoom_level:.0f}x zoom]" if self._zoom_level > 1.0 else ""
        self.ax.set_title(self._title.replace("Waveform", "Spectrogram") + zoom_text, color="#ffffff", fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Time (s)", color="#888888", fontsize=8)
        self.ax.set_ylabel("Frequency (Hz)", color="#888888", fontsize=8)
        self.ax.tick_params(colors="#888888", labelsize=8)

        self.canvas.draw()

        if self._current_position > 0:
            self.update_playhead(self._current_position)

    def _plot_spectrum_internal(self):
        """Plot a frequency spectrum with draggable threshold curve."""
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.selection_rect = None
        self.playhead = None
        self.played_area = None
        self.analyzer_img = None
        self._analyzer_levels = None
        self._analyzer_peak_levels = None
        self.analyzer_line = None
        self.analyzer_fill = None
        self._comparison_levels = None
        self._comparison_peak_levels = None
        self.comparison_line = None
        self.comparison_fill = None
        self.noise_floor_plot = None
        self.noise_threshold_plot = None

        audio = self._audio_data
        sr = self._sample_rate

        nyquist = max(2000.0, sr / 2)

        if self._analyzer_filter is None or self._analyzer_filter_sr != sr:
            self._build_analyzer_filter(sr)

        # Initialize with floor levels - will be updated with real audio data
        initial_levels = np.full_like(self._analyzer_freqs, self._analyzer_floor_db)

        # If comparison audio exists, draw it first (behind original)
        if self._comparison_audio is not None:
            self.comparison_fill = self.ax.fill_between(
                self._analyzer_freqs,
                self._analyzer_floor_db,
                initial_levels,
                color="#00d9ff",
                alpha=0.35,
                zorder=0.5,
            )
            (self.comparison_line,) = self.ax.plot(
                self._analyzer_freqs,
                initial_levels,
                color="#00d9ff",
                linewidth=1.5,
                alpha=0.8,
                zorder=1.5,
            )

        # Use line plot + fill for original spectrum (works correctly with log x-axis)
        self.analyzer_fill = self.ax.fill_between(
            self._analyzer_freqs,
            self._analyzer_floor_db,
            initial_levels,
            color="#ff9f43",
            alpha=0.5,
            zorder=1,
        )
        (self.analyzer_line,) = self.ax.plot(
            self._analyzer_freqs,
            initial_levels,
            color="#ffcc00",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )
        # Keep analyzer_img as a flag that the analyzer is initialized
        self.analyzer_img = True

        self._refresh_threshold_artists()

        self.ax.set_xlim(max(40, float(self._analyzer_freqs[0])), min(nyquist, float(self._analyzer_freqs[-1])))
        self.ax.set_ylim(self._analyzer_floor_db, self._analyzer_ceiling_db)
        self.ax.set_xscale("log")
        self.ax.set_xlabel("Frequency (Hz)", color="#888888", fontsize=8)
        self.ax.set_ylabel("Level (dB)", color="#888888", fontsize=8)
        self.ax.tick_params(colors="#888888", labelsize=8)
        self.ax.grid(True, color="#222222", linestyle="--", linewidth=0.5, alpha=0.6)
        self.ax.set_title("Frequency Analyzer", color="#ffffff", fontsize=10, fontweight="bold")
        self.ax.xaxis.set_major_formatter(mticker.LogFormatter())
        self.ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        self.update_frequency_analyzer(audio, sr, self._current_position)

        self.canvas.draw()

    def set_comparison_audio(self, audio: Optional[np.ndarray], sr: int = 44100):
        """Set comparison audio (e.g., denoised) for overlay on spectrum view."""
        if audio is None:
            self._comparison_audio = None
            self._comparison_sr = 44100
        else:
            if len(audio.shape) > 1:
                audio = audio[0] if audio.shape[0] <= 2 else audio[:, 0]
            self._comparison_audio = audio.copy()
            self._comparison_sr = sr
        self._comparison_levels = None
        self._comparison_peak_levels = None
        if self._view_mode == "spectrum":
            self._redraw_display()

    def _compute_spectrum_db(self, audio: np.ndarray, sr: int, position: float) -> Optional[np.ndarray]:
        """Compute mel spectrum in dB for audio at given position."""
        if audio is None or sr <= 0:
            return None

        center = int(position * len(audio))
        if center <= 0:
            center = len(audio) // 20

        window_len = int(sr * 0.10)  # Shorter window for better peak response
        window_len = min(max(1024, window_len), len(audio))
        start = max(0, center - window_len // 2)
        end = min(len(audio), start + window_len)
        if end - start < 512:
            return None

        window = audio[start:end]
        window = window * np.hanning(len(window))

        if self._analyzer_filter is None or self._analyzer_filter_sr != sr:
            self._build_analyzer_filter(sr)

        spec = np.abs(np.fft.rfft(window, n=self._analyzer_nfft))
        spec_power = spec ** 2

        mel_power = self._analyzer_filter @ spec_power[: self._analyzer_filter.shape[1]]
        mel_db = 10 * np.log10(np.maximum(mel_power, 1e-12))
        return mel_db

    def update_frequency_analyzer(self, audio: Optional[np.ndarray], sr: int, position: float):
        """Update the live frequency analyzer with a short FFT slice around the playhead."""
        if self._view_mode != "spectrum":
            return
        if audio is None or sr <= 0 or self.analyzer_img is None or self._analyzer_freqs is None:
            return
        if self.analyzer_line is None:
            return

        # Compute original audio spectrum
        mel_db = self._compute_spectrum_db(audio, sr, position)
        if mel_db is None:
            return

        # Peak detection algorithm: fast attack, slow decay
        if self._analyzer_levels is None or len(self._analyzer_levels) != len(mel_db):
            self._analyzer_levels = mel_db.copy()
            self._analyzer_peak_levels = mel_db.copy()
        else:
            # Fast attack (instant rise), slower decay for peaks
            self._analyzer_levels = np.maximum(mel_db, self._analyzer_levels * 0.85 + mel_db * 0.15)
            # Peak hold with slow decay
            self._analyzer_peak_levels = np.maximum(mel_db, self._analyzer_peak_levels * 0.95)

        # Clip levels to the display range
        display_levels = np.clip(self._analyzer_levels, self._analyzer_floor_db, self._analyzer_ceiling_db)

        # Update line plot data
        self.analyzer_line.set_ydata(display_levels)

        # Update fill - need to remove old and create new
        if self.analyzer_fill is not None:
            self.analyzer_fill.remove()
        self.analyzer_fill = self.ax.fill_between(
            self._analyzer_freqs,
            self._analyzer_floor_db,
            display_levels,
            color="#ff9f43",
            alpha=0.5,
            zorder=1,
        )
        # Ensure line stays on top
        self.analyzer_line.set_zorder(2)

        # Update comparison audio if present
        if self._comparison_audio is not None and self.comparison_line is not None:
            comp_db = self._compute_spectrum_db(self._comparison_audio, self._comparison_sr, position)
            if comp_db is not None:
                if self._comparison_levels is None or len(self._comparison_levels) != len(comp_db):
                    self._comparison_levels = comp_db.copy()
                    self._comparison_peak_levels = comp_db.copy()
                else:
                    # Same peak algorithm for comparison
                    self._comparison_levels = np.maximum(comp_db, self._comparison_levels * 0.85 + comp_db * 0.15)
                    self._comparison_peak_levels = np.maximum(comp_db, self._comparison_peak_levels * 0.95)

                comp_display = np.clip(self._comparison_levels, self._analyzer_floor_db, self._analyzer_ceiling_db)
                self.comparison_line.set_ydata(comp_display)

                if self.comparison_fill is not None:
                    self.comparison_fill.remove()
                self.comparison_fill = self.ax.fill_between(
                    self._analyzer_freqs,
                    self._analyzer_floor_db,
                    comp_display,
                    color="#00d9ff",
                    alpha=0.35,
                    zorder=0.5,
                )
                self.comparison_line.set_zorder(1.5)

        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        """Handle mouse click on canvas for seeking or spectrum dragging."""
        if event.inaxes != self.ax or self._duration <= 0:
            return

        if self._view_mode == "spectrum":
            return

        if self._selection_enabled:
            return

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
        """Handle mouse drag on canvas for seeking or spectrum drag."""
        if event.inaxes != self.ax or self._duration <= 0:
            return

        if self._view_mode == "spectrum":
            return

        if not self._drag_seeking:
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
        if self._view_mode == "spectrum":
            return

        if self._zoom_level <= 1.0:
            return

        pan_step = 0.1 / self._zoom_level

        if event.button == "up":
            self._zoom_offset = max(0, self._zoom_offset - pan_step)
        elif event.button == "down":
            max_offset = 1.0 - 1.0 / self._zoom_level
            self._zoom_offset = min(max_offset, self._zoom_offset + pan_step)

        self._redraw_display()

    def enable_selection(self, enable: bool = True):
        """Enable or disable region selection mode."""
        self._selection_enabled = enable

        if enable and self._duration > 0:
            self.span_selector = SpanSelector(
                self.ax,
                self._on_span_select,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="#ff6b6b"),
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

        xmin = max(0, xmin)
        xmax = min(self._duration, xmax)

        if xmax - xmin < 0.1:
            return

        self._selected_region = (xmin, xmax)
        self._draw_selection_rect(xmin, xmax)

        if self.on_region_select:
            self.on_region_select(xmin, xmax)

    def _draw_selection_rect(self, start: float, end: float):
        """Draw selection rectangle on waveform."""
        if self.selection_rect:
            try:
                self.selection_rect.remove()
            except Exception:
                pass
            self.selection_rect = None

        self.selection_rect = self.ax.axvspan(start, end, alpha=0.3, color="#ff6b6b", label="Noise Region")
        self.canvas.draw_idle()

    def add_selection_rect(self, start: float, end: float):
        """Add a selection rectangle without removing existing ones."""
        rect = self.ax.axvspan(start, end, alpha=0.3, color="#ff6b6b")
        if not hasattr(self, "_selection_rects"):
            self._selection_rects = []
        self._selection_rects.append(rect)
        self.canvas.draw_idle()

    def clear_selection(self):
        """Clear all selection rectangles."""
        if self.selection_rect:
            try:
                self.selection_rect.remove()
            except Exception:
                pass
            self.selection_rect = None

        if hasattr(self, "_selection_rects"):
            for rect in self._selection_rects:
                try:
                    rect.remove()
                except Exception:
                    pass
            self._selection_rects = []

        self._selected_region = None
        self.canvas.draw_idle()

    def hide_selection_rects(self):
        """Hide all selection rectangles (set alpha to 0)."""
        if hasattr(self, "_selection_rects"):
            for rect in self._selection_rects:
                try:
                    rect.set_alpha(0)
                except Exception:
                    pass
        if self.selection_rect:
            try:
                self.selection_rect.set_alpha(0)
            except Exception:
                pass
        self.canvas.draw_idle()

    def show_selection_rects(self):
        """Show all selection rectangles (restore alpha to 0.3)."""
        if hasattr(self, "_selection_rects"):
            for rect in self._selection_rects:
                try:
                    rect.set_alpha(0.3)
                except Exception:
                    pass
        if self.selection_rect:
            try:
                self.selection_rect.set_alpha(0.3)
            except Exception:
                pass
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

    def plot_waveform(self, audio: Optional[np.ndarray], sr: int, title: str = "Waveform", color: str = "#00d9ff"):
        """Plot audio waveform or spectrogram based on current view mode."""
        self._waveform_color = color
        self._title = title
        self._sample_rate = sr
        self._current_position = 0.0

        if audio is not None:
            if len(audio.shape) > 1:
                audio = audio[0] if audio.shape[0] <= 2 else audio[:, 0]

            self._audio_data = audio.copy()
            self._duration = len(audio) / sr
        else:
            self._duration = 0.0
            self._audio_data = None

        self._update_time_label(0)
        self._redraw_display()

    def update_playhead(self, position: float, skip_draw: bool = False):
        """Update playhead position and played area shading."""
        if self._duration <= 0:
            return

        time_pos = position * self._duration
        self._current_position = position

        self._update_time_label(position)

        if self._view_mode == "spectrum":
            return

        if skip_draw:
            return

        visible_start, visible_end = self.get_visible_time_range()

        if self._zoom_level > 1.0 and time_pos >= visible_end - 0.1:
            total_duration = self._duration
            visible_duration = visible_end - visible_start
            new_offset = min(time_pos / total_duration, 1.0 - (visible_duration / total_duration))
            new_offset = max(0.0, new_offset)

            if abs(new_offset - self._zoom_offset) > 0.01:
                self._zoom_offset = new_offset
                if self._view_mode == "spectrogram":
                    self._plot_spectrogram_internal()
                else:
                    self._plot_waveform_internal()
                visible_start, visible_end = self.get_visible_time_range()

        if self.playhead is not None:
            try:
                self.playhead.remove()
            except Exception:
                pass
            self.playhead = None

        if self.played_area is not None:
            try:
                self.played_area.remove()
            except Exception:
                pass
            self.played_area = None

        playhead_visible = visible_start <= time_pos <= visible_end

        if time_pos > visible_start and self._view_mode != "spectrogram":
            played_end = min(time_pos, visible_end)
            played_start = visible_start
            self.played_area = self.ax.axvspan(
                played_start,
                played_end,
                alpha=0.3,
                color="#333333",
                zorder=1,
            )

        if position > 0 and playhead_visible:
            self.playhead = self.ax.axvline(
                x=time_pos,
                color="#ffffff",
                linewidth=2,
                alpha=0.9,
                zorder=10,
            )

        self.canvas.draw_idle()

    def reset_playhead(self):
        """Reset playhead to start."""
        self._current_position = 0.0
        self._update_time_label(0)
        if self.playhead is not None:
            try:
                self.playhead.remove()
            except Exception:
                pass
            self.playhead = None
        if self.played_area is not None:
            try:
                self.played_area.remove()
            except Exception:
                pass
            self.played_area = None
        self.canvas.draw_idle()

