"""
Modern GUI for the Sound Denoiser application.

Provides an intuitive interface for loading audio, adjusting
denoising parameters, previewing results, and saving output.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import librosa
import librosa.display
import threading
from pathlib import Path
from typing import Optional
import queue

from .denoiser import AudioDenoiser
from .audio_player import AudioPlayer, PlaybackState


# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class WaveformDisplay(ctk.CTkFrame):
    """Widget for displaying audio waveforms."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 2.5), dpi=100, facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a2e')
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)
        
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
        
        # Playhead line
        self.playhead = None
        
    def plot_waveform(
        self,
        audio: Optional[np.ndarray],
        sr: int,
        title: str = "Waveform",
        color: str = '#00d9ff'
    ):
        """
        Plot audio waveform.
        
        Args:
            audio: Audio data
            sr: Sample rate
            title: Plot title
            color: Waveform color
        """
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        
        if audio is not None:
            # Downsample for display if needed
            if len(audio.shape) > 1:
                audio = audio[0] if audio.shape[0] <= 2 else audio[:, 0]
            
            # Create time axis
            duration = len(audio) / sr
            times = np.linspace(0, duration, len(audio))
            
            # Downsample for faster plotting
            if len(audio) > 10000:
                step = len(audio) // 10000
                times = times[::step]
                audio = audio[::step]
            
            self.ax.plot(times, audio, color=color, linewidth=0.5, alpha=0.8)
            self.ax.fill_between(times, audio, alpha=0.3, color=color)
            self.ax.set_xlim(0, duration)
            self.ax.set_ylim(-1, 1)
            
        self.ax.set_title(title, color='#ffffff', fontsize=10, fontweight='bold')
        self.ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        self.ax.tick_params(colors='#888888', labelsize=8)
        
        self.canvas.draw()
        
    def update_playhead(self, position: float, duration: float):
        """
        Update playhead position.
        
        Args:
            position: Position as fraction (0-1)
            duration: Total duration in seconds
        """
        if self.playhead:
            self.playhead.remove()
        
        time_pos = position * duration
        self.playhead = self.ax.axvline(x=time_pos, color='#ff6b6b', linewidth=2, alpha=0.8)
        self.canvas.draw_idle()


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


class SoundDenoiserApp(ctk.CTk):
    """Main application window for Sound Denoiser."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Sound Denoiser - Hiss Removal")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
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
        
        # Build UI
        self._create_ui()
        
        # Start update loop
        self._update_ui()
        
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
        
        self.waveform_original = WaveformDisplay(original_frame)
        self.waveform_original.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.waveform_original.plot_waveform(None, 44100, "Original Audio", "#ff9f43")
        
        # Original playback controls
        orig_controls = ctk.CTkFrame(original_frame, fg_color="transparent")
        orig_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.play_orig_btn = ctk.CTkButton(
            orig_controls,
            text="▶",
            width=40,
            height=32,
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
            width=40,
            height=32,
            command=lambda: self._stop_play("original"),
            fg_color="#555555",
            hover_color="#777777",
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.stop_orig_btn.pack(side="left", padx=2)
        
        self.progress_orig = ctk.CTkProgressBar(
            orig_controls,
            progress_color="#ff9f43",
            fg_color="#333333",
            height=8
        )
        self.progress_orig.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.progress_orig.set(0)
        
        # Processed waveform
        processed_frame = ctk.CTkFrame(left_panel, fg_color="#151525", corner_radius=10)
        processed_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        processed_frame.grid_rowconfigure(0, weight=1)
        processed_frame.grid_columnconfigure(0, weight=1)
        
        self.waveform_processed = WaveformDisplay(processed_frame)
        self.waveform_processed.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.waveform_processed.plot_waveform(None, 44100, "Processed Audio (Denoised)", "#00d9ff")
        
        # Processed playback controls
        proc_controls = ctk.CTkFrame(processed_frame, fg_color="transparent")
        proc_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        self.play_proc_btn = ctk.CTkButton(
            proc_controls,
            text="▶",
            width=40,
            height=32,
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
            width=40,
            height=32,
            command=lambda: self._stop_play("processed"),
            fg_color="#555555",
            hover_color="#777777",
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.stop_proc_btn.pack(side="left", padx=2)
        
        self.progress_proc = ctk.CTkProgressBar(
            proc_controls,
            progress_color="#00d9ff",
            fg_color="#333333",
            height=8
        )
        self.progress_proc.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.progress_proc.set(0)
        
    def _create_parameter_panel(self, parent):
        """Create parameter controls panel."""
        right_panel = ctk.CTkFrame(parent, fg_color="#0f0f1a", corner_radius=12)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # Title
        params_title = ctk.CTkLabel(
            right_panel,
            text="⚙️ Denoising Parameters",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ffffff"
        )
        params_title.pack(pady=(15, 10), padx=15, anchor="w")
        
        # Separator
        sep = ctk.CTkFrame(right_panel, height=1, fg_color="#333333")
        sep.pack(fill="x", padx=15, pady=(0, 15))
        
        # Parameters frame (scrollable)
        params_frame = ctk.CTkScrollableFrame(
            right_panel,
            fg_color="transparent",
            scrollbar_button_color="#333333"
        )
        params_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Max dB Reduction
        self.max_db_slider = ParameterSlider(
            params_frame,
            label="Max dB Reduction",
            from_=1.0,
            to=20.0,
            default=4.0,
            unit=" dB",
            command=self._on_parameter_change
        )
        self.max_db_slider.pack(fill="x", pady=(0, 15))
        
        # Blend Original
        self.blend_slider = ParameterSlider(
            params_frame,
            label="Blend Original Signal",
            from_=0.0,
            to=50.0,
            default=12.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.blend_slider.pack(fill="x", pady=(0, 15))
        
        # Noise Reduction Strength
        self.strength_slider = ParameterSlider(
            params_frame,
            label="Noise Reduction Strength",
            from_=0.0,
            to=100.0,
            default=70.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.strength_slider.pack(fill="x", pady=(0, 15))
        
        # Transient Protection
        self.transient_slider = ParameterSlider(
            params_frame,
            label="Transient Protection",
            from_=0.0,
            to=100.0,
            default=50.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.transient_slider.pack(fill="x", pady=(0, 15))
        
        # High Freq Rolloff
        self.rolloff_slider = ParameterSlider(
            params_frame,
            label="High Frequency Rolloff",
            from_=0.0,
            to=100.0,
            default=80.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.rolloff_slider.pack(fill="x", pady=(0, 15))
        
        # Smoothing Factor
        self.smoothing_slider = ParameterSlider(
            params_frame,
            label="Temporal Smoothing",
            from_=0.0,
            to=50.0,
            default=10.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.smoothing_slider.pack(fill="x", pady=(0, 15))
        
        # Separator
        sep2 = ctk.CTkFrame(right_panel, height=1, fg_color="#333333")
        sep2.pack(fill="x", padx=15, pady=(5, 15))
        
        # Process button
        self.process_btn = ctk.CTkButton(
            right_panel,
            text="🔄 Apply Denoising",
            command=self._process_audio,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            height=45,
            corner_radius=10,
            state="disabled"
        )
        self.process_btn.pack(fill="x", padx=15, pady=(0, 10))
        
        # Reset button
        self.reset_btn = ctk.CTkButton(
            right_panel,
            text="↺ Reset to Defaults",
            command=self._reset_parameters,
            font=ctk.CTkFont(size=12),
            fg_color="#555555",
            hover_color="#777777",
            height=35,
            corner_radius=8
        )
        self.reset_btn.pack(fill="x", padx=15, pady=(0, 15))
        
        # Info box
        info_frame = ctk.CTkFrame(right_panel, fg_color="#1a2a3a", corner_radius=8)
        info_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        info_text = ctk.CTkLabel(
            info_frame,
            text="💡 Tips:\n"
                 "• Lower dB reduction = gentler denoising\n"
                 "• Higher blend = more original character\n"
                 "• Protect transients for punchy drums",
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
        
    def _load_audio(self):
        """Load audio file."""
        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.flac *.aif *.aiff *.m4a *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("FLAC Files", "*.flac"),
            ("AIFF Files", "*.aif *.aiff"),
            ("M4A Files", "*.m4a"),
            ("All Files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes,
            initialdir=Path(__file__).parent.parent.parent.parent / "example_tracks"
        )
        
        if not file_path:
            return
            
        self.input_path = Path(file_path)
        self._set_status(f"Loading: {self.input_path.name}...")
        
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
            
        # Default output filename
        default_name = f"{self.input_path.stem}_denoised.wav"
        
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[
                ("WAV Files", "*.wav"),
                ("FLAC Files", "*.flac"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.denoiser.save(file_path)
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
            high_freq_rolloff=self.rolloff_slider.get() / 100.0,
            smoothing_factor=self.smoothing_slider.get() / 100.0,
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
            
    def _reset_parameters(self):
        """Reset parameters to defaults."""
        self.max_db_slider.set(4.0)
        self.blend_slider.set(12.0)
        self.strength_slider.set(70.0)
        self.transient_slider.set(50.0)
        self.rolloff_slider.set(80.0)
        self.smoothing_slider.set(10.0)
        
    def _toggle_play(self, which: str):
        """Toggle play/pause for original or processed audio."""
        if which == "original":
            player = self.player_original
            btn = self.play_orig_btn
            other_player = self.player_processed
        else:
            player = self.player_processed
            btn = self.play_proc_btn
            other_player = self.player_original
            
        # Stop other player
        other_player.stop()
        
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
            self.progress_orig.set(0)
        else:
            self.player_processed.stop()
            self.play_proc_btn.configure(text="▶")
            self.progress_proc.set(0)
            
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
                    
                elif msg[0] == "error":
                    _, error = msg
                    messagebox.showerror("Error", f"An error occurred:\n{error}")
                    self._set_status("Error occurred")
                    self.is_processing = False
                    self.process_btn.configure(state="normal", text="🔄 Apply Denoising")
                    
        except queue.Empty:
            pass
        
        # Update progress bars
        if self.player_original.is_playing():
            self.progress_orig.set(self.player_original.get_position())
        if self.player_processed.is_playing():
            self.progress_proc.set(self.player_processed.get_position())
            
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
        self.waveform_original.plot_waveform(display_audio, sr, "Original Audio", "#ff9f43")
        self.waveform_processed.plot_waveform(None, sr, "Processed Audio (Denoised)", "#00d9ff")
        
        # Load into player
        self.player_original.load(audio, sr)
        
        # Enable controls
        self.play_orig_btn.configure(state="normal")
        self.stop_orig_btn.configure(state="normal")
        self.process_btn.configure(state="normal")
        
        # Disable processed controls until processing
        self.play_proc_btn.configure(state="disabled")
        self.stop_proc_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        # Update status
        duration = len(audio[0] if audio.ndim == 2 else audio) / sr
        self._set_status(f"Loaded: {duration:.1f}s @ {sr}Hz")
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
        self.save_btn.configure(state="normal")
        self.process_btn.configure(state="normal", text="🔄 Apply Denoising", fg_color="#6c3483")
        
        # Update status
        self._set_status("Processing complete - Preview and adjust parameters as needed")
        
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
