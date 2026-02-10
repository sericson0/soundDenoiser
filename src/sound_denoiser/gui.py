"""
Modern GUI for the Sound Denoiser application.

Provides an intuitive interface for loading audio, adjusting
denoising parameters, previewing results, and saving output.
Includes noise profile learning with visual region selection.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import json
import librosa
import threading
from pathlib import Path
from typing import Optional, Tuple, List
import queue

WINDND_AVAILABLE = False
try:
    import windnd
    WINDND_AVAILABLE = True
except ImportError:
    pass

try:
    from .denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
    from .audio_player import AudioPlayer, PlaybackState
    from .waveform_display import WaveformDisplay
    from .ui_components import SeekBar, ParameterSlider, NoiseProfilePanel
except ImportError:
    from denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
    from audio_player import AudioPlayer, PlaybackState
    from waveform_display import WaveformDisplay
    from ui_components import SeekBar, ParameterSlider, NoiseProfilePanel

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
        self.active_waveform_view = "original"  # "original" or "processed"
        self.noise_selection_enabled = False
        self.track_title = "No Track"

        # State
        self.input_path: Optional[Path] = None
        self.output_path: Optional[Path] = None
        self.is_processing = False
        self.processing_queue = queue.Queue()
        self.selected_noise_region: Optional[Tuple[float, float]] = None
        self.config_path = Path.home() / ".sound_denoiser_config.json"
        self.config = self._load_config()

        # Build UI
        self._create_ui()

        # Setup drag and drop if available
        self._setup_drag_and_drop()

        # Start update loop
        self._update_ui()

    def _load_config(self) -> dict:
        """Load persisted config (last directories, etc.)."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_config(self):
        """Persist config to disk."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f)
        except Exception:
            pass

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
            text="🎧 Sound Denoiser",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#00d9ff"
        )
        title.pack(side="left")
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
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
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
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=0)
        left_panel.grid_columnconfigure(0, weight=1)

        waveform_frame = ctk.CTkFrame(left_panel, fg_color="#151525", corner_radius=10)
        waveform_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(6, 2))
        waveform_frame.grid_rowconfigure(0, weight=1)
        waveform_frame.grid_columnconfigure(0, weight=1)

        # Stack both waveforms in the same container and show one at a time
        self.waveform_original = WaveformDisplay(
            waveform_frame,
            on_region_select=self._on_noise_region_selected,
            on_seek=lambda pos: self._on_seek("original", pos),
        )
        self.waveform_original.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.waveform_original.plot_waveform(None, 44100, f"{self.track_title} (Original)", "#ff9f43")

        self.waveform_processed = WaveformDisplay(
            waveform_frame,
            on_seek=lambda pos: self._on_seek("processed", pos),
        )
        self.waveform_processed.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.waveform_processed.plot_waveform(None, 44100, f"{self.track_title} (Denoised)", "#00d9ff")
        self.waveform_processed.grid_remove()  # Start hidden for a single-panel view

        # Unified playback and view controls
        controls = ctk.CTkFrame(left_panel, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))

        self.play_btn = ctk.CTkButton(
            controls,
            text="▶",
            width=60,
            height=32,
            command=self._toggle_play,
            fg_color="#ff9f43",
            hover_color="#f39c12",
            text_color="#000000",
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled"
        )
        self.play_btn.pack(side="left", padx=4)

        self.stop_btn = ctk.CTkButton(
            controls,
            text="⏹",
            width=60,
            height=32,
            command=self._stop_play,
            fg_color="#555555",
            hover_color="#777777",
            font=ctk.CTkFont(size=14),
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)

        self.view_toggle_btn = ctk.CTkButton(
            controls,
            text="Show Processed",
            width=140,
            height=32,
            command=self._toggle_waveform_view,
            fg_color="#6c3483",
            hover_color="#8e44ad",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.view_toggle_btn.pack(side="left", padx=6)

        self.selection_btn = ctk.CTkButton(
            controls,
            text="Make Selection",
            width=130,
            height=32,
            command=self._toggle_selection_button,
            fg_color="#1a5276",
            hover_color="#2471a3",
            font=ctk.CTkFont(size=12, weight="bold"),
            state="disabled"
        )
        self.selection_btn.pack(side="left", padx=6)

        self.process_btn = ctk.CTkButton(
            controls,
            text="✨ Apply Denoising",
            command=self._process_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            height=32,
            corner_radius=8,
            state="disabled"
        )
        self.process_btn.pack(side="left", padx=6)

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
            on_toggle_selection=self._toggle_noise_selection,
            on_remove_selection=self._on_remove_noise_selection,
        )
        self.noise_profile_panel.pack(fill="x", padx=5, pady=(5, 10))

        # Parameters Section
        params_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=10)
        params_frame.pack(fill="x", padx=5, pady=(0, 10))

        # Title
        params_title = ctk.CTkLabel(
            params_frame,
            text="🎛️ Denoising Parameters",
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
            "Combined (All Methods)": DenoiseMethod.COMBINED,
            "Shellac/78rpm (Hiss+Groove)": DenoiseMethod.SHELLAC,
            "Spectral Gating (Learned Profile)": DenoiseMethod.SPECTRAL_GATING,
            "Adaptive Blend (Subtraction+Gating)": DenoiseMethod.ADAPTIVE_BLEND,
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
            "Combined (All Methods)": "Best for: Heavy noise. Uses Spectral + Wiener + Threshold",
            "Shellac/78rpm (Hiss+Groove)": "Best for: Old 78s. Key params: Strength, Noise Threshold",
            "Spectral Gating (Learned Profile)": "Best with learned profile. Soft gate based on noise floor",
            "Adaptive Blend (Subtraction+Gating)": "Blends both methods. Use Artifact Control to balance",
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
        self.noise_threshold_slider.pack(fill="x", pady=(0, 12))

        # Artifact Control (subtraction vs gating balance)
        self.artifact_control_slider = ParameterSlider(
            fine_tune_inner,
            label="Artifact Control",
            from_=0.0,
            to=100.0,
            default=50.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.artifact_control_slider.pack(fill="x", pady=(0, 8))

        # Artifact control description
        artifact_desc = ctk.CTkLabel(
            fine_tune_inner,
            text="0%=Subtraction (chirpy) ↔ 100%=Gating (pumping)",
            font=ctk.CTkFont(size=9),
            text_color="#666666"
        )
        artifact_desc.pack(anchor="w", pady=(0, 8))

        # Adaptive Blend checkbox
        self.adaptive_blend_var = ctk.BooleanVar(value=True)
        self.adaptive_blend_checkbox = ctk.CTkCheckBox(
            fine_tune_inner,
            text="Adaptive (varies by frequency/transients)",
            variable=self.adaptive_blend_var,
            command=self._on_parameter_change,
            font=ctk.CTkFont(size=11),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            text_color="#cccccc"
        )
        self.adaptive_blend_checkbox.pack(anchor="w", pady=(0, 5))

        # Buttons Section (reset only; Apply Denoising moved to transport row)
        buttons_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=10)
        buttons_frame.pack(fill="x", padx=5, pady=(0, 10))

        buttons_inner = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        buttons_inner.pack(fill="x", padx=10, pady=10)

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
        """Handle noise region selection from waveform - adds to multiple selections."""
        # Add to the panel's selections list
        self.noise_profile_panel.add_selection(start, end)
        count = len(self.noise_profile_panel.get_selections())
        self._set_status(f"Added noise region: {start:.2f}s - {end:.2f}s (Total: {count} selection(s))")

    def _learn_noise_profile_manual(self):
        """Learn noise profile from manually selected regions."""
        selections = self.noise_profile_panel.get_selections()

        if not selections:
            messagebox.showwarning(
                "No Selection",
                "Please select noise region(s) on the waveform first.\n\n"
                "1. Click 'Make Selection' button\n"
                "2. Click and drag on the waveform to select regions with only noise\n"
                "3. You can add multiple selections"
            )
            return

        try:
            self._set_status(f"Learning noise profile from {len(selections)} region(s)...")

            # Learn from all selections combined
            profile = self.denoiser.learn_noise_profile_from_regions(selections)
            self.noise_profile_panel.update_status(profile, selections)
            self._update_noise_floor_trace(profile)

            total_duration = sum(end - start for start, end in selections)
            self._set_status(f"Noise profile learned from {len(selections)} region(s) ({total_duration:.2f}s total)")

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

    def _update_noise_floor_trace(self, profile: Optional[NoiseProfile]):
        """Push learned noise spectrum into frequency view overlays."""
        if profile is None:
            self.waveform_original.set_noise_floor_curve(None, None)
            self.waveform_processed.set_noise_floor_curve(None, None)
            return

        # Recreate the frequency axis that matches the stored spectral mean
        n_fft = max(2, (len(profile.spectral_mean) - 1) * 2)
        freqs = librosa.fft_frequencies(sr=profile.sample_rate, n_fft=n_fft)
        floor_db = 20 * np.log10(np.maximum(profile.spectral_mean, 1e-12))

        self.waveform_original.set_noise_floor_curve(freqs, floor_db)
        self.waveform_processed.set_noise_floor_curve(freqs, floor_db)

    def _clear_noise_profile(self):
        """Clear the learned noise profile and all selections."""
        self.denoiser.clear_noise_profile()
        self.waveform_original.clear_selection()
        self.waveform_original.clear_noise_region()
        self._update_noise_floor_trace(None)
        self._set_status("Noise profile and selections cleared")

    def _toggle_use_profile(self, use: bool):
        """Toggle use of learned noise profile."""
        self.denoiser.set_use_learned_profile(use)

        if use:
            self._set_status("Using learned noise profile for denoising")
        else:
            self._set_status("Using adaptive noise estimation")

    def _toggle_noise_selection(self, enable: bool):
        """Toggle noise selection mode on the original waveform."""
        self.noise_selection_enabled = enable
        if enable:
            # Force view to original for selecting noise regions, keep position/play state
            self._set_active_waveform_view("original", preserve_position=True, preserve_play_state=True)
            self.waveform_original.set_view_mode("waveform")
            self.waveform_original.enable_selection(True)
        else:
            self.waveform_original.enable_selection(False)

        # Reflect state in panel
        self.noise_profile_panel.set_selection_enabled(enable)

        if enable:
            self._set_status("Selection mode ON - drag on waveform to add noise regions")
        else:
            self._set_status("Selection mode OFF - click on waveform to seek")

    def _on_remove_noise_selection(self, index: int, region: Tuple[float, float]):
        """Handle removal of a noise selection."""
        # Update the waveform display to remove the visual selection
        self.waveform_original.clear_selection()
        # Redraw all remaining selections
        selections = self.noise_profile_panel.get_selections()
        for start, end in selections:
            self.waveform_original.add_selection_rect(start, end)
        count = len(selections)
        self._set_status(f"Removed selection. {count} region(s) remaining.")

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

        # Try to start in last used folder, fall back to example_tracks then home
        last_dir = self.config.get("last_dir")
        if last_dir:
            initial_dir = Path(last_dir)
        else:
            initial_dir = Path(__file__).parent.parent.parent.parent / "example_tracks"
        if not initial_dir.exists():
            initial_dir = Path.home()

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes,
            initialdir=initial_dir
        )

        if file_path:
            # Persist last used directory
            self.config["last_dir"] = str(Path(file_path).parent)
            self._save_config()
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
        self._set_active_waveform_view("original")
        self.view_toggle_btn.configure(state="disabled")
        self.play_btn.configure(state="disabled", text="▶")
        self.stop_btn.configure(state="disabled")
        self.selection_btn.configure(state="disabled", text="Select Noise", fg_color="#1a5276")
        self.process_btn.configure(state="disabled", text="Apply Denoising")

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
            spectral_floor=self.spectral_floor_slider.get() / 100.0,
            noise_threshold=self.noise_threshold_slider.get(),
            artifact_control=self.artifact_control_slider.get() / 100.0,
            adaptive_blend=self.adaptive_blend_var.get(),
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

        # Spectral floor - relevant for Spectral, Wiener, and Spectral Gating
        floor_relevant = method_name in ["Spectral Subtraction", "Wiener Filter", "Combined (All Methods)", "Spectral Gating (Learned Profile)"]
        self.spectral_floor_slider.label.configure(text_color=active_color if floor_relevant else dim_color)

        # Noise threshold - relevant for all spectral methods
        threshold_relevant = True
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
        # Fine-tuning defaults
        self.spectral_floor_slider.set(5.0)
        self.noise_threshold_slider.set(1.0)
        self.artifact_control_slider.set(50.0)
        self.adaptive_blend_var.set(True)
        # Method default
        self.method_dropdown.set("Spectral Subtraction")
        self.denoiser.set_method(DenoiseMethod.SPECTRAL_SUBTRACTION)

    def _get_player(self, which: str) -> AudioPlayer:
        """Return the player for the requested view."""
        return self.player_original if which == "original" else self.player_processed

    def _get_waveform(self, which: str) -> WaveformDisplay:
        """Return the waveform widget for the requested view."""
        return self.waveform_original if which == "original" else self.waveform_processed

    def _refresh_play_button_label(self):
        """Update the unified play button text based on the active player's state."""
        active_player = self._get_player(self.active_waveform_view)
        self.play_btn.configure(text="⏸" if active_player.is_playing() else "▶")

    def _set_active_waveform_view(self, view: str, preserve_position: bool = False, preserve_play_state: bool = False):
        """Show the requested waveform and adjust controls, optionally preserving position/state."""
        if view not in ("original", "processed"):
            return

        if view == self.active_waveform_view:
            return

        # Capture current context
        current_view = self.active_waveform_view
        current_player = self._get_player(current_view)
        target_player = self._get_player(view)

        current_waveform = self._get_waveform(current_view)
        target_waveform = self._get_waveform(view)

        current_mode = current_waveform.get_view_mode()
        current_zoom = current_waveform.get_zoom_state()
        current_pos = current_player.get_position()
        was_playing = current_player.is_playing()

        # Sync position to target player for seamless bypass
        if preserve_position:
            target_player.seek(current_pos)
            target_waveform.update_playhead(current_pos)

        # Preserve play state: pause current, play target if it was playing
        if preserve_play_state and was_playing:
            current_player.pause()
            target_player.play()
        else:
            current_player.pause()
            target_player.pause()

        # Keep same view mode (waveform/spectrogram)
        target_waveform.set_view_mode(current_mode)
        target_waveform.set_zoom_state(*current_zoom)

        self.active_waveform_view = view

        if view == "original":
            self.waveform_processed.grid_remove()
            self.waveform_original.grid()
            self.view_toggle_btn.configure(text="Show Processed")
            self.play_btn.configure(fg_color="#ff9f43", hover_color="#f39c12")
            self.waveform_original.enable_selection(self.noise_selection_enabled)
            self.waveform_processed.enable_selection(False)
        else:
            self.waveform_original.grid_remove()
            self.waveform_processed.grid()
            self.view_toggle_btn.configure(text="Show Original")
            self.play_btn.configure(fg_color="#00d9ff", hover_color="#00b8d4")
            self.waveform_original.enable_selection(False)
            self.waveform_processed.enable_selection(False)

        self._refresh_play_button_label()

    def _toggle_waveform_view(self):
        """Toggle between original and processed views in the single waveform panel."""
        next_view = "processed" if self.active_waveform_view == "original" else "original"
        # Preserve position and play state for a bypass-like toggle
        self._set_active_waveform_view(next_view, preserve_position=True, preserve_play_state=True)

    def _toggle_selection_button(self):
        """Handle Make Selection button in the transport row."""
        enable = not self.noise_selection_enabled
        # Update button styling/text
        if enable:
            self.selection_btn.configure(text="Selection ON", fg_color="#ff6b6b", hover_color="#ff8f8f")
        else:
            self.selection_btn.configure(text="Make Selection", fg_color="#1a5276", hover_color="#2471a3")
        self._toggle_noise_selection(enable)

    def _toggle_play(self, which: Optional[str] = None):
        """Toggle play/pause for the active waveform (or a specific one)."""
        target = which or self.active_waveform_view
        player = self._get_player(target)
        other_player = self._get_player("processed" if target == "original" else "original")

        if other_player.is_playing():
            other_player.pause()

        if player.is_playing():
            player.pause()
        else:
            player.play()
            self.current_player = player

        self._refresh_play_button_label()

    def _stop_play(self, which: Optional[str] = None):
        """Stop playback for the active or specified track."""
        target = which or self.active_waveform_view
        player = self._get_player(target)
        waveform = self._get_waveform(target)
        player.stop()
        waveform.reset_playhead()
        self._refresh_play_button_label()

    def _on_seek(self, which: str, position: float):
        """Handle seek bar change."""
        player = self._get_player(which)
        waveform = self._get_waveform(which)
        player.seek(position)
        waveform.update_playhead(position)

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
                    self.process_btn.configure(state="normal", text="✨ Apply Denoising")

        except queue.Empty:
            pass

        # Check for drag-and-drop files (polled from instance variable for thread safety)
        if self._pending_drop_files is not None:
            file_list = self._pending_drop_files
            self._pending_drop_files = None
            self._process_dropped_files(file_list)

        # Update waveform playheads during playback
        # Note: update_playhead automatically skips canvas redraws in spectrogram mode
        # Update both frequency analyzers to keep them in sync during playback
        if self.player_original.is_playing():
            pos = self.player_original.get_position()
            self.waveform_original.update_playhead(pos)
            self.waveform_original.update_frequency_analyzer(self.waveform_original._audio_data, self.waveform_original._sample_rate, pos)
            self.waveform_processed.update_frequency_analyzer(self.waveform_processed._audio_data, self.waveform_processed._sample_rate, pos)

        if self.player_processed.is_playing():
            pos = self.player_processed.get_position()
            self.waveform_processed.update_playhead(pos)
            self.waveform_processed.update_frequency_analyzer(self.waveform_processed._audio_data, self.waveform_processed._sample_rate, pos)
            self.waveform_original.update_frequency_analyzer(self.waveform_original._audio_data, self.waveform_original._sample_rate, pos)

        # Update play buttons on completion
        if self.active_waveform_view == "original" and self.player_original.get_state() == PlaybackState.STOPPED:
            self.play_btn.configure(text="▶")
        if self.active_waveform_view == "processed" and self.player_processed.get_state() == PlaybackState.STOPPED:
            self.play_btn.configure(text="▶")

        # Schedule next update
        self.after(50, self._update_ui)

    def _on_audio_loaded(self, audio: np.ndarray, sr: int):
        """Handle audio loaded event."""
        # Update waveform display
        display_audio = audio[0] if audio.ndim == 2 else audio
        self.track_title = self.input_path.stem if self.input_path else "Audio"
        self.waveform_original.plot_waveform(
            display_audio, sr,
            f"{self.track_title} (Original)", "#ff9f43"
        )
        self.waveform_processed.plot_waveform(None, sr, f"{self.track_title} (Denoised)", "#00d9ff")

        # Clear any previous comparison audio
        self.waveform_original.set_comparison_audio(None)

        # Selection mode starts OFF - user enables via noise profile panel toggle
        self.noise_selection_enabled = False
        self.waveform_original.enable_selection(False)
        self.noise_profile_panel.set_selection_enabled(False)

        # Load into player
        self.player_original.load(audio, sr)

        # Enable controls
        self._set_active_waveform_view("original")
        self.play_btn.configure(state="normal", text="▶")
        self.stop_btn.configure(state="normal")
        self.view_toggle_btn.configure(state="disabled")
        self.selection_btn.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.noise_profile_panel.enable_controls(True)

        # Disable processed controls until processing
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
        self.waveform_processed.plot_waveform(display_audio, sr, f"{self.track_title} (Denoised)", "#00d9ff")

        # Set cross-comparison overlays so both views show both spectra
        # Original waveform shows denoised as comparison
        self.waveform_original.set_comparison_audio(display_audio, sr)
        # Processed waveform shows original as comparison
        original_audio = self.waveform_original._audio_data
        if original_audio is not None:
            self.waveform_processed.set_comparison_audio(original_audio, self.waveform_original._sample_rate)

        # Load into player
        self.player_processed.load(processed, sr)

        # Enable controls
        self.view_toggle_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.process_btn.configure(state="normal", text="✨ Apply Denoising", fg_color="#6c3483")
        self._refresh_play_button_label()

        # Update status
        profile_status = "with learned profile" if self.denoiser._use_learned_profile else "with adaptive estimation"
        self._set_status(f"Processing complete {profile_status} - Preview and adjust as needed")

    def _on_noise_profile_learned(self, profile: NoiseProfile, region: Tuple[float, float]):
        """Handle noise profile learned from auto-detect."""
        self.selected_noise_region = region
        self.waveform_original.set_noise_region(*region)
        # Add detected region to selections list/display
        self.noise_profile_panel.add_selection(*region)
        self.waveform_original.add_selection_rect(*region)
        self.noise_profile_panel.update_status(profile, region)
        self.noise_profile_panel.enable_learn_button(True)
        self._update_noise_floor_trace(profile)
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

