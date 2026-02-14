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
    from .ui_components import SeekBar, ParameterSlider, VerticalParameterSlider, NoiseProfilePanel, Tooltip
except ImportError:
    from denoiser import AudioDenoiser, NoiseProfile, DenoiseMethod
    from audio_player import AudioPlayer, PlaybackState
    from waveform_display import WaveformDisplay
    from ui_components import SeekBar, ParameterSlider, VerticalParameterSlider, NoiseProfilePanel, Tooltip

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
        self.player_noise_preview = AudioPlayer()
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

        self.output_format = ctk.CTkOptionMenu(
            file_frame,
            values=["FLAC", "WAV", "OGG"],
            font=ctk.CTkFont(size=12),
            fg_color="#2d5a27",
            button_color="#2d5a27",
            button_hover_color="#3d7a37",
            dropdown_fg_color="#1a1a2e",
            dropdown_hover_color="#333333",
            width=80,
            height=36,
        )
        self.output_format.set("FLAC")
        self.output_format.pack(side="left", padx=5)

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
        self.waveform_original.on_threshold_adjust = self._on_threshold_adjust
        self.waveform_original.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.waveform_original.plot_waveform(None, 44100, f"{self.track_title} (Original)", "#ff9f43")

        self.waveform_processed = WaveformDisplay(
            waveform_frame,
            on_seek=lambda pos: self._on_seek("processed", pos),
        )
        self.waveform_processed.on_threshold_adjust = self._on_threshold_adjust
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

        # (Apply Denoising button is placed in the parameter panel below the sliders)

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
            on_play_selection=self._on_play_selection,
            on_edit_selection=self._on_edit_selection,
            on_use_default=self._use_default_noise_profile,
        )
        self.noise_profile_panel.pack(fill="x", padx=4, pady=(4, 6))

        # Set the denoising method to Adaptive Blend
        self.denoiser.set_method(DenoiseMethod.ADAPTIVE_BLEND)

        # ═══════════════════════════════════════════════════════
        # Main Parameters — Reduction dB + Noise Threshold (vertical sliders)
        # ═══════════════════════════════════════════════════════
        params_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=8)
        params_frame.pack(fill="x", padx=4, pady=(0, 6))

        # Horizontal row for the two vertical sliders
        main_sliders_row = ctk.CTkFrame(params_frame, fg_color="transparent")
        main_sliders_row.pack(padx=8, pady=(6, 4))

        self.noise_threshold_slider = VerticalParameterSlider(
            main_sliders_row,
            label="Noise Threshold",
            from_=0.0,
            to=10.0,
            default=3.5,
            unit=" dB",
            command=self._on_parameter_change,
            slider_height=120,
        )
        self.noise_threshold_slider.pack(side="left", padx=(8, 15))

        self.reduction_db_slider = VerticalParameterSlider(
            main_sliders_row,
            label="Reduction",
            from_=0.0,
            to=40.0,
            default=18.0,
            unit=" dB",
            command=self._on_parameter_change,
            slider_height=120,
        )
        self.reduction_db_slider.pack(side="left", padx=(15, 8))

        # Apply Denoising button below the sliders
        self.process_btn = ctk.CTkButton(
            params_frame,
            text="✨ Apply Denoising",
            command=self._process_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            height=32,
            corner_radius=8,
            state="disabled"
        )
        self.process_btn.pack(fill="x", padx=10, pady=(0, 4))

        # Processing stats display (hidden until first process)
        self.stats_frame = ctk.CTkFrame(params_frame, fg_color="#1a2a3a", corner_radius=6)
        # Not packed yet — shown after first processing
        self._stats_widgets = []  # Track dynamically created stat rows
        self._stats_tooltip = None

        # ═══════════════════════════════════════════════════════
        # Fine-Tuning — Collapsible section with remaining params
        # ═══════════════════════════════════════════════════════
        fine_tune_frame = ctk.CTkFrame(scroll_frame, fg_color="#151525", corner_radius=8)
        fine_tune_frame.pack(fill="x", padx=4, pady=(0, 4))

        # Clickable header to expand/collapse, with reset button on the right
        self._fine_tune_expanded = False
        fine_tune_header = ctk.CTkFrame(fine_tune_frame, fg_color="transparent")
        fine_tune_header.pack(fill="x", padx=6, pady=(4, 2))

        self.fine_tune_toggle_btn = ctk.CTkButton(
            fine_tune_header,
            text="▶ Fine-Tuning",
            command=self._toggle_fine_tuning,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="transparent",
            hover_color="#222233",
            text_color="#bb8fce",
            anchor="w",
            height=24,
        )
        self.fine_tune_toggle_btn.pack(side="left", fill="x", expand=True)

        self.reset_btn = ctk.CTkButton(
            fine_tune_header,
            text="↺ Reset",
            command=self._reset_parameters,
            font=ctk.CTkFont(size=10),
            fg_color="#555555",
            hover_color="#777777",
            width=60,
            height=24,
            corner_radius=5,
        )
        self.reset_btn.pack(side="right", padx=(4, 0))

        # Collapsible content container
        self.fine_tune_inner = ctk.CTkFrame(fine_tune_frame, fg_color="transparent")
        # Starts collapsed — do NOT pack yet

        # Blend Original
        self.blend_slider = ParameterSlider(
            self.fine_tune_inner,
            label="Blend Original Signal",
            from_=0.0,
            to=50.0,
            default=5.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.blend_slider.pack(fill="x", padx=8, pady=(4, 4))

        # Transient Protection
        self.transient_slider = ParameterSlider(
            self.fine_tune_inner,
            label="Transient Protection",
            from_=0.0,
            to=50.0,
            default=15.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.transient_slider.pack(fill="x", padx=8, pady=(0, 4))

        # Spectral Floor (artifact prevention)
        self.spectral_floor_slider = ParameterSlider(
            self.fine_tune_inner,
            label="Spectral Floor",
            from_=0.0,
            to=10.0,
            default=2.5,
            unit="%",
            command=self._on_parameter_change,
            number_of_steps=100,
            decimal_places=2,
        )
        self.spectral_floor_slider.pack(fill="x", padx=8, pady=(0, 4))

        # Artifact Control (subtraction vs gating balance)
        self.artifact_control_slider = ParameterSlider(
            self.fine_tune_inner,
            label="Artifact Control",
            from_=0.0,
            to=100.0,
            default=70.0,
            unit="%",
            command=self._on_parameter_change
        )
        self.artifact_control_slider.pack(fill="x", padx=8, pady=(0, 2))

        artifact_desc = ctk.CTkLabel(
            self.fine_tune_inner,
            text="0%=Subtraction ↔ 100%=Gating",
            font=ctk.CTkFont(size=8),
            text_color="#666666"
        )
        artifact_desc.pack(anchor="w", padx=8, pady=(0, 4))

        # Adaptive Blend checkbox
        self.adaptive_blend_var = ctk.BooleanVar(value=True)
        self.adaptive_blend_checkbox = ctk.CTkCheckBox(
            self.fine_tune_inner,
            text="Adaptive (freq/transient-aware)",
            variable=self.adaptive_blend_var,
            command=self._on_parameter_change,
            font=ctk.CTkFont(size=10),
            fg_color="#6c3483",
            hover_color="#8e44ad",
            text_color="#cccccc"
        )
        self.adaptive_blend_checkbox.pack(anchor="w", padx=8, pady=(0, 6))

        # (Reset button is now in the Fine-Tuning header row)

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
        """Handle noise region selection from waveform.

        If the new selection overlaps significantly with an existing one,
        update that selection instead of adding a new one.
        """
        selections = self.noise_profile_panel.get_selections()

        # Check for overlap with existing selections
        best_overlap_idx = -1
        best_overlap_amount = 0.0
        for i, (s, e) in enumerate(selections):
            overlap_start = max(start, s)
            overlap_end = min(end, e)
            overlap = max(0, overlap_end - overlap_start)
            existing_duration = e - s
            # Consider it an edit if >=30% of existing region overlaps
            if existing_duration > 0 and overlap / existing_duration > 0.3:
                if overlap > best_overlap_amount:
                    best_overlap_amount = overlap
                    best_overlap_idx = i

        if best_overlap_idx >= 0:
            # Update existing selection
            self.noise_profile_panel._selections[best_overlap_idx] = (start, end)
            self.noise_profile_panel._update_selections_display()
            # Redraw waveform selections
            self.waveform_original.clear_selection()
            for s, e in self.noise_profile_panel.get_selections():
                self.waveform_original.add_selection_rect(s, e)
            self._set_status(f"Updated selection {best_overlap_idx + 1}: {start:.2f}s - {end:.2f}s")
        else:
            # Add new selection
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
            self._set_status(
                f"Noise profile learned from {len(selections)} region(s) "
                f"({total_duration:.2f}s total) — click Apply Denoising to process"
            )

            # Highlight the Apply button so the user knows to click it
            if self.denoiser.get_original() is not None and not self.is_processing:
                self.process_btn.configure(fg_color="#884499")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to learn noise profile:\n{str(e)}")

    def _learn_noise_profile_auto(self):
        """Auto-detect noise regions at beginning/end of file.

        Only finds regions — does NOT learn the profile or process.
        The user can review/edit the regions and then click Learn.
        """
        if self.denoiser.get_original() is None:
            return

        self._set_status("Auto-detecting noise regions at beginning/end of file...")

        def auto_detect_thread():
            try:
                regions = self.denoiser.auto_detect_noise_regions(min_duration=0.3)
                self.processing_queue.put(("noise_regions_detected", regions))
            except Exception as e:
                self.processing_queue.put(("error", str(e)))

        threading.Thread(target=auto_detect_thread, daemon=True).start()

    def _use_default_noise_profile(self):
        """Generate a default noise profile from the audio's statistical shape."""
        if self.denoiser.get_original() is None:
            return

        try:
            profile = self.denoiser.generate_default_noise_profile()
            self.noise_profile_panel.update_status(profile)
            self._update_noise_floor_trace(profile)
            self._set_status(
                "Default noise profile applied — drag control points on the "
                "Frequency view to fine-tune per-band threshold"
            )

            # Highlight Apply button
            if not self.is_processing:
                self.process_btn.configure(fg_color="#884499")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate default profile:\n{str(e)}")

    def _on_threshold_adjust(self, adjustments: dict):
        """Handle control-point threshold adjustments from the spectrum view."""
        self.denoiser.set_threshold_adjustments(adjustments)

        # Sync both waveform displays so the curve looks the same
        offsets = list(adjustments.values())
        self.waveform_original.set_ctrl_point_offsets(offsets)
        self.waveform_processed.set_ctrl_point_offsets(offsets)

        # Hint the user to reprocess
        if self.denoiser.get_original() is not None and not self.is_processing:
            self.process_btn.configure(fg_color="#884499")

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
        """Clear the learned noise profile, selections, and threshold adjustments."""
        self.denoiser.clear_noise_profile()
        self.denoiser.set_threshold_adjustments(None)
        self.waveform_original.clear_selection()
        self.waveform_original.clear_noise_region()
        self.waveform_original.reset_ctrl_points()
        self.waveform_processed.reset_ctrl_points()
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
            # Show selection rectangles when entering selection mode
            self.waveform_original.show_selection_rects()
        else:
            self.waveform_original.enable_selection(False)
            # Hide selection rectangles when leaving selection mode
            self.waveform_original.hide_selection_rects()

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

    def _on_play_selection(self, play: bool):
        """Handle play/stop selection button. Loops all selected noise regions."""
        if play:
            selections = self.noise_profile_panel.get_selections()
            if not selections or self.denoiser._audio is None:
                self.noise_profile_panel.set_playing_selection(False)
                return

            # Extract audio for all selected regions and concatenate
            sr = self.denoiser._sr
            audio = self.denoiser._audio  # shape (channels, samples)
            chunks = []
            for start_t, end_t in selections:
                s = int(start_t * sr)
                e = int(end_t * sr)
                s = max(0, min(s, audio.shape[1]))
                e = max(0, min(e, audio.shape[1]))
                if e > s:
                    chunks.append(audio[:, s:e])

            if not chunks:
                self.noise_profile_panel.set_playing_selection(False)
                return

            concat = np.concatenate(chunks, axis=1)
            self.player_noise_preview.stop()
            self.player_noise_preview.load(concat, sr)
            self.player_noise_preview.set_loop(True)
            self.player_noise_preview.play()
            self._set_status("Playing noise selection (looped)...")
        else:
            self.player_noise_preview.stop()
            self._set_status("Stopped noise preview.")

    def _on_edit_selection(self, index: int, new_start: float, new_end: float):
        """Handle editing of a noise selection's start/end time."""
        selections = self.noise_profile_panel.get_selections()
        if 0 <= index < len(selections):
            self.noise_profile_panel._selections[index] = (new_start, new_end)
            self.noise_profile_panel._update_selections_display()
            # Redraw waveform selections
            self.waveform_original.clear_selection()
            for start, end in self.noise_profile_panel.get_selections():
                self.waveform_original.add_selection_rect(start, end)
            self._set_status(f"Updated selection {index + 1}.")

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
        self.noise_profile_panel.make_selection_btn.configure(state="disabled")
        self.noise_profile_panel.set_selection_enabled(False)
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
            blend_original=self.blend_slider.get() / 100.0,
            reduction_db=self.reduction_db_slider.get(),
            transient_protection=self.transient_slider.get() / 100.0,
            spectral_floor=self.spectral_floor_slider.get() / 100.0,
            noise_threshold_db=self.noise_threshold_slider.get(),
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

    def _toggle_fine_tuning(self):
        """Toggle visibility of the fine-tuning section."""
        if self._fine_tune_expanded:
            self.fine_tune_inner.pack_forget()
            self.fine_tune_toggle_btn.configure(text="▶ Fine-Tuning")
            self._fine_tune_expanded = False
        else:
            self.fine_tune_inner.pack(fill="x", pady=(0, 5))
            self.fine_tune_toggle_btn.configure(text="▼ Fine-Tuning")
            self._fine_tune_expanded = True

    def _on_parameter_change(self, *args):
        """Handle parameter change - enable reprocessing hint."""
        if self.denoiser.get_original() is not None and not self.is_processing:
            self.process_btn.configure(fg_color="#884499")
        # Update the purple noise threshold line on the spectrum view
        # Convert dB slider value to multiplier for the visual display
        if hasattr(self, "noise_threshold_slider"):
            noise_thresh_db = self.noise_threshold_slider.get()
            noise_thresh_mult = 10 ** (noise_thresh_db / 20)
            self.waveform_original.set_noise_threshold_multiplier(noise_thresh_mult)
            self.waveform_processed.set_noise_threshold_multiplier(noise_thresh_mult)

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        # Main parameters
        self.reduction_db_slider.set(18.0)
        self.noise_threshold_slider.set(3.5)
        # Fine-tuning defaults
        self.blend_slider.set(5.0)
        self.transient_slider.set(15.0)
        self.spectral_floor_slider.set(2.5)
        self.artifact_control_slider.set(70.0)
        self.adaptive_blend_var.set(True)

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

    def _toggle_play(self, which: Optional[str] = None):
        """Toggle play/pause for the active waveform (or a specific one)."""
        target = which or self.active_waveform_view
        player = self._get_player(target)
        other_player = self._get_player("processed" if target == "original" else "original")

        if other_player.is_playing():
            other_player.pause()

        # Stop noise preview if it's playing
        if self.player_noise_preview.is_playing():
            self.player_noise_preview.stop()
            self.noise_profile_panel.set_playing_selection(False)

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

                elif msg[0] == "noise_profile_multi":
                    _, profile, regions = msg
                    self._on_noise_profile_auto_learned(profile, regions)

                elif msg[0] == "noise_regions_detected":
                    _, regions = msg
                    self._on_noise_regions_detected(regions)

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

        # Compute and display processing stats
        self._update_processing_stats(original_audio, display_audio, sr)

        # Update status
        profile_status = "with learned profile" if self.denoiser._use_learned_profile else "with adaptive estimation"
        self._set_status(f"Processing complete {profile_status} - Preview and adjust as needed")

    def _update_processing_stats(self, original: Optional[np.ndarray], processed: Optional[np.ndarray], sr: int):
        """Compute and display processing statistics segmented by volume level."""
        if original is None or processed is None or sr <= 0:
            return

        try:
            min_len = min(len(original), len(processed))
            orig = original[:min_len]
            proc = processed[:min_len]

            n_fft = 4096
            hop = 1024
            S_orig = np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)) ** 2
            S_proc = np.abs(librosa.stft(proc, n_fft=n_fft, hop_length=hop)) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Band definitions: Low (<1kHz), Mid (1-4kHz), HF (4kHz+)
            bands = [
                ("Low",  "0-1k",   0,    1000),
                ("Mid",  "1-4k",   1000, 4000),
                ("HF",   "4k+",    4000, sr // 2),
            ]

            # Frame-level energy for volume segmentation
            frame_energy = np.mean(S_orig, axis=0)
            frame_energy_db = 10 * np.log10(frame_energy + 1e-12)

            # Split frames into Quiet / Medium / Loud using percentiles
            p30 = np.percentile(frame_energy_db, 30)
            p70 = np.percentile(frame_energy_db, 70)
            quiet_mask = frame_energy_db <= p30
            mid_mask = (frame_energy_db > p30) & (frame_energy_db <= p70)
            loud_mask = frame_energy_db > p70

            volume_segments = [
                ("Quiet",  quiet_mask),
                ("Medium", mid_mask),
                ("Loud",   loud_mask),
            ]

            # Compute average dB reduction per band per volume segment
            # reduction = mean_band_dB(original) - mean_band_dB(processed)
            # Positive = noise/energy removed. For quiet sections this should
            # be large (noise removed). For loud, it should be small (signal kept).
            stats = {}  # stats[(vol_label, band_label)] = reduction_db
            for vol_label, vmask in volume_segments:
                if np.sum(vmask) < 2:
                    for b_label, _, _, _ in bands:
                        stats[(vol_label, b_label)] = 0.0
                    continue
                for b_label, _, lo, hi in bands:
                    fmask = (freqs >= lo) & (freqs < hi)
                    if np.sum(fmask) == 0:
                        stats[(vol_label, b_label)] = 0.0
                        continue
                    orig_band = S_orig[np.ix_(fmask, vmask)]
                    proc_band = S_proc[np.ix_(fmask, vmask)]
                    orig_mean_db = 10 * np.log10(np.mean(orig_band) + 1e-12)
                    proc_mean_db = 10 * np.log10(np.mean(proc_band) + 1e-12)
                    stats[(vol_label, b_label)] = orig_mean_db - proc_mean_db

            # HF detail preservation
            hf_mask = (freqs >= 4000) & (freqs <= 10000)
            S_orig_hf = np.sqrt(S_orig[hf_mask, :])
            S_proc_hf = np.sqrt(S_proc[hf_mask, :])
            orig_onset = np.mean(np.maximum(np.diff(S_orig_hf, axis=1), 0))
            proc_onset = np.mean(np.maximum(np.diff(S_proc_hf, axis=1), 0))
            hf_detail_pct = (proc_onset / (orig_onset + 1e-12)) * 100

            # Dynamics preservation
            orig_rms_frames = librosa.feature.rms(y=orig, frame_length=2048, hop_length=512)[0]
            proc_rms_frames = librosa.feature.rms(y=proc, frame_length=2048, hop_length=512)[0]
            orig_dyn = np.std(20 * np.log10(orig_rms_frames + 1e-12))
            proc_dyn = np.std(20 * np.log10(proc_rms_frames + 1e-12))
            dyn_pct = (proc_dyn / (orig_dyn + 1e-12)) * 100

            self._rebuild_stats_ui(bands, volume_segments, stats, hf_detail_pct, dyn_pct)
            self.stats_frame.pack(fill="x", padx=10, pady=(0, 6))

        except Exception as e:
            print(f"Stats computation error: {e}")
            import traceback; traceback.print_exc()
            self.stats_frame.pack_forget()

    def _rebuild_stats_ui(self, bands: list, volume_segments: list,
                          stats: dict, hf_detail_pct: float, dyn_pct: float):
        """Rebuild the stats panel with a columnar band layout."""
        # Clear previous widgets
        for w in self._stats_widgets:
            w.destroy()
        self._stats_widgets.clear()

        def _color_for_reduction(val: float, is_loud: bool = False) -> str:
            """Color code: for quiet/medium higher is better; for loud lower is better."""
            if is_loud:
                # Loud: low reduction = good (signal preserved)
                if val <= 1.5:
                    return "#4ade80"
                elif val <= 4.0:
                    return "#facc15"
                else:
                    return "#f87171"
            else:
                # Quiet/Medium: high reduction = good (noise removed)
                if val >= 6:
                    return "#4ade80"
                elif val >= 3:
                    return "#facc15"
                else:
                    return "#f87171"

        tiny = ctk.CTkFont(size=8)
        val_font = ctk.CTkFont(family="Consolas", size=11, weight="bold")
        label_font = ctk.CTkFont(size=9, weight="bold")
        sub_font = ctk.CTkFont(size=8)

        n_bands = len(bands)

        # ── Title row with help ──
        title_row = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=6, pady=(4, 1))
        self._stats_widgets.append(title_row)

        ctk.CTkLabel(
            title_row, text="dB Reduced by Volume Level",
            font=ctk.CTkFont(size=9, weight="bold"), text_color="#6c9bce",
        ).pack(side="left")

        help_lbl = ctk.CTkLabel(
            title_row, text="?",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#6c9bce", width=14, cursor="question_arrow",
        )
        help_lbl.pack(side="right", padx=(0, 2))

        tooltip_text = (
            "dB Reduced by Volume Level\n"
            "Shows how much energy (dB) was removed in\n"
            "each frequency band during quiet, medium,\n"
            "and loud passages.\n\n"
            "Quiet: Should be high — noise is being removed.\n"
            "Medium: Moderate — mix of noise and signal.\n"
            "Loud: Should be low — signal is preserved.\n\n"
            "HF Detail: % of high-frequency transients\n"
            "preserved. 100% = no detail lost.\n\n"
            "Dynamics: How well the original dynamic range\n"
            "is kept. 100% = unchanged."
        )
        if self._stats_tooltip is None:
            self._stats_tooltip = Tooltip(help_lbl, tooltip_text, delay=200)
        else:
            self._stats_tooltip._hide()
            self._stats_tooltip = Tooltip(help_lbl, tooltip_text, delay=200)

        # ── Band column headers: Low | Mid | HF ──
        header_frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=4, pady=(2, 0))
        self._stats_widgets.append(header_frame)
        for i in range(n_bands):
            header_frame.grid_columnconfigure(i, weight=1)

        for col, (b_label, b_range, _, _) in enumerate(bands):
            cell = ctk.CTkFrame(header_frame, fg_color="transparent")
            cell.grid(row=0, column=col, sticky="nsew", padx=1)
            ctk.CTkLabel(
                cell, text=b_label, font=label_font, text_color="#cccccc",
            ).pack()
            ctk.CTkLabel(
                cell, text=b_range, font=tiny, text_color="#666688",
            ).pack()

        # ── Volume segment rows ──
        for vol_label, _ in volume_segments:
            is_loud = (vol_label == "Loud")

            row_frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=4, pady=0)
            self._stats_widgets.append(row_frame)
            for i in range(n_bands):
                row_frame.grid_columnconfigure(i, weight=1)

            for col, (b_label, _, _, _) in enumerate(bands):
                val = stats.get((vol_label, b_label), 0.0)
                color = _color_for_reduction(val, is_loud)

                cell = ctk.CTkFrame(row_frame, fg_color="#151530", corner_radius=4)
                cell.grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

                ctk.CTkLabel(
                    cell, text=f"{val:.1f}", font=val_font,
                    text_color=color,
                ).pack(pady=(2, 0))
                ctk.CTkLabel(
                    cell, text=vol_label, font=tiny,
                    text_color="#555577",
                ).pack(pady=(0, 2))

        # ── Bottom summary: HF Detail + Dynamics ──
        sep = ctk.CTkFrame(self.stats_frame, height=1, fg_color="#252545")
        sep.pack(fill="x", padx=6, pady=(3, 2))
        self._stats_widgets.append(sep)

        summary = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        summary.pack(fill="x", padx=6, pady=(0, 4))
        self._stats_widgets.append(summary)
        summary.grid_columnconfigure(0, weight=1)
        summary.grid_columnconfigure(1, weight=1)

        # HF Detail
        hf_color = "#4ade80" if hf_detail_pct >= 75 else ("#facc15" if hf_detail_pct >= 55 else "#f87171")
        hf_cell = ctk.CTkFrame(summary, fg_color="transparent")
        hf_cell.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(hf_cell, text="HF Detail", font=sub_font, text_color="#666688").pack()
        ctk.CTkLabel(hf_cell, text=f"{hf_detail_pct:.0f}%", font=val_font, text_color=hf_color).pack()

        # Dynamics
        dyn_color = "#4ade80" if 90 <= dyn_pct <= 115 else ("#facc15" if 75 <= dyn_pct <= 130 else "#f87171")
        dyn_cell = ctk.CTkFrame(summary, fg_color="transparent")
        dyn_cell.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(dyn_cell, text="Dynamics", font=sub_font, text_color="#666688").pack()
        ctk.CTkLabel(dyn_cell, text=f"{dyn_pct:.0f}%", font=val_font, text_color=dyn_color).pack()

    def _on_noise_profile_learned(self, profile: NoiseProfile, region: Tuple[float, float]):
        """Handle noise profile learned from single-region auto-detect (legacy)."""
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

    def _on_noise_regions_detected(self, regions: List[Tuple[float, float]]):
        """Handle auto-detected noise regions — show them for user review."""
        if not regions:
            self._set_status("Auto-detect: no suitable noise regions found. Please select manually.")
            return

        # Clear any previous selections
        self.noise_profile_panel.clear_selections()
        self.waveform_original.clear_selection()

        # Switch to selection mode FIRST — this triggers a redraw (ax.clear),
        # so rectangles must be added AFTER this call.
        if not self.noise_selection_enabled:
            self._toggle_noise_selection(True)

        # Now add regions — these go on top of the freshly redrawn waveform
        for region in regions:
            self.noise_profile_panel.add_selection(*region)
            self.waveform_original.add_selection_rect(*region)

        # Build a descriptive status message
        parts = []
        track_dur = self.denoiser._audio.shape[1] / self.denoiser._sr if self.denoiser._audio is not None else 0
        for s, e in regions:
            loc = "beginning" if s < track_dur * 0.3 else "end"
            parts.append(f"{loc} ({s:.2f}s-{e:.2f}s)")
        self._set_status(
            f"Auto-detected {len(regions)} region(s): {', '.join(parts)} "
            f"— review and click 'Learn from Selections'"
        )

    def _on_noise_profile_auto_learned(self, profile: NoiseProfile, regions: List[Tuple[float, float]]):
        """Handle noise profile learned from auto-detect (legacy path)."""
        # Clear any previous selections
        self.noise_profile_panel.clear_selections()
        self.waveform_original.clear_selection()

        for region in regions:
            self.noise_profile_panel.add_selection(*region)
            self.waveform_original.add_selection_rect(*region)

        self.waveform_original.show_selection_rects()

        if regions:
            self.selected_noise_region = regions[0]
            self.waveform_original.set_noise_region(*regions[0])

        self.noise_profile_panel.update_status(profile, regions[0] if regions else (0, 0))
        self.noise_profile_panel.enable_learn_button(True)
        self._update_noise_floor_trace(profile)

        parts = []
        track_dur = self.denoiser._audio.shape[1] / self.denoiser._sr if self.denoiser._audio is not None else 0
        for s, e in regions:
            loc = "beginning" if s < track_dur * 0.3 else "end"
            parts.append(f"{loc} ({s:.2f}s-{e:.2f}s)")
        self._set_status(f"Auto-detected {len(regions)} noise region(s): {', '.join(parts)}")

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

