"""Reusable CustomTkinter components for the Sound Denoiser UI."""

from typing import List, Optional, Tuple, TYPE_CHECKING

import customtkinter as ctk

if TYPE_CHECKING:  # Only imported for type hints to avoid circular dependencies at runtime
    from .denoiser import NoiseProfile


class SeekBar(ctk.CTkFrame):
    """Seekable progress bar for audio playback with time display."""

    def __init__(self, master, color: str = "#00d9ff", on_seek=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.on_seek = on_seek
        self._duration = 0.0
        self._is_seeking = False

        self.time_current = ctk.CTkLabel(
            self,
            text="0:00",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=40,
        )
        self.time_current.pack(side="left", padx=(0, 5))

        self.slider = ctk.CTkSlider(
            self,
            from_=0,
            to=1,
            number_of_steps=1000,
            command=self._on_slider_change,
            progress_color=color,
            button_color=color,
            button_hover_color=color,
            height=12,
        )
        self.slider.set(0)
        self.slider.pack(side="left", fill="x", expand=True)

        self.slider.bind("<ButtonPress-1>", self._on_seek_start)
        self.slider.bind("<ButtonRelease-1>", self._on_seek_end)

        self.time_duration = ctk.CTkLabel(
            self,
            text="0:00",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
            width=40,
        )
        self.time_duration.pack(side="left", padx=(5, 0))

    def _format_time(self, seconds: float) -> str:
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
        if self._is_seeking and self.on_seek:
            self.on_seek(value)
        current_time = value * self._duration
        self.time_current.configure(text=self._format_time(current_time))

    def _on_seek_start(self, event):
        self._is_seeking = True

    def _on_seek_end(self, event):
        if self._is_seeking and self.on_seek:
            self.on_seek(self.slider.get())
        self._is_seeking = False

    def set_duration(self, duration: float):
        self._duration = duration
        self.time_duration.configure(text=self._format_time(duration))

    def set_position(self, position: float):
        if not self._is_seeking:
            self.slider.set(position)
            current_time = position * self._duration
            self.time_current.configure(text=self._format_time(current_time))

    def get_position(self) -> float:
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
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.unit = unit
        self.command = command

        self.label = ctk.CTkLabel(
            self,
            text=label,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#cccccc",
        )
        self.label.pack(anchor="w")

        slider_frame = ctk.CTkFrame(self, fg_color="transparent")
        slider_frame.pack(fill="x", pady=(1, 0))

        self.value_label = ctk.CTkLabel(
            slider_frame,
            text=f"{default:.1f}{unit}",
            font=ctk.CTkFont(size=10),
            text_color="#00d9ff",
            width=50,
        )
        self.value_label.pack(side="right", padx=(4, 0))

        self.slider = ctk.CTkSlider(
            slider_frame,
            from_=from_,
            to=to,
            number_of_steps=100,
            command=self._on_change,
            progress_color="#00d9ff",
            button_color="#00d9ff",
            button_hover_color="#00b8d4",
        )
        self.slider.set(default)
        self.slider.pack(side="left", fill="x", expand=True)

    def _on_change(self, value):
        self.value_label.configure(text=f"{value:.1f}{self.unit}")
        if self.command:
            self.command(value)

    def get(self) -> float:
        return self.slider.get()

    def set(self, value: float):
        self.slider.set(value)
        self.value_label.configure(text=f"{value:.1f}{self.unit}")


class VerticalParameterSlider(ctk.CTkFrame):
    """Vertical parameter slider with label on top, value display, and vertical slider."""

    def __init__(
        self,
        master,
        label: str,
        from_: float,
        to: float,
        default: float,
        unit: str = "",
        command=None,
        slider_height: int = 140,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.unit = unit
        self.command = command

        # Label at top
        self.label = ctk.CTkLabel(
            self,
            text=label,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#cccccc",
        )
        self.label.pack(pady=(0, 2))

        # Vertical slider
        self.slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            orientation="vertical",
            number_of_steps=100,
            command=self._on_change,
            progress_color="#00d9ff",
            button_color="#00d9ff",
            button_hover_color="#00b8d4",
            height=slider_height,
            width=18,
        )
        self.slider.set(default)
        self.slider.pack(pady=(0, 2))

        # Value display below slider
        self.value_label = ctk.CTkLabel(
            self,
            text=f"{default:.1f}{unit}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00d9ff",
        )
        self.value_label.pack(pady=(0, 0))

    def _on_change(self, value):
        self.value_label.configure(text=f"{value:.1f}{self.unit}")
        if self.command:
            self.command(value)

    def get(self) -> float:
        return self.slider.get()

    def set(self, value: float):
        self.slider.set(value)
        self.value_label.configure(text=f"{value:.1f}{self.unit}")


class NoiseProfilePanel(ctk.CTkFrame):
    """Panel for noise profile learning controls with multiple selection support."""

    def __init__(
        self,
        master,
        on_learn_manual,
        on_learn_auto,
        on_clear,
        on_toggle_use,
        on_toggle_selection=None,
        on_remove_selection=None,
        on_play_selection=None,
        on_edit_selection=None,
        **kwargs,
    ):
        super().__init__(master, fg_color="#151525", corner_radius=8, **kwargs)

        self.on_learn_manual = on_learn_manual
        self.on_learn_auto = on_learn_auto
        self.on_clear = on_clear
        self.on_toggle_use = on_toggle_use
        self.on_toggle_selection = on_toggle_selection
        self.on_remove_selection = on_remove_selection
        self.on_play_selection = on_play_selection
        self.on_edit_selection = on_edit_selection
        self._selection_enabled = False
        self._selections: List[Tuple[float, float]] = []
        self._selection_widgets = []
        self._is_playing_selection = False

        self._setup_ui()

    def _setup_ui(self):
        # ── Top row: Auto Detect + Make Selection + Clear ──
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=6, pady=(6, 3))

        self.auto_btn = ctk.CTkButton(
            top_row,
            text="Auto Detect",
            command=self.on_learn_auto,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#444444",
            hover_color="#555555",
            height=26,
            corner_radius=5,
            state="disabled",
        )
        self.auto_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.make_selection_btn = ctk.CTkButton(
            top_row,
            text="Select",
            command=self._on_toggle_selection,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1a5276",
            hover_color="#2471a3",
            height=26,
            corner_radius=5,
            state="disabled",
        )
        self.make_selection_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.clear_btn = ctk.CTkButton(
            top_row,
            text="Clear",
            command=self._on_clear,
            font=ctk.CTkFont(size=9),
            fg_color="#555555",
            hover_color="#777777",
            width=42,
            height=26,
            corner_radius=5,
            state="disabled",
        )
        self.clear_btn.pack(side="right")

        # ── Status + Use profile on one row ──
        profile_row = ctk.CTkFrame(self, fg_color="transparent")
        profile_row.pack(fill="x", padx=6, pady=(1, 3))

        self.status_label = ctk.CTkLabel(
            profile_row,
            text="No profile",
            font=ctk.CTkFont(size=10),
            text_color="#888888",
        )
        self.status_label.pack(side="left", padx=(2, 5))

        self.use_profile_var = ctk.BooleanVar(value=False)
        self.use_profile_switch = ctk.CTkSwitch(
            profile_row,
            text="Use",
            variable=self.use_profile_var,
            command=self._on_toggle,
            font=ctk.CTkFont(size=10),
            progress_color="#ff6b6b",
            button_color="#ff6b6b",
            button_hover_color="#ff8f8f",
            state="disabled",
            width=40,
        )
        self.use_profile_switch.pack(side="right", padx=(0, 2))

        sep = ctk.CTkFrame(self, height=1, fg_color="#333333")
        sep.pack(fill="x", padx=6, pady=(1, 3))

        # ── Selected Regions ──
        self.selections_frame = ctk.CTkFrame(self, fg_color="#1a2a3a", corner_radius=5)
        self.selections_frame.pack(fill="x", padx=6, pady=(0, 3))

        self.selections_title = ctk.CTkLabel(
            self.selections_frame,
            text="Selected Regions (0):",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#aaaaaa",
        )
        self.selections_title.pack(pady=(4, 2), padx=6, anchor="w")

        self.selections_list = ctk.CTkFrame(self.selections_frame, fg_color="transparent")
        self.selections_list.pack(fill="x", padx=3, pady=(0, 2))

        self.no_selections_label = ctk.CTkLabel(
            self.selections_list,
            text="No regions selected",
            font=ctk.CTkFont(size=9),
            text_color="#555555",
        )
        self.no_selections_label.pack(pady=2)

        # Play Selection button
        self.play_selection_btn = ctk.CTkButton(
            self.selections_frame,
            text="Play Selection",
            command=self._on_play_selection,
            font=ctk.CTkFont(size=9, weight="bold"),
            fg_color="#2a4a6a",
            hover_color="#3a5a7a",
            height=22,
            corner_radius=5,
            state="disabled",
        )
        self.play_selection_btn.pack(fill="x", padx=6, pady=(0, 4))

        # ── Learn button ──
        self.learn_btn = ctk.CTkButton(
            self,
            text="Learn from Selections",
            command=self.on_learn_manual,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#2d5a27",
            hover_color="#3d7a37",
            height=26,
            corner_radius=5,
            state="disabled",
        )
        self.learn_btn.pack(fill="x", padx=6, pady=(0, 6))

    def _on_toggle(self):
        self.on_toggle_use(self.use_profile_var.get())

    def _on_toggle_selection(self):
        """Handle Make Selection button click — toggles selection mode."""
        self._selection_enabled = not self._selection_enabled
        if self._selection_enabled:
            self.make_selection_btn.configure(
                text="Selection ON", fg_color="#ff6b6b", hover_color="#ff8f8f"
            )
        else:
            self.make_selection_btn.configure(
                text="Select", fg_color="#1a5276", hover_color="#2471a3"
            )
        if self.on_toggle_selection:
            self.on_toggle_selection(self._selection_enabled)

    def set_selection_enabled(self, enable: bool):
        self._selection_enabled = enable
        if enable:
            self.make_selection_btn.configure(
                text="Selection ON", fg_color="#ff6b6b", hover_color="#ff8f8f"
            )
        else:
            self.make_selection_btn.configure(
                text="Select", fg_color="#1a5276", hover_color="#2471a3"
            )

    def add_selection(self, start: float, end: float):
        self._selections.append((start, end))
        self._update_selections_display()
        self._update_learn_button()

    def remove_selection(self, index: int):
        if 0 <= index < len(self._selections):
            removed = self._selections.pop(index)
            self._update_selections_display()
            self._update_learn_button()
            if self.on_remove_selection:
                self.on_remove_selection(index, removed)

    def clear_selections(self):
        self._selections = []
        self._update_selections_display()
        self._update_learn_button()

    def get_selections(self) -> List[Tuple[float, float]]:
        return self._selections.copy()

    def _update_selections_display(self):
        for widget in self._selection_widgets:
            widget.destroy()
        self._selection_widgets = []

        count = len(self._selections)
        self.selections_title.configure(text=f"Selected Regions ({count}):")

        if count == 0:
            self.no_selections_label.pack(pady=5)
        else:
            self.no_selections_label.pack_forget()

            for i, (start, end) in enumerate(self._selections):
                duration = end - start

                row = ctk.CTkFrame(self.selections_list, fg_color="transparent")
                row.pack(fill="x", pady=1)
                self._selection_widgets.append(row)

                # Row number label
                num_label = ctk.CTkLabel(
                    row,
                    text=f"{i + 1}.",
                    font=ctk.CTkFont(size=10),
                    text_color="#aaaaaa",
                    width=18,
                )
                num_label.pack(side="left", padx=(2, 0))

                # Editable start time entry
                start_var = ctk.StringVar(value=f"{start:.2f}")
                start_entry = ctk.CTkEntry(
                    row,
                    textvariable=start_var,
                    font=ctk.CTkFont(size=10),
                    width=55,
                    height=20,
                    fg_color="#1a2a3a",
                    text_color="#cccccc",
                    border_color="#444444",
                    border_width=1,
                    corner_radius=3,
                )
                start_entry.pack(side="left", padx=(2, 0))

                dash_label = ctk.CTkLabel(
                    row, text="-", font=ctk.CTkFont(size=10), text_color="#888888", width=10,
                )
                dash_label.pack(side="left", padx=1)

                # Editable end time entry
                end_var = ctk.StringVar(value=f"{end:.2f}")
                end_entry = ctk.CTkEntry(
                    row,
                    textvariable=end_var,
                    font=ctk.CTkFont(size=10),
                    width=55,
                    height=20,
                    fg_color="#1a2a3a",
                    text_color="#cccccc",
                    border_color="#444444",
                    border_width=1,
                    corner_radius=3,
                )
                end_entry.pack(side="left", padx=(0, 2))

                # Bind both entries to the same callback (both vars now exist)
                start_entry.bind("<Return>", lambda e, idx=i, sv=start_var, ev=end_var: self._on_entry_edit(idx, sv, ev))
                start_entry.bind("<FocusOut>", lambda e, idx=i, sv=start_var, ev=end_var: self._on_entry_edit(idx, sv, ev))
                end_entry.bind("<Return>", lambda e, idx=i, sv=start_var, ev=end_var: self._on_entry_edit(idx, sv, ev))
                end_entry.bind("<FocusOut>", lambda e, idx=i, sv=start_var, ev=end_var: self._on_entry_edit(idx, sv, ev))

                # Duration label
                dur_label = ctk.CTkLabel(
                    row,
                    text=f"({duration:.2f}s)",
                    font=ctk.CTkFont(size=9),
                    text_color="#666666",
                    width=45,
                )
                dur_label.pack(side="left", padx=(2, 0))

                del_btn = ctk.CTkButton(
                    row,
                    text="X",
                    command=lambda idx=i: self.remove_selection(idx),
                    font=ctk.CTkFont(size=10, weight="bold"),
                    fg_color="#662222",
                    hover_color="#883333",
                    width=24,
                    height=20,
                    corner_radius=4,
                )
                del_btn.pack(side="right", padx=2)

    def _on_entry_edit(self, index: int, start_var: ctk.StringVar, end_var: ctk.StringVar):
        """Handle when user edits a start or end time entry field."""
        if index >= len(self._selections):
            return
        current_start, current_end = self._selections[index]
        try:
            new_start = float(start_var.get())
        except ValueError:
            new_start = current_start
            start_var.set(f"{current_start:.2f}")

        try:
            new_end = float(end_var.get())
        except ValueError:
            new_end = current_end
            end_var.set(f"{current_end:.2f}")

        # Validate: start < end, both >= 0
        new_start = max(0.0, new_start)
        new_end = max(new_start + 0.01, new_end)

        if new_start != current_start or new_end != current_end:
            self._selections[index] = (new_start, new_end)
            if self.on_edit_selection:
                self.on_edit_selection(index, new_start, new_end)

    def _on_play_selection(self):
        """Handle play/stop selection button click."""
        if self._is_playing_selection:
            self._is_playing_selection = False
            self.play_selection_btn.configure(text="Play Selection", fg_color="#2a4a6a", hover_color="#3a5a7a")
            if self.on_play_selection:
                self.on_play_selection(False)
        else:
            self._is_playing_selection = True
            self.play_selection_btn.configure(text="Stop", fg_color="#882222", hover_color="#aa3333")
            if self.on_play_selection:
                self.on_play_selection(True)

    def set_playing_selection(self, playing: bool):
        """Update the play button state externally (e.g., when playback finishes)."""
        self._is_playing_selection = playing
        if playing:
            self.play_selection_btn.configure(text="Stop", fg_color="#882222", hover_color="#aa3333")
        else:
            self.play_selection_btn.configure(text="Play Selection", fg_color="#2a4a6a", hover_color="#3a5a7a")

    def _update_learn_button(self):
        if len(self._selections) > 0:
            self.learn_btn.configure(state="normal")
            total_duration = sum(end - start for start, end in self._selections)
            self.learn_btn.configure(text=f"Learn from {len(self._selections)} Selection(s)")
            self.play_selection_btn.configure(state="normal")
        else:
            self.learn_btn.configure(state="disabled")
            self.learn_btn.configure(text="Learn from Selections")
            self.play_selection_btn.configure(state="disabled")

    def _on_clear(self):
        self.clear_selections()
        self.on_clear()
        self.update_status(None)

    def enable_controls(self, enable: bool = True):
        state = "normal" if enable else "disabled"
        self.auto_btn.configure(state=state)
        self.make_selection_btn.configure(state=state)

    def enable_learn_button(self, enable: bool = True):
        if enable and len(self._selections) > 0:
            self.learn_btn.configure(state="normal")
        elif not enable:
            self.learn_btn.configure(state="disabled")

    def update_status(self, profile: Optional["NoiseProfile"], regions: Optional[List[Tuple[float, float]]] = None):
        if profile is not None:
            self.status_label.configure(text="Profile learned", text_color="#4ade80")
            self.use_profile_switch.configure(state="normal")
            self.use_profile_var.set(True)
            self.clear_btn.configure(state="normal")
            self._on_toggle()
        else:
            self.status_label.configure(text="No profile", text_color="#888888")
            self.use_profile_switch.configure(state="disabled")
            self.use_profile_var.set(False)
            self.clear_btn.configure(state="disabled")

