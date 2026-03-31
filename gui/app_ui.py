from __future__ import annotations

import logging
import os
import subprocess
import sys
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui import analysis_handlers, io_handlers
from gui.progression import ProgressTracker
from utils.config import settings
from utils.helpers import show_citation_help
from utils.resources import asset_path, user_log_dir

logger = logging.getLogger(__name__)


def _normalize_workflow_config(workflow_config: dict | None) -> dict:
    config = workflow_config or {}
    return {
        "im_method": config.get("im_method", "PGA"),
        "period_mode": config.get("period_mode"),
        "period_value": config.get("period_value"),
        "resolved_period": config.get("resolved_period"),
    }


class mainUI(ctk.CTk):
    def __init__(self, workflow_config: dict | None = None) -> None:
        super().__init__()

        self.title("Automated Seismic Fragilty and Vulnerability Assessment for Buildings (ASFRAVA-B)")
        self.geometry("1440x900")
        try:
            self.iconbitmap(str(asset_path("Logo.ico")))
        except Exception:
            logger.warning("Failed setting icon", exc_info=True)

        self.workflow_config = _normalize_workflow_config(workflow_config)

        self.capacity_data: pd.DataFrame | None = None
        self.building_params: pd.DataFrame | None = None
        self.capacity_Sd = None
        self.capacity_Sa = None
        self.ideal_dt = None
        self.ideal_vb = None
        self.gmrs_folderpath: str | None = None
        self.analysis_cancelled: bool = False

        # State vars must be initialized before _create_widgets so radio buttons can bind to them
        self.IMs_selection = tk.StringVar(value=self.workflow_config["im_method"])
        self.period_mode_var = ctk.StringVar(
            value=self.workflow_config.get("period_mode") or "calculated"
        )
        self.period_value_var = ctk.StringVar(
            value=str(self.workflow_config.get("period_value") or "")
        )

        self._configure_grid()
        self._create_widgets()
        self._apply_initial_state()

    def open_logs(self) -> None:
        path = str(user_log_dir())
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception:
            logger.exception("Failed to open log folder: %s", path)
            messagebox.showerror("Open Logs", f"Failed to open log folder:\n{path}")

    def _configure_grid(self) -> None:
        self.columnconfigure(0, weight=1, uniform="a")
        self.columnconfigure((1, 2), weight=2, uniform="a")

        self.rowconfigure(0, weight=1, uniform="b")
        self.rowconfigure(1, weight=8, uniform="b")
        self.rowconfigure(2, weight=2, uniform="b")
        self.rowconfigure(3, weight=2, uniform="b")
        self.rowconfigure(4, weight=4, uniform="b")

    def _create_widgets(self) -> None:
        # ── Header frames (row 0) ────────────────────────────────────────────
        self.frame_0 = ctk.CTkFrame(self, fg_color="white")
        self.frame_0.grid(column=0, row=0, sticky="nsew", padx=1, pady=1)

        self.frame_1 = ctk.CTkFrame(self, fg_color="white")
        self.frame_1.grid(column=1, row=0, columnspan=2, sticky="nsew", padx=1, pady=1)

        # ── Right-side plot frames (columns 1-2) ─────────────────────────────
        self.output_idealization_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_idealization_frame.grid(column=1, row=1, sticky="nsew", padx=1, pady=1)

        self.output_EDPs_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_EDPs_frame.grid(column=2, row=1, sticky="nsew", padx=1, pady=1)

        self.output_fragility_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_fragility_frame.grid(column=1, row=2, rowspan=3, sticky="nsew", padx=1, pady=(1, 20))

        self.output_vulnerability_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_vulnerability_frame.grid(column=2, row=2, rowspan=3, sticky="nsew", padx=1, pady=(1, 20))

        # ── Progress tracker ─────────────────────────────────────────────────
        self.progress = ProgressTracker(self)

        # ── Section titles (row 0, col 0 and cols 1-2) ───────────────────────
        self.label_title_inputs = ctk.CTkLabel(
            self, text="App Input", font=("Inter", 24, "bold"), fg_color="white"
        )
        self.label_title_inputs.grid(row=0, column=0, sticky="wn", padx=15, pady=10)

        self.label_title_graph = ctk.CTkLabel(
            self, text="Graph", font=("Inter", 24, "bold"), fg_color="white"
        )
        self.label_title_graph.grid(row=0, column=1, sticky="n", padx=15, pady=10, columnspan=2)

        # ── Graph panel labels (columns 1-2) ─────────────────────────────────
        self.label_idealization = ctk.CTkLabel(
            self, text="Idealization curve", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_idealization.grid(row=1, column=1, sticky="n", padx=15, pady=(3, 0))

        self.label_EDPs = ctk.CTkLabel(
            self, text="Engineering Demand Parameters", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_EDPs.grid(row=1, column=2, sticky="n", padx=15, pady=(3, 0))

        self.label_fragility = ctk.CTkLabel(
            self, text="Fragility curve (PGA)", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_fragility.grid(row=2, column=1, sticky="n", padx=15, pady=(3, 0))

        self.label_vulnerability = ctk.CTkLabel(
            self, text="Vulnerability curve (PGA)", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_vulnerability.grid(row=2, column=2, sticky="n", padx=15, pady=(3, 0))

        # ── Matplotlib canvases ───────────────────────────────────────────────
        self.fig_idealization = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_idealization = self.fig_idealization.add_subplot(111)
        self.ax_idealization.grid(True, alpha=0.3)
        self.canvas_idealization = FigureCanvasTkAgg(self.fig_idealization, master=self.output_idealization_frame)
        self.canvas_idealization.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_EDPs = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_EDPs = self.fig_EDPs.add_subplot(111)
        self.ax_EDPs.grid(True, alpha=0.3)
        self.canvas_EDPs = FigureCanvasTkAgg(self.fig_EDPs, master=self.output_EDPs_frame)
        self.canvas_EDPs.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_fragility = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_fragility = self.fig_fragility.add_subplot(111)
        self.ax_fragility.grid(True, alpha=0.3)
        self.canvas_fragility = FigureCanvasTkAgg(self.fig_fragility, master=self.output_fragility_frame)
        self.canvas_fragility.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_vulnerability = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_vulnerability = self.fig_vulnerability.add_subplot(111)
        self.ax_vulnerability.grid(True, alpha=0.3)
        self.canvas_vulnerability = FigureCanvasTkAgg(self.fig_vulnerability, master=self.output_vulnerability_frame)
        self.canvas_vulnerability.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Utility buttons (bottom-right) ────────────────────────────────────
        self.button_open_logs = ctk.CTkButton(
            self,
            text="View Logs",
            command=self.open_logs,
            width=50,
            height=20,
            text_color="white",
            font=("Inter", 10, "bold"),
            fg_color="#727272",
        )
        self.button_open_logs.grid(row=4, column=2, sticky="se", padx=(5, 30), pady=(0, 4))

        self.btn_help = ctk.CTkButton(
            self,
            text="?",
            width=20,
            height=20,
            text_color="white",
            font=("Inter", 10, "bold"),
            fg_color="#727272",
            command=lambda: show_citation_help(self),
        )
        self.btn_help.grid(row=4, column=2, sticky="se", padx=(5, 5), pady=(0, 4))

        # =====================================================================
        # LEFT PANEL — scrollable, single-column, all items stacked vertically
        # =====================================================================
        self.left_panel = ctk.CTkScrollableFrame(self, fg_color="white")
        self.left_panel.grid(row=1, column=0, rowspan=4, sticky="nsew", padx=(1, 0), pady=1)
        self.left_panel.columnconfigure(0, weight=1)

        _r = 0  # running row counter — keeps numbering honest

        # ── Model setup subtitle ─────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Model setup", font=("Arial bold", 14),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(10, 4)); _r += 1

        # ── Workflow IM label + combobox ─────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Intensity Measure", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        self.im_selection_combo = ttk.Combobox(
            self.left_panel, values=["PGA", "Sa(T)", "Sa(avg)"],
            textvariable=self.IMs_selection, state="readonly", font=("Arial", 12),
        )
        self.im_selection_combo.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 6))
        self.im_selection_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_im_changed())
        _r += 1

        # ── Period section (hidden when PGA) ─────────────────────────────────
        # Entire block is one frame so grid_remove() hides it atomically
        self._period_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self._period_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 6))
        self._period_frame.columnconfigure(0, weight=1)
        _period_row = _r; _r += 1

        ctk.CTkLabel(
            self._period_frame, text="Period", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 2))

        self.radio_period_calculated = ctk.CTkRadioButton(
            self._period_frame, text="Calculated",
            variable=self.period_mode_var, value="calculated",
            command=self._on_period_mode_changed, font=("Inter", 12),
        )
        self.radio_period_calculated.grid(row=1, column=0, sticky="w", pady=(0, 2))

        self.label_calculated_period = ctk.CTkLabel(
            self._period_frame, text="", font=("Inter", 12),
            fg_color="transparent", anchor="w", text_color="#4b5563",
        )
        self.label_calculated_period.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(0, 2))

        # "Input" row: radio + T= entry inline
        _input_row_frame = ctk.CTkFrame(self._period_frame, fg_color="transparent")
        _input_row_frame.grid(row=2, column=0, sticky="ew")

        self.radio_period_input = ctk.CTkRadioButton(
            _input_row_frame, text="Input",
            variable=self.period_mode_var, value="specified",
            command=self._on_period_mode_changed, font=("Inter", 12),
        )
        self.radio_period_input.grid(row=0, column=0, sticky="w")

        self._t_entry_frame = ctk.CTkFrame(_input_row_frame, fg_color="transparent")
        self._t_entry_frame.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ctk.CTkLabel(
            self._t_entry_frame, text="T =", font=("Inter", 12, "bold"), fg_color="transparent",
        ).grid(row=0, column=0, padx=(0, 4))

        self.entry_period_value = ctk.CTkEntry(
            self._t_entry_frame, textvariable=self.period_value_var,
            width=70, height=24, font=("Inter Light", 11),
        )
        self.entry_period_value.grid(row=0, column=1, padx=(0, 4))

        ctk.CTkLabel(
            self._t_entry_frame, text="s", font=("Inter", 12), fg_color="transparent",
        ).grid(row=0, column=2)

        # ── Idealization ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Idealization", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        self._idealization_radio_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self._idealization_radio_frame.grid(row=_r, column=0, sticky="w", padx=15, pady=(0, 10))
        _r += 1

        self.idealization_option = tk.StringVar(value="EPP")
        self.radioEPP = ctk.CTkRadioButton(
            self._idealization_radio_frame, text="EPP", font=("Inter Light", 12),
            variable=self.idealization_option, value="EPP",
        )
        self.radioEPP.grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.radioSH = ctk.CTkRadioButton(
            self._idealization_radio_frame, text="SH", font=("Inter Light", 12),
            variable=self.idealization_option, value="SH",
        )
        self.radioSH.grid(row=0, column=1, sticky="w")

        # ── Output folder ────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Choose output folder", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        _out_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        _out_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 6))
        _out_frame.columnconfigure(0, weight=1); _r += 1

        self.entry_folder_output = ctk.CTkEntry(_out_frame, height=25, font=("Inter Light", 10))
        self.entry_folder_output.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.entry_folder_output.insert(0, settings.last_output_dir or "")
        ctk.CTkButton(
            _out_frame, text="Folder",
            command=lambda: io_handlers.select_output_folder(self, self.entry_folder_output),
            width=55, height=25, text_color="white", font=("Inter", 12, "bold"),
        ).grid(row=0, column=1)

        # ── Capacity curve ───────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Capacity curve", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        _cap_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        _cap_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 6))
        _cap_frame.columnconfigure(0, weight=1); _r += 1

        self.entry_capacity = ctk.CTkEntry(_cap_frame, height=25, font=("Inter Light", 10))
        self.entry_capacity.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(
            _cap_frame, text="File",
            command=lambda: io_handlers.load_capacity_csv(self, self.entry_capacity),
            width=45, height=25, text_color="white", font=("Inter", 12, "bold"),
        ).grid(row=0, column=1)

        # ── Building parameters ──────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Building parameters", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        _bp_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        _bp_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 6))
        _bp_frame.columnconfigure(0, weight=1); _r += 1

        self.entry_building_parameter = ctk.CTkEntry(_bp_frame, height=25, font=("Inter Light", 10))
        self.entry_building_parameter.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(
            _bp_frame, text="File",
            command=lambda: (
                io_handlers.load_building_params(self, self.entry_building_parameter),
                analysis_handlers.compute_adrs(self, show_errors=False),
            ),
            width=45, height=25, text_color="white", font=("Inter", 12, "bold"),
        ).grid(row=0, column=1)

        # ── Ground motion records ────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Ground motion records", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2)); _r += 1

        _gmr_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        _gmr_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 10))
        _gmr_frame.columnconfigure(0, weight=1); _r += 1

        self.entry_gmrs = ctk.CTkEntry(_gmr_frame, height=25, font=("Inter Light", 10))
        self.entry_gmrs.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(
            _gmr_frame, text="Folder",
            command=lambda: io_handlers.select_gmrs_folder(self, self.entry_gmrs),
            width=55, height=25, text_color="white", font=("Inter", 12, "bold"),
        ).grid(row=0, column=1)

        ctk.CTkButton(
            self.left_panel, text="Perform Idealization",
            command=lambda: analysis_handlers.plot_and_idealize(self),
            height=28, text_color="white", font=("Inter", 12, "bold"),
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(4, 10)); _r += 1

        # ── Scaling setup ────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Scaling setup", font=("Arial bold", 14),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(4, 4)); _r += 1

        ctk.CTkLabel(self.left_panel, text="Min. scale", font=("Inter", 12), fg_color="transparent", anchor="w").grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 1)); _r += 1
        self.entry_min_scale = ctk.CTkEntry(self.left_panel, height=24, font=("Inter Light", 10))
        self.entry_min_scale.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 4))
        self.entry_min_scale.insert(0, "0.000001"); _r += 1

        ctk.CTkLabel(self.left_panel, text="Max. scale", font=("Inter", 12), fg_color="transparent", anchor="w").grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 1)); _r += 1
        self.entry_max_scale = ctk.CTkEntry(self.left_panel, height=24, font=("Inter Light", 10))
        self.entry_max_scale.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 4))
        self.entry_max_scale.insert(0, "1.5"); _r += 1

        ctk.CTkLabel(self.left_panel, text="Increment", font=("Inter", 12), fg_color="transparent", anchor="w").grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 1)); _r += 1
        self.entry_increment = ctk.CTkEntry(self.left_panel, height=24, font=("Inter Light", 10))
        self.entry_increment.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 10))
        self.entry_increment.insert(0, "0.1"); _r += 1

        # ── Analysis ─────────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Analysis", font=("Arial bold", 14),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(4, 4)); _r += 1

        self.save_intersections = tk.BooleanVar(value=False)
        self.chk_save_intersections = ctk.CTkCheckBox(
            self.left_panel, text="Save intersection plots",
            variable=self.save_intersections, font=("Inter", 12),
            checkbox_width=15, checkbox_height=15,
        )
        self.chk_save_intersections.grid(row=_r, column=0, sticky="w", padx=15, pady=(0, 2)); _r += 1

        self.fast_mode = tk.BooleanVar(value=False)
        self.chk_fast_mode = ctk.CTkCheckBox(
            self.left_panel, text="Fast mode",
            variable=self.fast_mode, font=("Inter", 12),
            checkbox_width=15, checkbox_height=15,
        )
        self.chk_fast_mode.grid(row=_r, column=0, sticky="w", padx=15, pady=(0, 6)); _r += 1

        self.button_generate_EDPs = ctk.CTkButton(
            self.left_panel, text="Generate EDPs",
            command=lambda: analysis_handlers.generate_EDPs(self),
            height=28, text_color="white", font=("Inter", 12, "bold"),
        )
        self.button_generate_EDPs.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 10)); _r += 1

        # ── Statistical setup ────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Statistical setup", font=("Arial bold", 14),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(4, 4)); _r += 1

        ctk.CTkLabel(
            self.left_panel, text="Regression method", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 1)); _r += 1

        self.regression_selection = ttk.Combobox(
            self.left_panel, values=["MSA", "GLM", "LogregML"], font=("Arial", 12)
        )
        self.regression_selection.set("MSA")
        self.regression_selection.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2))
        self.regression_selection.bind("<<ComboboxSelected>>", self.on_selection_regression)
        _r += 1

        # Secondary combobox row (swapped by on_selection_regression)
        _secondary_row = _r; _r += 1

        self.link_function_selection = ttk.Combobox(
            self.left_panel, values=["Probit", "Logit"], font=("Arial", 12)
        )
        self.link_function_selection.set("Probit")
        self.link_function_selection.grid(row=_secondary_row, column=0, sticky="ew", padx=15, pady=(0, 2))
        self.link_function_selection.grid_forget()

        self.regulation_selection = ttk.Combobox(
            self.left_panel,
            values=["No Regulation", "Medium Regulation", "High Regulation"],
            font=("Arial", 12),
        )
        self.regulation_selection.set("No Regulation")
        self.regulation_selection.grid(row=_secondary_row, column=0, sticky="ew", padx=15, pady=(0, 2))
        self.regulation_selection.grid_forget()

        self.msa_variant_selection = ttk.Combobox(
            self.left_panel, values=["J-MLE", "MLE"], font=("Arial", 12)
        )
        self.msa_variant_selection.set("J-MLE")
        self.msa_variant_selection.grid(row=_secondary_row, column=0, sticky="ew", padx=15, pady=(0, 6))

        # Store so on_selection_regression can re-grid them at the right row
        self._secondary_combo_row = _secondary_row

        # ── Loss ratios ──────────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_panel, text="Loss ratio", font=("Inter", 13),
            fg_color="transparent", anchor="w",
        ).grid(row=_r, column=0, sticky="ew", padx=15, pady=(4, 2)); _r += 1

        for ds_label, default_val, attr in [
            ("DS1", "0.33", "entry_loss_ratio_ds1"),
            ("DS2", "0.67", "entry_loss_ratio_ds2"),
            ("DS3", "1.0",  "entry_loss_ratio_ds3"),
        ]:
            _ds_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
            _ds_frame.grid(row=_r, column=0, sticky="ew", padx=15, pady=(0, 2))
            _ds_frame.columnconfigure(0, weight=1); _r += 1
            ctk.CTkLabel(
                _ds_frame, text=ds_label, font=("Inter", 12), fg_color="transparent", anchor="w",
            ).grid(row=0, column=0, sticky="w")
            entry = ctk.CTkEntry(_ds_frame, width=65, height=22, font=("Inter Light", 10))
            entry.grid(row=0, column=1, sticky="e")
            entry.insert(0, default_val)
            setattr(self, attr, entry)

        self.button_perform_statistical_fit = ctk.CTkButton(
            self.left_panel, text="Perform statistical fit",
            command=lambda: analysis_handlers.perform_statistical_fit(self),
            height=28, text_color="white", font=("Inter", 12, "bold"),
        )
        self.button_perform_statistical_fit.grid(row=_r, column=0, sticky="ew", padx=15, pady=(6, 14))

    # =========================================================================
    # Workflow state management
    # =========================================================================

    def _apply_initial_state(self) -> None:
        self.IMs_selection.set(self.workflow_config["im_method"])
        self.period_mode_var.set(self.workflow_config.get("period_mode") or "calculated")
        val = self.workflow_config.get("period_value")
        self.period_value_var.set(str(val) if val is not None else "")
        self._refresh_period_row_visibility()
        self.refresh_plot_labels()

    def _on_im_changed(self) -> None:
        self._update_workflow_config()
        self._refresh_period_row_visibility()
        self.refresh_plot_labels()

    def _on_period_mode_changed(self) -> None:
        self._update_workflow_config()
        self._refresh_t_entry_visibility()

    def _update_workflow_config(self) -> None:
        im = self.IMs_selection.get()
        mode = self.period_mode_var.get() if im != "PGA" else None
        raw_val = self.period_value_var.get().strip()
        period_val = None
        if mode == "specified" and raw_val:
            try:
                period_val = float(raw_val)
            except ValueError:
                period_val = None
        self.workflow_config["im_method"] = im
        self.workflow_config["period_mode"] = mode
        self.workflow_config["period_value"] = period_val
        if mode != "specified":
            self.workflow_config["resolved_period"] = None
        elif period_val is not None:
            self.workflow_config["resolved_period"] = period_val

    def _refresh_period_row_visibility(self) -> None:
        if self.IMs_selection.get() == "PGA":
            self._period_frame.grid_remove()
        else:
            self._period_frame.grid()
            self._refresh_t_entry_visibility()

    def _refresh_t_entry_visibility(self) -> None:
        if self.period_mode_var.get() == "specified":
            self._t_entry_frame.grid()
        else:
            self._t_entry_frame.grid_remove()

    def refresh_workflow_info(self) -> None:
        """Called by analysis_handlers after resolved_period is updated."""
        resolved = self.workflow_config.get("resolved_period")
        mode = self.workflow_config.get("period_mode")
        if resolved is not None and mode == "calculated":
            self.label_calculated_period.configure(text=f"| T = {resolved:.4g} sec")
        else:
            self.label_calculated_period.configure(text="")
        self.refresh_plot_labels()

    def refresh_plot_labels(self) -> None:
        im_method = self.workflow_config.get("im_method", "PGA")
        self.label_fragility.configure(text=f"Fragility curve ({im_method})")
        self.label_vulnerability.configure(text=f"Vulnerability curve ({im_method})")

    def on_selection_regression(self, event):
        method = self.regression_selection.get()

        self.link_function_selection.grid_forget()
        self.regulation_selection.grid_forget()
        self.msa_variant_selection.grid_forget()

        r = self._secondary_combo_row
        if method == "MSA":
            self.msa_variant_selection.grid(row=r, column=0, sticky="ew", padx=15, pady=(0, 6))
        elif method == "GLM":
            self.link_function_selection.grid(row=r, column=0, sticky="ew", padx=15, pady=(0, 2))
        elif method == "LogregML":
            self.regulation_selection.grid(row=r, column=0, sticky="ew", padx=15, pady=(0, 2))
