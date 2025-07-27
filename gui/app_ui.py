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


class mainUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Automated Seismic Fragilty and Vulnerability Assessment for Buildings (ASFRAVA-B)")
        self.geometry("1366x768")
        try:
            self.iconbitmap(str(asset_path("Logo.ico")))
        except Exception:
            logger.warning("Failed setting icon", exc_info=True)

        # App-wide state (no globals)
        self.capacity_data: pd.DataFrame | None = None
        self.building_params: pd.DataFrame | None = None
        self.ideal_dt = None
        self.ideal_vb = None
        self.gmrs_folderpath: str | None = None
        self.analysis_cancelled: bool = False

        # Configure layout and create UI elements
        self._configure_grid()
        self._create_widgets()

    # ------------------------------------------------------------------
    #                     LOG HANDLERS
    # ------------------------------------------------------------------
    def open_logs(self) -> None:
        """Open the application log folder in the OS file explorer."""
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

    # ------------------------------------------------------------------
    #                     UI LAYOUT / WIDGETS
    # ------------------------------------------------------------------
    def _configure_grid(self) -> None:
        # Columns
        self.columnconfigure(0, weight=1, uniform="a")
        self.columnconfigure((1, 2), weight=2, uniform="a")
        # Rows
        self.rowconfigure(0, weight=1, uniform="b")
        self.rowconfigure(1, weight=8, uniform="b")
        self.rowconfigure(2, weight=2, uniform="b")
        self.rowconfigure(3, weight=2, uniform="b")
        self.rowconfigure(4, weight=4, uniform="b")

    def _create_widgets(self) -> None:
        # ============ Frames ============ #
        self.frame_0 = ctk.CTkFrame(self, fg_color="white")
        self.frame_0.grid(column=0, row=0, sticky="nsew", padx=1, pady=1)
        self.frame_0.columnconfigure(0, weight=1)
        self.frame_0.rowconfigure(0, weight=0)

        self.frame_1 = ctk.CTkFrame(self, fg_color="white")
        self.frame_1.grid(column=1, row=0, columnspan=2, sticky="nsew", padx=1, pady=1)
        self.frame_1.columnconfigure(0, weight=1)
        self.frame_1.rowconfigure(0, weight=0)

        self.frame_2 = ctk.CTkFrame(self, fg_color="white")
        self.frame_2.grid(column=0, row=1, rowspan=2, sticky="nsew", padx=1, pady=1)
        self.frame_2.columnconfigure(0, weight=1)
        self.frame_2.rowconfigure(0, weight=1)

        self.frame_3 = ctk.CTkFrame(self, fg_color="white")
        self.frame_3.grid(column=0, row=3, rowspan=1, sticky="nsew", padx=1, pady=1)
        self.frame_3.columnconfigure(0, weight=1)
        self.frame_3.rowconfigure(0, weight=1)

        self.frame_4 = ctk.CTkFrame(self, fg_color="white")
        self.frame_4.grid(column=0, row=4, rowspan=1, sticky="nsew", padx=1, pady=1)
        self.frame_4.columnconfigure(0, weight=1)
        self.frame_4.rowconfigure(0, weight=1)

        self.output_idealization_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_idealization_frame.grid(column=1, row=1, rowspan=1, sticky="nsew", padx=(1, 1), pady=(1, 1))

        self.output_EDPs_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_EDPs_frame.grid(column=2, row=1, rowspan=1, sticky="nsew", padx=(1, 1), pady=(1, 1))

        self.output_fragility_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_fragility_frame.grid(column=1, row=2, rowspan=3, sticky="nsew", padx=(1, 1), pady=(1, 20))

        self.output_vulnerability_frame = ctk.CTkFrame(self, width=400, height=200, fg_color="white")
        self.output_vulnerability_frame.grid(column=2, row=2, rowspan=3, sticky="nsew", padx=(1, 1), pady=(1, 20))

        self.progress = ProgressTracker(self)

        # ============ Labels ============ #
        # Title
        self.label_Title = ctk.CTkLabel(self, text="App Input", font=("Inter", 24, "bold"), fg_color="white")
        self.label_Title.grid(row=0, column=0, sticky="wn", padx=15, pady=10, columnspan=1)
        self.label_Title = ctk.CTkLabel(self, text="Graph", font=("Inter", 24, "bold"), fg_color="white")
        self.label_Title.grid(row=0, column=1, sticky="n", padx=15, pady=10, columnspan=2)

        # Subtitle
        self.label_subtitle_0 = ctk.CTkLabel(self, text="Model setup", font=("Arial bold", 14), fg_color="white")
        self.label_subtitle_0.grid(row=1, column=0, sticky="wn", padx=15, pady=(5, 0), columnspan=1)
        self.label_subtitle_1 = ctk.CTkLabel(self, text="Scaling setup", font=("Arial bold", 14), fg_color="white")
        self.label_subtitle_1.grid(row=2, column=0, sticky="wn", padx=15, pady=(5, 0), columnspan=1)
        self.label_subtitle_3 = ctk.CTkLabel(self, text="Analysis", font=("Arial bold", 14), fg_color="white")
        self.label_subtitle_3.grid(row=3, column=0, sticky="wn", padx=15, pady=(5, 0), columnspan=1)
        self.label_subtitle_2 = ctk.CTkLabel(self, text="Statistical setup", font=("Arial bold", 14), fg_color="white")
        self.label_subtitle_2.grid(row=4, column=0, sticky="wn", padx=15, pady=(5, 0), columnspan=1)

        # Settings
        self.label_0 = ctk.CTkLabel(self, text="Idealization:", font=("Inter", 14), fg_color="white")
        self.label_0.grid(row=1, column=0, sticky="wn", padx=15, pady=(35, 0), columnspan=1)
        self.label_1 = ctk.CTkLabel(self, text="IM:", font=("Inter", 14), fg_color="white")
        self.label_1.grid(row=1, column=0, sticky="wn", padx=15, pady=(70, 0), columnspan=1)
        self.label_2 = ctk.CTkLabel(self, text="Choose output folder", font=("Inter", 14), fg_color="white")
        self.label_2.grid(row=1, column=0, sticky="wn", padx=15, pady=(100, 0), columnspan=1)
        self.label_3 = ctk.CTkLabel(self, text="Capacity curve", font=("Inter", 14), fg_color="white")
        self.label_3.grid(row=1, column=0, sticky="wn", padx=15, pady=(160, 0), columnspan=1)
        self.label_4 = ctk.CTkLabel(self, text="Building parameters", font=("Inter", 14), fg_color="white")
        self.label_4.grid(row=1, column=0, sticky="wn", padx=15, pady=(240, 0), columnspan=1)
        self.label_5 = ctk.CTkLabel(self, text="Ground motion records", font=("Inter", 14), fg_color="white")
        self.label_5.grid(row=1, column=0, sticky="wn", padx=15, pady=(300, 0), columnspan=1)

        self.label_6 = ctk.CTkLabel(self, text="Min. scale", font=("Inter", 12), fg_color="white")
        self.label_6.grid(row=2, column=0, sticky="wn", padx=15, pady=(30, 0), columnspan=1)
        self.label_7 = ctk.CTkLabel(self, text="Max. scale", font=("Inter", 12), fg_color="white")
        self.label_7.grid(row=2, column=0, sticky="n", padx=10, pady=(30, 0), columnspan=1)
        self.label_8 = ctk.CTkLabel(self, text="Increment", font=("Inter", 12), fg_color="white")
        self.label_8.grid(row=2, column=0, sticky="en", padx=10, pady=(30, 0), columnspan=1)

        self.label_9 = ctk.CTkLabel(self, text="Regression method:", font=("Inter", 14), fg_color="white")
        self.label_9.grid(row=4, column=0, sticky="wn", padx=15, pady=(30, 0), columnspan=1)
        self.label_10 = ctk.CTkLabel(self, text="Loss ratio:", font=("Inter", 14), fg_color="white")
        self.label_10.grid(row=4, column=0, sticky="wn", padx=15, pady=(55, 0), columnspan=1)
        self.label_11 = ctk.CTkLabel(self, text="Loss ratio DS1", font=("Inter", 12), fg_color="white")
        self.label_11.grid(row=4, column=0, sticky="wn", padx=15, pady=(75, 0), columnspan=1)
        self.label_12 = ctk.CTkLabel(self, text="Loss ratio DS2", font=("Inter", 12), fg_color="white")
        self.label_12.grid(row=4, column=0, sticky="wn", padx=15, pady=(95, 0), columnspan=1)
        self.label_13 = ctk.CTkLabel(self, text="Loss ratio DS3", font=("Inter", 12), fg_color="white")
        self.label_13.grid(row=4, column=0, sticky="wn", padx=15, pady=(115, 0), columnspan=1)

        # Plot titles
        self.label_idealization = ctk.CTkLabel(
            self, text="Idealization curve", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_idealization.grid(row=1, column=1, sticky="n", padx=15, pady=(3, 0), columnspan=1)
        self.label_EDPs = ctk.CTkLabel(
            self, text="Engineering Demand Parameters", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_EDPs.grid(row=1, column=2, sticky="n", padx=15, pady=(3, 0), columnspan=1)
        self.label_fragility = ctk.CTkLabel(
            self, text="Fragility curve", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_fragility.grid(row=2, column=1, sticky="n", padx=15, pady=(3, 0), columnspan=1)
        self.label_vulnerability = ctk.CTkLabel(
            self, text="Vulnerability curve", font=("Inter bold", 14, "bold"), fg_color="white"
        )
        self.label_vulnerability.grid(row=2, column=2, sticky="n", padx=15, pady=(3, 0), columnspan=1)

        # ============ Entries ============ #
        self.entry_folder_output = ctk.CTkEntry(self, width=400, height=25, font=("Inter Light", 10))
        self.entry_folder_output.grid(row=1, column=0, sticky="nw", padx=(15, 70), pady=(125, 0))
        default_out = settings.last_output_dir or ""
        self.entry_folder_output.insert(0, default_out)

        self.entry_capacity = ctk.CTkEntry(self, width=400, height=25, font=("Inter Light", 10))
        self.entry_capacity.grid(row=1, column=0, sticky="nw", padx=(15, 70), pady=(185, 0))

        self.entry_building_parameter = ctk.CTkEntry(self, width=400, height=25, font=("Inter Light", 10))
        self.entry_building_parameter.grid(row=1, column=0, sticky="nw", padx=(15, 70), pady=(270, 0))

        self.entry_gmrs = ctk.CTkEntry(self, width=400, height=25, font=("Inter Light", 10))
        self.entry_gmrs.grid(row=1, column=0, sticky="wn", padx=(15, 70), pady=(330, 0))

        self.entry_min_scale = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_min_scale.grid(row=2, column=0, sticky="nw", padx=15, pady=(55, 0))
        self.entry_min_scale.insert(0, "0.000001")

        self.entry_max_scale = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_max_scale.grid(row=2, column=0, sticky="n", padx=15, pady=(55, 0))
        self.entry_max_scale.insert(0, "1.5")

        self.entry_increment = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_increment.grid(row=2, column=0, sticky="ne", padx=15, pady=(55, 0))
        self.entry_increment.insert(0, "0.1")

        self.entry_loss_ratio_ds1 = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_loss_ratio_ds1.grid(row=4, column=0, sticky="en", padx=15, pady=(80, 0))
        self.entry_loss_ratio_ds1.insert(0, "0.33")

        self.entry_loss_ratio_ds2 = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_loss_ratio_ds2.grid(row=4, column=0, sticky="en", padx=15, pady=(100, 0))
        self.entry_loss_ratio_ds2.insert(0, "0.67")

        self.entry_loss_ratio_ds3 = ctk.CTkEntry(self, width=50, height=10, font=("Inter Light", 10))
        self.entry_loss_ratio_ds3.grid(row=4, column=0, sticky="en", padx=15, pady=(120, 0))
        self.entry_loss_ratio_ds3.insert(0, "1.0")

        # ============ Checkboxes ============ #
        self.save_intersections = tk.BooleanVar(value=False)
        self.chk_save_intersections = ctk.CTkCheckBox(
            self,
            text="Save intersection plots",
            variable=self.save_intersections,
            font=("Inter", 12),
            bg_color="white",
            checkbox_width=15,
            checkbox_height=15,
        )
        self.chk_save_intersections.grid(row=3, column=0, sticky="nw", padx=15, pady=(30, 0))

        self.fast_mode = tk.BooleanVar(value=False)
        self.chk_fast_mode = ctk.CTkCheckBox(
            self,
            text="Fast mode",
            variable=self.fast_mode,
            font=("Inter", 12),
            bg_color="white",
            checkbox_width=15,
            checkbox_height=15,
        )
        self.chk_fast_mode.grid(row=3, column=0, sticky="ne", padx=0, pady=(30, 0))

        # ============ Plots ============ #
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

        # ============ Buttons ============ #
        # Output folder
        self.button_output = ctk.CTkButton(
            self,
            text="Folder",
            command=lambda: io_handlers.select_output_folder(self, self.entry_folder_output),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.button_output.grid(row=1, column=0, sticky="ne", padx=(15, 15), pady=(125, 0))

        # Capacity CSV
        self.btn_browse_load_capacity = ctk.CTkButton(
            self,
            text="File",
            command=lambda: io_handlers.load_capacity_csv(self, self.entry_capacity),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.btn_browse_load_capacity.grid(row=1, column=0, sticky="ne", padx=(15, 15), pady=(185, 0))

        # Plot capacity & idealization
        self.button_plot = ctk.CTkButton(
            self,
            text="Plot curve",
            command=lambda: analysis_handlers.plot_and_idealize(self),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.button_plot.grid(row=1, column=0, sticky="ne", padx=(15, 15), pady=(213, 0))

        # Idealization radio buttons
        self.idealization_option = tk.StringVar(value="EPP")
        self.radioEPP = ctk.CTkRadioButton(
            self, text="EPP", font=("Inter Light", 12), bg_color="white", variable=self.idealization_option, value="EPP"
        )
        self.radioEPP.grid(row=1, column=0, sticky="ne", padx=(0, 30), pady=(40, 15), columnspan=1)
        self.radioSH = ctk.CTkRadioButton(
            self,
            width=20,
            text="SH",
            font=("Inter Light", 12),
            bg_color="white",
            variable=self.idealization_option,
            value="SH",
        )
        self.radioSH.grid(row=1, column=0, sticky="ne", padx=(0, 15), pady=(40, 15), columnspan=1)

        # IMs
        self.IMs_selection = tk.StringVar(value="PGA")
        self.radioIMs = ctk.CTkRadioButton(
            self, text="PGA", font=("Inter Light", 12), bg_color="white", variable=self.IMs_selection, value="PGA"
        )
        self.radioIMs.grid(row=1, column=0, sticky="ne", padx=(0, 30), pady=(70, 0), columnspan=1)

        # Building parameters
        self.btn_browse_building_parameters = ctk.CTkButton(
            self,
            text="File",
            command=lambda: (
                io_handlers.load_building_params(self, self.entry_building_parameter),
                analysis_handlers.compute_adrs(self),
            ),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.btn_browse_building_parameters.grid(row=1, column=0, sticky="ne", padx=(15, 15), pady=(270, 0))

        # Ground motion folder
        self.btn_browse_gmrs = ctk.CTkButton(
            self,
            text="Folder",
            command=lambda: io_handlers.select_gmrs_folder(self, self.entry_gmrs),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.btn_browse_gmrs.grid(row=1, column=0, sticky="ne", padx=(15, 15), pady=(330, 0))

        # Generate EDPs
        self.button_generate_EDPs = ctk.CTkButton(
            self,
            text="Generate EDPs",
            command=lambda: analysis_handlers.generate_EDPs(self),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.button_generate_EDPs.grid(row=3, column=0, padx=(0, 0), pady=(50, 0))

        # Regression combobox
        self.regression_selection = ttk.Combobox(self, values=["MSA", "GLM", "LogregML"], width=8, font=("Arial", 12))
        self.regression_selection.set("MSA")
        self.regression_selection.grid(row=4, column=0, sticky="ne", padx=(30, 15), pady=(55, 0))
        self.regression_selection.bind("<<ComboboxSelected>>", self.on_selection_regression)

        self.link_function_selection = ttk.Combobox(self, values=["Probit", "Logit"], width=8, font=("Arial", 12))
        self.link_function_selection.set("Probit")
        self.link_function_selection.grid(row=4, column=0, sticky="ne", padx=(30, 15), pady=(90, 0))
        self.link_function_selection.grid_forget()

        self.regulation_selection = ttk.Combobox(
            self, values=["No Regulation", "Medium Regulation", "High Regulation"], width=15, font=("Arial", 12)
        )
        self.regulation_selection.set("No Regulation")
        self.regulation_selection.grid(row=4, column=0, sticky="ne", padx=(30, 15), pady=(90, 0))
        self.regulation_selection.grid_forget()

        # Statistical fit
        self.button_perform_statistical_fit = ctk.CTkButton(
            self,
            text="Perform statistical fit",
            command=lambda: analysis_handlers.perform_statistical_fit(self),
            width=50,
            height=25,
            text_color="white",
            font=("Inter", 12, "bold"),
        )
        self.button_perform_statistical_fit.grid(row=4, column=0, sticky="s", padx=(0, 0), pady=(0, 10))

        # open logs button
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

        # Helpers:
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

    # ------------------------------------------------------------------
    #                       UI-only handlers
    # ------------------------------------------------------------------
    def on_selection_regression(self, event):
        method = self.regression_selection.get()

        if method == "GLM":
            self.link_function_selection.grid(row=4, column=0, sticky="ne", padx=(30, 15), pady=(90, 0))
            self.regulation_selection.grid_forget()
        elif method == "LogregML":
            self.regulation_selection.grid(row=4, column=0, sticky="ne", padx=(30, 15), pady=(90, 0))
            self.link_function_selection.grid_forget()
        else:
            self.link_function_selection.grid_forget()
            self.regulation_selection.grid_forget()
