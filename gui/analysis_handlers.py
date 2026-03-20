from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

import gui.plotting as ploti
import modules.adrs_trans as ADRS
from modules import fragility, vulnerability
from utils.config import settings

logger = logging.getLogger(__name__)

_MODULE_CACHE: dict[str, object] = {}
_ANALYSIS_FILE_MAP = {
    "PGA": "analysis_PGA.py",
    "Sa(T)": "analysis_Sa(T).py",
    "Sa(avg)": "analysis_SA(avg).py",
}
_GRAVITY = 9.81


def _selected_im(app) -> str:
    workflow_config = getattr(app, "workflow_config", {}) or {}
    return str(workflow_config.get("im_method") or app.IMs_selection.get() or "PGA")


def _analysis_file_for_im(im_method: str) -> Path:
    try:
        filename = _ANALYSIS_FILE_MAP[im_method]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow IM: {im_method}") from exc
    return Path(__file__).resolve().parent.parent / "modules" / filename


def _load_analysis_module(im_method: str):
    module_path = _analysis_file_for_im(im_method)
    if not module_path.exists():
        raise FileNotFoundError(f"Analysis module not found: {module_path}")

    cache_key = str(module_path)
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    safe_name = "modules._dynamic_" + re.sub(r"[^0-9A-Za-z_]+", "_", module_path.stem)
    spec = importlib.util.spec_from_file_location(safe_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load analysis module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[safe_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(safe_name, None)
        raise
    _MODULE_CACHE[cache_key] = module
    return module


def _derive_period_from_elastic_branch(sd_values, sa_values) -> float:
    sd = np.asarray(sd_values, dtype=float).reshape(-1)
    sa = np.asarray(sa_values, dtype=float).reshape(-1)

    if sd.size < 2 or sa.size < 2:
        raise ValueError("At least two ADRS points are required to derive the elastic period.")

    delta_sd = float(sd[1] - sd[0])
    delta_sa = float(sa[1] - sa[0])

    if delta_sd <= 0.0 or delta_sa <= 0.0:
        raise ValueError("Elastic ADRS branch must have positive Sd and Sa increments to derive the period.")

    slope = delta_sa / delta_sd  # Sa[g] / Sd[m]
    period = 2.0 * np.pi / np.sqrt(_GRAVITY * slope)

    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Failed to derive a valid positive period from the idealized elastic branch.")

    return float(period)


def resolve_base_period(app) -> float | None:
    im_method = _selected_im(app)
    if im_method == "PGA":
        return None

    workflow_config = getattr(app, "workflow_config", {}) or {}
    period_mode = workflow_config.get("period_mode")

    if period_mode == "specified":
        period_value = workflow_config.get("period_value")
        if period_value is None:
            raise ValueError(f"Please provide T for the {im_method} workflow.")
        period = float(period_value)
        if period <= 0.0:
            raise ValueError(f"T for the {im_method} workflow must be greater than 0.")
        workflow_config["resolved_period"] = period
        if hasattr(app, "refresh_workflow_info"):
            app.refresh_workflow_info()
        return period

    if getattr(app, "capacity_Sd", None) is None or getattr(app, "capacity_Sa", None) is None:
        ok = compute_adrs(app, show_errors=False)
        if not ok:
            raise ValueError(
                "The period could not be derived yet. Plot the capacity curve and load building parameters first."
            )

    period = _derive_period_from_elastic_branch(app.capacity_Sd, app.capacity_Sa)
    workflow_config["resolved_period"] = period
    if hasattr(app, "refresh_workflow_info"):
        app.refresh_workflow_info()
    logger.info("Derived period for %s workflow from idealized elastic branch: %.6f s", im_method, period)
    return period


def plot_and_idealize(app):
    if app.capacity_data is None:
        messagebox.showerror("Error", "Please load a valid CSV file first.")
        return

    choice = app.idealization_option.get()
    ideal_dt, ideal_vb = ploti.capacity_and_idealization_curve(
        app.ax_idealization, app.canvas_idealization, app.capacity_data, choice=choice
    )

    app.ideal_dt = ideal_dt
    app.ideal_vb = ideal_vb
    app.capacity_Sd = None
    app.capacity_Sa = None

    workflow_config = getattr(app, "workflow_config", {}) or {}
    if workflow_config.get("period_mode") == "calculated":
        workflow_config["resolved_period"] = None

    logger.info("Idealized Dt: %s", ideal_dt)
    logger.info("Idealized Vb: %s", ideal_vb)

    if getattr(app, "building_params", None) is not None:
        compute_adrs(app, show_errors=False)
    elif hasattr(app, "refresh_workflow_info"):
        app.refresh_workflow_info()


def compute_adrs(app, show_errors: bool = True) -> bool:
    if app.building_params is None:
        if show_errors:
            messagebox.showerror("Error", "Please load Building Parameters CSV first.")
        return False

    if app.ideal_dt is None or app.ideal_vb is None:
        if show_errors:
            messagebox.showerror("Error", "Please perform Idealization first.")
        return False

    try:
        Sd_coef, Sa_coef = ADRS.adrs_transformation(app.building_params)
        capacity_Sd, capacity_Sa = ADRS.adrs_capacity(app.ideal_dt, app.ideal_vb, Sd_coef, Sa_coef)

        app.capacity_Sd = capacity_Sd
        app.capacity_Sa = capacity_Sa

        workflow_config = getattr(app, "workflow_config", {}) or {}
        if workflow_config.get("period_mode") == "calculated" and _selected_im(app) != "PGA":
            try:
                workflow_config["resolved_period"] = _derive_period_from_elastic_branch(capacity_Sd, capacity_Sa)
            except ValueError:
                workflow_config["resolved_period"] = None

        if hasattr(app, "refresh_workflow_info"):
            app.refresh_workflow_info()

        logger.info("Spectral Sd: %s", capacity_Sd)
        logger.info("Spectral Sa: %s", capacity_Sa)
        return True

    except (ValueError, KeyError) as exc:
        logger.exception("ADRS input error: %s", exc)
        if show_errors:
            messagebox.showerror("ADRS Computation Error", f"Invalid input:\n{exc}")
        return False
    except Exception as exc:
        logger.exception("Unexpected ADRS error: %s", exc)
        if show_errors:
            messagebox.showerror("ADRS Computation Error", f"Unexpected error:\n{exc}")
        return False


def generate_EDPs(app):
    app.analysis_cancelled = False

    output_folder = app.entry_folder_output.get().strip()
    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder first.")
        return

    if app.capacity_data is None or app.ideal_dt is None:
        messagebox.showerror("Error", "Capacity curve not plotted.\nClick 'Plot curve' first.")
        return

    if app.building_params is None:
        messagebox.showerror("Error", "Building parameters CSV not loaded.")
        return

    if app.gmrs_folderpath is None:
        messagebox.showerror("Error", "Please browse a folder with ground-motion records (GMRS).")
        return

    raw_min = app.entry_min_scale.get().strip()
    raw_max = app.entry_max_scale.get().strip()
    raw_inc = app.entry_increment.get().strip()

    if not raw_min or not raw_max or not raw_inc:
        messagebox.showerror("Invalid Input", "Min scale, Max scale and Increment must all be provided.")
        return

    try:
        min_scale = float(raw_min)
        max_scale = float(raw_max)
        increment = float(raw_inc)
    except ValueError:
        messagebox.showerror("Invalid Input", "Scale fields must be valid numbers (e.g. 0.25).")
        return

    if increment <= 0:
        messagebox.showerror("Invalid Input", "Increment must be greater than 0.")
        return
    if min_scale > max_scale:
        messagebox.showerror("Invalid Input", "Min scale must be ≤ Max scale.")
        return

    try:
        resolve_base_period(app)
    except ValueError as exc:
        messagebox.showerror("Workflow Error", str(exc))
        return

    user_confirmed = messagebox.askokcancel(
        "Confirmation",
        "Proceed with analysis? (Please do not interrupt the analysis to prevent possible errors.)",
    )
    if not user_confirmed:
        messagebox.showinfo("Cancelled", "Analysis was cancelled by the user.")
        return

    gm_count = sum(1 for f in os.scandir(app.gmrs_folderpath) if f.name.lower().endswith((".txt", ".csv")))
    steps_per_rec = int((max_scale - min_scale) / increment) + 1
    total_steps = gm_count * steps_per_rec
    app.progress.start(total_steps)

    analyzing_dialog = ctk.CTkToplevel(app)
    analyzing_dialog.title("Running")
    analyzing_dialog.geometry("300x100")
    analyzing_dialog.transient(app)
    analyzing_dialog.grab_set()

    def on_dialog_close():
        app.analysis_cancelled = True
        analyzing_dialog.destroy()
        messagebox.showinfo("Cancelled", "Analysis was cancelled by the user.")

    analyzing_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
    analyzing_label = ctk.CTkLabel(analyzing_dialog, text="Analyzing...", font=("Helvetica", 16))
    analyzing_label.pack(pady=20)

    app.update()
    analysis_thread = threading.Thread(target=_model_analysis, args=(app, analyzing_dialog), daemon=True)
    analysis_thread.start()


def _model_analysis(app, analyzing_dialog: ctk.CTkToplevel):
    try:
        min_scale = float(app.entry_min_scale.get().strip())
        max_scale = float(app.entry_max_scale.get().strip())
        increment = float(app.entry_increment.get().strip())

        output_folder = app.entry_folder_output.get().strip()
        gmrs_folderpath = app.gmrs_folderpath
        im_method = _selected_im(app)
        analysis_module = _load_analysis_module(im_method)

        analysis_kwargs = {
            "fast_mode": app.fast_mode.get(),
            "progress_callback": app.progress.tick,
            "im_column": im_method,
        }
        base_period = resolve_base_period(app)
        if base_period is not None:
            analysis_kwargs["base_period"] = base_period

        try:
            (
                point1,
                point2,
                point3,
                data_pushover,
                ds1_threshold,
                ds2_threshold,
                ds3_threshold,
                plot_data,
                idealized_adrs_curve,
            ) = analysis_module.analyze(
                app.entry_capacity.get(),
                app.entry_building_parameter.get(),
                gmrs_folderpath,
                min_scale,
                max_scale,
                increment,
                app.idealization_option.get(),
                output_folder,
                **analysis_kwargs,
            )
        finally:
            app.progress.finish()

        EDPs_filename = "EDPs_data_" + os.path.basename(app.entry_capacity.get())
        EDPs_filepath = os.path.join(output_folder, EDPs_filename)

        def _post_ui():
            try:
                _ = pd.read_csv(EDPs_filepath, delimiter=settings.CSV_SEP)
                ploti.plot_EDPs(
                    app.fig_EDPs,
                    app.canvas_EDPs,
                    ds1_threshold,
                    ds2_threshold,
                    ds3_threshold,
                    EDPs_filepath,
                    im_method,
                )
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to plot EDPs: {exc}")

            if app.save_intersections.get():
                figs_dir = os.path.join(output_folder, "figs")
                ploti.save_intersection_plots(plot_data, figs_dir)

            if hasattr(app, "refresh_plot_labels"):
                app.refresh_plot_labels()

            if not app.analysis_cancelled and analyzing_dialog.winfo_exists():
                analyzing_dialog.destroy()

        app.after(0, _post_ui)

    except Exception as exc:
        app.after(0, lambda: messagebox.showerror("Analysis Error", str(exc)))
        if analyzing_dialog.winfo_exists():
            app.after(0, lambda: analyzing_dialog.destroy())


def perform_statistical_fit(app):
    selected_IMs = _selected_im(app)
    selected_regression = app.regression_selection.get()
    regulation = app.regulation_selection.get()
    output_folder = app.entry_folder_output.get().strip()

    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    edp_file = f"EDPs_data_{os.path.basename(app.entry_capacity.get())}"
    edp_path = os.path.join(output_folder, edp_file)
    if not os.path.exists(edp_path):
        messagebox.showerror("Error", "EDPs file not found. Please generate EDPs first.")
        return

    edps = pd.read_csv(edp_path, delimiter=settings.CSV_SEP)
    damage_states = ["ds1", "ds2", "ds3"]
    min_scale = float(app.entry_min_scale.get().strip())
    max_scale = float(app.entry_max_scale.get().strip())
    step_size = max(200, int(round((max_scale - min_scale) / 0.01)) + 1)
    increment = float(app.entry_increment.get().strip())

    models, IM_range, probabilities, scatter = fragility.fit_fragility_models(
        edps,
        damage_states,
        selected_regression,
        selected_IMs,
        app.link_function_selection.get(),
        regulation,
        app.gmrs_folderpath,
        min_scale,
        max_scale,
        step_size,
        increment,
    )

    ploti.plot_fragility(
        app.fig_fragility,
        app.canvas_fragility,
        app,
        IM_range,
        probabilities,
        edps,
        selected_IMs,
        damage_states,
        app.regression_selection,
        scatter,
    )

    LRs = np.array(
        [
            float(app.entry_loss_ratio_ds1.get()),
            float(app.entry_loss_ratio_ds2.get()),
            float(app.entry_loss_ratio_ds3.get()),
        ]
    )
    vul_LR = vulnerability.vulnerability_model(probabilities, LRs)

    ploti.plot_vulnerability(app.fig_vulnerability, app.canvas_vulnerability, app, IM_range, vul_LR, selected_IMs)

    if hasattr(app, "refresh_plot_labels"):
        app.refresh_plot_labels()

    try:
        fragility_df = pd.DataFrame(
            {
                selected_IMs: IM_range,
                "DS1": probabilities["ds1"],
                "DS2": probabilities["ds2"],
                "DS3": probabilities["ds3"],
            }
        )
        fragility_csv = os.path.join(output_folder, "fragility.csv")
        fragility_df.to_csv(fragility_csv, index=False, sep=settings.CSV_SEP)

        vulnerability_df = pd.DataFrame(
            {
                selected_IMs: IM_range,
                "Loss ratio": vul_LR,
            }
        )
        vulnerability_csv = os.path.join(output_folder, "vulnerability.csv")
        vulnerability_df.to_csv(vulnerability_csv, index=False, sep=settings.CSV_SEP)

        logger.info("Saved fragility CSV -> %s", fragility_csv)
        logger.info("Saved vulnerability CSV -> %s", vulnerability_csv)
        app.after(
            0,
            lambda: messagebox.showinfo(
                "Export complete", "Fragility & vulnerability datasets were written to the output folder."
            ),
        )
    except Exception as exc:
        logger.error("Failed to write CSV files: %s", exc, exc_info=True)
        app.after(
            0,
            lambda: messagebox.showerror(
                "CSV export failed", f"Could not write fragility/vulnerability CSV files:\n{exc}"
            ),
        )
