import logging
import os
import threading
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import pandas as pd

import gui.plotting as ploti
import modules.adrs_trans as ADRS
from modules import analysis, fragility, vulnerability
from utils.config import settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
#               Plot and Idealize Capacity Curve
# ----------------------------------------------------------------------
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

    logger.info("Idealized Dt: %s", ideal_dt)
    logger.info("Idealized Vb: %s", ideal_vb)


# ----------------------------------------------------------------------
#                        ADRS computation
# ----------------------------------------------------------------------
def compute_adrs(app):
    """Compute ADRS coefficients + transform capacity curve to Sd/Sa AFTER:
    - capacity curve has been idealized, AND
    - building parameters are loaded.
    """
    if app.building_params is None:
        messagebox.showerror("Error", "Please load Building Parameters CSV first.")
        return

    if app.ideal_dt is None or app.ideal_vb is None:
        messagebox.showerror("Error", "Please perform Idealization first.")
        return

    try:
        Sd_coef, Sa_coef = ADRS.adrs_transformation(app.building_params)
        capacity_Sd, capacity_Sa = ADRS.adrs_capacity(app.ideal_dt, app.ideal_vb,
                                                    Sd_coef, Sa_coef)

        app.capacity_Sd = capacity_Sd
        app.capacity_Sa = capacity_Sa

        logger.info("Spectral Sd: %s", capacity_Sd)
        logger.info("Spectral Sa: %s", capacity_Sa)

    # catch **expected** data problems 
    except (ValueError, KeyError) as exc:          
        logger.exception("ADRS input error: %s", exc)
        messagebox.showerror("ADRS Computation Error",
                            f"Invalid input:\n{exc}")

    # keep a *very* last-resort guard, but silence W0718 
    except Exception as exc:  
        logger.exception("Unexpected ADRS error: %s", exc)
        messagebox.showerror("ADRS Computation Error",
                            f"Unexpected error:\n{exc}")



# ----------------------------------------------------------------------
#                           INITIALIZE EDPs ANALYSIS
# ----------------------------------------------------------------------
def generate_EDPs(app):
    """Run the IDA loop after *all* critical inputs are present."""
    app.analysis_cancelled = False

    # Pre-flight checks
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

    # 1. Scaling inputs
    raw_min = app.entry_min_scale.get().strip()
    raw_max = app.entry_max_scale.get().strip()
    raw_inc = app.entry_increment.get().strip()

    # 1a. empty-string guard
    if not raw_min or not raw_max or not raw_inc:
        messagebox.showerror("Invalid Input", "Min scale, Max scale and Increment must all be provided.")
        return

    # 1b. numeric & logical checks
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

    # 2. User confirmation
    user_confirmed = messagebox.askokcancel(
        "Confirmation", "Proceed with analysis? (Please do not interrupt the analysis " "to prevent possible errors.)"
    )
    if not user_confirmed:
        messagebox.showinfo("Cancelled", "Analysis was cancelled by the user.")
        return

    # progress‑bar initialisation
    gm_count = sum(1 for f in os.scandir(app.gmrs_folderpath) if f.name.lower().endswith((".txt", ".csv")))

    steps_per_rec = int((max_scale - min_scale) / increment) + 1
    total_steps = gm_count * steps_per_rec

    app.progress.start(total_steps)

    # 3.  Launch analysis thread
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


# ----------------------------------------------------------------------
#                   EDPs ANALYSIS (Perform IDA)
# ----------------------------------------------------------------------
def _model_analysis(app, analyzing_dialog: ctk.CTkToplevel):
    try:
        min_scale = float(app.entry_min_scale.get().strip())
        max_scale = float(app.entry_max_scale.get().strip())
        increment = float(app.entry_increment.get().strip())

        output_folder = app.entry_folder_output.get().strip()
        gmrs_folderpath = app.gmrs_folderpath

        # 2. Perform Analysis
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
            ) = analysis.analyze(
                app.entry_capacity.get(),
                app.entry_building_parameter.get(),
                gmrs_folderpath,
                min_scale,
                max_scale,
                increment,
                app.idealization_option.get(),
                output_folder,
                fast_mode=app.fast_mode.get(),
                progress_callback=app.progress.tick,
            )
        finally:

            app.progress.finish()

        # 3. Post-processing & UI update (schedule on UI thread)
        EDPs_filename = "EDPs_data_" + os.path.basename(app.entry_capacity.get())
        EDPs_filepath = os.path.join(output_folder, EDPs_filename)

        def _post_ui():
            try:
                _ = pd.read_csv(EDPs_filepath, delimiter=settings.CSV_SEP)
                ploti.plot_EDPs(
                    app.fig_EDPs, app.canvas_EDPs, ds1_threshold, ds2_threshold, ds3_threshold, EDPs_filepath
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot EDPs: {e}")

            if app.save_intersections.get():
                figs_dir = os.path.join(output_folder, "figs")
                ploti.save_intersection_plots(plot_data, figs_dir)

            if not app.analysis_cancelled and analyzing_dialog.winfo_exists():
                analyzing_dialog.destroy()

        app.after(0, _post_ui)

    except Exception as e:
        app.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        if analyzing_dialog.winfo_exists():
            app.after(0, lambda: analyzing_dialog.destroy())


# ----------------------------------------------------------------------
#                         STATISTICAL ANALYSIS
# ----------------------------------------------------------------------
def perform_statistical_fit(app):
    """
    Runs fragility fitting, vulnerability post-processing, updates the plots,
    and **exports two CSVs** (fragility.csv & vulnerability.csv) to the
    selected output folder.
    """
    selected_IMs = app.IMs_selection.get()
    selected_regression = app.regression_selection.get()
    regulation = app.regulation_selection.get()
    output_folder = app.entry_folder_output.get().strip()

    if not output_folder:
        messagebox.showerror("Error", "Please select an output folder.")
        return

    # 1.  Read EDPs ­­­— file must exist (produced by the capacity analysis)
    edp_file = f"EDPs_data_{os.path.basename(app.entry_capacity.get())}"
    edp_path = os.path.join(output_folder, edp_file)
    if not os.path.exists(edp_path):
        messagebox.showerror("Error", "EDPs file not found. Please generate EDPs first.")
        return

    edps = pd.read_csv(edp_path, delimiter=settings.CSV_SEP)
    damage_states = ["ds1", "ds2", "ds3"]
    min_scale = float(app.entry_min_scale.get().strip())
    max_scale = float(app.entry_max_scale.get().strip())
    step_size = int(max_scale / 0.01)
    increment = float(app.entry_increment.get().strip())

    # 2. Generate fragility curves
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

    # 3. Generate vulnerability curve (Loss-ratio vs IM)
    LRs = np.array(
        [
            float(app.entry_loss_ratio_ds1.get()),
            float(app.entry_loss_ratio_ds2.get()),
            float(app.entry_loss_ratio_ds3.get()),
        ]
    )
    vul_LR = vulnerability.vulnerability_model(probabilities, LRs)

    ploti.plot_vulnerability(app.fig_vulnerability, app.canvas_vulnerability, app, IM_range, vul_LR, selected_IMs)

    # 4.  CSV export
    try:
        # fragility.csv  (Probability, DS1, DS2, DS3)
        fragility_df = pd.DataFrame(
            {
                "Probability": IM_range,
                "DS1": probabilities["ds1"],
                "DS2": probabilities["ds2"],
                "DS3": probabilities["ds3"],
            }
        )
        fragility_csv = os.path.join(output_folder, "fragility.csv")
        fragility_df.to_csv(fragility_csv, index=False, sep=settings.CSV_SEP)

        # vulnerability.csv  (Loss ratio, IM)
        vulnerability_df = pd.DataFrame(
            {
                "Loss ratio": vul_LR,
                "IM": IM_range,
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
    except Exception as e:
        logger.error("Failed to write CSV files: %s", e, exc_info=True)
        app.after(
            0,
            lambda: messagebox.showerror(
                "CSV export failed", f"Could not write fragility/vulnerability CSV files:\n{e}"
            ),
        )
