import os
import sys
import tkinter as tk
import tkinter.messagebox as messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils.config import settings

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import modules.idealization as IP


def _toolbar(canvas: FigureCanvasTkAgg) -> None:
    """
    Attach a NavigationToolbar2Tk under *canvas* exactly once.

    The toolbar object is cached in canvas._toolbar to avoid duplicates.
    """
    if not hasattr(canvas, "_toolbar"):
        toolbar = NavigationToolbar2Tk(canvas, canvas.get_tk_widget().master)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas._toolbar = toolbar


def capacity_and_idealization_curve(ax, canvas, capacity_data, choice=None):
    ideal_dt: list[float] | None = None
    ideal_vb: list[float] | None = None

    point1: tuple[float, float] | None = None
    point2: tuple[float, float] | None = None
    point3: tuple[float, float] | None = None
    label: str = ""

    if capacity_data is None or capacity_data.empty:
        messagebox.showerror("Error", "No data loaded. Please load a valid CSV file first.")
        return

    if "Dt(m)" not in capacity_data.columns or "Vb(kN)" not in capacity_data.columns:
        messagebox.showerror("Error", "CSV must have 'Dt(m)' and 'Vb(kN)' columns.")
        return

    # Clear previous plots
    ax.clear()

    # Plot the original capacity curve first
    ax.plot(
        capacity_data["Dt(m)"],
        capacity_data["Vb(kN)"],
        marker="o",
        linestyle="--",
        color="black",
        label="Capacity Curve",
    )

    # If idealization is selected, plot it
    if choice in ["EPP", "SH"]:
        try:
            if choice == "EPP":
                point1, point2, point3 = IP.EPP(capacity_data, 0.001)
                label = "EPP Idealization"
            elif choice == "SH":
                point1, point2, point3 = IP.SH(capacity_data, 0.001)
                label = "SH Idealization"

            # Plot idealization
            ideal_dt = [point1[0], point2[0], point3[0]]
            ideal_vb = [point1[1], point2[1], point3[1]]
            ax.plot(ideal_dt, ideal_vb, linestyle="-", linewidth=3, color="blue", marker="s", label=label)

        except Exception as e:
            messagebox.showerror("Error", f"Idealization failed: {e}")
            return

    # Set labels, title, grid, legend
    ax.set_xlabel("Displacement (Dt, m)")
    ax.set_ylabel("Base Shear (Vb, kN)")
    ax.grid(True)
    ax.legend()

    # Refresh the canvas
    canvas.draw()
    _toolbar(canvas)

    return ideal_dt, ideal_vb


def plot_EDPs(
    fig: plt.Figure,
    canvas: FigureCanvasTkAgg,
    ds1_threshold: float,
    ds2_threshold: float,
    ds3_threshold: float,
    edp_filepath: str,
):
    """Redraw the Sd‑PGA cloud in *fig* and refresh *canvas*."""
    fig.clear()
    ax = fig.add_subplot(111)

    sns.set_theme(context="notebook", style="ticks", palette="deep", font="Arial", font_scale=1)

    data_EDPs = pd.read_csv(edp_filepath, delimiter=settings.CSV_SEP)

    sns.regplot(
        data=data_EDPs,
        x="Sd",
        y="PGA",
        ax=ax,
        fit_reg=True,
        line_kws={"color": "grey", "label": "Linear model fit"},
        scatter_kws={"alpha": 0.5, "edgecolor": "black", "s": 50},
        ci=95,
        truncate=True,
    )

    ax.axvline(ds1_threshold, color="g", linestyle="--", label="DS1 Threshold")
    ax.axvline(ds2_threshold, color="y", linestyle="--", label="DS2 Threshold")
    ax.axvline(ds3_threshold, color="r", linestyle="--", label="DS3 Threshold")

    ax.set_xlim(0, np.max(data_EDPs["Sd"]))
    ax.set_ylim(0, np.max(data_EDPs["PGA"]))
    ax.set_xlabel("Sd")
    ax.set_ylabel("PGA")
    ax.legend(loc="best")

    canvas.draw()
    _toolbar(canvas)


def plot_curves(rs, idealized, intersection, record, scale, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 3), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(rs["Sd"], rs["Sa"], label="RS", lw=1)
    ax.plot(idealized["Sd"], idealized["Sa"], ls="--", lw=1.5, label="Idealised ADRS")

    if intersection is not None:
        ax.plot(*intersection, "ro", label="Intersection")

    ax.set_xlabel("Sd")
    ax.set_ylabel("Sa")
    ax.legend()

    fname = f"{Path(record).stem}_scale-{scale}.png"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
#  Batch-save helper
# ------------------------------------------------------------------
def save_intersection_plots(plot_data, figs_dir):
    """
    Loop over `plot_data` (list of dicts) and save every intersection
    plot in <figs_dir>.
    """
    for d in plot_data:
        plot_curves(d["rs"], d["idealized_adrs_curve"], d["intersection_point"], d["record"], d["scale"], figs_dir)


def plot_fragility(
    fig,
    canvas_fragility,
    tk_root,
    IM_range,
    probabilities,
    edps,
    selected_IMs,
    damage_states,
    regression_selection_combobox,
    scatter_data,
):

    DATA_LABEL = {"MSA": "Binned", "GLM": "Sampled", "LogregML": "Sampled"}

    def _draw():
        fig.clear()
        ax = fig.add_subplot(111)

        sns.set_style("ticks")
        colors = ["g", "y", "r"]
        linestyles = "-"

        sel_reg = regression_selection_combobox.get()

        for i, ds in enumerate(damage_states):
            # fitted curve
            ax.plot(IM_range, probabilities[ds], color=colors[i], linestyle=linestyles, label=f"{ds.upper()} {sel_reg}")

            # points / markers
            point_label = f"{ds.upper()} {DATA_LABEL[sel_reg]}"

            if sel_reg == "MSA" and scatter_data is not None:
                # MSA uses the per-bin fractions you saved in scatter_data
                ax.scatter(scatter_data[ds]["x"], scatter_data[ds]["y"], alpha=0.5, color=colors[i], label=point_label)
            else:
                # GLM / LogregML use the raw EDP rows
                ax.scatter(edps[selected_IMs], edps[ds], alpha=0.10, color=colors[i], label=point_label)

        ax.set_xlabel(selected_IMs)
        ax.set_ylabel("Probability of Exceedance")
        ax.set_xlim(min(IM_range), max(IM_range))
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        ax.legend()

        canvas_fragility.draw()
        _toolbar(canvas_fragility)

    tk_root.after(0, _draw)


## function to plot vulnerabilty
def plot_vulnerability(fig, canvas, tk_root, IM_range, vul_LR, selected_IMs):
    def _draw():
        fig.clear()
        ax = fig.add_subplot(111)

        sns.set_style("ticks")
        ax.plot(IM_range, vul_LR, color="b", label="Vulnerability")
        ax.set_xlabel(selected_IMs)
        ax.set_ylabel("Loss Ratio")
        ax.set_xlim(min(IM_range), max(IM_range))
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        ax.legend(loc="lower right")

        canvas.draw()
        _toolbar(canvas)

    tk_root.after(0, _draw)
