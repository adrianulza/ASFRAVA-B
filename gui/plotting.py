from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import modules.idealization as IP
from utils.config import settings


def _toolbar(canvas: FigureCanvasTkAgg) -> None:
    old_frame = getattr(canvas, "_toolbar_frame", None)
    if old_frame is not None:
        try:
            old_frame.destroy()
        except Exception:
            pass

    parent = canvas.get_tk_widget().master
    toolbar_frame = tk.Frame(parent)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.LEFT)

    canvas._toolbar_frame = toolbar_frame  # type: ignore[attr-defined]
    canvas._toolbar = toolbar  # type: ignore[attr-defined]


def capacity_and_idealization_curve(ax, canvas: FigureCanvasTkAgg, capacity_data: pd.DataFrame, choice: str = "EPP"):
    if capacity_data is None or capacity_data.empty:
        messagebox.showerror("Error", "Please load a valid CSV file first.")
        return None, None

    if "Dt(m)" not in capacity_data.columns or "Vb(kN)" not in capacity_data.columns:
        messagebox.showerror("Error", "CSV must have 'Dt(m)' and 'Vb(kN)' columns.")
        return None, None

    ax.clear()
    ax.plot(
        capacity_data["Dt(m)"],
        capacity_data["Vb(kN)"],
        marker="o",
        linestyle="--",
        color="black",
        label="Capacity Curve",
    )

    if choice in ["EPP", "SH"]:
        try:
            if choice == "EPP":
                point1, point2, point3 = IP.EPP(capacity_data, 0.001)
                label = "EPP Idealization"
            else:
                point1, point2, point3 = IP.SH(capacity_data, 0.001)
                label = "SH Idealization"

            ideal_dt = [point1[0], point2[0], point3[0]]
            ideal_vb = [point1[1], point2[1], point3[1]]
            ax.plot(ideal_dt, ideal_vb, linestyle="-", linewidth=3, color="blue", marker="s", label=label)
        except Exception as exc:
            messagebox.showerror("Error", f"Idealization failed: {exc}")
            return None, None
    else:
        messagebox.showerror("Error", f"Unknown idealization choice: {choice}")
        return None, None

    ax.set_xlabel("Displacement (Dt, m)")
    ax.set_ylabel("Base Shear (Vb, kN)")
    ax.grid(True)
    ax.legend()
    canvas.draw()
    _toolbar(canvas)
    return ideal_dt, ideal_vb


def _resolve_im_column(data_EDPs: pd.DataFrame, selected_IMs: str) -> str:
    if selected_IMs in data_EDPs.columns:
        return selected_IMs
    if "PGA" in data_EDPs.columns and selected_IMs == "PGA":
        return "PGA"
    raise KeyError(f"IM column '{selected_IMs}' was not found in the EDP dataset.")


def plot_EDPs(
    fig: plt.Figure,
    canvas: FigureCanvasTkAgg,
    ds1_threshold: float,
    ds2_threshold: float,
    ds3_threshold: float,
    edp_filepath: str,
    selected_IMs: str = "PGA",
):
    fig.clear()
    ax = fig.add_subplot(111)
    sns.set_theme(context="notebook", style="ticks", palette="deep", font="Arial", font_scale=1)

    data_EDPs = pd.read_csv(edp_filepath, delimiter=settings.CSV_SEP)
    im_column = _resolve_im_column(data_EDPs, selected_IMs)

    gmr_groups = data_EDPs.groupby("GMR") if "GMR" in data_EDPs.columns else None
    if gmr_groups is not None:
        for idx, (_, grp) in enumerate(gmr_groups):
            grp_sorted = grp.sort_values(im_column)
            ax.plot(
                grp_sorted["Sd"],
                grp_sorted[im_column],
                color="#222D3E",
                marker="None",
                linewidth=1.0,
                alpha=0.5,
                label="IDA curves" if idx == 0 else "_nolegend_",
            )

    ax.scatter(
        data_EDPs["Sd"],
        data_EDPs[im_column],
        alpha=0.5,
        edgecolors="black",
        s=50,
        label="EDPs",
        zorder=3,
    )
    _lines_before = set(ax.lines)
    sns.regplot(
        data=data_EDPs,
        x="Sd",
        y=im_column,
        ax=ax,
        fit_reg=True,
        scatter=False,
        line_kws={"color": "grey"},
        ci=95,
        truncate=True,
    )
    for _line in set(ax.lines) - _lines_before:
        _line.set_label("Linear model fit")

    ax.axvline(ds1_threshold, color="g", linestyle="--", label="DS1 Threshold")
    ax.axvline(ds2_threshold, color="y", linestyle="--", label="DS2 Threshold")
    ax.axvline(ds3_threshold, color="r", linestyle="--", label="DS3 Threshold")

    x_max = float(np.nanmax(data_EDPs["Sd"])) if not data_EDPs["Sd"].empty else 1.0
    y_max = float(np.nanmax(data_EDPs[im_column])) if not data_EDPs[im_column].empty else 1.0
    ax.set_xlim(0, max(x_max, 1e-6))
    ax.set_ylim(0, max(y_max, 1e-6))
    ax.set_xlabel("Sd")
    ax.set_ylabel(im_column)
    ax.legend(loc="best")

    canvas.draw()
    _toolbar(canvas)


def plot_curves(rs, idealized, intersection, record, scale, out_dir):
    fig = Figure(figsize=(4, 3), dpi=100)
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


def save_intersection_plots(plot_data, figs_dir):
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
    DATA_LABEL = {"MSA": "Binned", "J-MLE": "Binned", "GLM": "Sampled", "LogregML": "Sampled"}

    def _draw():
        fig.clear()
        ax = fig.add_subplot(111)
        sns.set_style("ticks")
        colors = ["g", "y", "r"]
        sel_reg = regression_selection_combobox.get()

        for i, ds in enumerate(damage_states):
            ax.plot(IM_range, probabilities[ds], color=colors[i], linestyle="-", label=f"{ds.upper()} {sel_reg}")
            point_label = f"{ds.upper()} {DATA_LABEL[sel_reg]}"
            if sel_reg in ("MSA", "J-MLE") and scatter_data is not None:
                ax.scatter(scatter_data[ds]["x"], scatter_data[ds]["y"], alpha=0.5, color=colors[i], label=point_label)
            else:
                ax.scatter(edps[selected_IMs], edps[ds], alpha=0.10, color=colors[i], label=point_label)

        ax.set_title(f"Fragility curve ({selected_IMs})")
        ax.set_xlabel(selected_IMs)
        ax.set_ylabel("Probability of Exceedance")
        ax.set_xlim(float(np.min(IM_range)), float(np.max(IM_range)))
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        ax.legend()
        canvas_fragility.draw()
        _toolbar(canvas_fragility)

    tk_root.after(0, _draw)


def plot_vulnerability(fig, canvas, tk_root, IM_range, vul_LR, selected_IMs):
    def _draw():
        fig.clear()
        ax = fig.add_subplot(111)
        sns.set_style("ticks")
        ax.plot(IM_range, vul_LR, color="b", label="Vulnerability")
        ax.set_title(f"Vulnerability curve ({selected_IMs})")
        ax.set_xlabel(selected_IMs)
        ax.set_ylabel("Loss Ratio")
        ax.set_xlim(float(np.min(IM_range)), float(np.max(IM_range)))
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        ax.legend(loc="lower right")
        canvas.draw()
        _toolbar(canvas)

    tk_root.after(0, _draw)
