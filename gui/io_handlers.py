import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd

from utils.config import settings

logger = logging.getLogger(__name__)


# 1. Select output (folder)
def select_output_folder(app, entry_widget: tk.Entry) -> None:
    path = filedialog.askdirectory(title="Select output folder")
    if not path:
        return

    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, path)
    settings.last_output_dir = path
    settings.save()

    logger.info("Output folder set to: %s", path)


# 2. Select pushover file (csv)
def load_capacity_csv(app, entry_widget):
    """Browse and load CSV in a single step (capacity curve)."""
    capacity_filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not capacity_filepath:
        return

    if not capacity_filepath.lower().endswith(".csv"):
        messagebox.showerror("Error", "Please select a valid CSV file.")
        return

    try:
        df = pd.read_csv(capacity_filepath, delimiter=settings.CSV_SEP)

        if "Dt(m)" not in df.columns or "Vb(kN)" not in df.columns:
            messagebox.showerror("Error", "CSV must have 'Dt(m)' and 'Vb(kN)' column headers.")
            return

        app.capacity_data = df

        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, capacity_filepath)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV:\n{e}")


# 3. Select building parameters (csv)
def load_building_params(app, entry_widget):
    building_params_filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not building_params_filepath:
        return

    if not building_params_filepath.lower().endswith(".csv"):
        messagebox.showerror("Error", "Please select a valid CSV file.")
        return

    try:
        df = pd.read_csv(building_params_filepath, delimiter=settings.CSV_SEP)

        required_columns = {"Floor(number)", "Mass(ton)", "Mode(unitless)"}
        missing = required_columns.difference(df.columns)
        if missing:
            messagebox.showerror("Error", f"CSV is missing columns: {missing}")
            return

        app.building_params = df
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, building_params_filepath)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")


# 4. Select ground motion records (folder)
def select_gmrs_folder(app, entry_widget):
    folderpath = filedialog.askdirectory()
    if not folderpath:
        return

    if not os.listdir(folderpath):
        messagebox.showerror("Error", "Folder is empty.")
        return

    app.gmrs_folderpath = folderpath
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, folderpath)
