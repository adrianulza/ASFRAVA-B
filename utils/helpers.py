import tkinter as tk
from tkinter import messagebox


def show_citation_help(parent: tk.Tk) -> None:
    """
    Display an information dialog prompting users to cite the original paper if they use this program.
    """
    # Refined support message with correct APA citation
    message = (
        "Thank you for using ASFRAVA-B!\n\n"
        "If you find this program useful for your study or work, please support us by citing our original publication:\n\n"
        "Ulza, A., Idris, Y., Ramadhan, M. A., Syamsidik, Amalia, Z. (2025). Automated seismic fragility and vulnerability assessment for buildings (ASFRAVA‑B): Integrating probabilistic seismic design into performance‑based engineering practices. International Journal of Disaster Risk Reduction, 127, Article 105679. https://doi.org/10.1016/j.ijdrr.2025.105679"
    )
    messagebox.showinfo(title="Support Our Work", message=message, parent=parent)
