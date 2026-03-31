import logging
import time

APP_VERSION = "1.1.0"

import customtkinter as ctk

for _attempt in range(5):
    try:
        import pandas as _pd
        import matplotlib as _mpl
        break
    except Exception:
        if _attempt < 4:
            time.sleep(0.4)
        else:
            raise

_mpl.use("TkAgg")

from gui.app_ui import mainUI
from utils.config import settings
from utils.logging_conf import setup_logging

logger = logging.getLogger(__name__)

_log_path, _redirect_stdout = setup_logging(level=settings.log_level, console=settings.log_to_console)
logger.info("ASFRAVA-B v%s starting", APP_VERSION)


def run():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("dark-blue")

    try:
        app = mainUI()
        # Redirect stdout only AFTER Tkinter window is created to avoid
        # interfering with Tcl's console I/O initialization on Windows
        if _redirect_stdout:
            _redirect_stdout()
        app.mainloop()
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user")
        raise
    except Exception:
        logger.exception("Fatal error in GUI main loop")
        raise


if __name__ == "__main__":
    run()
