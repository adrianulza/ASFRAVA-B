import logging

import customtkinter as ctk

from gui.app_ui import mainUI
from utils.config import settings
from utils.logging_conf import setup_logging

logger = logging.getLogger(__name__)

setup_logging(level=settings.log_level, console=settings.log_to_console)


def run():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("dark-blue")
    try:
        app = mainUI()
        app.mainloop()
    except Exception:
        logger.exception("Fatal error in GUI main loop")
        raise


if __name__ == "__main__":
    run()
