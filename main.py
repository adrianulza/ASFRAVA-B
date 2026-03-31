import logging

import customtkinter as ctk

from gui.app_ui import mainUI
from utils.config import settings
from utils.logging_conf import setup_logging

logger = logging.getLogger(__name__)


def _is_alive(widget) -> bool:
    """Return True if the Tk widget still exists (safe to call after any exception)."""
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return False


def run() -> None:
    # ── Logging ──────────────────────────────────────────────────────────────
    _, _redirect = setup_logging(
        level=settings.log_level,
        console=settings.log_to_console,
    )

    # ── Appearance ────────────────────────────────────────────────────────────
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("dark-blue")

    # ── Window ────────────────────────────────────────────────────────────────
    app = mainUI()

    # ── stdout → log ──────────────────────────────────────────────────────────
    if _redirect:
        _redirect()

    # ── Event loop ────────────────────────────────────────────────────────────
    for _attempt in range(2):
        try:
            app.mainloop()
            return                          
        except KeyboardInterrupt:
            if _attempt == 0 and _is_alive(app):
                logger.debug(
                    "Absorbed spurious startup KeyboardInterrupt "
                    "(Windows console initialisation); re-entering event loop"
                )
                continue                   
            return                         
        except Exception:
            logger.exception("Fatal error in GUI main loop")
            raise


if __name__ == "__main__":
    run()
