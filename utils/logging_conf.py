# utils/logging_conf.py
from __future__ import annotations

import io
import logging
import logging.config
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .config import settings
from .resources import ensure_dir, user_log_dir

LOG_FILENAME = "asfravab.log"


# ───────────────────────── helper: dictConfig ──────────────────────────
def _dict_config(
    log_file: Path,
    *,
    level: str = "INFO",
    console: bool = False,
) -> dict:
    """
    Build a dictConfig dictionary.

    * If *console* is False → no StreamHandler (quiet terminal).
    * If *console* is True  → StreamHandler prints **only** WARNING+ to
      the original stderr (sys.__stderr__) so redirection cannot recurse.
    """
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    handlers: dict[str, dict] = {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "std",
            "level": level,
            "filename": str(log_file),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf-8",
        }
    }

    if console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": "WARNING",  # tweak if you need INFO
            # use the **original** stream to avoid recursion
            "stream": "ext://sys.__stderr__",
        }

    root_handlers = ["file"] + (["console"] if console else [])

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"std": {"format": fmt}},
        "handlers": handlers,
        "root": {"handlers": root_handlers, "level": level},
    }


# ──────────────────────── helper: stream wrapper ───────────────────────
class _StreamToLogger(io.TextIOBase):
    """Redirect writes to a logger (used to capture print())."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf: str) -> int:  # type: ignore[override]
        if buf.rstrip():
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())
        return len(buf)

    def flush(self) -> None:  # type: ignore[override]
        pass


# ───────────────────────────── public API ──────────────────────────────
def setup_logging(
    *,
    level: Optional[str] = None,
    console: Optional[bool] = None,
) -> Path:
    """
    Call **once** near application start.

    Parameters
    ----------
    level
        Override log level (defaults to settings.log_level).
    console
        • True  – echo WARNING+ to terminal (uses sys.__stderr__).
        • False – keep terminal silent (default).
        • None  – follow settings.log_to_console.
    """
    if console is None:
        console = getattr(settings, "log_to_console", False)

    level = level or settings.log_level

    # pick writable log directory
    log_file = user_log_dir() / LOG_FILENAME

    def _apply(target: Path) -> Path:
        cfg = _dict_config(target, level=level, console=console)
        logging.config.dictConfig(cfg)
        logging.captureWarnings(True)

        # Redirect print() if requested and safe (i.e. console handler isn't
        # using the redirected stream).
        if settings.log_capture_prints and not console:
            sys.stdout = _StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
            sys.stderr = _StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

        _install_exception_hooks()
        os.environ["ASFRAVAB_LOG_DIR"] = str(target.parent)
        return target

    try:
        return _apply(log_file)
    except Exception:
        # fallback to temp if the normal dir is not writable
        tmp = Path(tempfile.gettempdir()) / "ASFRAVA-B" / "logs"
        ensure_dir(tmp)
        return _apply(tmp / LOG_FILENAME)


# ────────────────────── global exception hooks ─────────────────────────
def _handle_uncaught(exc_type, exc_value, exc_tb):
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


def _thread_excepthook(args: threading.ExceptHookArgs):
    logging.critical(
        "Uncaught thread exception",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _install_exception_hooks():
    sys.excepthook = _handle_uncaught
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook  # type: ignore[assignment]
