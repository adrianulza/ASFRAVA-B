from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Union

ENV_LOG_DIR = "ASFRAVAB_LOG_DIR"


# ---------------------------------------------------------------------------
#                         Frozen / base path helpers
# ---------------------------------------------------------------------------
def _is_frozen() -> bool:
    # PyInstaller/py2exe style
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def base_path() -> Path:
    """
    Project base in dev, or PyInstaller temp folder when frozen.
    """
    if _is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]  # utils -> ASFRAVA-B


def asset_path(*relative: Union[str, os.PathLike]) -> Path:
    """
    Use for bundled files (icons, templates, etc.).
    Works in both dev and frozen (PyInstaller) modes.
    """
    return base_path() / "assets" / Path(*map(str, relative))


# ---------------------------------------------------------------------------
#                         Platform-specific base dirs
# ---------------------------------------------------------------------------
def _windows_appdata() -> Path:
    return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))


def _windows_localappdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))


def _mac_app_support() -> Path:
    return Path.home() / "Library" / "Application Support"


def _linux_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def _platform_user_root_candidates() -> list[Path]:
    if sys.platform.startswith("win"):
        # Try Roaming first, then Local
        return [
            _windows_appdata() / "ASFRAVA-B",
            _windows_localappdata() / "ASFRAVA-B",
        ]
    if sys.platform == "darwin":
        return [
            _mac_app_support() / "ASFRAVA-B",
        ]
    # Linux
    xdg = _linux_data_home()
    return [
        xdg / "ASFRAVA-B",
    ]


# ---------------------------------------------------------------------------
#              Writable-dir resolution (with fallback to temp)
# ---------------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_writable(dir_path: Path) -> bool:
    try:
        ensure_dir(dir_path)
        test_file = dir_path / ".permcheck"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _pick_writable_dir(candidates: list[Path]) -> Path:
    for c in candidates:
        if _is_writable(c):
            return c
    # Fallback: system temp (always last resort)
    fallback = Path(tempfile.gettempdir()) / "ASFRAVA-B"
    ensure_dir(fallback)
    return fallback


def user_data_dir() -> Path:
    """
    Returns a **writable** per-user data directory.
    Tries OS-standard locations first, falls back to system temp.
    """
    return _pick_writable_dir(_platform_user_root_candidates())


def user_config_dir() -> Path:
    return ensure_dir(user_data_dir() / "config")


def user_log_dir() -> Path:
    # If logging_conf selected a fallback dir, it writes it here:
    env_override = os.environ.get(ENV_LOG_DIR)
    if env_override:
        return ensure_dir(Path(env_override))
    return ensure_dir(user_data_dir() / "logs")
