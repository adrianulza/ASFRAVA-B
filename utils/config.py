# utils/config.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict

from .resources import ensure_dir, user_config_dir

CONFIG_FILENAME = "settings.json"
CONFIG_VERSION = 1


@dataclass
class Settings:
    # core constants
    CSV_SEP: str = ";"

    # paths
    last_output_dir: str = ""

    # logging
    log_level: str = "INFO"
    log_capture_prints: bool = True
    log_to_console: bool = False

    # meta
    version: int = CONFIG_VERSION

    # forward-compat bucket
    extra: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------
    #                      persistence helpers
    # ----------------------------------------------------------------------
    @classmethod
    def load(cls) -> "Settings":
        cfg_dir = ensure_dir(user_config_dir())
        cfg_file = cfg_dir / CONFIG_FILENAME

        # 1. if no file → create fresh defaults
        if not cfg_file.exists():
            s = cls()
            s.save()
            return s

        # 2. read JSON
        try:
            data: dict[str, Any] = json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            # corrupt file → back-up & reset
            cfg_file.rename(cfg_file.with_suffix(".bak"))
            s = cls()
            s.save()
            return s

        # 3. keep only fields defined in the dataclass
        allowed = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in allowed}

        # 4. back-fill missing keys with defaults
        s = cls(**clean)
        s.save()
        return s

    def save(self) -> None:
        cfg_dir = ensure_dir(user_config_dir())
        cfg_file = cfg_dir / CONFIG_FILENAME
        cfg_file.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


settings = Settings.load()
