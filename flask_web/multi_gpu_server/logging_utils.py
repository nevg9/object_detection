from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_file_logging(default_name: str, log_dir: Path) -> logging.Logger:
    """Configure one append-only file per service.

    Rotation and retention are deliberately handled by logrotate and the
    cleanup timer, so all processes use normal FileHandlers here.
    """

    requested = os.getenv("OD_LOG_FILE", str(log_dir / default_name))
    log_path = Path(requested)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s service=%(name)s %(message)s"
        )
    )
    root.addHandler(handler)
    return logging.getLogger(default_name.rsplit(".", 1)[0])
