from __future__ import annotations

import time
from pathlib import Path

from .config import ServiceConfig, ensure_runtime_directories
from .logging_utils import configure_file_logging
from .spool import remove_orphan_spool_files


def cleanup_logs(
    log_dir: Path,
    retention_seconds: int,
    total_limit_bytes: int,
    now: float | None = None,
) -> list[Path]:
    now = time.time() if now is None else now
    files = [path for path in log_dir.iterdir() if path.is_file()]
    removed: list[Path] = []

    # Active *.log files are managed by logrotate; this cleanup only removes
    # rotated files and old auxiliary logs.
    for path in files:
        try:
            if path.suffix == ".log":
                continue
            if now - path.stat().st_mtime > retention_seconds:
                path.unlink()
                removed.append(path)
        except FileNotFoundError:
            continue

    remaining = [
        path
        for path in log_dir.iterdir()
        if path.is_file() and path.suffix != ".log"
    ]
    total = sum(path.stat().st_size for path in remaining)
    if total > total_limit_bytes:
        for path in sorted(remaining, key=lambda item: item.stat().st_mtime):
            if total <= total_limit_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
                removed.append(path)
            except FileNotFoundError:
                continue
    return removed


def main() -> int:
    config = ServiceConfig.from_env()
    ensure_runtime_directories(config)
    logger = configure_file_logging("cleanup.log", config.log_dir)
    spool_removed = remove_orphan_spool_files(
        config.spool_dir, config.orphan_max_age_seconds
    )
    logs_removed = cleanup_logs(
        config.log_dir,
        config.log_retention_seconds,
        config.log_total_limit_bytes,
    )
    logger.info(
        "cleanup complete spool_removed=%s logs_removed=%s",
        len(spool_removed),
        len(logs_removed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
