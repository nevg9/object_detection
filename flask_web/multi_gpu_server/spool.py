from __future__ import annotations

import fcntl
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .config import ServiceConfig


class LowDiskSpaceError(RuntimeError):
    pass


def _assert_free_space(config: ServiceConfig) -> None:
    free_bytes = shutil.disk_usage(config.spool_dir).free
    if free_bytes < config.minimum_free_bytes:
        raise LowDiskSpaceError(
            f"spool free space {free_bytes} is below "
            f"the required {config.minimum_free_bytes} bytes"
        )


@dataclass
class SpoolFile:
    path: Path
    handle: BinaryIO
    size: int

    def cleanup(self) -> None:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
            pass
        try:
            self.handle.close()
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def stream_to_spool(stream: BinaryIO, config: ServiceConfig) -> SpoolFile:
    """Stream an unbounded request to disk while preserving a safety reserve."""

    _assert_free_space(config)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix="request-",
        suffix=".bin",
        dir=config.spool_dir,
        delete=False,
    )
    path = Path(handle.name)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    total = 0
    bytes_since_check = 0
    try:
        while True:
            chunk = stream.read(1024**2)
            if not chunk:
                break
            handle.write(chunk)
            chunk_size = len(chunk)
            total += chunk_size
            bytes_since_check += chunk_size
            if bytes_since_check >= config.disk_check_bytes:
                handle.flush()
                _assert_free_space(config)
                bytes_since_check = 0
        handle.flush()
        handle.seek(0)
        return SpoolFile(path=path, handle=handle, size=total)
    except Exception:
        SpoolFile(path=path, handle=handle, size=total).cleanup()
        raise


def remove_orphan_spool_files(
    spool_dir: Path, max_age_seconds: int, now: float | None = None
) -> list[Path]:
    now = time.time() if now is None else now
    removed: list[Path] = []
    for path in spool_dir.glob("request-*.bin"):
        try:
            if now - path.stat().st_mtime < max_age_seconds:
                continue
            with path.open("r+b") as handle:
                try:
                    fcntl.flock(
                        handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError:
                    continue
                path.unlink()
                removed.append(path)
        except FileNotFoundError:
            continue
    return removed
