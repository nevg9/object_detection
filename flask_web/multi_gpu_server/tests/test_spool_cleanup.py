from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
from pathlib import Path

from flask_web.multi_gpu_server.cleanup import cleanup_logs
from flask_web.multi_gpu_server.config import ServiceConfig
from flask_web.multi_gpu_server.spool import (
    remove_orphan_spool_files,
    stream_to_spool,
)


def config_for(root: Path) -> ServiceConfig:
    return ServiceConfig(
        gpu_count=8,
        worker_host="127.0.0.1",
        worker_base_port=5100,
        spool_dir=root / "spool",
        log_dir=root / "logs",
        minimum_free_bytes=0,
        disk_check_bytes=4,
        image_queue_size=128,
        video_queue_size=16,
        image_timeout_seconds=15,
        frame_timeout_seconds=30,
        video_timeout_seconds=1800,
        worker_connect_timeout_seconds=2,
        worker_read_timeout_seconds=120,
        health_interval_seconds=5,
        orphan_max_age_seconds=7200,
        log_retention_seconds=7 * 86400,
        log_total_limit_bytes=20 * 1024**3,
    )


class SpoolCleanupTest(unittest.TestCase):
    def test_spool_is_locked_until_request_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            cfg = config_for(Path(raw_root))
            cfg.spool_dir.mkdir()
            spooled = stream_to_spool(io.BytesIO(b"test-body"), cfg)
            old = time.time() - 10_000
            os.utime(spooled.path, (old, old))
            self.assertEqual(
                remove_orphan_spool_files(cfg.spool_dir, 1), []
            )
            path = spooled.path
            spooled.handle.close()
            self.assertEqual(
                remove_orphan_spool_files(cfg.spool_dir, 1), [path]
            )

    def test_log_cleanup_keeps_active_logs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            log_dir = Path(raw_root)
            active = log_dir / "gateway.log"
            rotated = log_dir / "gateway.log.1.gz"
            active.write_text("active")
            rotated.write_text("old")
            old = time.time() - 10 * 86400
            os.utime(active, (old, old))
            os.utime(rotated, (old, old))
            removed = cleanup_logs(log_dir, 7 * 86400, 1024)
            self.assertEqual(removed, [rotated])
            self.assertTrue(active.exists())


if __name__ == "__main__":
    unittest.main()
