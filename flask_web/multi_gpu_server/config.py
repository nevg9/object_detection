from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class ServiceConfig:
    gpu_count: int
    worker_host: str
    worker_base_port: int
    spool_dir: Path
    log_dir: Path
    minimum_free_bytes: int
    disk_check_bytes: int
    image_queue_size: int
    video_queue_size: int
    image_timeout_seconds: float
    frame_timeout_seconds: float
    video_timeout_seconds: float
    worker_connect_timeout_seconds: float
    worker_read_timeout_seconds: float
    health_interval_seconds: float
    orphan_max_age_seconds: int
    log_retention_seconds: int
    log_total_limit_bytes: int

    @property
    def worker_urls(self) -> tuple[str, ...]:
        return tuple(
            f"http://{self.worker_host}:{self.worker_base_port + index}"
            for index in range(self.gpu_count)
        )

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        gib = 1024**3
        return cls(
            gpu_count=_env_int("OD_GPU_COUNT", 8),
            worker_host=os.getenv("OD_WORKER_HOST", "127.0.0.1"),
            worker_base_port=_env_int("OD_WORKER_BASE_PORT", 5100),
            spool_dir=Path(
                os.getenv(
                    "OD_SPOOL_DIR", "/mnt/data2/object_detection/spool"
                )
            ),
            log_dir=Path(
                os.getenv("OD_LOG_DIR", "/mnt/data2/object_detection/logs")
            ),
            minimum_free_bytes=_env_int("OD_MINIMUM_FREE_GB", 50) * gib,
            disk_check_bytes=_env_int("OD_DISK_CHECK_MB", 64) * 1024**2,
            image_queue_size=_env_int("OD_IMAGE_QUEUE_SIZE", 128),
            video_queue_size=_env_int("OD_VIDEO_QUEUE_SIZE", 16),
            image_timeout_seconds=_env_float("OD_IMAGE_TIMEOUT_SECONDS", 15.0),
            frame_timeout_seconds=_env_float("OD_FRAME_TIMEOUT_SECONDS", 30.0),
            video_timeout_seconds=_env_float("OD_VIDEO_TIMEOUT_SECONDS", 1800.0),
            worker_connect_timeout_seconds=_env_float(
                "OD_WORKER_CONNECT_TIMEOUT_SECONDS", 2.0
            ),
            worker_read_timeout_seconds=_env_float(
                "OD_WORKER_READ_TIMEOUT_SECONDS", 120.0
            ),
            health_interval_seconds=_env_float(
                "OD_HEALTH_INTERVAL_SECONDS", 5.0
            ),
            orphan_max_age_seconds=_env_int("OD_ORPHAN_MAX_AGE_SECONDS", 7200),
            log_retention_seconds=_env_int("OD_LOG_RETENTION_DAYS", 7) * 86400,
            log_total_limit_bytes=_env_int("OD_LOG_TOTAL_LIMIT_GB", 20) * gib,
        )


def ensure_runtime_directories(config: ServiceConfig) -> None:
    for path in (config.spool_dir, config.log_dir):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700 if path == config.spool_dir else 0o750)
