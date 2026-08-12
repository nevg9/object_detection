from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import (
    CancelledError,
    Future,
    TimeoutError as FutureTimeoutError,
)

import av
from flask import Flask, Response, jsonify, request

from .config import ServiceConfig, ensure_runtime_directories
from .logging_utils import configure_file_logging
from .scheduler import (
    InferenceScheduler,
    SchedulerQueueFull,
    TaskResult,
)
from .spool import LowDiskSpaceError, SpoolFile, stream_to_spool


LOGGER = logging.getLogger("gateway")


def _proxy_worker_response(result: TaskResult) -> Response:
    # The response body and status come directly from the compatibility worker.
    # Do not deserialize and rebuild classifier/detector payloads here.
    return Response(
        result.body,
        status=result.status_code,
        content_type=result.content_type,
    )


def _error_response(reason: str, status: int):
    return jsonify({"error-reason": reason}), status


def create_app() -> Flask:
    config = ServiceConfig.from_env()
    ensure_runtime_directories(config)
    configure_file_logging("gateway.log", config.log_dir)
    scheduler = InferenceScheduler(
        worker_urls=config.worker_urls,
        image_queue_size=config.image_queue_size,
        video_queue_size=config.video_queue_size,
        connect_timeout=config.worker_connect_timeout_seconds,
        read_timeout=config.worker_read_timeout_seconds,
        health_interval=config.health_interval_seconds,
    )
    app = Flask(__name__)
    app.config["SERVICE_CONFIG"] = config
    app.config["INFERENCE_SCHEDULER"] = scheduler
    video_semaphore = threading.BoundedSemaphore(1)

    @app.get("/health/live")
    def health_live():
        return jsonify({"status": "ok"})

    @app.get("/health/ready")
    def health_ready():
        online = scheduler.pool.online_gpu_indices()
        payload = {
            "status": "ready" if len(online) == config.gpu_count else "degraded",
            "online_gpus": list(online),
            "expected_gpus": config.gpu_count,
        }
        return jsonify(payload), 200 if len(online) == config.gpu_count else 503

    def proxy_image(action: str):
        started = time.monotonic()
        spooled: SpoolFile | None = None
        future: Future[TaskResult] | None = None
        try:
            spooled = stream_to_spool(request.stream, config)
            future = scheduler.enqueue_image(
                action=action,
                spool_path=spooled.path,
                resource_id=request.headers.get("resourceId"),
                media_type=request.headers.get("mediaType"),
                timeout_seconds=config.image_timeout_seconds,
            )
            result = future.result(timeout=config.image_timeout_seconds)
            image_count, video_count = scheduler.broker.counts()
            LOGGER.info(
                "endpoint=%s gpu=%s body_bytes=%s total_ms=%.3f "
                "image_queue=%s video_queue=%s status=%s",
                action,
                result.gpu_index,
                spooled.size,
                (time.monotonic() - started) * 1000,
                image_count,
                video_count,
                result.status_code,
            )
            return _proxy_worker_response(result)
        except LowDiskSpaceError:
            LOGGER.exception("spool disk reserve reached endpoint=%s", action)
            return _error_response("service storage is temporarily unavailable", 503)
        except SchedulerQueueFull:
            return _error_response("service queue is full", 429)
        except (CancelledError, FutureTimeoutError):
            if future is not None:
                future.cancel()
            return _error_response("inference timed out", 504)
        finally:
            if spooled is not None:
                spooled.cleanup()

    @app.post("/predict_classifier")
    def predict_classifier():
        return proxy_image("predict_classifier")

    @app.post("/predict_detect")
    def predict_detect():
        return proxy_image("predict_detect")

    @app.post("/predict_video")
    def predict_video():
        if not video_semaphore.acquire(blocking=False):
            return _error_response("another video request is running", 429)
        started_wall = time.time()
        deadline = time.monotonic() + config.video_timeout_seconds
        spooled: SpoolFile | None = None
        outstanding: deque[tuple[int, float, Future[TaskResult]]] = deque()
        frames_json: list[dict] = []
        species_accumulate = 0
        last_ts_ms = 0.0
        frame_idx = -1

        def consume_oldest() -> None:
            nonlocal species_accumulate
            current_frame_idx, ts_ms, future = outstanding.popleft()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                future.cancel()
                raise FutureTimeoutError()
            result = future.result(timeout=remaining)
            if result.status_code != 200:
                raise RuntimeError(
                    f"video frame worker returned {result.status_code}: "
                    f"{result.body[:200]!r}"
                )
            frame_result = json.loads(result.body)
            if not frame_result["has_target"]:
                return
            frame_dict = frame_result["frame"]
            species_accumulate += len(frame_dict["agg_results"])
            frame_dict["frame_id"] = current_frame_idx
            frame_dict["ts_ms"] = round(ts_ms)
            frames_json.append(frame_dict)

        try:
            spooled = stream_to_spool(request.stream, config)
            resource_id = request.headers.get("resourceId")
            media_type = request.headers.get("mediaType")
            container = av.open(str(spooled.path))
            try:
                for frame_idx, frame in enumerate(container.decode(video=0)):
                    if time.monotonic() >= deadline:
                        raise FutureTimeoutError()
                    while len(outstanding) >= config.video_queue_size:
                        consume_oldest()
                    ts_ms = frame.time * 1000
                    last_ts_ms = ts_ms
                    image_bgr = frame.to_ndarray(format="bgr24")
                    remaining = max(0.001, deadline - time.monotonic())
                    future = scheduler.enqueue_video_frame(
                        frame=image_bgr.tobytes(order="C"),
                        height=image_bgr.shape[0],
                        width=image_bgr.shape[1],
                        timeout_seconds=min(
                            config.frame_timeout_seconds, remaining
                        ),
                    )
                    outstanding.append((frame_idx, ts_ms, future))
                while outstanding:
                    consume_oldest()

                if container.duration is not None:
                    duration_ms = container.duration / av.time_base * 1000
                else:
                    duration_ms = last_ts_ms
            finally:
                container.close()

            response = {
                "frames": frames_json,
                "summary": {
                    "species_accumulate": species_accumulate,
                    "duration_frames": frame_idx + 1,
                    "duration_ms": round(duration_ms),
                },
                "processing_ms": round((time.time() - started_wall) * 1000),
            }
            if resource_id:
                response["resourceId"] = resource_id
            if media_type:
                response["mediaType"] = media_type
            LOGGER.info(
                "endpoint=predict_video body_bytes=%s frames=%s "
                "processing_ms=%s status=200",
                spooled.size,
                frame_idx + 1,
                response["processing_ms"],
            )
            return jsonify(response)
        except LowDiskSpaceError:
            return _error_response("service storage is temporarily unavailable", 503)
        except SchedulerQueueFull:
            return _error_response("service queue is full", 429)
        except (CancelledError, FutureTimeoutError):
            for _, _, future in outstanding:
                future.cancel()
            return _error_response("video inference timed out", 504)
        finally:
            for _, _, future in outstanding:
                future.cancel()
            if spooled is not None:
                spooled.cleanup()
            video_semaphore.release()

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("OD_GATEWAY_PORT", "5090"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
