from __future__ import annotations

import importlib
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


LOGGER = logging.getLogger("scheduler")
TaskKind = Literal["image", "video"]


class SchedulerQueueFull(RuntimeError):
    pass


class WorkerTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskResult:
    status_code: int
    body: bytes
    content_type: str
    gpu_index: int


@dataclass
class InferenceTask:
    kind: TaskKind
    action: str
    payload: dict[str, Any]
    deadline: float
    future: Future[TaskResult] = field(default_factory=Future)


class FairTaskBroker:
    """Two bounded FIFO queues with strict 1:1 selection when both are busy."""

    def __init__(self, image_capacity: int, video_capacity: int) -> None:
        self._image_capacity = image_capacity
        self._video_capacity = video_capacity
        self._images: deque[InferenceTask] = deque()
        self._videos: deque[InferenceTask] = deque()
        self._condition = threading.Condition()
        self._prefer_video = False
        self._closed = False

    def put(self, task: InferenceTask) -> None:
        with self._condition:
            queue = self._images if task.kind == "image" else self._videos
            capacity = (
                self._image_capacity
                if task.kind == "image"
                else self._video_capacity
            )
            if len(queue) >= capacity:
                raise SchedulerQueueFull(f"{task.kind} queue is full")
            queue.append(task)
            self._condition.notify()

    def requeue_front(self, task: InferenceTask) -> None:
        if task.future.done() or task.deadline <= time.monotonic():
            return
        with self._condition:
            queue = self._images if task.kind == "image" else self._videos
            queue.appendleft(task)
            self._condition.notify()

    def get_next(self) -> InferenceTask | None:
        with self._condition:
            while not self._closed and not self._images and not self._videos:
                self._condition.wait()
            if self._closed:
                return None

            if self._images and self._videos:
                if self._prefer_video:
                    task = self._videos.popleft()
                else:
                    task = self._images.popleft()
                self._prefer_video = not self._prefer_video
                return task
            if self._images:
                return self._images.popleft()
            return self._videos.popleft()

    def counts(self) -> tuple[int, int]:
        with self._condition:
            return len(self._images), len(self._videos)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class WorkerTransport(Protocol):
    def probe(self, worker_url: str) -> bool: ...

    def execute(
        self, worker_url: str, gpu_index: int, task: InferenceTask
    ) -> TaskResult: ...


class RequestsWorkerTransport:
    def __init__(self, connect_timeout: float, read_timeout: float) -> None:
        self._requests = importlib.import_module("requests")
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._local = threading.local()

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._requests.Session()
            self._local.session = session
        return session

    def probe(self, worker_url: str) -> bool:
        try:
            response = self._session().get(
                f"{worker_url}/health/ready",
                timeout=(self._connect_timeout, self._connect_timeout),
            )
            return response.status_code == 200
        except self._requests.RequestException:
            return False

    def execute(
        self, worker_url: str, gpu_index: int, task: InferenceTask
    ) -> TaskResult:
        try:
            if task.kind == "image":
                response = self._session().post(
                    f"{worker_url}/internal/{task.action}",
                    json=task.payload,
                    timeout=(self._connect_timeout, self._read_timeout),
                )
            else:
                frame = task.payload["frame"]
                response = self._session().post(
                    f"{worker_url}/internal/predict_video_frame",
                    data=frame,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Frame-Height": str(task.payload["height"]),
                        "X-Frame-Width": str(task.payload["width"]),
                    },
                    timeout=(self._connect_timeout, self._read_timeout),
                )
        except self._requests.RequestException as exc:
            raise WorkerTransportError(str(exc)) from exc

        return TaskResult(
            status_code=response.status_code,
            body=response.content,
            content_type=response.headers.get(
                "Content-Type", "application/octet-stream"
            ),
            gpu_index=gpu_index,
        )


class WorkerPool:
    def __init__(
        self,
        worker_urls: tuple[str, ...],
        broker: FairTaskBroker,
        transport: WorkerTransport,
        health_interval_seconds: float = 5.0,
    ) -> None:
        self._worker_urls = worker_urls
        self._broker = broker
        self._transport = transport
        self._health_interval_seconds = health_interval_seconds
        self._online = [False] * len(worker_urls)
        self._state_condition = threading.Condition()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for gpu_index, worker_url in enumerate(self._worker_urls):
            thread = threading.Thread(
                target=self._dispatch_loop,
                args=(gpu_index, worker_url),
                name=f"gpu-dispatch-{gpu_index}",
                daemon=True,
            )
            thread.start()
        threading.Thread(
            target=self._health_loop,
            name="gpu-health-monitor",
            daemon=True,
        ).start()

    def _set_online(self, gpu_index: int, online: bool) -> None:
        with self._state_condition:
            changed = self._online[gpu_index] != online
            self._online[gpu_index] = online
            if changed:
                LOGGER.info("gpu=%s online=%s", gpu_index, online)
            self._state_condition.notify_all()

    def _wait_until_online(self, gpu_index: int) -> None:
        with self._state_condition:
            while not self._online[gpu_index]:
                self._state_condition.wait(timeout=self._health_interval_seconds)

    def _health_loop(self) -> None:
        while True:
            for gpu_index, worker_url in enumerate(self._worker_urls):
                self._set_online(
                    gpu_index, self._transport.probe(worker_url)
                )
            time.sleep(self._health_interval_seconds)

    def _dispatch_loop(self, gpu_index: int, worker_url: str) -> None:
        while True:
            self._wait_until_online(gpu_index)
            task = self._broker.get_next()
            if task is None:
                return
            if task.future.done():
                continue
            if task.deadline <= time.monotonic():
                task.future.cancel()
                continue
            try:
                result = self._transport.execute(worker_url, gpu_index, task)
            except WorkerTransportError:
                LOGGER.exception(
                    "worker transport failed gpu=%s action=%s",
                    gpu_index,
                    task.action,
                )
                self._set_online(gpu_index, False)
                self._broker.requeue_front(task)
                continue
            if not task.future.done():
                task.future.set_result(result)

    def online_gpu_indices(self) -> tuple[int, ...]:
        with self._state_condition:
            return tuple(i for i, online in enumerate(self._online) if online)


class InferenceScheduler:
    def __init__(
        self,
        worker_urls: tuple[str, ...],
        image_queue_size: int,
        video_queue_size: int,
        connect_timeout: float,
        read_timeout: float,
        health_interval: float,
        transport: WorkerTransport | None = None,
    ) -> None:
        self.broker = FairTaskBroker(image_queue_size, video_queue_size)
        actual_transport = transport or RequestsWorkerTransport(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        self.pool = WorkerPool(
            worker_urls=worker_urls,
            broker=self.broker,
            transport=actual_transport,
            health_interval_seconds=health_interval,
        )
        self.pool.start()

    def enqueue_image(
        self,
        action: str,
        spool_path: Path,
        resource_id: str | None,
        media_type: str | None,
        timeout_seconds: float,
    ) -> Future[TaskResult]:
        task = InferenceTask(
            kind="image",
            action=action,
            payload={
                "spool_path": str(spool_path),
                "resource_id": resource_id,
                "media_type": media_type,
            },
            deadline=time.monotonic() + timeout_seconds,
        )
        self.broker.put(task)
        return task.future

    def enqueue_video_frame(
        self,
        frame: bytes,
        height: int,
        width: int,
        timeout_seconds: float,
    ) -> Future[TaskResult]:
        task = InferenceTask(
            kind="video",
            action="predict_video_frame",
            payload={"frame": frame, "height": height, "width": width},
            deadline=time.monotonic() + timeout_seconds,
        )
        self.broker.put(task)
        return task.future
