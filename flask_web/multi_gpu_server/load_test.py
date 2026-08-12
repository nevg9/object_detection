from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import statistics
import threading
import time
from pathlib import Path

import requests

from .consistency_check import image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate-limited mixed API load test")
    parser.add_argument("--url", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--qps", type=float, default=40)
    parser.add_argument("--classifier-ratio", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=20)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    args = parse_args()
    paths = image_paths(args.images)
    if not paths:
        raise SystemExit("no images found")
    payloads = [(path.name, path.read_bytes()) for path in paths]
    payload_cycle = itertools.cycle(payloads)
    latencies: list[float] = []
    statuses: list[int] = []
    lock = threading.Lock()
    session_local = threading.local()

    def one_request(sequence: int) -> None:
        session = getattr(session_local, "session", None)
        if session is None:
            session = requests.Session()
            session_local.session = session
        name, body = next(payload_cycle)
        ratio_position = (sequence % 10) / 10
        endpoint = (
            "predict_classifier"
            if ratio_position < args.classifier_ratio
            else "predict_detect"
        )
        started = time.perf_counter()
        try:
            response = session.post(
                f"{args.url.rstrip('/')}/{endpoint}",
                data=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "resourceId": f"load-{sequence}-{name}",
                    "mediaType": "photo",
                },
                timeout=args.timeout,
            )
            status = response.status_code
        except requests.RequestException:
            status = 0
        latency = (time.perf_counter() - started) * 1000
        with lock:
            latencies.append(latency)
            statuses.append(status)

    interval = 1.0 / args.qps
    stop_at = time.monotonic() + args.duration
    futures: set[concurrent.futures.Future[None]] = set()
    sequence = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        next_submit = time.monotonic()
        while time.monotonic() < stop_at:
            now = time.monotonic()
            if now < next_submit:
                time.sleep(next_submit - now)
            futures = {future for future in futures if not future.done()}
            if len(futures) >= args.concurrency:
                concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                futures = {future for future in futures if not future.done()}
            futures.add(executor.submit(one_request, sequence))
            sequence += 1
            next_submit += interval
        concurrent.futures.wait(futures)
    elapsed = time.monotonic() - started

    successes = sum(status == 200 for status in statuses)
    errors = len(statuses) - successes
    print(f"requests={len(statuses)} successes={successes} errors={errors}")
    print(f"achieved_qps={len(statuses) / elapsed:.2f}")
    print(
        "latency_ms "
        f"avg={statistics.mean(latencies):.2f} "
        f"p50={percentile(latencies, 0.50):.2f} "
        f"p95={percentile(latencies, 0.95):.2f} "
        f"p99={percentile(latencies, 0.99):.2f} "
        f"max={max(latencies):.2f}"
    )
    error_rate = errors / len(statuses) if statuses else 1.0
    return 0 if percentile(latencies, 0.95) <= 1000 and error_rate < 0.001 else 1


if __name__ == "__main__":
    raise SystemExit(main())
