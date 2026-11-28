#!/usr/bin/env python3
"""Measure warmed single-image TensorRT detector latency on verified images."""

import argparse
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

from pbiomon_train import check_file, checksum, write_json


def read_images(manifest):
    samples = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        digest = row["sha256"]
        if digest in samples:
            raise ValueError("Benchmark image list repeats content")
        path = (manifest.parent / row["image"]).resolve()
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Image digest mismatch: {path}")
        samples[digest] = payload
    if not samples:
        raise ValueError("Benchmark needs verified images")
    return samples


def run(args, engine, samples, stage):
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(stage / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(stage / "matplotlib")

    import cv2
    from PIL import Image
    import tensorrt
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose one evaluation GPU")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "T4" not in gpu:
        raise ValueError(f"Expected the benchmark T4, got {gpu}")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    settings.update({key: False for key in ("sync", "hub") if key in settings})
    model = YOLO(str(engine), task="detect")

    def predict(payload):
        with Image.open(io.BytesIO(payload)) as image, torch.inference_mode():
            result, = model.predict(
                source=image, imgsz=640, device=0, conf=args.confidence, iou=0.7, max_det=300,
                augment=False, verbose=False, save=False, project=str(stage), name="predict",
            )
            return result.boxes.data.cpu().tolist()

    for _ in range(args.warmup_rounds):
        for payload in samples.values():
            predict(payload)
    torch.cuda.synchronize(device)
    per_image = {key: [] for key in samples}
    outputs = {}
    mismatches = 0
    for _ in range(args.rounds):
        for digest, payload in samples.items():
            torch.cuda.synchronize(device)
            tick = time.perf_counter_ns()
            detections = predict(payload)
            torch.cuda.synchronize(device)
            per_image[digest].append((time.perf_counter_ns() - tick) / 1e6)
            mismatches += outputs.setdefault(digest, detections) != detections
    intervals = sorted(t for runs in per_image.values() for t in runs)
    return {
        "status": "complete", "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "engine_sha256": args.engine_sha256, "class_count": len(model.names), "images": len(samples),
        "warmup_rounds": args.warmup_rounds, "measured_rounds": args.rounds, "requests": len(intervals),
        "latency_ms": {"p50": statistics.median(intervals), "p95": intervals[math.ceil(.95 * len(intervals)) - 1]},
        "measurement": "in-memory image decode, predictor, and detection tensor copied to host",
        "options": {"imgsz": 640, "batch": 1, "confidence": args.confidence, "nms_iou": 0.7, "max_det": 300},
        "runtime": {"gpu": gpu, "tensorrt": tensorrt.__version__, "ultralytics": ultralytics.__version__,
                    "torch": torch.__version__, "cuda": torch.version.cuda},
        "repeat_output_mismatches": mismatches, "detections": outputs, "durations_ms_by_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--images", type=Path, required=True, help="JSONL with image paths and SHA256")
    parser.add_argument("--confidence", type=float, default=0.256)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.confidence < 1 and args.warmup_rounds > 0 and args.rounds > 0):
        parser.error("Invalid confidence, warm-up or measurement settings")
    os.umask(0o077)
    engine = args.engine.resolve()
    if engine.suffix != ".engine":
        parser.error("TensorRT benchmark needs a serialized .engine")
    check_file(engine, args.engine_sha256)
    manifest = args.images.resolve()
    manifest_digest = checksum(manifest)
    samples = read_images(manifest)
    out = args.out.resolve()
    if out.exists():
        raise ValueError(f"Benchmark output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + out.name + "-", dir=out.parent) as temporary:
        stage = Path(temporary)
        with tempfile.TemporaryDirectory(dir=stage) as runtime:
            result = run(args, engine, samples, Path(runtime))
        check_file(engine, args.engine_sha256)
        check_file(manifest, manifest_digest)
        result["images_sha256"] = manifest_digest
        write_json(stage / "receipt.json", result)
        stage.rename(out)
    print(json.dumps({k: result[k] for k in ("runtime", "images", "requests", "latency_ms",
                                             "repeat_output_mismatches")}, ensure_ascii=False))
    print(checksum(out / "receipt.json"), out / "receipt.json")


if __name__ == "__main__":
    main()
