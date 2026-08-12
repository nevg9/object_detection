from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from safetensors.torch import load_file
from ultralytics import YOLO
from ultralytics.data.dataset import classify_transforms
from ultralytics.nn.tasks import ClassificationModel, torch_safe_load

from .config import ServiceConfig, ensure_runtime_directories
from .logging_utils import configure_file_logging


ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger("gpu-worker")


def load_yolo_model(path: str, nc: int):
    """Keep the classifier loading path identical to server_241124.py."""

    ckpt, _ = torch_safe_load(path)
    model = ckpt["model"]
    ClassificationModel.reshape_outputs(model, nc)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.float()


class ModelRuntime:
    def __init__(self) -> None:
        torch.set_num_threads(int(os.getenv("OD_TORCH_NUM_THREADS", "1")))
        self.device = 0
        self.worker_index = int(os.getenv("OD_GPU_INDEX", "0"))
        classifier_model_path = os.getenv(
            "OD_CLASSIFIER_MODEL_PATH",
            "/mnt/data1/model_1122/image_classifier/yolov8m-cls.pt",
        )
        classifier_checkpoint_path = os.getenv(
            "OD_CLASSIFIER_CHECKPOINT_PATH",
            "/mnt/data1/model_1122/image_classifier/epoch_30/model.safetensors",
        )
        detector_path = os.getenv(
            "OD_DETECTOR_PATH",
            "/mnt/data1/model_1122/image_detection/best.engine",
        )

        LOGGER.info(
            "loading models gpu=%s cuda_visible_devices=%s",
            self.worker_index,
            os.getenv("CUDA_VISIBLE_DEVICES"),
        )
        classifier = load_yolo_model(classifier_model_path, 3)
        weights = load_file(classifier_checkpoint_path)
        classifier.load_state_dict(weights)
        classifier.to(self.device)
        self.classifier_model = classifier
        self.torch_transforms = classify_transforms(size=224)
        self.classifier_detail = {0: "动物", 1: "人类", 2: "无目标"}
        self.detect_model = YOLO(detector_path, task="detect")
        self.gender_set = {"雄性", "雌性"}
        self.age_set = {"成体", "亚成体", "幼体"}
        LOGGER.info("models loaded gpu=%s", self.worker_index)

    def process_classifier_image(self, image_binary: bytes):
        try:
            nparr = np.frombuffer(image_binary, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            return self.torch_transforms(image).to(self.device)
        except Exception as exc:
            LOGGER.warning("Error decoding image: %s", exc)
            return None

    def classify_tensor(self, sample: torch.Tensor) -> tuple[float, int]:
        with torch.inference_mode():
            predict_value = self.classifier_model(sample.unsqueeze(0))
            max_prob, predicted_class = torch.max(
                F.softmax(predict_value, dim=1), dim=1
            )
        return max_prob.item(), predicted_class.item()

    def classifier_response(
        self,
        binary_data: bytes,
        resource_id: str | None,
        media_type: str | None,
    ):
        start_time = time.time()
        processed_image = self.process_classifier_image(binary_data)
        if processed_image is None:
            response_data = {}
            if resource_id:
                response_data["resourceId"] = resource_id
            if media_type:
                response_data["mediaType"] = media_type
            response_data["error-reason"] = "image is corrupt"
            return jsonify(response_data)

        process_image_end_time = time.time()
        image_time = (process_image_end_time - start_time) * 1000
        prediction, label = self.classify_tensor(processed_image)
        processing_time = (time.time() - start_time) * 1000
        LOGGER.info(
            "endpoint=classifier gpu=%s processing_ms=%.3f image_ms=%.3f",
            self.worker_index,
            processing_time,
            image_time,
        )
        response_data = {"prediction": prediction, "label": label}
        if label in self.classifier_detail:
            response_data["desc"] = self.classifier_detail[label]
        if resource_id:
            response_data["resourceId"] = resource_id
        if media_type:
            response_data["mediaType"] = media_type
        return jsonify(response_data)

    def process_predict(self, results):
        detect_class = []
        agg_detect_class = []
        class_id_to_num = {}
        class_id_to_detail = {}
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                if class_id not in class_id_to_num:
                    class_id_to_num[class_id] = 0
                class_id_to_num[class_id] += 1
                class_name = result.names[class_id]

                class_name_copy = class_name
                gender = "难以鉴定"
                for value in self.gender_set:
                    if value in class_name_copy:
                        class_name_copy = class_name_copy.replace(value, "")
                        gender = value
                age = "难以鉴定"
                for value in self.age_set:
                    if value in class_name_copy:
                        class_name_copy = class_name_copy.replace(value, "")
                        age = value
                class_id_to_detail[class_id] = [class_name_copy, gender, age]
                detect_class.append(
                    (
                        class_name,
                        (x1, y1, x2, y2),
                        confidence,
                        [class_name_copy, gender, age],
                    )
                )
        for class_id, count in class_id_to_num.items():
            agg_detect_class.append((class_id_to_detail[class_id], count))
        return {"results": detect_class, "agg_results": agg_detect_class}

    def detect_image(self, image_content, conf: float):
        with torch.inference_mode():
            results = self.detect_model.predict(
                image_content, conf=conf, verbose=False
            )
        return self.process_predict(results)

    def detect_response(
        self,
        binary_data: bytes,
        resource_id: str | None,
        media_type: str | None,
    ):
        start_time = time.time()
        image_content = Image.open(io.BytesIO(binary_data))
        results_dict = self.detect_image(image_content, conf=0.256)
        processing_time = (time.time() - start_time) * 1000
        LOGGER.info(
            "endpoint=detect gpu=%s processing_ms=%.3f",
            self.worker_index,
            processing_time,
        )
        if resource_id:
            results_dict["resourceId"] = resource_id
        if media_type:
            results_dict["mediaType"] = media_type
        return jsonify(results_dict)

    def process_video_frame(self, image_bgr: np.ndarray) -> dict:
        image = Image.fromarray(image_bgr[..., ::-1])
        sample = self.torch_transforms(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            classifier_logits = self.classifier_model(sample)
            _, top = classifier_logits.softmax(1).max(1)
        if top.item() != 0:
            return {"has_target": False}
        frame_dict = self.detect_image(image_bgr, conf=0.25)
        return {"has_target": True, "frame": frame_dict}


def _validated_spool_path(raw_path: str, spool_dir: Path) -> Path:
    spool_root = spool_dir.resolve()
    path = Path(raw_path).resolve(strict=True)
    if path.parent != spool_root or not path.name.startswith("request-"):
        raise ValueError("invalid spool path")
    return path


def create_app() -> Flask:
    config = ServiceConfig.from_env()
    ensure_runtime_directories(config)
    worker_index = int(os.getenv("OD_GPU_INDEX", "0"))
    configure_file_logging(f"worker-{worker_index}.log", config.log_dir)
    app = Flask(__name__)
    runtime = ModelRuntime()
    app.config["MODEL_RUNTIME"] = runtime
    app.config["SERVICE_CONFIG"] = config

    @app.get("/health/live")
    def health_live():
        return jsonify({"status": "ok", "gpu": worker_index})

    @app.get("/health/ready")
    def health_ready():
        return jsonify({"status": "ready", "gpu": worker_index})

    # Direct routes are intentionally compatible with server_241124.py. They
    # are bound to loopback in production and are used for parity tests and
    # emergency fallback only.
    @app.post("/predict_classifier")
    def predict_classifier():
        return runtime.classifier_response(
            request.data,
            request.headers.get("resourceId"),
            request.headers.get("mediaType"),
        )

    @app.post("/predict_detect")
    def predict_detect():
        return runtime.detect_response(
            request.data,
            request.headers.get("resourceId"),
            request.headers.get("mediaType"),
        )

    @app.post("/predict_video")
    def predict_video():
        start = time.time()
        video_bytes = request.data
        resource_id = request.headers.get("resourceId")
        media_type = request.headers.get("mediaType")
        container = av.open(io.BytesIO(video_bytes))
        frames_json = []
        stride = 1
        last_ts_ms = 0
        species_accumulate = 0
        frame_idx = -1
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx % stride:
                continue
            ts_ms = frame.time * 1000
            last_ts_ms = ts_ms
            frame_result = runtime.process_video_frame(
                frame.to_ndarray(format="bgr24")
            )
            if not frame_result["has_target"]:
                continue
            frame_dict = frame_result["frame"]
            species_accumulate += len(frame_dict["agg_results"])
            frame_dict["frame_id"] = frame_idx
            frame_dict["ts_ms"] = round(ts_ms)
            frames_json.append(frame_dict)
        if container.duration is not None:
            duration_ms = container.duration / av.time_base * 1000
        else:
            duration_ms = last_ts_ms
        response = {
            "frames": frames_json,
            "summary": {
                "species_accumulate": species_accumulate,
                "duration_frames": frame_idx + 1,
                "duration_ms": round(duration_ms),
            },
            "processing_ms": round((time.time() - start) * 1000),
        }
        if resource_id:
            response["resourceId"] = resource_id
        if media_type:
            response["mediaType"] = media_type
        return jsonify(response)

    @app.post("/internal/predict_classifier")
    def internal_predict_classifier():
        descriptor = request.get_json(force=True)
        path = _validated_spool_path(descriptor["spool_path"], config.spool_dir)
        return runtime.classifier_response(
            path.read_bytes(),
            descriptor.get("resource_id"),
            descriptor.get("media_type"),
        )

    @app.post("/internal/predict_detect")
    def internal_predict_detect():
        descriptor = request.get_json(force=True)
        path = _validated_spool_path(descriptor["spool_path"], config.spool_dir)
        return runtime.detect_response(
            path.read_bytes(),
            descriptor.get("resource_id"),
            descriptor.get("media_type"),
        )

    @app.post("/internal/predict_video_frame")
    def internal_predict_video_frame():
        height = int(request.headers["X-Frame-Height"])
        width = int(request.headers["X-Frame-Width"])
        frame = np.frombuffer(request.data, dtype=np.uint8).reshape(
            height, width, 3
        )
        return jsonify(runtime.process_video_frame(frame))

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("OD_WORKER_PORT", "5100"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=False)
