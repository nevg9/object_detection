from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any


DYNAMIC_VALUE_KEYS = {"processing_ms"}


class CompatibilityMismatch(AssertionError):
    pass


def compare_values(old: Any, new: Any, path: str = "$", tolerance: float = 1e-5) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            raise CompatibilityMismatch(
                f"{path}: keys differ old={sorted(old)} new={sorted(new)}"
            )
        for key in old:
            if key in DYNAMIC_VALUE_KEYS:
                if not isinstance(new[key], (int, float)):
                    raise CompatibilityMismatch(
                        f"{path}.{key}: expected numeric timing value"
                    )
                continue
            compare_values(old[key], new[key], f"{path}.{key}", tolerance)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            raise CompatibilityMismatch(
                f"{path}: list lengths differ old={len(old)} new={len(new)}"
            )
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            compare_values(old_item, new_item, f"{path}[{index}]", tolerance)
        return
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if type(old) is not type(new):
            raise CompatibilityMismatch(
                f"{path}: numeric types differ "
                f"old={type(old).__name__} new={type(new).__name__}"
            )
        if isinstance(old, bool) or isinstance(new, bool):
            if old != new:
                raise CompatibilityMismatch(f"{path}: {old!r} != {new!r}")
            return
        if not math.isclose(float(old), float(new), abs_tol=tolerance, rel_tol=0):
            raise CompatibilityMismatch(f"{path}: {old!r} != {new!r}")
        return
    if old != new:
        raise CompatibilityMismatch(f"{path}: {old!r} != {new!r}")


def post_media(
    base_url: str,
    endpoint: str,
    path: Path,
    resource_id: str,
    media_type: str,
    timeout: float,
) -> Any:
    requests = importlib.import_module("requests")
    try:
        with path.open("rb") as handle:
            return requests.post(
                f"{base_url.rstrip('/')}/{endpoint}",
                data=handle,
                headers={
                    "Content-Type": "application/octet-stream",
                    "resourceId": resource_id,
                    "mediaType": media_type,
                },
                timeout=timeout,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"request failed for {endpoint} {path}: {exc}") from exc


def check_one(
    old_url: str,
    new_url: str,
    endpoint: str,
    path: Path,
    media_type: str,
    tolerance: float,
    timeout: float,
) -> None:
    resource_id = f"compat-{path.name}"
    old_response = post_media(
        old_url, endpoint, path, resource_id, media_type, timeout
    )
    new_response = post_media(
        new_url, endpoint, path, resource_id, media_type, timeout
    )
    if old_response.status_code != new_response.status_code:
        raise CompatibilityMismatch(
            f"{path} {endpoint}: status differs "
            f"old={old_response.status_code} new={new_response.status_code}"
        )
    old_content_type = old_response.headers.get("Content-Type", "").split(";", 1)[0]
    new_content_type = new_response.headers.get("Content-Type", "").split(";", 1)[0]
    if old_content_type != new_content_type:
        raise CompatibilityMismatch(
            f"{path} {endpoint}: content type differs "
            f"old={old_content_type!r} new={new_content_type!r}"
        )
    try:
        old_payload = old_response.json()
        new_payload = new_response.json()
    except ValueError:
        if old_response.content != new_response.content:
            raise CompatibilityMismatch(
                f"{path} {endpoint}: non-JSON response bodies differ"
            )
    else:
        compare_values(old_payload, new_payload, tolerance=tolerance)


def image_paths(inputs: list[str]) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    result: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            result.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in supported
            )
        elif path.is_file():
            result.append(path)
        else:
            raise FileNotFoundError(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and multi-GPU API responses."
    )
    parser.add_argument("--old-url", required=True)
    parser.add_argument("--new-url", required=True)
    parser.add_argument("--images", nargs="+", default=[])
    parser.add_argument("--video")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = 0
    try:
        for path in image_paths(args.images):
            for endpoint in ("predict_classifier", "predict_detect"):
                check_one(
                    args.old_url,
                    args.new_url,
                    endpoint,
                    path,
                    "photo",
                    args.tolerance,
                    args.timeout,
                )
                checks += 1
                print(f"PASS {endpoint} {path}")
        if args.video:
            video_path = Path(args.video)
            check_one(
                args.old_url,
                args.new_url,
                "predict_video",
                video_path,
                "video",
                args.tolerance,
                args.timeout,
            )
            checks += 1
            print(f"PASS predict_video {video_path}")
    except (CompatibilityMismatch, OSError, RuntimeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    if checks == 0:
        print("FAIL no media files were selected", file=sys.stderr)
        return 2
    print(f"All {checks} compatibility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
