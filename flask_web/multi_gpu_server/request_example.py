from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the classifier or detector using the legacy API format."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--endpoint",
        choices=("predict_classifier", "predict_detect"),
        default="predict_classifier",
    )
    parser.add_argument("--url", default="http://127.0.0.1:5002")
    parser.add_argument("--resource-id", default="example-resource-id")
    parser.add_argument("--media-type", default="photo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.image.open("rb") as handle:
        response = requests.post(
            f"{args.url.rstrip('/')}/{args.endpoint}",
            data=handle.read(),
            headers={
                "Content-Type": "application/octet-stream",
                "resourceId": args.resource_id,
                "mediaType": args.media_type,
            },
            timeout=60,
        )
    print(f"HTTP {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
