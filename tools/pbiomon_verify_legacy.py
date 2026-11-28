#!/usr/bin/env python3
"""Read-only media and provenance check for the historical Pbiomon dataset."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import gzip
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageOps


def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(output, status):
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "status.json")


def read_candidates(manifest, root):
    candidates = {}
    with gzip.open(manifest, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            split = row["historical_split"]
            path = Path(row["source_image"])
            if split not in ("train", "test") or path.parent != root / split / "images":
                raise ValueError(f"Candidate outside the historical split: {path}")
            key = (split, path.name)
            if key in candidates:
                raise ValueError(f"Repeated image in candidate manifest: {path}")
            candidates[key] = row
    return candidates


def image_slots(root, candidates):
    for split in ("train", "test"):
        with os.scandir(root / split / "images") as entries:
            files = sorted(
                ((entry.inode(), entry.name) for entry in entries
                 if entry.name.lower().endswith(".jpg") and entry.is_file()),
            )
        for _, name in files:
            yield split, root / split / "images" / name, candidates.pop((split, name), {})


def verify(slot):
    split, path, candidate = slot
    row = {
        "historical_split": split,
        "image": str(path),
        "phase": candidate.get("phase", "absent_from_candidate_manifest"),
        "event_candidate": candidate.get("filename_event_candidate"),
    }
    try:
        stat = path.stat()
        data = path.read_bytes()
        row.update(
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
            device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
        )
        after = path.stat()
        if (stat.st_ino, stat.st_mtime_ns, stat.st_size) != (
            after.st_ino, after.st_mtime_ns, len(data)
        ) or after.st_size != len(data):
            raise ValueError("Image changed during verification")
        with Image.open(io.BytesIO(data)) as image:
            orientation = image.getexif().get(274, 1)
            width, height = image.size
            if image.format != "JPEG" or min(width, height) < 10 or not data.endswith(b"\xff\xd9"):
                raise ValueError("Invalid JPEG format, size or end marker")
            image.draft("L", (64, 64))
            image = ImageOps.exif_transpose(image)
            thumb = image.convert("L").resize((16, 16), Image.Resampling.BOX)
        row.update(
            dimensions=[width, height], orientation=orientation,
            gray16=thumb.tobytes().hex(),
        )
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        row.update(status="image_reject", error=str(error))
        return row

    if row["phase"] == "positive_label_text_valid":
        label = Path(candidate["source_label"])
        if (
            label.parent != path.parent.parent / "labels"
            or label.stem != path.stem or label.suffix.lower() != ".txt"
        ):
            raise ValueError(f"Unexpected source label: {label}")
        try:
            digest = hashlib.sha256(label.read_bytes()).hexdigest()
        except OSError:
            digest = None
        if digest != candidate["source_label_sha256"]:
            row["status"] = "source_label_changed"
        else:
            row.update(status="positive", label_sha256=digest, rows=candidate["rows"])
    elif row["phase"] == "source_name_match_no_target":
        # A matching filename alone does not establish the background provenance.
        source = Path(candidate["cohort_candidate_image"])
        expected = path.parents[3] / "no_target"
        if source.parent not in (expected / "train/images", expected / "test/images"):
            raise ValueError(f"Unexpected no_target source: {source}")
        try:
            matched = source.stat().st_size == len(data) and source.read_bytes() == data
        except OSError:
            matched = False
        row.update(
            status="background" if matched else "background_source_unverified",
            cohort_source=str(source),
        )
    else:
        row["status"] = "review_unlabeled"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Historical animal_action directory")
    parser.add_argument("--candidates", type=Path, required=True, help="Legacy candidate JSONL.GZ")
    parser.add_argument("--out", type=Path, required=True, help="New receipt directory")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.root.resolve()
    output = args.out.resolve()
    if root == output or root in output.parents:
        parser.error("--out must be outside the source dataset")
    os.umask(0o077)
    output.mkdir(mode=0o700, exist_ok=False)
    state = {
        "started_at": stamp(), "status": "reading_candidates",
        "root": str(root), "candidates": str(args.candidates.resolve()),
        "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
    }
    write_status(output, state)
    candidates = read_candidates(args.candidates, root)
    counts = Counter()
    started = time.monotonic()
    temporary = output / "images.jsonl.gz.partial"
    state["status"] = "scanning"
    write_status(output, state)
    with ThreadPoolExecutor(max_workers=args.workers) as executor, gzip.open(
        temporary, "wt", encoding="utf-8", compresslevel=1,
    ) as stream:
        slots = image_slots(root, candidates)
        while batch := list(itertools.islice(slots, 2048)):
            for row in executor.map(verify, batch):
                counts[f"{row['historical_split']}:{row['status']}"] += 1
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            state.update(
                checked=sum(counts.values()), counts=dict(counts), heartbeat_at=stamp(),
                seconds=round(time.monotonic() - started, 1),
            )
            write_status(output, state)
        if candidates:
            raise ValueError(f"{len(candidates)} candidate images disappeared")
    digest = hashlib.sha256()
    with temporary.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    temporary.replace(output / "images.jsonl.gz")
    state.update(status="complete", finished_at=stamp(), sha256=digest.hexdigest())
    write_status(output, state)


if __name__ == "__main__":
    main()
