#!/usr/bin/env python3
"""Prepare legacy training and regression candidates and record archive overlap."""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

from pbiomon_train import check_file, checksum, write_json


def records(path, compressed=True):
    reader = gzip.open if compressed else open
    with reader(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def emit(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def photo_keys(row):
    stem = Path(row["image"]).stem.casefold()
    match = re.fullmatch(r"(.+)-(20[0-9]{6})-[0-9]+", stem)
    event = None
    if match:
        try:
            day = dt.datetime.strptime(match[2], "%Y%m%d").date().isoformat()
            event = (match[1], day)
        except ValueError:
            pass
    gray = row.get("gray16")
    thumb = hashlib.blake2b(bytes.fromhex(gray), digest_size=16).digest() if gray else None
    return stem, event, row.get("sha256"), thumb


def signature(row, background_sources):
    if row["status"] == "positive":
        labels = json.dumps(sorted(row["rows"]), separators=(",", ":"))
        return hashlib.sha256(labels.encode()).hexdigest()
    if row["status"] == "background" and Path(row["cohort_source"]).name in background_sources:
        return "background"
    return None


def index_legacy(verified, background_sources):
    indices = {split: tuple(set() for _ in range(4)) for split in ("train", "test")}
    canonical = {}
    conflicts = set()
    for row in records(verified):
        split = row["historical_split"]
        keys = photo_keys(row)
        for lookup, value in zip(indices[split], keys):
            if value is not None:
                lookup.add(value)
        label = signature(row, background_sources)
        if label is None:
            continue
        sha = keys[2]
        identity = (split != "train", row["image"])
        previous = canonical.get(sha)
        if previous and previous[0] != label:
            conflicts.add(sha)
        if previous is None or identity < previous[1]:
            canonical[sha] = (label, identity)
    return indices, canonical, conflicts


def possible_training_overlap(row, training):
    return any(key is not None and key in lookup for key, lookup in zip(photo_keys(row), training))


def prepare_legacy(verified, output, background_sources, indices, canonical, conflicts):
    train = indices["train"]
    # A content or name match exposes its entire camera-day to the historical training split.
    exposed_events = set(train[1])
    for row in records(verified):
        if row["historical_split"] == "test" and possible_training_overlap(row, train):
            event = photo_keys(row)[1]
            if event:
                exposed_events.add(event)
    counts = Counter()
    classes = defaultdict(Counter)
    name = "legacy-partitions.jsonl.gz"
    with gzip.open(output / name, "wt", encoding="utf-8", compresslevel=1) as stream:
        for row in records(verified):
            split = row["historical_split"]
            _, event, sha, _ = photo_keys(row)
            label = signature(row, background_sources)
            if label is None:
                decision = (
                    "background_receipt_missing" if row["status"] == "background"
                    else row["status"]
                )
            elif sha in conflicts:
                decision = "conflicting_labels_same_content"
            elif canonical[sha][1][1] != row["image"]:
                decision = "duplicate_content"
            elif split == "train":
                decision = "training_candidate"
            elif possible_training_overlap(row, train) or event in exposed_events:
                decision = "historical_train_overlap"
            elif event is None:
                decision = "regression_event_unresolved"
            else:
                decision = "regression_candidate"
            counts[f"{split}:{decision}"] += 1
            export = {
                "image": row["image"], "sha256": sha, "historical_split": split,
                "decision": decision, "camera_day_from_filename": event, "source_status": row["status"],
            }
            if decision in ("training_candidate", "regression_candidate"):
                labels = row.get("rows", [])
                classes[decision].update(box[0] for box in labels)
                export.update(
                    rows=labels, dimensions=row["dimensions"], orientation=row["orientation"],
                    source_label_sha256=row.get("label_sha256"), background_source=row.get("cohort_source"),
                )
            emit(stream, export)
    return name, counts, classes


def prepare_archive(photos, output, indices):
    counts = Counter()
    name = "archive-photo-overlap.jsonl.gz"
    with gzip.open(output / name, "wt", encoding="utf-8", compresslevel=1) as stream:
        for row in records(photos):
            sha, stem = row["media_sha256"], row["filename_stem"]
            exact = [split for split, lookup in indices.items() if sha and sha in lookup[2]]
            same_name = [split for split, lookup in indices.items() if stem and stem in lookup[0]]
            decision = (
                "exact_content_match" if exact else "same_filename_candidate" if same_name
                else "no_exact_or_filename_match"
            )
            counts[decision] += 1
            emit(stream, {
                "resource_key": row["resource_key"], "media_sha256": sha,
                "exact_legacy_splits": exact, "same_name_legacy_splits": same_name, "decision": decision,
                "monitor_unit": row["monitor_unit"], "captured_at": row["captured_at"],
            })
    return name, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-verification", type=Path, required=True)
    parser.add_argument("--archive-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    legacy_receipt = args.legacy_verification / "status.json"
    archive_receipt = args.archive_inventory / "receipt.json"
    receipt = json.loads(legacy_receipt.read_bytes())
    if receipt["status"] != "complete":
        raise ValueError("Legacy media verification has not completed")
    archive = json.loads(archive_receipt.read_bytes())
    if archive["status"] != "indexed":
        raise ValueError("Archive identity index has not completed")
    verified = check_file(args.legacy_verification / "images.jsonl.gz", receipt["sha256"])
    photos = check_file(
        args.archive_inventory / "photo-identities.jsonl.gz",
        archive["files"]["photo-identities.jsonl.gz"],
    )
    root = Path(receipt["root"]).resolve()
    backgrounds = root.parent / "no_target/success.json"
    background_sources = {Path(row["本地路径"]).name for row in records(backgrounds, compressed=False)}
    output = args.out.resolve()
    if root.parent == output or root.parent in output.parents:
        raise ValueError("Partition directory must be outside the original dataset")
    os.umask(0o077)
    output.mkdir(mode=0o700, exist_ok=False)
    indices, canonical, conflicts = index_legacy(verified, background_sources)
    legacy, old_counts, classes = prepare_legacy(
        verified, output, background_sources, indices, canonical, conflicts,
    )
    novel, new_counts = prepare_archive(photos, output, indices)
    if sum(old_counts.values()) != receipt["checked"] or sum(new_counts.values()) != archive["counts"]["photos"]:
        raise ValueError("Partition row counts did not match both source receipts")
    write_json(output / "receipt.json", {
        "status": "candidate_partitions",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "legacy_verification_sha256": checksum(legacy_receipt),
        "archive_inventory_sha256": checksum(archive_receipt),
        "legacy_background_success_sha256": checksum(backgrounds),
        "counts_legacy": old_counts, "counts_archive_photo": new_counts,
        "class_box_counts": classes,
        "files": {name: checksum(output / name) for name in (legacy, novel)},
    })


if __name__ == "__main__":
    main()
