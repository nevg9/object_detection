#!/usr/bin/env python3
"""Freeze reviewed archive and legacy samples into a shared YOLO comparison dataset."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import datetime as dt
import fcntl
import gzip
from itertools import chain
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageOps

from pbiomon_classes import AGES, GENDERS, load_registry
from pbiomon_legacy_partitions import emit, records
from pbiomon_sync import digest, resource_key
from pbiomon_train import check_file, checksum, local_child, write_json


SPLITS = ("train", "val", "test")


def evidence(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing review evidence")
    return value


def index(value):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid zero-based index: {value}")
    return value


def version(path_root, review):
    kind, uid = review["type"], review["resource_id"]
    if kind not in ("PHOTO", "VIDEO") or not isinstance(uid, str) or not uid:
        raise ValueError("Invalid archive resource identity")
    key = resource_key(kind, uid)
    revision = review["bundle_sha256"]
    if not isinstance(revision, str) or len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Invalid immutable bundle digest")
    relative = Path("raw/annotations") / kind / key[:2] / key / (revision + ".json")
    bundle_path = local_child(path_root, relative)
    bundle = json.loads(bundle_path.read_bytes())
    resource, media = bundle["resource"], bundle["file"]
    if (
        digest(bundle) != revision or bundle["schemaVersion"] != 1
        or resource["resourceType"] != kind or resource["resourceId"] != uid
        or media["status"] != "done" or media["sha256"] != review["media_sha256"]
    ):
        raise ValueError(f"Approval no longer matches source media and annotation: {key}")
    source = local_child(path_root, media["path"])
    annotations = {row["speciesId"]: row for row in bundle["annotations"]}
    provenance = {row["speciesId"]: row["annotationHash"] for row in bundle["provenance"]}
    if (
        len(annotations) != len(bundle["annotations"]) or len(provenance) != len(bundle["provenance"])
        or annotations.keys() != provenance.keys()
        or any(digest(a) != provenance[k] for k, a in annotations.items())
    ):
        raise ValueError(f"Annotation versions do not match their source receipts: {key}")
    return key, relative, bundle_path, source, bundle, annotations, provenance


def check_box(corners, size):
    left, top, right, bottom = corners
    width, height = size
    if not (
        all(type(value) in (int, float) for value in corners)
        and 0 <= left < right <= width and 0 <= top < bottom <= height
    ):
        raise ValueError("Approved box is outside its reference image")
    return left, top, right, bottom


def box_pixels(raw, space, display_size, photo_geometry=None):
    """Convert a source box using a coordinate space confirmed in its review."""
    if space not in ("display_pixels", "raw_pixels", "raw_65536") or raw is None:
        raise ValueError("Each box needs resolved pixels or a confirmed source coordinate space")
    left, top, width, height = (raw.get(key) for key in ("left", "top", "width", "height"))
    if not all(type(value) in (int, float) and math.isfinite(value) for value in (left, top, width, height)):
        raise ValueError("Source box has missing or invalid coordinates")
    corners = left, top, left + width, top + height
    if space == "display_pixels":
        return check_box(corners, display_size)

    if photo_geometry is None:
        raise ValueError("Raw-image coordinates require verified photo geometry")
    raw_width, raw_height, orientation = photo_geometry
    expected = (raw_height, raw_width) if orientation in (5, 6, 7, 8) else (raw_width, raw_height)
    if tuple(display_size) != expected:
        raise ValueError("Reviewed display dimensions disagree with the photo orientation")
    ref_width, ref_height = (65536, 65536) if space == "raw_65536" else (raw_width, raw_height)
    left, top, right, bottom = check_box(corners, (ref_width, ref_height))
    x1, y1, x2, y2 = left/ref_width, top/ref_height, right/ref_width, bottom/ref_height
    # Match the transformations applied to the pixels by ImageOps.exif_transpose.
    oriented = {
        0: (x1, y1, x2, y2),
        1: (x1, y1, x2, y2),
        2: (1-x2, y1, 1-x1, y2),
        3: (1-x2, 1-y2, 1-x1, 1-y1),
        4: (x1, 1-y2, x2, 1-y1),
        5: (y1, x1, y2, x2),
        6: (1-y2, x1, 1-y1, x2),
        7: (1-y2, 1-x2, 1-y1, 1-x1),
        8: (y1, 1-x2, y2, 1-x1),
    }
    if orientation not in oriented:
        raise ValueError(f"Unsupported EXIF orientation: {orientation}")
    w, h = display_size
    return check_box(tuple(value * size for value, size in zip(oriented[orientation], (w, h, w, h))), display_size)


def reviewed_boxes(frame, annotations, class_ids, registry, photo_geometry=None):
    if frame["complete"] is not True or not frame["boxes"]:
        raise ValueError("The review must cover all targets in the frame")
    evidence(frame["review_evidence"])
    w, h = frame["display_size"]
    if type(w) is not int or type(h) is not int or min(w, h) < 10:
        raise ValueError("Invalid reviewed display dimensions")
    frame_id, timestamp = frame["frame_id"], frame["ts_ms"]
    expected = {
        (species, number)
        for species, row in annotations.items()
        for number, raw in enumerate(row["finalIdentification"]["boxes"])
        if frame_id is None or raw["frameId"] == frame_id
    }
    seen = set()
    species_seen = set()
    labels = []
    resolved_boxes = []
    for box in frame["boxes"]:
        species_id = box["species_id"]
        original = annotations[species_id]["finalIdentification"]
        species_seen.add(species_id)
        number = box["box_index"]
        source_box = None
        if number is not None:
            identity = (species_id, index(number))
            if identity not in expected or identity in seen:
                raise ValueError(f"Repeated or off-frame source box: {identity}")
            source_box = original["boxes"][number]
            if frame_id is not None and source_box["tsMs"] != timestamp:
                raise ValueError(f"Source box timestamp conflict: {identity}")
            seen.add(identity)
        evidence(box["coordinate_evidence"])
        evidence(box["association_evidence"])
        gender, age = box["gender"], box["age"]
        if gender not in GENDERS.values() or age not in AGES.values():
            raise ValueError("Each accepted box needs explicit standardized gender and age")
        attribute_index = box["attribute_index"]
        if attribute_index is not None:
            item = original["attributes"][index(attribute_index)]
            if GENDERS.get(item["gender"]) != gender or AGES.get(item["age"]) != age:
                raise ValueError("Reviewed gender/age disagrees with the linked identification item")
        code = class_ids[box["species"], gender, age]
        registered_ids = registry[code].get("species_ids")
        if registered_ids and species_id not in registered_ids:
            raise ValueError("Approved species is outside the frozen class registration")
        pixels = box.get("xyxy")
        if pixels is None:
            pixels = box_pixels(source_box, box.get("coordinate_space"), (w, h), photo_geometry)
        x1, y1, x2, y2 = check_box(pixels, (w, h))
        resolved_boxes.append({**box, "xyxy": (x1, y1, x2, y2)})
        labels.append((code, (x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h))
    if seen != expected or (frame_id is None and species_seen != annotations.keys()):
        raise ValueError("Reviewed boxes do not cover the archived targets")
    return labels, {**frame, "boxes": resolved_boxes}


def video_frames(source, frames, work):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source),
    ], check=True, capture_output=True, text=True)
    if probe.stderr:
        raise ValueError(f"Video frame probe reported a decoding error: {source.name}")
    times = json.loads(probe.stdout)["frames"]
    start = Decimal(times[0]["best_effort_timestamp_time"])
    offsets = {}
    for frame in frames:
        n = index(frame["frame_id"])
        actual = (Decimal(times[n]["best_effort_timestamp_time"]) - start) * 1000
        if abs(actual - index(frame["ts_ms"])) > 1:
            raise ValueError(f"Decoded video timestamp disagrees at frame {n}")
        offsets[n] = str(actual)
    expr_path = work / "frames.filter"
    expr_path.write_text("select=" + "+".join(f"eq(n\\,{n})" for n in sorted(offsets)), encoding="ascii")
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(source), "-map", "0:v:0",
        "-filter_script:v", str(expr_path), "-vsync", "0", "-q:v", "2", "-start_number", "0",
        str(work / "frame-%08d.jpg"),
    ], check=True)
    images = sorted(work.glob("frame-*.jpg"))
    if len(images) != len(offsets):
        raise ValueError("Decoded frame count does not match the reviewed frame list")
    return dict(zip(sorted(offsets), images)), offsets


def training_photo(source, path, size):
    with source.open("rb") as stream, Image.open(stream) as image:
        orientation = image.getexif().get(274, 1)
        if orientation == 1 and image.format == "JPEG" and image.mode == "RGB":
            if tuple(size) != image.size:
                raise ValueError(f"Reviewed dimensions disagree with the decoded photo: {source.name}")
            image.load()
            stream.seek(-2, os.SEEK_END)
            if stream.read() == b"\xff\xd9":
                shutil.copyfile(source, path)
                return
        display = ImageOps.exif_transpose(image)
        if tuple(size) != display.size:
            raise ValueError(f"Reviewed dimensions disagree with the decoded photo: {source.name}")
        display.convert("RGB").save(path, "JPEG", quality=95)


def approved_archive(args, plans, stage, registry, class_ids, photo_overlaps):
    seen = set()
    for review in plans:
        key, relative, bundle_path, source, bundle, annotations, provenance = version(args.archive, review)
        split, kind = review["split"], review["type"]
        if split not in SPLITS or key in seen:
            raise ValueError(f"Repeated archive resource or invalid split: {key}")
        seen.add(key)
        group = evidence(review["group_id"])
        exposure = evidence(review["exposure_evidence"]) if split != "train" else None
        filename_clearance = None
        if kind == "PHOTO":
            overlap = photo_overlaps[key]
            if overlap["media_sha256"] != review["media_sha256"]:
                raise ValueError(f"Photo differs from the cross-library overlap inventory: {key}")
            if split != "train" and "train" in overlap["exact_legacy_splits"]:
                raise ValueError(f"Held-out photo overlaps with historical training: {key}")
            if split != "train" and "train" in overlap["same_name_legacy_splits"]:
                filename_clearance = evidence(review.get("filename_match_clearance"))
        before = source.stat()
        check_file(source, review["media_sha256"])
        frames = review["frames"]
        frame_ids = [row["frame_id"] for row in frames]
        if not frames or len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"Empty or repeated frame approval: {key}")
        if kind == "PHOTO" and (frame_ids != [None] or frames[0]["ts_ms"] is not None):
            raise ValueError(f"Photo approval must describe a single still: {key}")
        photo_geometry = None
        if kind == "PHOTO":
            with Image.open(source) as original:
                photo_geometry = (*original.size, original.getexif().get(274, 1))
        rows = [reviewed_boxes(frame, annotations, class_ids, registry, photo_geometry) for frame in frames]
        saved = stage / "metadata/source-bundles" / relative.relative_to("raw/annotations")
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle_path, saved)
        if digest(json.loads(saved.read_bytes())) != review["bundle_sha256"]:
            raise ValueError(f"Annotation snapshot changed while copying: {key}")
        with tempfile.TemporaryDirectory(dir=stage) as directory:
            work = Path(directory)
            if kind == "VIDEO":
                images, times = video_frames(source, frames, work)
            else:
                path = work / "photo.jpg"
                training_photo(source, path, frames[0]["display_size"])
                images, times = {None: path}, {None: None}
            for labels, frame in rows:
                n = frame["frame_id"]
                sample = key if n is None else f"{key}__f{n}"
                with Image.open(images[n]) as image:
                    if image.size != tuple(frame["display_size"]):
                        raise ValueError(f"Decoded frame size disagrees with approval: {sample}")
                yield sample, split, f"archive_{kind.lower()}", images[n], labels, {
                    "resource_key": key, "resource_id": review["resource_id"], "source_media": bundle["file"]["path"],
                    "original_sha256": review["media_sha256"], "group_id": group, "exposure_evidence": exposure,
                    "filename_match_clearance": filename_clearance,
                    "annotation_bundle": str(saved.relative_to(stage)), "annotation_hashes": provenance,
                    "archive_bundle_path": str(relative), "reviewed_frame": frame, "decoded_timestamp_ms": times[n],
                }
        after = source.stat()
        if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise ValueError(f"Archive media changed during export: {source.name}")


def legacy_samples(args, rows, reviewed_regression, excluded):
    for row in rows:
        decision, sha = row["decision"], row["sha256"]
        if decision not in ("training_candidate", "regression_candidate"):
            emit(excluded, {"origin": "legacy", "source_media": row["image"], "reason": decision})
            continue
        if decision == "regression_candidate" and sha not in reviewed_regression:
            emit(excluded, {"origin": "legacy", "source_media": row["image"], "reason": "regression_review_pending"})
            continue
        split, exposure = ("train", None) if decision == "training_candidate" else reviewed_regression.pop(sha)
        original = Path(row["image"])
        source = local_child(args.legacy_root, Path(row["historical_split"]) / "images" / original.name)
        yield f"legacy_{sha}", split, "legacy", source, row["rows"], {
            "original_sha256": sha, "source_media": str(original),
            "group_id": json.dumps(row["camera_day_from_filename"]) if row["camera_day_from_filename"] else sha,
            "source_label_sha256": row.get("source_label_sha256"), "background_source": row.get("background_source"),
            "exposure_evidence": exposure,
        }
    if reviewed_regression:
        raise ValueError(f"{len(reviewed_regression)} reviewed legacy test samples are missing from the candidates")


def register(groups, key, split):
    previous = groups.setdefault(key, split)
    if previous != split:
        raise ValueError(f"Source content or capture group crosses held-out splits: {key}")


def add_samples(samples, stage, manifest, excluded, counters, groups, images_seen, lists, class_count):
    for key, split, origin, path, labels, metadata in samples:
        register(groups, ("original", metadata["original_sha256"]), split)
        register(groups, ("event", metadata["group_id"]), split)
        normalized = set()
        for code, x, y, w, h in labels:
            if (
                type(code) is not int or not 0 <= code < class_count
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h))
                or not (w > 0 and h > 0)
            ):
                raise ValueError(f"Invalid normalized YOLO label on {key}")
            normalized.add((code, x, y, w, h))
        label_text = "".join(f"{code} {x:.10g} {y:.10g} {w:.10g} {h:.10g}\n"
                             for code, x, y, w, h in sorted(normalized))
        signature = digest(label_text)
        original = metadata["original_sha256"]
        frame = metadata.get("reviewed_frame", {}).get("frame_id")
        source_identity = ("source", original, frame)
        if source_identity in images_seen:
            if images_seen[source_identity] != signature:
                raise ValueError(f"Identical source image or video frame has conflicting labels: {key}")
            counters["duplicate"][origin] += 1
            emit(excluded, {"origin": origin, "key": key, "reason": "duplicate_source"})
            continue
        image_path = stage / "images" / split / (key + ".jpg")
        target = stage / "labels" / split / (key + ".txt")
        shutil.copyfile(path, image_path)
        image_sha = checksum(image_path)
        if origin == "legacy" and image_sha != metadata["original_sha256"]:
            raise ValueError(f"Historical photo no longer matches its scan: {path}")
        register(groups, ("image", image_sha), split)
        image_identity = ("image", image_sha)
        if image_identity in images_seen:
            if images_seen[image_identity] != signature:
                raise ValueError(f"Identical image has conflicting labels: {key}")
            images_seen[source_identity] = signature
            image_path.unlink()
            counters["duplicate"][origin] += 1
            emit(excluded, {"origin": origin, "key": key, "reason": "duplicate_image"})
            continue
        images_seen[image_identity] = images_seen[source_identity] = signature
        target.write_text(label_text, encoding="ascii")
        image_path.chmod(0o400)
        target.chmod(0o400)
        relative = "./" + str(image_path.relative_to(stage))
        lists[split].append(relative)
        if split == "test":
            cohort = "legacy" if origin == "legacy" else "archive"
            lists[f"test-{cohort}"].append(relative)
        counters[split][origin] += 1
        class_ids = sorted({row[0] for row in normalized})
        counters[f"{split}_class_boxes"].update(row[0] for row in normalized)
        counters[f"{split}_class_images"].update(class_ids)
        emit(manifest, {
            **metadata, "key": key, "split": split, "origin": origin, "image": relative,
            "image_sha256": image_sha, "label": str(target.relative_to(stage)),
            "label_sha256": checksum(target), "class_ids": class_ids,
        })


def freeze(args, stage):
    legacy_receipt = args.legacy_partitions / "receipt.json"
    inputs = {
        "archive_reviews": args.accepted, "legacy_partitions": legacy_receipt,
        "legacy_regression_reviews": args.legacy_test, "class_registry": args.classes,
    }
    input_hashes = {key: checksum(path) for key, path in inputs.items()}
    registry, class_ids, _ = load_registry(args.classes)
    old = json.loads(legacy_receipt.read_bytes())
    if old["status"] != "candidate_partitions":
        raise ValueError("Legacy partitions have not completed")
    old_rows = check_file(args.legacy_partitions / "legacy-partitions.jsonl.gz", old["files"]["legacy-partitions.jsonl.gz"])
    overlaps = check_file(args.legacy_partitions / "archive-photo-overlap.jsonl.gz", old["files"]["archive-photo-overlap.jsonl.gz"])
    old_approvals = {}
    for row in records(args.legacy_test, compressed=args.legacy_test.suffix == ".gz"):
        sha = row["sha256"]
        split = row.get("split", "test")
        if sha in old_approvals or split not in ("val", "test"):
            raise ValueError("Repeated legacy regression approval or invalid held-out split")
        old_approvals[sha] = (split, evidence(row["exposure_evidence"]))
    if not old_approvals:
        raise ValueError("The historic regression set needs at least one accepted sample")
    plans = records(args.accepted, compressed=args.accepted.suffix == ".gz")
    first = next(plans, None)
    if first is None:
        raise ValueError("No approved archive data")

    for split in SPLITS:
        (stage / "images" / split).mkdir(parents=True)
        (stage / "labels" / split).mkdir(parents=True)
    photo_overlaps = {row["resource_key"]: row for row in records(overlaps)}
    new_samples = approved_archive(args, chain((first,), plans), stage, registry, class_ids, photo_overlaps)
    counters, groups, images_seen, lists = defaultdict(Counter), {}, {}, defaultdict(list)
    with (
        gzip.open(stage / "manifest.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as manifest,
        gzip.open(stage / "excluded.jsonl.gz", "wt", encoding="utf-8", compresslevel=1) as excluded,
    ):
        old_samples = legacy_samples(args, records(old_rows), old_approvals, excluded)
        for samples in (old_samples, new_samples):
            add_samples(samples, stage, manifest, excluded, counters, groups, images_seen, lists, len(registry))
    if not counters["train"]["legacy"] or not lists["test-legacy"]:
        raise ValueError("The approved historic data needs training and regression samples")
    if not (counters["train"]["archive_photo"] + counters["train"]["archive_video"]) or not lists["test-archive"]:
        raise ValueError("The approved archive needs distinct training and independent test samples")
    if not lists["val"]:
        raise ValueError("The approved dataset needs a validation split")

    for split, paths in lists.items():
        (stage / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    shutil.copyfile(args.classes, stage / "classes.json")
    for key, path in inputs.items():
        check_file(path, input_hashes[key])
    check_file(stage / "classes.json", input_hashes["class_registry"])
    write_json(stage / "class-counts.json", {
        split: {"images": counters[f"{split}_class_images"], "boxes": counters[f"{split}_class_boxes"]}
        for split in SPLITS
    })
    write_json(stage / "data.yaml", {
        **{split: f"{split}.txt" for split in SPLITS}, "names": [row["name"] for row in registry],
    })
    write_json(stage / "receipt.json", {
        "purpose": "comparison", "status": "frozen",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": input_hashes,
        "split_sizes": {split: len(lists[split]) for split in SPLITS},
        "train_sources": {name: counters["train"][name] for name in ("legacy", "archive_photo", "archive_video")},
        "split_sources": {split: counters[split] for split in SPLITS},
        "duplicates_skipped": counters["duplicate"],
        "files": {name: checksum(stage / name) for name in (
            "data.yaml", "classes.json", "manifest.jsonl.gz", "excluded.jsonl.gz", "class-counts.json",
            "train.txt", "val.txt", "test.txt", "test-legacy.txt", "test-archive.txt",
        )},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("archive", "accepted", "legacy-root", "legacy-partitions", "legacy-test", "classes", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    args.archive, args.legacy_root, out = args.archive.resolve(), args.legacy_root.resolve(), args.out.resolve()
    if out == args.legacy_root or args.legacy_root in out.parents:
        parser.error("Output must be outside the historical dataset")
    if (out == args.archive or args.archive in out.parents) and args.archive / "exports/yolo" not in out.parents:
        parser.error("Training versions within the archive must be under exports/yolo")
    os.umask(0o077)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise ValueError(f"Output version already exists: {out}")
        with tempfile.TemporaryDirectory(prefix="." + out.name + "-", dir=out.parent) as temporary:
            stage = Path(temporary)
            freeze(args, stage)
            stage.rename(out)
    print(f"{checksum(out / 'receipt.json')}  {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
