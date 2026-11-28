#!/usr/bin/env python3
"""Index source identities and unresolved annotations for a versioned Pbiomon export."""

import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def emit(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def photo_identities(db, stream, counts):
    for key, sha, size, path, saved, metadata, geometry in db.execute("""
        SELECT resource_key, sha256, byte_size, file_path, saved_revision, metadata_json, media_json
          FROM resources
         WHERE resource_type='PHOTO' AND status='done'
         ORDER BY resource_id
    """):
        data = json.loads(metadata)
        filename = data.get("fileName")
        monitor_unit = data.get("monitorUnitId")
        captured = data.get("capturedAt")
        counts["photos"] += 1
        if not (monitor_unit and captured):
            counts["photos_without_monitor_or_timestamp"] += 1
        emit(stream, {
            "resource_key": key, "media_sha256": sha, "bytes": size, "media_path": path,
            "saved_revision": saved, "filename_stem": Path(filename).stem.casefold() if filename else None,
            "monitor_unit": monitor_unit, "captured_at": captured,
            "orientation": json.loads(geometry or "{}").get("exif_orientation"),
        })


def annotation_reviews(db, stream, counts, value_counts):
    query = """
        SELECT a.resource_type, r.resource_key, r.sha256, r.annotation_path,
               r.saved_revision, r.width, r.height, a.species_id, a.species_name,
               a.annotation_hash, a.source_page, a.annotation_json
          FROM annotations a JOIN resources r USING(resource_type, resource_id)
         WHERE r.status='done'
         ORDER BY a.resource_type, a.resource_id, a.species_id
    """
    for row in db.execute(query):
        kind, key, sha, bundle, revision, width, height, species, name, version, page, payload = row
        final = json.loads(payload)["finalIdentification"]
        attribute_status = "present"
        attributes = final.get("attributes")
        if "attributes" not in final:
            attribute_status = "missing"
        elif attributes is None:
            attribute_status = "null"
        elif not attributes:
            attribute_status = "empty"
        attributes = attributes or []
        boxes = final["boxes"]
        reasons = []
        source_units = Counter()
        for box in boxes:
            is_ai = box.get("isAI")
            source = "ai" if is_ai is True else "manual" if is_ai is False else "unspecified"
            unit = box.get("coordinateUnit") or "UNSPECIFIED"
            source_units[f"{source}:{unit}"] += 1
        counts[f"{kind}:records"] += 1
        counts[f"{kind}:boxes"] += len(boxes)
        for group, count in source_units.items():
            counts[f"{kind}:box:{group}"] += count
        if not boxes:
            reasons.append("missing_boxes")
        else:
            if any(group.endswith((":UNKNOWN", ":UNSPECIFIED")) for group in source_units):
                reasons.append("coordinate_contract_required")
            if len(boxes) == 1 and len(attributes) == 1:
                reasons.append("single_box_and_item_review")
            else:
                reasons.append("box_to_attribute_link_review")
            if kind == "VIDEO":
                reasons.append("decoded_frame_and_geometry_review")
        if attribute_status != "present":
            reasons.append("identification_items_" + attribute_status)
        elif any(not item.get("age") or not item.get("gender") for item in attributes):
            reasons.append("incomplete_age_or_gender")
        for item in attributes:
            for field in ("age", "gender"):
                value_counts[field][json.dumps(item.get(field), ensure_ascii=False)] += 1
        for reason in reasons:
            counts[f"{kind}:review:{reason}"] += 1
        emit(stream, {
            "type": kind, "resource_key": key, "species_id": species, "species_name": name,
            "media_sha256": sha, "media_dimensions": [width, height],
            "annotation_sha256": version, "saved_revision": revision, "bundle_path": bundle,
            "source_page": page, "box_count": len(boxes), "attribute_count": len(attributes),
            "attribute_status": attribute_status,
            "box_source_units": source_units, "reasons": reasons,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    out = args.out.resolve()
    if out == archive or archive in out.parents:
        parser.error("--out must be outside the source archive")
    os.umask(0o077)
    out.mkdir(mode=0o700, exist_ok=False)
    counts = Counter()
    values = {"age": Counter(), "gender": Counter()}
    files = {}
    with sqlite3.connect(
        (archive / "state/state.sqlite").as_uri() + "?mode=ro", uri=True, timeout=10,
    ) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        run = db.execute(
            "SELECT run_id, status, end_date, finished_at FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        for name, export in (
            ("photo-identities.jsonl.gz", lambda stream: photo_identities(db, stream, counts)),
            ("annotation-reviews.jsonl.gz", lambda stream: annotation_reviews(db, stream, counts, values)),
        ):
            part = out / (name + ".partial")
            with gzip.open(part, "wt", compresslevel=1, encoding="utf-8") as stream:
                export(stream)
            files[name] = digest(part)
            part.rename(out / name)
    receipt = {
        "status": "indexed", "archive": str(archive), "source_run": run,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "box_and_attribute_identity": "zero-based indices in finalIdentification arrays, keyed by annotation_sha256",
        "counts": dict(counts), "attribute_values_as_json": values, "files": files,
    }
    temporary = out / "receipt.json.partial"
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.rename(out / "receipt.json")


if __name__ == "__main__":
    main()
