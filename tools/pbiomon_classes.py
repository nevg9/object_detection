#!/usr/bin/env python3
"""Audit Pbiomon class coverage and append approved classes to the legacy registry."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sqlite3

from pbiomon_train import checksum, write_json


UNKNOWN = "无法区分"
GENDERS = {
    "雄": "雄性", "雄性": "雄性", "雌": "雌性", "雌性": "雌性",
    "难以鉴定": UNKNOWN, "难以辨认": UNKNOWN, UNKNOWN: UNKNOWN,
}
AGES = {
    "成年": "成体", "成体": "成体", "亚成体": "亚成体", "幼体": "幼体",
    "难以鉴定": UNKNOWN, "难以辨认": UNKNOWN, UNKNOWN: UNKNOWN,
}


def class_key(row):
    return row["species"], row["gender"], row["age"]


def load_registry(path):
    entries = json.loads(path.read_bytes())
    if [row["id"] for row in entries] != list(range(len(entries))):
        raise ValueError("Registry IDs must be in their original contiguous order")
    by_key = {class_key(row): row["id"] for row in entries}
    by_name = {row["name"]: row["id"] for row in entries}
    if len(by_key) != len(entries) or len(by_name) != len(entries):
        raise ValueError("Registry contains duplicate classes")
    return entries, by_key, by_name


def expert_aliases(paths):
    aliases = defaultdict(set)
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row["性别"] == row["年龄"] == UNKNOWN and row["标签"] != row["物种"]:
                    aliases[row["标签"]].add(row["物种"])
    return aliases


def write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def audit(archive, registry, alias_paths, output):
    entries, old_keys, _ = load_registry(registry)
    legacy_species = {row["species"] for row in entries}
    aliases = expert_aliases(alias_paths)
    species_stats = defaultdict(Counter)
    candidate_stats = defaultdict(Counter)
    counts = Counter()
    query = """
        SELECT a.species_id, a.species_name, a.resource_type, a.box_count, a.annotation_json
          FROM annotations a JOIN resources r USING(resource_type, resource_id)
         WHERE r.status='done'
    """
    with sqlite3.connect(
        (archive / "state/state.sqlite").resolve().as_uri() + "?mode=ro", uri=True, timeout=10,
    ) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        run = db.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        for species_id, name, kind, boxes, annotation in db.execute(query):
            mapped = aliases.get(name, set())
            canonical = next(iter(mapped)) if name not in legacy_species and len(mapped) == 1 else name
            status = (
                "legacy_model_species" if canonical in legacy_species
                else "alias_conflict" if len(mapped) > 1 else "species_absent_from_legacy_model"
            )
            species = species_stats[species_id, name, canonical, status]
            species.update({"records": 1, kind: 1, "records_with_boxes": int(boxes > 0), "boxes": boxes})
            valid = Counter()
            for attribute in json.loads(annotation)["finalIdentification"].get("attributes") or []:
                gender, age = GENDERS.get(attribute.get("gender")), AGES.get(attribute.get("age"))
                if gender is None or age is None or status == "alias_conflict":
                    counts["traits_held_for_review"] += 1
                    continue
                valid[canonical, gender, age] += 1
            for key, occurrences in valid.items():
                candidate_stats[species_id, name, key].update({
                    "items": occurrences, "records": 1, "records_with_boxes": int(boxes > 0),
                })
    output.mkdir(mode=0o700, exist_ok=False)
    write_csv(output / "species-coverage.csv", (
        "species_id", "api_species_name", "canonical_name_candidate", "coverage",
        "records", "photos", "videos", "records_with_boxes", "boxes",
    ), (
        (*key, c["records"], c["PHOTO"], c["VIDEO"], c["records_with_boxes"], c["boxes"])
        for key, c in sorted(species_stats.items())
    ))
    write_csv(output / "class-candidates.csv", (
        "species_id", "api_species_name", "canonical_name_candidate", "gender", "age",
        "legacy_id", "items", "records", "records_with_boxes",
    ), (
        (species_id, name, *key, old_keys.get(key), c["items"], c["records"], c["records_with_boxes"])
        for (species_id, name, key), c in sorted(candidate_stats.items())
    ))
    novel = {
        key[2] for key in species_stats
        if key[3] == "species_absent_from_legacy_model"
    }
    novel_boxed = {
        key[2] for key, c in species_stats.items()
        if c["records_with_boxes"] and key[3] == "species_absent_from_legacy_model"
    }
    write_json(output / "receipt.json", {
        "status": "class_coverage_audit", "source_run": run[0],
        "registry_sha256": checksum(registry),
        "expert_alias_sha256": {p.name: checksum(p) for p in alias_paths},
        "legacy_class_count": len(entries), "legacy_species_names": len(legacy_species),
        "archive_species_ids": len({key[0] for key in species_stats}),
        "archive_standard_names": len({key[1] for key in species_stats}),
        "legacy_species_in_archive": len({key[2] for key in species_stats if key[3] == "legacy_model_species"}),
        "names_absent_from_legacy_model": len(novel),
        "names_absent_with_box_records": len(novel_boxed),
        "novel_joint_keys_in_box_records": len({
            key for (_, _, key), c in candidate_stats.items() if c["records_with_boxes"] and key not in old_keys
        }),
        **counts,
        "files": {name: checksum(output / name) for name in ("species-coverage.csv", "class-candidates.csv")},
    })


def extend(registry, approved, output):
    entries, old_keys, old_names = load_registry(registry)
    additions = defaultdict(set)
    for row in json.loads(approved.read_bytes()):
        species, gender, age = class_key(row)
        species_id = row["species_id"]
        if (
            not isinstance(species, str) or not species or species != species.strip()
            or not isinstance(species_id, str) or not species_id
            or gender not in GENDERS.values() or age not in AGES.values()
        ):
            raise ValueError(f"Approved class has incomplete or non-canonical traits: {row}")
        if (species, gender, age) not in old_keys:
            additions[species, gender, age].add(species_id)
    for (species, gender, age), species_ids in sorted(additions.items()):
        name = "".join(value for value in (species, gender, age) if value != UNKNOWN)
        if name in old_names:
            raise ValueError(f"Display name conflicts with another class: {name}")
        old_names[name] = len(entries)
        entries.append({
            "id": len(entries), "name": name, "species": species,
            "gender": gender, "age": age, "species_ids": sorted(species_ids),
        })
    with output.open("x", encoding="utf-8") as stream:
        json.dump(entries, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    scan = subcommands.add_parser("audit")
    scan.add_argument("--archive", type=Path, required=True)
    scan.add_argument("--expert-aliases", type=Path, nargs="*", default=[])
    append = subcommands.add_parser("extend")
    append.add_argument("--approved", type=Path, required=True)
    for command in (scan, append):
        command.add_argument("--legacy-registry", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "audit":
        archive = args.archive.resolve()
        output = args.out.resolve()
        if output == archive or archive in output.parents:
            parser.error("--out must be outside the archive")
        audit(archive, args.legacy_registry, args.expert_aliases, output)
    else:
        extend(args.legacy_registry, args.approved, args.out)


if __name__ == "__main__":
    main()
