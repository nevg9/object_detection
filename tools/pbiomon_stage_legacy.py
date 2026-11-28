#!/usr/bin/env python3
"""Stream legacy photos selected by a partition receipt into a read-only media cache."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile

from pbiomon_legacy_partitions import records
from pbiomon_train import check_file, local_child, write_json


LAYOUT = "pbiomon-legacy-jpeg-by-sha256-v1"


def media_key(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Invalid media SHA256")
    return value


def image_path(cache, sha):
    media_key(sha)
    path = cache / "objects" / sha[:2] / (sha + ".jpg")
    if path.resolve() != path:
        raise ValueError(f"Cached images must use their managed paths: {path}")
    return path


def ready_image(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o400:
        raise ValueError(f"Cached image is not an owner-only read-only file: {path}")
    return True


def check_cache(cache):
    if json.loads((cache / "layout.json").read_bytes()) != {"layout": LAYOUT}:
        raise ValueError(f"Unexpected media-cache layout: {cache}")
    return cache


def candidates(folder, wanted=None):
    receipt = json.loads((folder / "receipt.json").read_bytes())
    if receipt["status"] != "candidate_partitions":
        raise ValueError("Legacy partitions have not completed")
    name = "legacy-partitions.jsonl.gz"
    manifest = check_file(folder / name, receipt["files"][name])
    selected = None
    if wanted is not None:
        selected = {media_key(row) for row in wanted.read_text(encoding="ascii").splitlines() if row}
    entries = {}
    for row in records(manifest):
        if row["decision"] not in ("training_candidate", "regression_candidate"):
            continue
        sha = media_key(row["sha256"])
        if selected is not None and sha not in selected:
            continue
        split = row["historical_split"]
        if split not in ("train", "test") or sha in entries:
            raise ValueError("Unexpected legacy split or duplicate content in the partition receipt")
        entries[sha] = Path(split) / "images" / Path(row["image"]).name
    if selected is not None and selected != entries.keys():
        raise ValueError("Requested image is outside the verified legacy partitions")
    return entries


def pack(entries, legacy_root, output):
    with tarfile.open(mode="w|", fileobj=output) as stream:
        for sha, relative in entries.items():
            source = local_child(legacy_root, relative)
            with source.open("rb") as file:
                info = tarfile.TarInfo(sha + ".jpg")
                info.size = os.fstat(file.fileno()).st_size
                info.mode = 0o400
                stream.addfile(info, file)


def receive_file(cache, stream, member, sha):
    target = image_path(cache, sha)
    if ready_image(target):
        check_file(target, sha)
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    with stream.extractfile(member) as received:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".photo-", delete=False) as temp:
            pathname = Path(temp.name)
            result = hashlib.sha256()
            try:
                while block := received.read(1024**2):
                    result.update(block)
                    temp.write(block)
                if result.hexdigest() != sha:
                    raise ValueError(f"Transmitted image differs from its verified digest: {sha}")
                temp.flush()
                os.fsync(temp.fileno())
                os.fchmod(temp.fileno(), 0o400)
                pathname.replace(target)
            finally:
                pathname.unlink(missing_ok=True)
    return "stored"


def receive(entries, cache, input_stream):
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".receive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (cache / "layout.json").exists():
            if any(path.name != ".receive.lock" for path in cache.iterdir()):
                raise ValueError("Use a dedicated directory for the legacy media cache")
            write_json(cache / "layout.json", {"layout": LAYOUT})
        check_cache(cache)
        seen = set()
        counts = Counter()
        with tarfile.open(mode="r|", fileobj=input_stream) as stream:
            for member in stream:
                if not member.isreg() or not member.name.endswith(".jpg"):
                    raise ValueError("The legacy media stream accepts regular JPEG files only")
                sha = media_key(member.name[:-4])
                if sha not in entries or sha in seen:
                    raise ValueError(f"Unexpected or repeated photo in the media stream: {sha}")
                counts[receive_file(cache, stream, member, sha)] += 1
                seen.add(sha)
        if seen != entries.keys():
            raise ValueError(f"The media stream ended with {len(entries) - len(seen)} selected photos missing")
        return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    send = modes.add_parser("pack")
    send.add_argument("--legacy-root", type=Path, required=True)
    arrive = modes.add_parser("receive")
    missing = modes.add_parser("missing")
    for mode in (send, arrive, missing):
        mode.add_argument("--legacy-partitions", type=Path, required=True)
        mode.add_argument("--wanted", type=Path, help="Optional text list of SHA256 values to transfer")
    for mode in (arrive, missing):
        mode.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    entries = candidates(args.legacy_partitions, args.wanted)
    if args.mode == "pack":
        pack(entries, args.legacy_root.resolve(), sys.stdout.buffer)
    elif args.mode == "receive":
        print(json.dumps(receive(entries, args.cache.resolve(), sys.stdin.buffer)))
    else:
        cache = args.cache.resolve()
        if cache.exists():
            check_cache(cache)
        for sha in entries:
            if not ready_image(image_path(cache, sha)):
                print(sha)


if __name__ == "__main__":
    main()
