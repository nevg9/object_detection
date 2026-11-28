#!/usr/bin/env python3
"""Transfer the verified legacy partition to the training cache over a private TLS link."""

import argparse
from collections import Counter, deque
from datetime import datetime
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import selectors
import socket
import ssl
import tarfile
import time

from pbiomon_stage_legacy import candidates, image_path, media_key, pack, ready_image, receive
from pbiomon_train import checksum, write_json


MAX_BATCH = 1024
REQUEST_LIMIT = 67 * MAX_BATCH + 512


def load_job(path):
    cfg = json.loads(path.read_bytes())
    for route in cfg["routes"]:
        source = ipaddress.ip_address(route["source"])
        peer = ipaddress.ip_address(route["target"])
        if source.version != peer.version or not 1 <= route["port"] <= 65535:
            raise ValueError(f"Invalid addresses or port on {route['interface']}")
        if not 1 <= route["batch"] <= MAX_BATCH:
            raise ValueError(f"Invalid transfer batch size on {route['interface']}")
    return cfg


def bind_address(route, side, port):
    ip = route[side]
    return (ip, port, 0, socket.if_nametoindex(route["interface"])) if ":" in ip else (ip, port)


def snapshot(cfg):
    directory = Path(cfg["partitions"]).resolve()
    if checksum(directory / "receipt.json") != cfg["partition_receipt_sha256"]:
        raise ValueError("Partition receipt does not match the transfer plan")
    return candidates(directory)


def send_batch(connection, ctx, entries, source, receipt_sha):
    with ctx.wrap_socket(connection, server_side=True) as tls:
        with tls.makefile("rb") as reader:
            header = reader.readline(REQUEST_LIMIT + 1)
        if not header or len(header) > REQUEST_LIMIT:
            raise ValueError("Invalid transfer request length")
        query = json.loads(header)
        if not isinstance(query, dict) or query.get("receipt") != receipt_sha:
            raise ValueError("Transfer request does not match the verified legacy partition")
        wanted = query.get("wanted")
        if (not isinstance(wanted, list) or not 1 <= len(wanted) <= MAX_BATCH
                or any(media_key(k) not in entries for k in wanted) or len(set(wanted)) != len(wanted)):
            raise ValueError("Transfer request does not match the verified legacy partition")
        selection = {k: entries[k] for k in wanted}
        with tls.makefile("wb", buffering=1024**2) as stream:
            pack(selection, source, stream)


def serve(cfg):
    entries, root = snapshot(cfg), Path(cfg["legacy_root"]).resolve()
    state = Path(cfg["source_status"])
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cfg["server_cert"], cfg["server_key"])
    bindings = []
    with selectors.DefaultSelector() as poller:
        for route in cfg["routes"]:
            addr = bind_address(route, "source", route["port"])
            server = socket.socket(socket.AF_INET6 if len(addr) == 4 else socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(addr)
                server.listen(2)
            except OSError as error:
                server.close()
                print(f"{route['interface']}: {error}", flush=True)
                continue
            poller.register(server, selectors.EVENT_READ, route)
            bindings.append(route["interface"])
        if not bindings:
            raise RuntimeError("No source interface is listening")
        write_json(state, {"status": "listening", "pid": os.getpid(), "routes": bindings, "images": len(entries)})
        while True:
            for key, _ in poller.select():
                connection, addr = key.fileobj.accept()
                with connection:
                    route = key.data
                    if ipaddress.ip_address(addr[0]) != ipaddress.ip_address(route["target"]):
                        continue
                    connection.settimeout(40)
                    try:
                        send_batch(connection, ctx, entries, root, cfg["partition_receipt_sha256"])
                    except (OSError, ValueError, KeyError) as error:
                        print(f"{route['interface']}: {error}", flush=True)


def fetch(route, ctx, wanted, entries, cache, receipt_sha):
    destination = bind_address(route, "source", route["port"])
    with socket.socket(socket.AF_INET6 if len(destination) == 4 else socket.AF_INET, socket.SOCK_STREAM) as raw:
        raw.settimeout(3)
        raw.bind(bind_address(route, "target", 0))
        raw.connect(destination)
        raw.settimeout(40)
        with ctx.wrap_socket(raw, server_hostname="pbiomon-stage") as tls:
            data = json.dumps({"receipt": receipt_sha, "wanted": wanted}, separators=(",", ":"))
            tls.sendall(data.encode("ascii") + b"\n")
            with tls.makefile("rb", buffering=1024**2) as stream:
                receive({k: entries[k] for k in wanted}, cache, stream)


def transfer(cfg):
    status = Path(cfg["target_status"])
    cache = Path(cfg["cache"]).resolve()
    entries = snapshot(cfg)
    receipt_sha = cfg["partition_receipt_sha256"]
    ctx = ssl.create_default_context(cafile=cfg["server_cert"])
    with (status.parent / ".transfer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = deque(k for k in entries if not ready_image(image_path(cache, k)))
        original_missing = len(pending)
        added_bytes = 0
        batches = Counter()
        failures = Counter()
        route_bytes = Counter()
        route_seconds = Counter()
        started = time.monotonic()

        def report(phase, **extra):
            write_json(status, {
                "status": phase, "pid": os.getpid(),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "partition_receipt_sha256": receipt_sha, "total_images": len(entries),
                "present_at_start": len(entries) - original_missing,
                "present_images": len(entries) - len(pending), "remaining_images": len(pending),
                "new_bytes": added_bytes, "batches": dict(batches), "route_failures": dict(failures),
                "route_bytes": dict(route_bytes), "route_seconds": dict(route_seconds),
                "elapsed_seconds": round(time.monotonic() - started, 3), **extra,
            })

        report("running")
        stalled = 0
        try:
            while pending:
                progressed = False
                for route in cfg["routes"]:
                    route_name = route["interface"]
                    wanted = [pending.popleft() for _ in range(min(route["batch"], len(pending)))]
                    tick = time.monotonic()
                    error = None
                    try:
                        fetch(route, ctx, wanted, entries, cache, receipt_sha)
                    except (OSError, tarfile.ReadError, EOFError) as problem:
                        error = str(problem)
                    except ValueError as problem:
                        if not str(problem).startswith("The media stream ended with "):
                            pending.extendleft(reversed(wanted))
                            raise
                        error = str(problem)
                    fresh = 0
                    written = 0
                    for key in reversed(wanted):
                        path = image_path(cache, key)
                        if ready_image(path):
                            fresh += 1
                            written += path.stat().st_size
                        else:
                            pending.appendleft(key)
                    elapsed = time.monotonic() - tick
                    added_bytes += written
                    route_bytes[route_name] += written
                    route_seconds[route_name] += elapsed
                    if error is not None:
                        failures[route_name] += 1
                    else:
                        batches[route_name] += 1
                    report("running", interface=route_name, last_error=error,
                           last_batch={"images": fresh, "bytes": written, "seconds": round(elapsed, 3)})
                    if fresh:
                        progressed = True
                        break
                if progressed:
                    stalled = 0
                else:
                    stalled += 1
                    if stalled >= 5:
                        raise RuntimeError("No photos arrived after five attempts on the available interfaces")
                    time.sleep(5)
        except Exception as error:
            report("interrupted", last_error=str(error))
            raise
        report("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "transfer"))
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    cfg = load_job(args.job)
    (serve if args.mode == "serve" else transfer)(cfg)


if __name__ == "__main__":
    main()
