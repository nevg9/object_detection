#!/usr/bin/env python3
"""Run a pinned Pbiomon comparison and resume from an intact epoch checkpoint."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, expected):
    if not path.is_file() or checksum(path) != expected:
        raise ValueError(f"Digest mismatch or missing file: {path}")
    return path


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def local_child(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Path outside dataset: {relative}")
    return path


def verify_dataset(spec, purpose):
    receipt = check_file(Path(spec["receipt"]).resolve(), spec["sha256"])
    seal = json.loads(receipt.read_bytes())
    if seal["purpose"] != purpose or seal["status"] != "frozen":
        raise ValueError("Dataset purpose or freeze status does not match the experiment")
    if purpose == "comparison":
        sources = seal["train_sources"]
        if not sources["legacy"] or not (sources["archive_photo"] + sources["archive_video"]):
            raise ValueError("A formal comparison needs accepted legacy and archive samples")
        if not seal["split_sizes"]["test"]:
            raise ValueError("Independent test set is empty")
        for artifact in ("manifest.jsonl.gz", "classes.json", "train.txt", "val.txt", "test.txt"):
            if artifact not in seal["files"]:
                raise ValueError(f"Dataset receipt is missing {artifact}")
    root = receipt.parent
    files = {
        name: check_file(local_child(root, name), digest)
        for name, digest in seal["files"].items()
    }
    return files["data.yaml"], seal["split_sizes"]


def continuation(run, experiment_digest, epoch_limit):
    for folder in sorted((run / "continuations").glob("epoch-*"), reverse=True):
        receipt = json.loads((folder / "receipt.json").read_bytes())
        if receipt["experiment_sha256"] != experiment_digest:
            raise ValueError(f"Checkpoint belongs to a different experiment: {folder}")
        if not 0 < receipt["completed_epochs"] < epoch_limit:
            continue
        return check_file(folder / "last.pt", receipt["checkpoint_sha256"]), receipt
    raise ValueError("No resumable epoch checkpoint")


def restore_outputs(run, checkpoint, previous):
    for name, relative in {"best.pt": "weights/best.pt", "results.csv": "results.csv"}.items():
        source = check_file(checkpoint.parent / name, previous["outputs"][name])
        target = run / relative
        temporary = target.with_name("." + target.name + ".restore")
        shutil.copyfile(source, temporary)
        temporary.replace(target)


def preserve_legacy_head(model, weights, old_count, new_count):
    import torch
    from ultralytics.nn.modules.head import Detect

    original, expanded = weights.model[-1], model.model[-1]
    if (
        type(original) is not Detect or type(expanded) is not Detect
        or original.nc != old_count or expanded.nc != new_count
        or original.end2end or expanded.end2end
        or len(original.cv3) != len(expanded.cv3)
    ):
        raise ValueError("Strategy A needs a compatible YOLO11m detection head")
    with torch.no_grad():
        for old_branch, new_branch in zip(original.cv3, expanded.cv3):
            old_out, new_out = old_branch[-1], new_branch[-1]
            if (
                not isinstance(old_out, torch.nn.Conv2d)
                or not isinstance(new_out, torch.nn.Conv2d)
                or old_out.weight.shape != new_out.weight[:old_count].shape
                or old_out.bias is None or new_out.bias is None
                or old_out.bias.shape != new_out.bias[:old_count].shape
            ):
                raise ValueError("Strategy A classification output cannot be expanded in place")
            new_out.weight[:old_count].copy_(old_out.weight)
            new_out.bias[:old_count].copy_(old_out.bias)
            if not (
                torch.equal(new_out.weight[:old_count], old_out.weight)
                and torch.equal(new_out.bias[:old_count], old_out.bias)
            ):
                raise ValueError("Strategy A legacy class weights failed transfer verification")
    return len(original.cv3)


def run_training(args, experiment, experiment_digest, run, data, counts, initial):
    purpose = experiment["purpose"]
    options = experiment["training"]
    gpus = options.get("gpus", 1)
    rank = int(os.environ.get("RANK", "-1"))
    leader = rank in (-1, 0)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(run / "config" / str(rank))
    os.environ["MPLCONFIGDIR"] = str(run / "matplotlib" / str(rank))

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO, settings
    from ultralytics.data import utils as dataset_utils
    from ultralytics.models.yolo.detect import DetectionTrainer

    requested = experiment["runtime"]
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    }
    if requested != runtime:
        raise ValueError(f"Runtime mismatch: expected {requested}, got {runtime}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != gpus:
        raise ValueError(f"Set CUDA_VISIBLE_DEVICES to exactly {gpus} available GPU(s)")
    device = ",".join(str(index) for index in range(gpus))
    settings.update({
        k: False for k in ("sync", "hub", "wandb", "mlflow", "neptune", "clearml",
                          "comet", "tensorboard") if k in settings
    })
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font.is_file():
        dataset_utils.check_font = lambda *unused, **kwargs: font
    definition = yaml.safe_load(data.read_text(encoding="utf-8"))
    if definition.get("download"):
        raise ValueError("Frozen data YAML must reference local data")
    names = definition["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    if sorted(names) != list(range(len(names))):
        raise ValueError("Dataset class IDs must be contiguous")
    if purpose == "comparison":
        registry = json.loads(data.with_name("classes.json").read_bytes())
        if [(row["id"], row["name"]) for row in registry] != sorted(names.items()):
            raise ValueError("Class registry and dataset YAML disagree")

    trainer_type = None
    if args.resume:
        checkpoint, previous = continuation(run, experiment_digest, options["epochs"])
        model = YOLO(str(checkpoint), task="detect")
        if (
            model.ckpt.get("optimizer") is None
            or not model.ckpt.get("scaler")
            or model.ckpt["epoch"] + 1 != previous["completed_epochs"]
            or Path(model.ckpt["train_args"]["data"]).resolve() != data
            or model.names != names
        ):
            raise ValueError("Resume checkpoint does not contain the expected data and training state")
    else:
        model = YOLO(str(initial), task="detect")
        if args.strategy == "A":
            old_names = model.names
            if any(names.get(index) != name for index, name in old_names.items()):
                raise ValueError("Strategy A requires all original class IDs and names")
            if len(names) > len(old_names):
                class ExpandedDetectionTrainer(DetectionTrainer):
                    def get_model(self, cfg=None, weights=None, verbose=True):
                        target = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                        levels = preserve_legacy_head(target, weights, len(old_names), len(names))
                        if leader:
                            write_json(run / "head-transfer.json", {
                                "legacy_classes": len(old_names), "total_classes": len(names),
                                "added_classes": len(names) - len(old_names), "verified_head_levels": levels,
                            })
                        return target

                trainer_type = ExpandedDetectionTrainer

    def check_loaded(trainer):
        actual = {"train": len(trainer.train_loader.dataset), "val": len(trainer.test_loader.dataset)}
        if actual != {key: counts[key] for key in actual}:
            raise ValueError(f"Ultralytics loaded unexpected split sizes: {actual}")
        if not trainer.amp:
            raise ValueError("FP16 AMP did not pass runtime checks")

    def save_continuation(trainer):
        completed = trainer.epoch + 1
        if completed >= options["epochs"]:
            return
        base = run / "continuations"
        base.mkdir(exist_ok=True)
        temporary = base / f".epoch-{completed:06d}-{os.getpid()}"
        temporary.mkdir()
        hashes = {}
        for source in (trainer.last, trainer.best, trainer.csv):
            copied = temporary / source.name
            shutil.copyfile(source, copied)
            with copied.open("rb") as stream:
                os.fsync(stream.fileno())
            hashes[copied.name] = checksum(copied)
        write_json(temporary / "receipt.json", {
            "completed_epochs": completed, "experiment_sha256": experiment_digest,
            "checkpoint_sha256": hashes.pop("last.pt"), "outputs": hashes,
        })
        temporary.rename(base / f"epoch-{completed:06d}")
        if args.pause_after == completed:
            trainer.stop = True

    def mark_resume(trainer):
        if not args.resume or not leader:
            return
        if trainer.start_epoch != previous["completed_epochs"] or not trainer.optimizer.state:
            raise ValueError("Optimizer or epoch was not restored")
        write_json(run / "resume-verified.json", {
            **previous, "strategy": args.strategy, "optimizer_states": len(trainer.optimizer.state),
            "scaler": trainer.scaler.state_dict(), "resumed_at": dt.datetime.now().astimezone().isoformat(),
        })

    model.add_callback("on_train_start", check_loaded)
    model.add_callback("on_train_start", mark_resume)
    model.add_callback("on_model_save", save_continuation)
    common = {
        "data": str(data), "epochs": options["epochs"], "batch": options["batch"],
        "imgsz": 640, "workers": options["workers"], "device": device, "amp": True,
        "optimizer": "AdamW", "lr0": options["lr0"], "nbs": options["nbs"],
        "seed": options["seed"], "patience": 0, "val": True, "plots": False,
        "cache": False, "compile": False, "pretrained": True,
        "project": str(run.parent), "name": run.name, "exist_ok": True,
    }
    if purpose == "launch-check":
        common.update(close_mosaic=0, mosaic=0.0, verbose=False)
    if args.resume:
        model.train(resume=str(checkpoint), data=str(data), device=device, save_dir=str(run))
    else:
        model.train(trainer=trainer_type, **common)
    if not leader:
        return
    finished = model.trainer.epoch + 1 >= options["epochs"]
    write_json(run / "training-status.json", {
        "status": "complete" if finished else "paused",
        "strategy": args.strategy, "purpose": purpose,
        "experiment_sha256": experiment_digest, "completed_epochs": model.trainer.epoch + 1,
        "last_sha256": checksum(model.trainer.last), "best_sha256": checksum(model.trainer.best),
        "updated_at": dt.datetime.now().astimezone().isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after", type=int, help="Pause a launch check after the specified epoch")
    parser.add_argument("--ddp-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    raw = args.experiment.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    purpose = plan["purpose"]
    if purpose not in ("launch-check", "comparison") or set(plan["weights"]) != {"A", "B", "C"}:
        parser.error("Invalid experiment purpose or strategy list")
    if args.pause_after is not None and (
        purpose != "launch-check" or args.resume or not 0 < args.pause_after < plan["training"]["epochs"]
    ):
        parser.error("--pause-after requires a new launch check and a non-final epoch")
    data, counts = verify_dataset(plan["dataset"], purpose)
    spec = plan["weights"][args.strategy]
    initial = check_file(Path(spec["path"]).resolve(), spec["sha256"])
    root = Path(plan["runs"]).resolve() / digest[:16]
    root.mkdir(parents=True, exist_ok=True)
    run = root / args.strategy
    gpus = plan["training"].get("gpus", 1)
    if type(gpus) is not int or gpus < 1 or plan["training"]["batch"] % gpus:
        parser.error("Training batch must be divisible by the GPU count")
    if args.ddp_worker:
        if (
            int(os.environ.get("WORLD_SIZE", "0")) != gpus
            or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != gpus
            or (run / "experiment.json").read_bytes() != raw
        ):
            raise ValueError("Distributed workers must use the prepared single-node experiment")
        run_training(args, plan, digest, run, data, counts, initial)
        return
    reached = run / "training-status.json"
    with (root / f"{args.strategy}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if (run / "experiment.json").read_bytes() != raw:
                raise ValueError("Existing run has a different experiment")
            if reached.is_file() and json.loads(reached.read_bytes())["status"] == "complete":
                raise ValueError("Run already completed")
            checkpoint, previous = continuation(run, digest, plan["training"]["epochs"])
            restore_outputs(run, checkpoint, previous)
        else:
            run.mkdir()
            (run / "experiment.json").write_bytes(raw)
        if gpus == 1:
            run_training(args, plan, digest, run, data, counts, initial)
        else:
            subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc-per-node", str(gpus), str(Path(__file__).resolve()), *sys.argv[1:], "--ddp-worker",
            ], check=True)


if __name__ == "__main__":
    main()
