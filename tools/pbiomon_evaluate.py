#!/usr/bin/env python3
"""Evaluate the baseline and completed strategies on the same sealed Pbiomon split."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import gzip
import json
import os
from pathlib import Path
import tempfile

from pbiomon_legacy_partitions import records
from pbiomon_train import check_file, checksum, local_child, verify_dataset, write_json


AP_CONFIDENCE = 0.001
STRATEGIES = ("baseline", "A", "B", "C")


def read_recipe(path, experiment_digest):
    raw = json.loads(path.read_bytes())
    if raw["experiment_sha256"] != experiment_digest:
        raise ValueError("Evaluation recipe belongs to a different experiment")
    opts = {"batch": 8, "half": True, "nms_iou": 0.7, "max_det": 300, "workers": 2}
    opts.update({k: raw[k] for k in (*opts, "confidence") if k in raw})
    if (type(opts.get("confidence")) not in (int, float) or not AP_CONFIDENCE < opts["confidence"] <= 1
            or type(opts["nms_iou"]) not in (int, float) or not 0 < opts["nms_iou"] <= 1
            or type(opts["half"]) is not bool
            or any(type(opts[k]) is not int or opts[k] < 1 for k in ("max_det", "batch"))
            or type(opts["workers"]) is not int or opts["workers"] < 0):
        raise ValueError("Invalid confidence, preprocessing or detection limits in evaluation recipe")
    return opts


def checkpoint(plan, experiment_digest, strategy):
    if strategy == "baseline":
        spec = plan["weights"]["A"]
        return check_file(Path(spec["path"]).resolve(), spec["sha256"]), spec["sha256"]
    run = Path(plan["runs"]).resolve() / experiment_digest[:16] / strategy
    state = json.loads((run / "training-status.json").read_bytes())
    if (state["status"] != "complete" or state["purpose"] != "comparison" or state["strategy"] != strategy
            or state["completed_epochs"] != plan["training"]["epochs"]
            or state["experiment_sha256"] != experiment_digest
            or checksum(run / "experiment.json") != experiment_digest):
        raise ValueError(f"Strategy {strategy} does not have a completed matching run")
    return check_file(run / "weights/best.pt", state["best_sha256"]), state["best_sha256"]


def identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def prepare_view(root, split, count, names, view):
    paths = [local_child(root, line) for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]
    expected = set(paths)
    if len(paths) != count or len(expected) != count or not count:
        raise ValueError(f"Frozen {split} inventory has an unexpected size or repeats an image")
    images, labels = view / "images", view / "labels"
    images.mkdir()
    labels.mkdir()
    samples, pinned = {}, {}
    for row in records(root / "manifest.jsonl.gz"):
        if row["split"] != split:
            continue
        image = local_child(root, row["image"])
        if image not in expected or image.name in samples:
            raise ValueError(f"Manifest and {split} list disagree on {image.name}")
        label = local_child(root, row["label"])
        for file, sha in ((image, row["image_sha256"]), (label, row["label_sha256"])):
            before = identity(file)
            check_file(file, sha)
            if identity(file) != before:
                raise ValueError(f"Frozen file changed while verifying: {file}")
            pinned[file] = before
        (images / image.name).symlink_to(image)
        (labels / f"{image.stem}.txt").symlink_to(label)
        samples[image.name] = {
            "key": row["key"], "origin": row["origin"], "group_id": row["group_id"],
            "truth": Counter(int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()),
        }
        expected.remove(image)
    if expected or len(samples) != count:
        raise ValueError(f"Manifest is missing frozen {split} images")
    dataset = view / "evaluation.yaml"
    write_json(dataset, {
        "path": str(view), "train": str(root / "train.txt"), "val": str(root / "val.txt"),
        "test": str(root / "test.txt"), split: "images", "names": names,
    })
    return dataset, samples, pinned


def class_counts(ids):
    return Counter(int(value) for value in ids)


def summarize(rows, names, train_images):
    import numpy as np
    from ultralytics.utils.metrics import ap_per_class

    gt, predictions, true_positives = Counter(), Counter(), Counter()
    backgrounds = false_alarm_images = 0
    for row in rows:
        gt.update(row["truth"])
        predictions.update(row["predicted"])
        true_positives.update(row["true_positives"])
        if not row["truth"]:
            backgrounds += 1
            false_alarm_images += bool(row["predicted"])
    ap_by_class = {}
    if gt:
        tp, conf, pred, target = (np.concatenate([r["ap"][key] for r in rows], axis=0)
                                  for key in ("tp", "conf", "pred_cls", "target_cls"))
        results = ap_per_class(tp, conf, pred, target, plot=False)
        ap_by_class = {int(i): (float(v[0]), float(v.mean())) for i, v in zip(results[6], results[5])}
        if ap_by_class.keys() != gt.keys():
            raise ValueError("AP and fixed-confidence metrics used different ground-truth classes")
    by_class = []
    for i in sorted(gt.keys() | predictions.keys()):
        tp, count, det = true_positives[i], gt[i], predictions[i]
        if tp > min(count, det):
            raise ValueError("Fixed-confidence matching returned inconsistent counts")
        a50, amap = ap_by_class.get(i, (None, None))
        by_class.append({
            "class_id": i, "name": names[i], "train_images": train_images.get(str(i)),
            "targets": count, "detections": det, "tp": tp, "fp": det - tp, "fn": count - tp,
            "recall": tp / count if count else None,
            "f1": 2 * tp / (count + det) if count else None,
            "ap50": a50, "ap50_95": amap,
        })
    supported = [row for row in by_class if row["targets"]]
    result = {
        "images": len(rows), "targets": sum(gt.values()), "classes_with_truth": len(supported),
        "background_images": backgrounds, "background_false_alarm_images": false_alarm_images,
        "background_false_alarm_rate": false_alarm_images / backgrounds if backgrounds else None,
        "per_class": by_class,
    }
    for key in ("ap50", "ap50_95", "f1", "recall"):
        result[key] = sum(row[key] for row in supported) / len(supported) if supported else None
    return result


def score(model, dataset, samples, split, opts, names, train_images, out):
    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect import DetectionValidator

    image_results = {}

    class SealedValidator(DetectionValidator):
        def get_dataloader(self, dataset_path, batch_size):
            # The standalone validator normally drops classes absent from an old model.
            dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
            return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False,
                                    rank=-1, pin_memory=False, device=self.device)

        def update_metrics(self, preds, batch):
            first = len(self.metrics.stats["conf"])
            super().update_metrics(preds, batch)
            for i, pred in enumerate(preds):
                filename = Path(batch["im_file"][i]).name
                if filename not in samples or filename in image_results:
                    raise ValueError(f"Evaluation loaded an unexpected or repeated image: {filename}")
                sample = samples[filename]
                stats = {key: self.metrics.stats[key][first + i] for key in ("tp", "conf", "pred_cls", "target_cls")}
                if class_counts(stats["target_cls"]) != sample["truth"]:
                    raise ValueError(f"Loader targets differ from the frozen labels: {filename}")
                # Rematch after filtering so low-confidence boxes cannot claim a fixed-threshold match.
                keep = pred["conf"].float() >= opts["confidence"]
                fixed = {key: pred[key][keep] for key in ("bboxes", "cls")}
                correct = self._process_batch(fixed, self._prepare_batch(i, batch))["tp"][:, 0]
                ids = fixed["cls"].cpu().numpy()
                image_results[filename] = {
                    **sample, "predicted": class_counts(ids), "true_positives": class_counts(ids[correct]), "ap": stats,
                }

    metrics = model.val(
        validator=SealedValidator, data=str(dataset), split=split, device="0", imgsz=640, rect=False,
        conf=AP_CONFIDENCE, iou=opts["nms_iou"], batch=opts["batch"], half=opts["half"], max_det=opts["max_det"],
        workers=opts["workers"], plots=False, verbose=False, augment=False, save_json=False, save_txt=False,
        single_cls=False, classes=None, agnostic_nms=False, project=str(out), name="validator",
    )
    if image_results.keys() != samples.keys():
        raise ValueError("Evaluation did not cover the entire frozen split")
    rows = list(image_results.values())
    with gzip.open(out / "image-counts.jsonl.gz", "wt", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps({k: v for k, v in item.items() if k != "ap"}, ensure_ascii=False) + "\n")
    groups = {"all": rows, "legacy": [], "archive": [], "archive_photo": [], "archive_video": []}
    for row in rows:
        groups[row["origin"]].append(row)
        if row["origin"].startswith("archive_"):
            groups["archive"].append(row)
    return {"cohorts": {k: summarize(v, names, train_images) for k, v in groups.items()},
            "validator_mean_ms_per_image": metrics.speed}


def evaluate(args, plan, experiment_digest, opts, root, count, output):
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO, settings
    from ultralytics.data import utils as dataset_utils

    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__}
    if runtime != plan["runtime"] or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Use the pinned runtime and expose exactly one evaluation GPU")
    settings.update({k: False for k in ("sync", "hub") if k in settings})
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font.is_file():
        dataset_utils.check_font = lambda *unused, **kwargs: font
    registry = json.loads((root / "classes.json").read_bytes())
    names = {r["id"]: r["name"] for r in registry}
    original_names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(original_names, list):
        original_names = dict(enumerate(original_names))
    if names != original_names or list(names) != list(range(len(registry))):
        raise ValueError("Frozen data and class registry disagree")
    train_images = json.loads((root / "class-counts.json").read_bytes())["train"]["images"]
    results = {}
    with tempfile.TemporaryDirectory(dir=output) as temp:
        dataset, samples, pinned = prepare_view(root, args.split, count, names, Path(temp))
        for group in (STRATEGIES if args.strategy == "all" else (args.strategy,)):
            path, sha = checkpoint(plan, experiment_digest, group)
            model = YOLO(str(path), task="detect")
            expected = dict(list(names.items())[:len(model.names)]) if group == "baseline" else names
            if not model.names or model.names != expected:
                raise ValueError(f"Class IDs do not match the frozen registry for {group}")
            target = output / group
            target.mkdir()
            results[group] = score(model, dataset, samples, args.split, opts, names, train_images, target)
            results[group].update(
                weights_sha256=sha, model_classes=len(model.names),
                end_to_end=bool(getattr(model.model, "end2end", False)),
                image_counts_sha256=checksum(target / "image-counts.jsonl.gz"),
            )
            del model
        if any(identity(path) != sig for path, sig in pinned.items()):
            raise ValueError("A frozen image or label changed during evaluation")
    return runtime, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    experiment = checksum(args.experiment)
    recipe = checksum(args.recipe)
    plan = json.loads(args.experiment.read_bytes())
    if plan["purpose"] != "comparison":
        parser.error("Independent evaluation requires the comparison experiment")
    opts = read_recipe(args.recipe, experiment)
    data, sizes = verify_dataset(plan["dataset"], "comparison")
    out = args.out.resolve()
    if out == data.parent or data.parent in out.parents:
        parser.error("Evaluation output must be outside the frozen dataset")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists():
            raise ValueError(f"Evaluation output already exists: {out}")
        with tempfile.TemporaryDirectory(prefix="." + out.name + "-", dir=out.parent) as temporary:
            stage = Path(temporary)
            runtime, results = evaluate(args, plan, experiment, opts, data.parent, sizes[args.split], stage)
            check_file(args.experiment, experiment)
            check_file(args.recipe, recipe)
            write_json(stage / "receipt.json", {
                "status": "complete", "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "experiment_sha256": experiment, "dataset_receipt_sha256": plan["dataset"]["sha256"],
                "evaluation_recipe_sha256": recipe, "runtime": runtime, "split": args.split,
                "images": sizes[args.split], "options": {**opts, "imgsz": 640, "ap_confidence": AP_CONFIDENCE,
                                                          "fixed_match_iou": 0.50},
                "models": results,
            })
            stage.rename(out)
    print(f"{checksum(out / 'receipt.json')}  {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
