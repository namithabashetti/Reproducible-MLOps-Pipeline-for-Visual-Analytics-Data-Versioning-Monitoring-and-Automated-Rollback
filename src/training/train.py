"""Check CUDA, stage isolated training inputs, and train the approved YOLO baseline."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

import yaml


def check_device(device):
    import torch
    import torchvision

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fix the GPU environment or explicitly choose --device cpu")
    tensor = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    product = tensor @ tensor.T
    if not torch.isfinite(product).all() or product[0, 0].item() != 14:
        raise RuntimeError("Device tensor test failed")
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    selected = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=device), 0.5)
    if selected.cpu().tolist() != [0, 2]:
        raise RuntimeError("torchvision NMS device test failed")
    result = {"device": device, "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_runtime": torch.version.cuda, "tensor_test": True, "nms_test": True}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        result.update(gpu=torch.cuda.get_device_name(device),
                      total_vram_bytes=torch.cuda.get_device_properties(device).total_memory)
    return result


def prepare_data(project, destination, smoke, reviewed_data=None):
    """Copies keep Ultralytics label caches and possible repairs outside DVC outputs."""
    manifest = json.loads((project / "data/splits.json").read_text())
    source = project / "data/processed"
    if reviewed_data is not None:
        from scripts.review_annotations import check_review, cohort_selection

        source = reviewed_data.resolve()
        if source.parent != (project / "data").resolve() or not re.fullmatch(r"person-corrected-[A-Za-z0-9_-]+", source.name):
            raise ValueError("Reviewed data must be an isolated data/person-corrected-* export")
        review = json.loads((source / "review.json").read_text())
        if not check_review(project, review)["ready_for_export"]:
            raise ValueError("Reviewed candidate is incomplete")
        selected_export = json.loads((source / "selection.json").read_text())
        if selected_export != cohort_selection(project, review):
            raise ValueError("Candidate selection disagrees with the frozen review")
        manifest = {**manifest, "splits": selected_export}
        expected_config = {"train": "images/train", "val": "images/validation",
                           "names": {0: "helmet", 1: "head", 2: "person"}}
        if yaml.safe_load((source / "dataset.yaml").read_text()) != expected_config:
            raise ValueError("Candidate dataset configuration changed")
        for split, names in selected_export.items():
            for category, suffix in (("images", ".png"), ("labels", ".txt")):
                if {p.stem for p in (source / category / split).glob('*' + suffix)} != set(names):
                    raise ValueError("Candidate files disagree with selected membership")
    destination.mkdir(parents=True, exist_ok=False)
    selected = {}
    for split, count in (("train", 64), ("validation", 32)):
        names = list(manifest["splits"][split])
        if smoke:
            random.Random(manifest["seed"]).shuffle(names)
            covered, required = set(), []
            for name in names:
                classes = {int(row.split()[0]) for row in
                           (source / "labels" / split / f"{name}.txt").read_text().splitlines()}
                if classes - covered:
                    required.append(name)
                    covered.update(classes)
                if covered == {0, 1, 2}:
                    break
            names = (required + [n for n in names if n not in required])[:count]
        selected[split] = names
        for category, suffix in (("images", ".png"), ("labels", ".txt")):
            folder = destination / category / split
            folder.mkdir(parents=True)
            for name in names:
                shutil.copy2(source / category / split / f"{name}{suffix}", folder / f"{name}{suffix}")
    config = yaml.safe_load((source / "dataset.yaml").read_text())
    (destination / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (destination / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    if reviewed_data is not None:
        for name in ("review.json", "annotation-cohort.json"):
            shutil.copy2(source / name, destination / name)
        hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for category in ("images", "labels") for p in sorted((destination / category).rglob('*')) if p.is_file()}
        (destination / "input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return destination / "dataset.yaml", {s: len(n) for s, n in selected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One epoch on 64 train/32 validation images")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-name", help="New run directory name; existing runs are never overwritten")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), help="Explicit device override; no silent fallback")
    parser.add_argument("--reviewed-data", type=Path, help="DVC-tracked, completed person-corrected-* export")
    args = parser.parse_args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name allows only letters, digits, hyphens and underscores")
    if args.reviewed_data and (not args.run_name or args.smoke):
        parser.error("Reviewed-data experiments require a new --run-name and the full frozen cohort (no --smoke)")
    project = Path(__file__).resolve().parents[2]
    params = yaml.safe_load((project / "params.yaml").read_text())
    training = params["training"]
    device = args.device or training["device"]
    # Configure cuBLAS before the hardware probe creates the first CUDA context.
    if training["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    hardware = check_device(device)
    print(json.dumps(hardware, indent=2), flush=True)
    (project / "reports").mkdir(exist_ok=True)
    (project / "reports/gpu-check.json").write_text(json.dumps(hardware, indent=2) + "\n")
    if args.check_only:
        return
    # These are project-local settings, not the user's global Ultralytics settings.
    settings_directory = project / "artifacts/ultralytics-settings"
    settings_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_directory)
    from ultralytics import YOLO, settings
    from src.training.evaluate import evaluate

    settings.update({"sync": False, "mlflow": False})
    changed = subprocess.check_output([sys.executable, "-m", "dvc", "status", "--json"],
                                      cwd=project, text=True)
    if json.loads(changed):
        raise RuntimeError("DVC data stages have changed; run dvc repro before training")
    output = project / "artifacts/training" / (args.run_name or ("smoke" if args.smoke else "baseline"))
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}; preserve it before starting another run")
    candidate = None
    if args.reviewed_data:
        candidate_path = args.reviewed_data.resolve()
        pointer = candidate_path.with_suffix('.dvc')
        metadata = yaml.safe_load(pointer.read_text())["outs"][0]
        if metadata["path"] != candidate_path.name or not metadata["md5"].endswith('.dir'):
            raise ValueError("Invalid candidate DVC pointer")
        candidate = {"path": candidate_path.relative_to(project).as_posix(), "dvc_md5": metadata["md5"],
                     "review_sha256": hashlib.sha256((candidate_path / "review.json").read_bytes()).hexdigest(),
                     "cohort_sha256": hashlib.sha256((candidate_path / "annotation-cohort.json").read_bytes()).hexdigest()}
    data, counts = prepare_data(project, output / "inputs", args.smoke, args.reviewed_data)
    weights = project / "artifacts/pretrained" / training["model"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    lock = yaml.safe_load((project / "dvc.lock").read_text())
    # The checkout may be uncommitted; retain the actual source, not only a Git SHA.
    sources = sorted(project.glob("src/**/*.py")) + [project / p for p in
              ("params.yaml", "configs/config.yaml", "requirements-lock.txt", "dvc.yaml", "dvc.lock", "dataset.dvc")]
    if candidate:
        sources += [pointer, project / "docs/annotation-policy.md", project / "data/annotation-cohort.json",
                    project / "scripts/review_annotations.py"]
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(project)
        copy = output / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        source_hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    lineage = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "smoke_test": args.smoke,
        "purpose": "hardware/pipeline test" if args.smoke else f"{training['epochs']}-epoch YOLOv8n training",
        "dataset_dvc_md5": yaml.safe_load((project / "dataset.dvc").read_text())["outs"][0]["md5"],
        "processed_dvc_md5": lock["stages"]["preprocess"]["outs"][0]["md5"],
        "splits_dvc_md5": lock["stages"]["split"]["outs"][0]["md5"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project)),
        "file_sha256": source_hashes,
        "initial_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "ultralytics": importlib.metadata.version("ultralytics"), "hardware": hardware,
        "parameters": training, "seed": params["data"]["seed"], "input_counts": counts,
        "epochs_requested": 1 if args.smoke else training["epochs"],
    }
    if candidate:
        lineage["reviewed_data"] = candidate
    (output / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    model.train(data=str(data), epochs=lineage["epochs_requested"], imgsz=training["image_size"],
                batch=training["batch_size"], workers=training["workers"], device=device,
                cache=training["cache_images"], seed=params["data"]["seed"],
                amp=training["amp"], deterministic=training["deterministic"],
                optimizer=training["optimizer"], lr0=training["learning_rate"],
                weight_decay=training["weight_decay"], patience=training["epochs"],
                project=str(output), name="fit", exist_ok=False, plots=False, verbose=False)
    best = Path(model.trainer.best)
    with (Path(model.trainer.save_dir) / "results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    losses = [float(value) for row in history for key, value in row.items() if key.strip().startswith("train/")]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training loss history is missing or non-finite")
    report = evaluate(best, data, output, device, training["image_size"], training["batch_size"])
    report.update(smoke_test=args.smoke, input_counts=counts, epochs_completed=len(history), finite_training_losses=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not args.smoke:
        from src.tracking.mlflow_utils import log_completed_run

        print(json.dumps(log_completed_run(project, output), indent=2), flush=True)


if __name__ == "__main__":
    main()
