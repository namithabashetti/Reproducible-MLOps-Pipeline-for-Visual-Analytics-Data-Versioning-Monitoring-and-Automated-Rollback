"""Compare registered checkpoints on the same isolated labeled monitoring batch."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from src.monitoring.health import iso, model_identity, save


def labeled_audit(project, output):
    from src.tracking.mlflow_utils import local_store
    from src.serving.deploy import stage_model

    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    receipt = json.loads((project / "artifacts/serving/deployment.json").read_text())
    client, _, registry = local_store(project)
    aliases = client.get_registered_model(registry["model_name"]).aliases
    baseline_version = aliases.get(config["rollback"]["stable_alias"])
    if baseline_version is None:
        report = {"status": "unavailable", "reason": "no_approved_stable_baseline",
                  "generated_at": iso(time.time()), "candidate": model_identity(receipt["model"])}
        save(output / "audit.json", report)
        return report
    status = subprocess.check_output([sys.executable, "-m", "dvc", "status", "--json"], cwd=project, text=True)
    if json.loads(status):
        raise ValueError("Labeled audit requires unchanged DVC data stages")
    baseline_path, baseline_manifest = stage_model(project, project, str(baseline_version))
    candidate_path, candidate_manifest = stage_model(project, project, receipt["model"]["model_version"],
                                                    receipt["model"]["deployment_mode"] == "experimental")
    if model_identity(candidate_manifest) != model_identity(receipt["model"]):
        raise ValueError("Candidate settings differ from the deployed manifest")
    if baseline_manifest["dataset_dvc_md5"] != candidate_manifest["dataset_dvc_md5"]:
        raise ValueError("Baseline and candidate need a common labeled dataset identity")
    splits_path = project / "data/splits.json"
    splits = json.loads(splits_path.read_text())["splits"]
    names = splits["monitoring_current"]
    if len(set(names)) != len(names) or len(names) < config["health"]["labeled_audit_min_images"] or \
            set(names) & {n for key, values in splits.items() if key != "monitoring_current" for n in values}:
        raise ValueError("Audit must use a sufficiently large isolated monitoring pool")
    inputs = output / "inputs"
    hashes = {}
    for category, suffix in (("images", ".png"), ("labels", ".txt")):
        destination = inputs / category
        destination.mkdir(parents=True)
        for name in names:
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Invalid audit image identifier")
            source = project / "data/processed" / category / "monitoring_current" / f"{name}{suffix}"
            copied = destination / source.name
            shutil.copy2(source, copied)  # Ultralytics writes caches only into this isolated copy.
            hashes[f"{category}/{source.name}"] = hashlib.sha256(copied.read_bytes()).hexdigest()
    batch_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    save(output / "batch-sha256.json", hashes)
    data = inputs / "dataset.yaml"
    data.write_text(yaml.safe_dump({"path": inputs.resolve().as_posix(), "train": "images", "val": "images",
                                    "names": config["dataset"]["classes"]}), encoding="utf-8")
    os.environ["YOLO_CONFIG_DIR"] = str(project / "artifacts/ultralytics-settings")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "mlflow": False})
    torch.set_num_threads(candidate_manifest["inference"]["cpu_threads"])
    evaluation = {"device": "cpu", "imgsz": candidate_manifest["inference"]["image_size"],
                  "batch": 4, "workers": 0, "conf": .001, "iou": .7, "max_det": 300,
                  "rect": False, "half": False, "augment": False, "plots": False, "verbose": False}
    scores = {}
    for name, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        if name == "candidate" and baseline_manifest["checkpoint_sha256"] == candidate_manifest["checkpoint_sha256"]:
            scores[name] = scores["baseline"]
        else:
            result = YOLO(str(path / "best.pt")).val(data=str(data), split="val", project=str(output),
                                                    name=name, exist_ok=False, **evaluation)
            scores[name] = float(result.box.map)
        if not math.isfinite(scores[name]) or not 0 <= scores[name] <= 1:
            raise ValueError("Audit produced invalid mAP50-95")
    if any(hashlib.sha256((inputs / relative).read_bytes()).hexdigest() != value for relative, value in hashes.items()):
        raise ValueError("Audit changed input image or label bytes")
    if json.loads((project / "artifacts/serving/deployment.json").read_text())["container_id"] != receipt["container_id"]:
        raise ValueError("Deployment changed during labeled audit")
    report = {"status": "available", "generated_at": iso(time.time()), "candidate": model_identity(candidate_manifest),
              "baseline": {key: baseline_manifest[key] for key in ("model_name", "model_version", "checkpoint_sha256")},
              "images": len(names), "split_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
              "baseline_batch_sha256": batch_sha, "candidate_batch_sha256": batch_sha,
              "baseline_map50_95": scores["baseline"], "candidate_map50_95": scores["candidate"],
              "evaluation_settings": evaluation, "note": "Offline labeled AP comparison, not confidence or stable approval"}
    save(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = labeled_audit(Path(__file__).resolve().parents[2], args.output)
    print(json.dumps({key: report[key] for key in ("status", "reason") if key in report}))
