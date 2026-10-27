"""Import completed YOLO runs into local MLflow and verify registry retrieval."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml

# Tracking is local; neither the SDK nor the dashboard needs usage telemetry.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_store(project):
    """Resolve the approved SQLite/artifact paths independently of the working directory."""
    config = yaml.safe_load((project / "configs/config.yaml").read_text())["mlflow"]
    prefix = "sqlite:///"
    if not config["backend_store_uri"].startswith(prefix):
        raise ValueError("This project requires a local SQLite MLflow store")
    database = (project / config["backend_store_uri"][len(prefix):]).resolve()
    artifacts = (project / config["artifact_directory"]).resolve()
    for path in (database, artifacts):
        if not path.is_relative_to((project / "artifacts").resolve()):
            raise ValueError("MLflow storage must stay inside project artifacts")
    database.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    uri = prefix + database.as_posix()
    return MlflowClient(tracking_uri=uri, registry_uri=uri), artifacts, config


def read_completed_run(directory):
    """Validate immutable training evidence before creating any MLflow record."""
    directory = directory.resolve()
    lineage = json.loads((directory / "lineage.json").read_text())
    report = json.loads((directory / "metrics.json").read_text())
    with (directory / "fit/results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if (report["evaluation_split"] != "validation" or report["stable_model"] is not False
            or len(history) != report["epochs_completed"] or not history
            or report["smoke_test"] != lineage["smoke_test"]
            or len(history) != lineage["epochs_requested"]
            or not report["finite_training_losses"]):
        raise ValueError("Expected a completed, evaluated, experimental training run")
    if [int(row["epoch"]) for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Epoch history is incomplete")
    if not all(math.isfinite(float(v)) for row in history for v in row.values()):
        raise ValueError("Epoch history contains non-finite values")
    values = dict(report["metrics"])
    for row in report["per_class"]:
        for source, name in (("Box-P", "precision"), ("Box-R", "recall"),
                             ("mAP50", "map50"), ("mAP50-95", "map50_95")):
            values[f"{row['Class']}/{name}"] = row[source]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()):
        raise ValueError("Validation metrics must be finite values in [0, 1]")
    selected = json.loads((directory / "inputs/selection.json").read_text())
    if (set(selected) != {"train", "validation"}
            or {s: len(v) for s, v in selected.items()} != report["input_counts"]
            or report["input_counts"] != lineage["input_counts"]
            or any(len(v) != len(set(v)) for v in selected.values())
            or set(selected["train"]) & set(selected["validation"])):
        raise ValueError("Invalid train/validation membership")
    files = ["lineage.json", "metrics.json", "fit/results.csv", "fit/args.yaml",
             "inputs/selection.json", "inputs/dataset.yaml", "fit/weights/best.pt"]
    if not lineage["file_sha256"]:
        raise ValueError("Missing source snapshot")
    for name, digest in lineage["file_sha256"].items():
        path = (directory / "source" / name).resolve()
        if not path.is_relative_to(directory / "source") or file_hash(path) != digest:
            raise ValueError(f"Source snapshot mismatch: {name}")
        files.append(path.relative_to(directory).as_posix())
    checkpoint = directory / "fit/weights/best.pt"
    if file_hash(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash does not match the evaluation")
    raw = yaml.safe_load((directory / "source/dataset.dvc").read_text())
    lock = yaml.safe_load((directory / "source/dvc.lock").read_text())
    base_identity = lineage['base_data'] if 'expanded_data' in lineage else lineage
    if (raw["outs"][0]["md5"] != base_identity["dataset_dvc_md5"]
            or lock["stages"]["split"]["outs"][0]["md5"] != base_identity["splits_dvc_md5"]
            or lock["stages"]["preprocess"]["outs"][0]["md5"] != base_identity["processed_dvc_md5"]):
        raise ValueError("DVC snapshot identities disagree with lineage")
    if 'expanded_data' in lineage:
        candidate = lineage['expanded_data']
        relative = Path(candidate['path'])
        if relative.parent.as_posix() != 'data' or not re.fullmatch(r'expanded-[A-Za-z0-9_-]+',relative.name):
            raise ValueError("Invalid expanded candidate path")
        metadata = yaml.safe_load((directory/'source'/relative.with_suffix('.dvc')).read_text())['outs'][0]
        if (metadata['path'] != relative.name or metadata['md5'] != candidate['dvc_md5']
                or lineage['dataset_dvc_md5'] != candidate['dvc_md5']
                or lineage['processed_dvc_md5'] != candidate['dvc_md5']
                or lineage['splits_dvc_md5'] is not None
                or file_hash(directory/'inputs/selection.json') != lineage['selection_sha256']):
            raise ValueError("Expanded dataset identity mismatch")
        manifest = json.loads((directory/'inputs/manifest.json').read_text())
        if (file_hash(directory/'inputs/manifest.json') != candidate['manifest_sha256']
                or file_hash(directory/'inputs/audit.json') != manifest['audit_sha256']):
            raise ValueError("Expanded provenance mismatch")
        expected_selection = {s:manifest['selection'][s] for s in ('train','validation')}
        if lineage['smoke_test']:
            if any(not set(ids) <= set(expected_selection[s]) for s,ids in selected.items()):
                raise ValueError("Smoke inputs escaped expanded train/validation membership")
        elif selected != expected_selection:
            raise ValueError("Expanded training membership changed")
        input_hashes = json.loads((directory/'inputs/input-sha256.json').read_text())
        expected = {f'{kind}/{split}/{name}{suffix}' for split,ids in selected.items() for name in ids
                    for kind,suffix in (('images','.png'),('labels','.txt'))}
        if set(input_hashes) != expected:
            raise ValueError("Incomplete expanded input hashes")
        for name,digest in input_hashes.items():
            path = (directory/'inputs'/name).resolve()
            if not path.is_relative_to(directory/'inputs') or file_hash(path) != digest:
                raise ValueError("Expanded training input was changed")
        files += ['inputs/'+name for name in ('manifest.json','audit.json','input-sha256.json','README.dataset.txt','README.roboflow.txt')]
    if "reviewed_data" in lineage:
        candidate = lineage["reviewed_data"]
        relative = Path(candidate["path"])
        if relative.parent.as_posix() != "data" or not re.fullmatch(r"person-corrected-[A-Za-z0-9_-]+", relative.name):
            raise ValueError("Invalid reviewed candidate path")
        pointer = directory / "source" / relative.with_suffix('.dvc')
        metadata = yaml.safe_load(pointer.read_text())["outs"][0]
        if metadata["md5"] != candidate["dvc_md5"] or metadata["path"] != relative.name:
            raise ValueError("Reviewed candidate DVC identity mismatch")
        for name, key in (("review.json", "review_sha256"), ("annotation-cohort.json", "cohort_sha256")):
            if file_hash(directory / "inputs" / name) != candidate[key]:
                raise ValueError("Reviewed candidate provenance mismatch")
            files.append("inputs/" + name)
        if "training_proposal_sha256" in candidate:
            review = json.loads((directory / "inputs/review.json").read_text())
            proposal_hash = candidate["training_proposal_sha256"]
            if (file_hash(directory / "inputs/training-proposal.json") != proposal_hash
                    or review.get("training_proposal", {}).get("draft_sha256") != proposal_hash):
                raise ValueError("Training proposal provenance mismatch")
            files.append("inputs/training-proposal.json")
        cohort = json.loads((directory / "inputs/annotation-cohort.json").read_text())
        if selected != cohort["selection"]:
            raise ValueError("Reviewed training membership differs from the frozen cohort")
        input_hashes = json.loads((directory / "inputs/input-sha256.json").read_text())
        expected = {f"{category}/{split}/{name}{suffix}" for split, names in selected.items()
                    for name in names for category, suffix in (("images", ".png"), ("labels", ".txt"))}
        if set(input_hashes) != expected:
            raise ValueError("Incomplete reviewed input hashes")
        for name, expected_hash in input_hashes.items():
            path = (directory / "inputs" / name).resolve()
            if not path.is_relative_to(directory / "inputs") or file_hash(path) != expected_hash:
                raise ValueError("Reviewed training input was changed")
        files.append("inputs/input-sha256.json")
    for receipt_path in sorted((directory/'resumes').glob('*/resume.json')):
        resume=json.loads(receipt_path.read_text())
        for relative,digest in resume['sha256'].items():
            path=(receipt_path.parent/relative).resolve()
            if not path.is_relative_to(receipt_path.parent.resolve()) or file_hash(path)!=digest:
                raise ValueError('Resume evidence changed')
            files.append(path.relative_to(directory).as_posix())
        files.append(receipt_path.relative_to(directory).as_posix())
    hashes = {name: file_hash(directory / name) for name in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return lineage, report, history, values, hashes, digest


def verify_registration(client, name, version, directory):
    """Resolve a numeric registry version and verify its bytes, lineage, and metrics."""
    # MLflow 3.16's models:/ resolver consults these defaults even with explicit URI arguments.
    mlflow.set_tracking_uri(client.tracking_uri)
    mlflow.set_registry_uri(client.tracking_uri)
    lineage, report, history, values, hashes, digest = read_completed_run(directory)
    model = client.get_model_version(name, str(version))
    run = client.get_run(model.run_id)
    if (model.status != "READY" or run.info.status != "FINISHED"
            or model.tags.get("bundle_sha256") != digest
            or run.data.tags.get("bundle_sha256") != digest
            or run.data.tags.get("dataset_dvc_md5") != lineage["dataset_dvc_md5"]):
        raise ValueError("Registry version does not match this completed run")
    if any(run.data.metrics.get(f"validation/{key}") != value for key, value in values.items()):
        raise ValueError("Stored MLflow metrics differ from the evaluation")
    curve = client.get_metric_history(model.run_id, "epoch/train/box_loss")
    if sorted((p.step, p.value) for p in curve) != [(int(r["epoch"]), float(r["train/box_loss"])) for r in history]:
        raise ValueError("Stored epoch history differs from the training results")
    uri = f"models:/{name}/{version}/best.pt"
    with tempfile.TemporaryDirectory(prefix="visual-analytics-registry-") as temporary:
        restored = mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=temporary, tracking_uri=client.tracking_uri,
            registry_uri=client.tracking_uri)
        if file_hash(Path(restored)) != report["checkpoint_sha256"]:
            raise ValueError("Registry checkpoint content mismatch")
        evidence = Path(client.download_artifacts(model.run_id, "", temporary))
        for relative, expected in hashes.items():
            stored = "model/best.pt" if relative == "fit/weights/best.pt" else relative
            if file_hash(evidence / stored) != expected:
                raise ValueError(f"Stored training evidence mismatch: {relative}")
    return {"verified": True, "tracking_uri": client.tracking_uri,
            "experiment_id": run.info.experiment_id, "run_id": model.run_id,
            "model_name": name, "model_version": str(version), "model_uri": uri,
            "checkpoint_sha256": report["checkpoint_sha256"], "bundle_sha256": digest,
            "dataset_dvc_md5": lineage["dataset_dvc_md5"],
            "registry_aliases": {alias: str(value) for alias, value in client.get_registered_model(name).aliases.items()},
            "registration_status": model.tags.get("approval_status"),
            "evaluation_split": "validation", "metrics": report["metrics"]}


def log_completed_run(project, directory):
    """Log once per evidence bundle; resume serial retries without another model version."""
    lineage, report, history, values, hashes, digest = read_completed_run(directory)
    client, artifacts, config = local_store(project)
    name = config["model_name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use a simple model name with letters, digits, hyphens or underscores")
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(artifacts / name).as_uri())
    existing = client.search_runs([experiment_id], f"tags.bundle_sha256 = '{digest}'", max_results=2)
    if len(existing) > 1:
        raise ValueError("Duplicate import records need review; run only one importer at a time")
    tags = {key: str(lineage[key]) for key in ("dataset_dvc_md5", "processed_dvc_md5", "splits_dvc_md5",
                                             "git_commit", "git_worktree_dirty", "started_at_utc")}
    tags.update(bundle_sha256=digest, approval_status="experimental", logging_mode="completed_run_import",
                smoke_test=str(report["smoke_test"]).lower(), mlflow_version=mlflow.__version__,
                model_format="ultralytics_checkpoint", importer_sha256=file_hash(Path(__file__)))
    tags["mlflow.source.git.commit"] = lineage["git_commit"]
    if "reviewed_data" in lineage:
        tags.update({"reviewed_data_" + key: str(value) for key, value in lineage["reviewed_data"].items()})
    if 'expanded_data' in lineage:
        tags.update({'expanded_data_'+key:str(value) for key,value in lineage['expanded_data'].items()})
        tags['selection_sha256'] = lineage['selection_sha256']
        tags['label_status'] = 'source annotations; incomplete visual review'
    tags["mlflow.source.name"] = "src/training/train.py"
    run = existing[0] if existing else client.create_run(experiment_id, tags=tags, run_name=directory.name)
    run_id = run.info.run_id
    try:
        if run.data.tags.get("artifacts_logged") != "true":
            params = {**lineage["parameters"], "seed": lineage["seed"],
                      "epochs_completed": len(history), "effective_device": lineage["hardware"]["device"],
                      "training_ultralytics": lineage["ultralytics"],
                      **{f"images_{s}": n for s, n in report["input_counts"].items()}}
            client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items()])
            # Historical values use actual epoch steps; timestamps describe this import, not live training.
            timestamp = run.info.start_time
            for row in history:
                client.log_batch(run_id, metrics=[Metric(f"epoch/{k.replace('(B)', '')}", float(v),
                                                        timestamp, int(row["epoch"]))
                                                 for k, v in row.items() if k not in {"epoch", "time"}])
            client.log_batch(run_id, metrics=[Metric(f"validation/{k}", float(v), timestamp, 0)
                                             for k, v in values.items()])
            for relative in hashes:
                path = directory / relative
                destination = "model" if relative == "fit/weights/best.pt" else path.relative_to(directory).parent.as_posix()
                client.log_artifact(run_id, str(path), None if destination == "." else destination)
            client.log_dict(run_id, {"format": "ultralytics_checkpoint", "loader": "ultralytics.YOLO",
                                    "checkpoint_sha256": report["checkpoint_sha256"],
                                    "bundle_sha256": digest, "approval_status": "experimental"}, "model/metadata.json")
            client.log_dict(run_id, hashes, "bundle-sha256.json")
            client.log_artifact(run_id, __file__, "tracking")
            client.log_artifact(run_id, str(project / "requirements-lock.txt"), "tracking")
            client.set_tag(run_id, "artifacts_logged", "true")
        client.set_terminated(run_id, "FINISHED")
        try:
            client.get_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(name, description="YOLOv8n detector. Registration is not stable approval.")
        versions = client.search_model_versions(f"name = '{name}' AND run_id = '{run_id}'")
        if len(versions) > 1:
            raise ValueError("Multiple registry versions reference this import; review before retrying")
        version = versions[0] if versions else client.create_model_version(
            name, source=f"runs:/{run_id}/model", run_id=run_id,
            tags={"bundle_sha256": digest, "checkpoint_sha256": report["checkpoint_sha256"],
                  "approval_status": "experimental", "dataset_dvc_md5": lineage["dataset_dvc_md5"]},
            description="Validation-only experimental detector. Review per-class quality before stable approval.")
        receipt = verify_registration(client, name, version.version, directory)
    except Exception:
        # A registry/retrieval failure does not erase successfully logged training evidence.
        if client.get_run(run_id).data.tags.get("artifacts_logged") != "true":
            client.set_terminated(run_id, "FAILED")
        raise
    (directory / "mlflow.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "verify", "serve"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/training/baseline"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    if args.command == "import":
        result = log_completed_run(project, (project / args.run_dir).resolve())
    else:
        client, artifacts, config = local_store(project)
        if args.command == "serve":
            subprocess.run([sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1",
                            "--workers", "1",
                            "--port", str(config["port"]), "--backend-store-uri", client.tracking_uri,
                            "--default-artifact-root", artifacts.as_uri(), "--no-serve-artifacts"], check=True)
            return
        directory = (project / args.run_dir).resolve()
        receipt = json.loads((directory / "mlflow.json").read_text())
        result = verify_registration(client, receipt["model_name"], receipt["model_version"], directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
