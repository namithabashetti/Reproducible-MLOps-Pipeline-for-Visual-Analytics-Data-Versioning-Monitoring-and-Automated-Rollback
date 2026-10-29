"""Stage a verified MLflow version and deploy it through one local Docker procedure."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image
import yaml

from src.tracking.mlflow_utils import file_hash, local_store


def stage_model(project, store_project, version=None, experimental=False, restore_manifest=None):
    """Default to the explicit stable alias; unapproved numeric versions require opt-in."""
    client, _, registry = local_store(store_project)
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    name = registry["model_name"]
    if experimental and version is None:
        raise ValueError("Experimental deployment requires an explicit numeric version")
    if version is None:
        model = client.get_model_version_by_alias(name, config["rollback"]["stable_alias"])
    else:
        if not str(version).isdigit() or int(version) < 1:
            raise ValueError("Model version must be a positive integer")
        model = client.get_model_version(name, str(version))
    aliases = client.get_registered_model(name).aliases
    approved = {str(aliases.get(config["rollback"][key])) for key in ("stable_alias", "previous_stable_alias")}
    if not experimental and str(model.version) not in approved:
        raise ValueError("Version is not referenced by stable or previous_stable; review it first")
    run = client.get_run(model.run_id)
    if model.status != "READY" or run.info.status != "FINISHED":
        raise ValueError("Expected a ready version from a finished run")
    if model.source != f"runs:/{model.run_id}/model":
        raise ValueError("Unexpected registered checkpoint source")
    with tempfile.TemporaryDirectory(prefix="visual-analytics-stage-") as temporary:
        evidence = Path(client.download_artifacts(model.run_id, "", temporary))
        hashes = json.loads((evidence / "bundle-sha256.json").read_text())
        bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        if not hashes or bundle != model.tags.get("bundle_sha256") or bundle != run.data.tags.get("bundle_sha256"):
            raise ValueError("Registry evidence bundle mismatch")
        for relative, expected in hashes.items():
            path = (evidence / ("model/best.pt" if relative == "fit/weights/best.pt" else relative)).resolve()
            if not path.is_relative_to(evidence.resolve()) or file_hash(path) != expected:
                raise ValueError(f"Registry artifact hash mismatch: {relative}")
        checkpoint = evidence / "model/best.pt"
        digest = file_hash(checkpoint)
        if digest != model.tags.get("checkpoint_sha256") or hashes.get("fit/weights/best.pt") != digest:
            raise ValueError("Checkpoint hash does not match the registered version")
        lineage = json.loads((evidence / "lineage.json").read_text())
        if lineage["dataset_dvc_md5"] != model.tags.get("dataset_dvc_md5"):
            raise ValueError("Registered dataset identity mismatch")
        params = yaml.safe_load((evidence / "source/params.yaml").read_text())
        settings = {key: value for key, value in config["serving"].items()
                    if key not in {"host_bind_address", "port"}}
        settings.update(image_size=lineage["parameters"]["image_size"],
                        pad_value=params["preprocessing"]["pad_value"])
        manifest = {"model_name": name, "model_version": str(model.version), "run_id": model.run_id,
                    "checkpoint_sha256": digest, "bundle_sha256": bundle,
                    "dataset_dvc_md5": lineage["dataset_dvc_md5"],
                    "deployment_mode": "experimental" if experimental else "approved",
                    "classes": config["dataset"]["classes"], "inference": settings}
        if restore_manifest is not None:
            if experimental or restore_manifest.get("deployment_mode") != "approved" or any(
                    restore_manifest.get(key) != manifest[key] for key in
                    ("model_name", "model_version", "run_id", "checkpoint_sha256", "bundle_sha256", "dataset_dvc_md5")):
                raise ValueError("Recorded rollback model does not match the verified registry artifact")
            # Restore the recorded model settings, even if the candidate changed config.yaml.
            manifest = json.loads(json.dumps(restore_manifest))
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        identity = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        destination = store_project / "artifacts/serving" / f"v{model.version}-{identity}"
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and manifest_path.read_text() != encoded:
            raise ValueError("Existing staged manifest differs; refusing to overwrite")
        if (destination / "best.pt").exists():
            if file_hash(destination / "best.pt") != digest:
                raise ValueError("Existing staged checkpoint is corrupt; refusing to overwrite")
        else:
            shutil.copyfile(checkpoint, destination / "best.pt")
        manifest_path.write_text(encoded, encoding="utf-8")
    return destination, manifest


@contextmanager
def deployment_lock(store_project):
    """Serialize normal deployment and rollback; a crash leaves a fail-closed lock."""
    path = store_project / "artifacts/serving/deployment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        path.unlink()


def write_receipt(path, value):
    """Publish a complete JSON document, never a partially written current pointer."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers may briefly hold the destination without delete sharing.
    # Retain atomic replacement; a bounded retry must not publish partial JSON.
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.02 * (attempt + 1))


def snapshot_target(project, store_project, old, docker):
    """Capture an approved, ready predecessor before the candidate replaces it."""
    path = store_project / "artifacts/serving/deployment.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt["container_id"] != old["Id"]:
        raise ValueError("Current container and deployment receipt disagree")
    if receipt["model"]["deployment_mode"] != "approved":
        return None
    if receipt["status"] != "ready" or not old["State"]["Running"]:
        raise ValueError("Cannot preserve an unready approved deployment as a rollback target")
    bindings = old["HostConfig"]["PortBindings"]["8000/tcp"]
    if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1" or old["Image"] != receipt["image_id"]:
        raise ValueError("Approved deployment image or loopback binding differs from its receipt")
    port = int(bindings[0]["HostPort"])
    with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
        if json.load(response) != {"status": "ready", **receipt["model"]}:
            raise ValueError("Approved predecessor readiness identity mismatch")
    staged, manifest = stage_model(project, store_project, receipt["model"]["model_version"],
                                   restore_manifest=receipt["model"])
    return {"model": manifest, "staged_directory": str(staged), "image_id": receipt["image_id"],
            "source_container_id": old["Id"], "captured_at": datetime.now(timezone.utc).isoformat(),
            "serving": {"host_bind_address": "127.0.0.1", "port": port}}


def deploy(project, staged, manifest, image, docker, *, recovery_target=None):
    """Shared deployment procedure; callers hold deployment_lock for the local store."""
    settings = yaml.safe_load((project / "configs/config.yaml").read_text())["serving"]
    if recovery_target is not None:
        settings = recovery_target["serving"]
    if settings["host_bind_address"] != "127.0.0.1":
        raise ValueError("Only loopback deployment is approved")
    image_id = subprocess.check_output([docker, "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    # A stable network/DNS name lets Prometheus keep scraping across replacements.
    network = subprocess.run([docker, "network", "inspect", "visual-analytics"], capture_output=True, text=True)
    if network.returncode != 0:
        subprocess.run([docker, "network", "create", "--label", "visual-analytics.project=hard-hat-detector",
                        "visual-analytics"], check=True, capture_output=True)
    elif (json.loads(network.stdout)[0].get("Labels") or {}).get("visual-analytics.project") != "hard-hat-detector":
        raise ValueError("Network name belongs to another application; refusing to use it")
    name = "visual-analytics-api"
    previous = subprocess.run([docker, "container", "inspect", name], text=True, capture_output=True)
    previous_id = None
    target = None
    old = None
    if previous.returncode == 0:
        old = json.loads(previous.stdout)[0]
        if (old["Config"].get("Labels") or {}).get("visual-analytics.project") != "hard-hat-detector":
            raise ValueError("Container name belongs to another application; refusing to replace it")
        previous_id = old["Id"]
        if recovery_target is None:
            target = snapshot_target(project, staged.parent.parent.parent, old, docker)
    intent = {"model": manifest, "image_id": image_id, "previous_container_id": previous_id,
              "rollback_target": target, "serving": {"host_bind_address": "127.0.0.1", "port": settings["port"]}}
    intent_path = staged.parent / f"deployment-intent-{uuid4().hex}.json"
    with intent_path.open("x", encoding="utf-8") as handle:
        json.dump(intent, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if old is not None:
        if recovery_target is not None and (old["Id"] != recovery_target["failed_container_id"] or
                old["State"]["StartedAt"] != recovery_target["failed_runtime_started"] or
                datetime.now(timezone.utc) >= datetime.fromisoformat(recovery_target["decision_valid_until"])):
            raise ValueError("Deployment/process changed or health evidence expired before replacement")
        subprocess.run([docker, "rm", "-f", name], check=True, capture_output=True)
    elif recovery_target is not None:
        raise ValueError("Failed deployment disappeared before replacement")
    container_id = subprocess.check_output([
        docker, "run", "-d", "--name", name, "--label", "visual-analytics.project=hard-hat-detector",
        "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=256m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "128",
        "--network", "visual-analytics", "--network-alias", "api",
        "--publish", f"127.0.0.1:{settings['port']}:8000",
        "--mount", f"type=bind,source={staged.resolve()},target=/model,readonly", image_id], text=True).strip()
    receipt = {"status": "starting", "container_id": container_id, "image_id": image_id,
               "previous_container_id": previous_id, "staged_directory": str(staged),
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "model": manifest,
               "serving": intent["serving"], "rollback_target": target, "intent_file": str(intent_path),
               "rollback_of": previous_id if recovery_target is not None else None,
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
               "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project, text=True).strip())}
    receipt_path = staged.parent / "deployment.json"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            running = subprocess.check_output([docker, "inspect", name, "--format", "{{.State.Running}}"], text=True).strip()
            if running != "true":
                raise RuntimeError("Serving container exited; inspect docker logs visual-analytics-api")
            try:
                with urlopen(f"http://127.0.0.1:{settings['port']}/health", timeout=5) as response:
                    health = json.load(response)
                expected = json.loads(json.dumps(manifest))
                if health != {"status": "ready", **expected}:
                    raise RuntimeError("Readiness returned a different model or inference configuration")
                probe = BytesIO()
                Image.new("RGB", (32, 32), (114, 114, 114)).save(probe, format="PNG")
                with urlopen(Request(f"http://127.0.0.1:{settings['port']}/predict", data=probe.getvalue(),
                                     headers={"Content-Type": "image/png"}), timeout=30) as response:
                    prediction = json.load(response)
                if (prediction["model_version"] != manifest["model_version"]
                        or prediction["checkpoint_sha256"] != manifest["checkpoint_sha256"]
                        or not isinstance(prediction["detections"], list)):
                    raise RuntimeError("Prediction probe returned an unexpected model or response")
                receipt["prediction_probe_passed"] = True
                receipt["status"] = "ready"
                break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(1)
        else:
            raise RuntimeError("Serving readiness timed out; inspect docker logs visual-analytics-api")
    except Exception:
        receipt["status"] = "failed"
        raise
    finally:
        # Preserve each attempt as well as the pointer used for current deployment inspection.
        write_receipt(staged.parent / f"deployment-{container_id[:12]}.json", receipt)
        write_receipt(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--version")
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--image", default="visual-analytics-api:local")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--docker", default=shutil.which("docker") or str(
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    with deployment_lock(args.store_project.resolve()):
        staged, manifest = stage_model(project, args.store_project.resolve(), args.version, args.experimental)
        result = {"staged_directory": str(staged), "model": manifest} if args.stage_only else deploy(
            project, staged, manifest, args.image, args.docker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
