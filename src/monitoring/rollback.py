"""One bounded host-side rollback check; never approves a model or retries a failed switch."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from threading import Event
import time

import requests
import yaml

from src.monitoring.health import collect, evaluate, iso, model_identity, read_optional, save, timestamp
from src.serving.deploy import deploy, deployment_lock, stage_model, write_receipt
from src.tracking.mlflow_utils import local_store


def current_container(docker, receipt):
    state = json.loads(subprocess.check_output(
        [docker, "inspect", "visual-analytics-api"], text=True, timeout=10))[0]
    if state["Id"] != receipt["container_id"] or (state["Config"].get("Labels") or {}).get(
            "visual-analytics.project") != "hard-hat-detector" or state["Image"] != receipt["image_id"]:
        raise ValueError("Deployment changed or container ownership/image does not match")
    return state


def validate_decision(report, receipt, state, config, now):
    """Re-evaluate evidence under today's policy and reject stale deployment/process identity."""
    evidence = report["evidence"]
    identity = model_identity(receipt["model"])
    if report.get("policy") != config["health"] or not (
            timestamp(report["generated_at"]) <= now < timestamp(report["valid_until"])):
        raise ValueError("Health decision is expired, future-dated, or uses another policy")
    if report.get("identity") != identity or evidence.get("identity") != identity or any(
            doc.get("deployment_id") != receipt["container_id"] for doc in (report, evidence)):
        raise ValueError("Health evidence belongs to another deployment")
    if evidence.get("runtime_started") != timestamp(state["State"]["StartedAt"]):
        raise ValueError("Serving process restarted after health evidence was collected")
    decision = evaluate(evidence, config["health"], config["drift"], now)
    if report.get("status") != "unhealthy" or decision["status"] != "unhealthy":
        raise ValueError("Fresh combined health evidence does not establish an unhealthy deployment")
    return decision


def prepare_target(project, receipt, docker):
    """Resolve the recorded numeric version, recheck approval/artifacts and use its pinned image."""
    target = deepcopy(receipt.get("rollback_target"))
    if not target:
        raise ValueError("No approved rollback target was recorded before this deployment")
    if target["source_container_id"] == receipt["container_id"] or target["model"]["deployment_mode"] != "approved":
        raise ValueError("Rollback target must be an approved predecessor")
    if target["model"] == receipt["model"] and target["image_id"] == receipt["image_id"] and \
            target["serving"] == receipt.get("serving"):
        raise ValueError("Recorded target is identical to the failed deployment")
    if target["serving"]["host_bind_address"] != "127.0.0.1" or \
            not isinstance(target["serving"]["port"], int) or not 1 <= target["serving"]["port"] <= 65535:
        raise ValueError("Invalid recorded serving binding")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", target["image_id"]):
        raise ValueError("Rollback requires a recorded immutable image ID")
    image = subprocess.check_output([docker, "image", "inspect", target["image_id"],
                                     "--format", "{{.Id}}"], text=True, timeout=10).strip()
    if image != target["image_id"]:
        raise ValueError("Recorded rollback image is not available locally")
    staged, manifest = stage_model(project, project, target["model"]["model_version"],
                                   restore_manifest=target["model"])
    if manifest != target["model"]:
        raise ValueError("Verified model differs from recorded target")
    target["staged_directory"] = str(staged)
    return target


def recovery_images(project, count):
    """Use existing validation images; do not use final-test data or write to original inputs."""
    names = json.loads((project / "data/splits.json").read_text())["splits"]["validation"][:count]
    if len(names) != count or len(set(names)) != count:
        raise ValueError("Not enough distinct validation images for recovery probes")
    directory = (project / "dataset/images").resolve()
    result = []
    for name in names:
        path = (directory / f"{name}.png").resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Recovery input path escapes the dataset")
        body = path.read_bytes()
        result.append((name, body, hashlib.sha256(body).hexdigest()))
    return result


def verify_recovery(project, receipt, images, config, docker, result=None):
    """Verify operational recovery and new Prometheus identity; do not claim model-quality health."""
    expected = receipt["model"]
    base = f"http://127.0.0.1:{receipt['serving']['port']}"
    inference_hash = hashlib.sha256(json.dumps(expected["inference"], sort_keys=True).encode()).hexdigest()
    if result is None:
        result = {}
    result.update(readiness=[], predictions=[], monitoring_resumed=False)
    with requests.Session() as session:
        session.trust_env = False
        for index in range(config["rollback"]["recovery_readiness_successes"]):
            if index:
                time.sleep(config["health"]["readiness_interval_seconds"])
            current_container(docker, receipt)
            response = session.get(base + "/health", timeout=5)
            response.raise_for_status()
            if response.json() != {"status": "ready", **expected}:
                raise ValueError("Recovery readiness returned the wrong model/settings")
            result["readiness"].append({"at": iso(time.time()), "ready": True})
        for name, body, digest in images:
            response = session.post(base + "/predict", data=body, headers={"Content-Type": "image/png"}, timeout=30)
            response.raise_for_status()
            value = response.json()
            if any(value.get(key) != expected[key] for key in
                    ("model_name", "model_version", "checkpoint_sha256", "deployment_mode")) or \
                    value.get("inference_sha256") != inference_hash or not isinstance(value.get("detections"), list):
                raise ValueError("Recovery prediction returned the wrong model/settings or response")
            result["predictions"].append({"image_id": name, "sha256": digest, "status": "success"})
        current_container(docker, receipt)
        # A fresh sample of the new container identity proves that monitoring has resumed.
        deadline = time.monotonic() + 2 * config["health"]["max_scrape_gap_seconds"]
        while time.monotonic() < deadline:
            try:
                labels = f'model_version={json.dumps(expected["model_version"])},deployment_id={json.dumps(receipt["container_id"][:12])}'
                samples = []
                for expression in ('up{job="visual-analytics-api"}', f'va_model_info{{{labels}}}', f'va_model_ready{{{labels}}}'):
                    response = session.get(f"http://127.0.0.1:{config['monitoring']['prometheus_port']}/api/v1/query",
                                           params={"query": f"{expression}[{config['health']['max_scrape_gap_seconds']}s]"}, timeout=5)
                    response.raise_for_status()
                    payload = response.json()
                    values = payload["data"]["result"]
                    if payload["status"] != "success" or payload.get("warnings") or \
                            payload["data"]["resultType"] != "matrix" or len(values) != 1 or not values[0].get("values"):
                        raise ValueError("Missing or ambiguous monitoring samples")
                    sample = {"metric": values[0]["metric"], "value": values[0]["values"][-1]}
                    if sample["value"][1] != "1" or not timestamp(receipt["started_at_utc"]) <= float(
                            sample["value"][0]) <= time.time() or time.time() - float(sample["value"][0]) > config["health"]["max_scrape_gap_seconds"]:
                        raise ValueError("Monitoring sample is stale or not ready")
                    samples.append(sample)
                if samples[1]["metric"].get("checkpoint_sha256") != expected["checkpoint_sha256"] or \
                        samples[1]["metric"].get("inference_sha256") != inference_hash:
                    raise ValueError("Monitoring model identity mismatch")
                result.update(monitoring_resumed=True, prometheus_samples=samples)
                break
            except (requests.RequestException, ValueError, KeyError):
                time.sleep(1)
        if not result["monitoring_resumed"]:
            raise RuntimeError("Serving recovered but fresh monitoring of the restored model was not verified")
    current_container(docker, receipt)
    return result


def run(project, docker, *, report=None, drift=None, audit=None, latency_baseline=None):
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    policy = config["rollback"]
    if policy["max_attempts_per_failed_deployment"] != 1 or policy["recovery_readiness_successes"] != 3 or \
            policy["recovery_prediction_successes"] != 20:
        raise ValueError("Expected the approved one-attempt, three-readiness, twenty-prediction policy")
    with deployment_lock(project):
        receipt_path = project / "artifacts/serving/deployment.json"
        receipt = json.loads(receipt_path.read_text())
        failed_id = receipt["container_id"]
        if not re.fullmatch(r"[0-9a-f]{64}", failed_id):
            raise ValueError("Invalid deployment container identity")
        attempts = receipt_path.parent / "rollback"
        attempts.mkdir(exist_ok=True)
        attempt_path = attempts / f"attempt-{failed_id}.json"
        if attempt_path.exists():
            return {"status": "blocked", "reason": "rollback_attempt_already_recorded", "attempt_file": str(attempt_path)}
        try:
            current_container(docker, receipt)
            target = prepare_target(project, receipt, docker)
            images = recovery_images(project, policy["recovery_prediction_successes"])
        except Exception as error:
            return {"status": "unavailable", "reason": str(error), "deployment_id": failed_id}
        if report is None:
            evidence = collect(project, config["health"], config["health"]["readiness_failure_count"])
            evidence.update(drift=drift, audit=audit, latency_baseline=latency_baseline)
            report = evaluate(evidence, config["health"], config["drift"])
            report["evidence"] = evidence
        try:
            state = current_container(docker, receipt)
            decision = validate_decision(report, receipt, state, config, time.time())
            client, _, registry = local_store(project)
            aliases = client.get_registered_model(registry["model_name"]).aliases
            approved = {str(aliases.get(policy[key])) for key in ("stable_alias", "previous_stable_alias")}
            if target["model"]["model_name"] != registry["model_name"] or target["model"]["model_version"] not in approved:
                raise ValueError("Rollback approval changed during evidence collection")
            if "quality" in decision["reasons"]:
                stable = client.get_model_version_by_alias(registry["model_name"], policy["stable_alias"])
                baseline = {"model_name": registry["model_name"], "model_version": str(stable.version),
                            "checkpoint_sha256": stable.tags["checkpoint_sha256"]}
                if report["evidence"].get("stable_baseline") != baseline:
                    raise ValueError("Approved quality baseline changed after evidence collection")
        except Exception as error:
            return {"status": "observe", "reason": str(error), "deployment_id": failed_id, "health": report}
        target.update(failed_container_id=failed_id, failed_runtime_started=state["State"]["StartedAt"],
                      decision_valid_until=decision["valid_until"])
        # Durable claim precedes every destructive action. Interrupted/failed attempts are never retried.
        attempt = {"status": "started", "failed_deployment_id": failed_id, "target": target,
                   "started_at": iso(time.time()), "health": report, "decision": decision,
                   "attempt_file": str(attempt_path)}
        with attempt_path.open("x", encoding="utf-8") as handle:
            json.dump(attempt, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            restored = deploy(project, Path(target["staged_directory"]), target["model"], target["image_id"], docker,
                              recovery_target=target)
            attempt["restored_deployment"] = restored
            attempt["recovery"] = {}
            verify_recovery(project, restored, images, config, docker, attempt["recovery"])
            attempt["status"] = "recovered"
            attempt["quality_health"] = "not_established_by_operational_recovery"
        except Exception as error:
            attempt.update(status="failed", error_type=type(error).__name__, reason=str(error))
        finally:
            attempt["completed_at"] = iso(time.time())
            write_receipt(attempt_path, attempt)
        return attempt


def watch(project, docker, output, interval=30, *, stop=None, drift=None, audit=None, latency_baseline=None):
    """Continuously collect fresh evidence; preserve the existing bounded recovery gate."""
    if interval < 10:
        raise ValueError("Watch interval must be at least 10 seconds")
    stop = stop if stop is not None else Event()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    while not stop.is_set():
        result = run(project, docker, drift=read_optional(drift), audit=read_optional(audit),
                     latency_baseline=read_optional(latency_baseline))
        result = {**result, "watch_checked_at": iso(time.time())}
        write_receipt(output, result)
        summary = (result["status"], result.get("reason"))
        if summary != previous:
            print(json.dumps({"status": summary[0], "reason": summary[1], "output": str(output)}), flush=True)
            previous = summary
        if result["status"] == "failed":
            raise RuntimeError("Recovery failed; inspect the saved attempt before restarting the watcher")
        stop.wait(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, help="Fresh Phase 10 JSON; otherwise collect current evidence")
    parser.add_argument("--drift", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--latency-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Repeat fresh health checks until interrupted")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between watch cycles (minimum 10)")
    parser.add_argument("--docker", default=shutil.which("docker") or str(
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"))
    args = parser.parse_args()
    if args.watch and args.decision:
        parser.error("Watch mode collects fresh decisions; --decision is only for a single check")
    if args.watch and args.interval < 10:
        parser.error("Watch interval must be at least 10 seconds")
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[2]
    if args.watch:
        try:
            watch(project, args.docker, args.output, args.interval, drift=args.drift, audit=args.audit,
                  latency_baseline=args.latency_baseline)
        except KeyboardInterrupt:
            print("Rollback watch stopped.")
        return
    result = run(project, args.docker, report=read_optional(args.decision), drift=read_optional(args.drift),
                 audit=read_optional(args.audit), latency_baseline=read_optional(args.latency_baseline))
    save(args.output, result)
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "output": str(args.output)}))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
