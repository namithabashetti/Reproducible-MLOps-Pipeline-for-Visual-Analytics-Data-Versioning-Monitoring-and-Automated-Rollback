"""Host-side, bounded health checks. Emits evidence/decisions; never deploys a model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import requests
import yaml


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Evidence timestamps must include a timezone")
    return result.timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def model_identity(manifest):
    return {key: manifest[key] for key in
            ("model_name", "model_version", "checkpoint_sha256", "inference", "dataset_dvc_md5")}


def unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def histogram_p95(buckets):
    """Linear interpolation on cumulative bucket deltas; overflow is unavailable."""
    ordered = sorted((float(bound), count) for bound, count in buckets.items())
    if not ordered or not math.isinf(ordered[-1][0]) or ordered[-1][1] <= 0:
        return None
    if any(not math.isfinite(count) or count < 0 for _, count in ordered) or \
            any(b[1] < a[1] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Histogram bucket counts are not cumulative")
    rank = .95 * ordered[-1][1]
    lower, previous = 0., 0.
    for upper, count in ordered:
        if count < previous:
            raise ValueError("Histogram bucket counts are not cumulative")
        if count >= rank:
            if not math.isfinite(upper):
                return None
            return lower + (upper - lower) * (rank - previous) / (count - previous)
        lower, previous = upper, count
    return None


def traffic_windows(series, end, runtime_started, policy):
    """Use shared scrape boundaries and exact deltas, without rate extrapolation."""
    width, gap = policy["window_seconds"], policy["max_scrape_gap_seconds"]
    result = []
    for index in range(policy["consecutive_non_overlapping_windows"]):
        stop = end - width * (policy["consecutive_non_overlapping_windows"] - index - 1)
        start = stop - width
        window = {"start": start, "end": stop}
        try:
            up = [s for s in series if s["metric"]["__name__"] == "up"]
            if len(up) != 1:
                raise ValueError("missing_or_ambiguous_scrapes")
            points = [(float(t), float(v)) for t, v in up[0]["values"]]
            first = max(t for t, _ in points if t <= start)
            last = max(t for t, _ in points if t <= stop)
            anchors = [(t, v) for t, v in points if first <= t <= last]
            if start - first > gap or stop - last > gap or first < runtime_started or \
                    len(anchors) < 2 or any(v != 1 for _, v in anchors) or \
                    any(not 0 < b[0] - a[0] <= gap for a, b in zip(anchors, anchors[1:])):
                raise ValueError("scrape_gap_outage_or_restart")
            times = [t for t, _ in anchors]
            counters, buckets, created = {}, {}, {}
            for sample in series:
                labels = sample["metric"]
                metric, outcome = labels["__name__"], labels.get("outcome")
                if outcome not in {"success", "server_error"}:
                    continue
                values = [(float(t), float(v)) for t, v in sample["values"] if first <= float(t) <= last]
                if [t for t, _ in values] != times or any(not math.isfinite(v) or v < 0 for _, v in values):
                    raise ValueError("incomplete_or_invalid_counter_samples")
                numbers = [v for _, v in values]
                if metric == "va_prediction_requests_created":
                    if len(set(numbers)) != 1 or outcome in created:
                        raise ValueError("counter_restart")
                    created[outcome] = numbers[0]
                    continue
                if any(b < a for a, b in zip(numbers, numbers[1:])):
                    raise ValueError("counter_reset")
                delta = numbers[-1] - numbers[0]
                if metric == "va_prediction_requests_total":
                    if outcome in counters:
                        raise ValueError("ambiguous_counter_series")
                    counters[outcome] = delta
                elif metric == "va_prediction_duration_seconds_bucket":
                    key = (outcome, labels["le"])
                    if key in buckets:
                        raise ValueError("ambiguous_histogram_series")
                    buckets[key] = delta
            if set(counters) != {"success", "server_error"} or set(created) != set(counters):
                raise ValueError("missing_counters")
            bounds = {bound for outcome, bound in buckets if outcome == "success"}
            if bounds != {bound for outcome, bound in buckets if outcome == "server_error"} or \
                    "+Inf" not in bounds:
                raise ValueError("missing_histogram_buckets")
            combined = {bound: sum(buckets[(outcome, bound)] for outcome in counters) for bound in bounds}
            count = sum(counters.values())
            if abs(combined["+Inf"] - count) > 1e-6:
                raise ValueError("counter_histogram_mismatch")
            window.update(samples=count, server_errors=counters["server_error"],
                          observed_start=first, observed_end=last)
            if count < policy["min_predictions_per_window"]:
                raise ValueError("insufficient_predictions")
            window.update(status="available", server_error_rate=counters["server_error"] / count,
                          p95_seconds=histogram_p95(combined), histogram=combined)
        except (ValueError, KeyError, TypeError) as error:
            window.update(unavailable(str(error)))
        result.append(window)
    return result


def fresh(document, now, age):
    try:
        return isinstance(document, dict) and document.get("status") == "available" and \
            0 <= now - timestamp(document["generated_at"]) <= age
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def evaluate(evidence, policy, drift_policy, now=None):
    """A breach wins; otherwise require all signals before declaring healthy."""
    now = time.time() if now is None else now
    if policy["consecutive_non_overlapping_windows"] != 2 or policy["readiness_failure_count"] < 1:
        raise ValueError("This health policy requires two non-overlapping windows")
    if any(not math.isfinite(v) or v <= 0 for key, v in policy.items() if key != "policy_status") or \
            any(policy[key] > 1 for key in ("server_error_rate_threshold", "map50_95_absolute_drop_threshold",
                                           "drift_and_server_error_rate_threshold")):
        raise ValueError("Invalid health thresholds")
    signals = {}
    identity = evidence["identity"]
    if evidence.get("deployment_changed"):
        return {"status": "unknown", "reasons": ["deployment_changed_during_collection"], "signals": {},
                "identity": identity, "generated_at": iso(now), "valid_until": iso(now), "action": "observe"}
    probes = evidence.get("readiness", [])
    latest = probes[-1] if probes else None
    started = evidence.get("runtime_started")
    if not latest or not 0 <= now - latest["at"] <= policy["readiness_max_age_seconds"]:
        signals["readiness"] = unavailable("missing_or_stale_probe")
    elif latest["ready"] is True:
        signals["readiness"] = {"status": "healthy"}
    elif started is None or now < started + policy["startup_grace_seconds"]:
        signals["readiness"] = unavailable("startup_grace_or_unknown_runtime_start")
    else:
        failures, previous = 0, None
        for probe in reversed(probes):
            if probe["ready"] is not False or probe["at"] < started + policy["startup_grace_seconds"]:
                break
            if previous is not None and not policy["readiness_interval_seconds"] <= previous - probe["at"] <= \
                    policy["readiness_interval_seconds"] + policy["max_scrape_gap_seconds"]:
                break
            failures += 1
            previous = probe["at"]
        signals["readiness"] = {"status": "unhealthy" if failures >= policy["readiness_failure_count"] else "unavailable",
                                "reason": "persistent_readiness_failure" if failures >= policy["readiness_failure_count"] else "insufficient_failure_history",
                                "consecutive_failures": failures}
    windows = evidence.get("windows", [])
    valid_windows = len(windows) == 2 and all(w.get("status") == "available" for w in windows)
    if valid_windows:
        valid_windows = windows[0]["end"] == windows[1]["start"] and \
            all(w["end"] - w["start"] == policy["window_seconds"] and
                w["samples"] >= policy["min_predictions_per_window"] and
                0 <= w["server_error_rate"] <= 1 for w in windows) and \
            0 <= now - windows[-1]["end"] <= policy["window_seconds"] + policy["max_scrape_gap_seconds"]
    if valid_windows:
        bad = [w["server_error_rate"] >= policy["server_error_rate_threshold"] for w in windows]
        signals["server_errors"] = {"status": "unhealthy" if all(bad) else "unavailable" if any(bad) else "healthy",
                                    "reason": "persistent_server_errors" if all(bad) else "pending_persistence" if any(bad) else "below_threshold"}
    else:
        signals["server_errors"] = unavailable("missing_stale_or_nonqualifying_windows")
    latency_valid = False
    try:
        baseline = evidence.get("latency_baseline")
        latency_valid = fresh(baseline, now, policy["latency_baseline_max_age_seconds"]) and \
            baseline.get("inference") == identity["inference"] and baseline.get("hardware") == evidence.get("hardware") and \
            math.isfinite(baseline["p95_seconds"]) and baseline["p95_seconds"] > 0
        if valid_windows and latency_valid and all(w.get("p95_seconds") is not None and
                math.isfinite(w["p95_seconds"]) and w["p95_seconds"] >= 0 for w in windows):
            limit = max(policy["latency_p95_floor_seconds"], policy["latency_baseline_multiplier"] * baseline["p95_seconds"])
            bad = [w["p95_seconds"] > limit for w in windows]
            signals["latency"] = {"status": "unhealthy" if all(bad) else "unavailable" if any(bad) else "healthy",
                                  "reason": "persistent_latency" if all(bad) else "pending_persistence" if any(bad) else "below_threshold",
                                  "threshold_seconds": limit}
        else:
            signals["latency"] = unavailable("missing_or_incompatible_baseline_or_windows")
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        signals["latency"] = unavailable("malformed_latency_evidence")
    try:
        audit = evidence.get("audit")
        if fresh(audit, now, policy["audit_max_age_seconds"]) and audit.get("candidate") == identity and \
                audit.get("baseline") == evidence.get("stable_baseline") and evidence.get("stable_baseline") and \
                audit.get("split_sha256") == evidence.get("split_sha256") and \
                audit.get("images", 0) >= policy["labeled_audit_min_images"] and \
                audit.get("baseline_batch_sha256") == audit.get("candidate_batch_sha256") and audit.get("baseline_batch_sha256"):
            before, after = audit["baseline_map50_95"], audit["candidate_map50_95"]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (before, after)):
                raise ValueError("Invalid labeled audit metrics")
            drop = before - after
            signals["quality"] = {"status": "unhealthy" if drop + 1e-12 >= policy["map50_95_absolute_drop_threshold"] else "healthy",
                                  "absolute_map50_95_drop": drop}
        else:
            signals["quality"] = unavailable("missing_stale_or_incompatible_stable_audit")
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        signals["quality"] = unavailable("malformed_quality_evidence")
    drift_valid = False
    try:
        drift = evidence.get("drift")
        usable_drift = isinstance(drift, dict) and (drift.get("status") == "available" or
                       drift.get("status") == "partial" and drift.get("drift_detected") is True)
        drift_valid = usable_drift and fresh({**drift, "status": "available"}, now, drift_policy["current_max_age_seconds"]) and \
            timestamp(drift["current_valid_until"]) >= now and drift.get("policy") == drift_policy and \
            drift.get("identity") == drift.get("current_identity") and \
            all(drift["identity"].get(k) == v for k, v in identity.items()) and \
            drift["identity"].get("split_sha256") == evidence.get("split_sha256") and \
            type(drift.get("drift_detected")) is bool
        if drift_valid:
            population = drift["populations"]["current"]
            first, last = timestamp(population["started_at"]), timestamp(population["completed_at"])
            drift_valid = 0 <= now - first <= drift_policy["current_max_age_seconds"] and \
                first <= last <= timestamp(drift["generated_at"]) <= now
        signals["drift"] = {"status": "available" if drift["status"] == "available" else "unavailable",
                            "detected": drift["drift_detected"], "reason": "complete" if drift["status"] == "available" else "partial_drift_evidence"} if drift_valid else unavailable("missing_stale_or_incompatible_drift")
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        signals["drift"] = unavailable("malformed_drift_evidence")
        drift_valid = False
    combined = False
    if valid_windows and drift_valid and drift["drift_detected"]:
        population = drift["populations"]["current"]
        combined = timestamp(population["started_at"]) <= windows[-1]["end"] and \
            timestamp(population["completed_at"]) >= windows[-1]["start"] and \
            all(w["server_error_rate"] >= policy["drift_and_server_error_rate_threshold"] for w in windows)
    signals["drift_and_errors"] = {"status": "unhealthy" if combined else "not_triggered"}
    breaches = [name for name, signal in signals.items() if signal["status"] == "unhealthy"]
    missing = [name for name, signal in signals.items() if signal["status"] == "unavailable"]
    status = "unhealthy" if breaches else "unknown" if missing else "healthy"
    expires = [now + policy["readiness_max_age_seconds"]]
    if latest and signals["readiness"]["status"] in {"healthy", "unhealthy"}:
        expires.append(latest["at"] + policy["readiness_max_age_seconds"])
    if valid_windows:
        expires.append(windows[-1]["end"] + policy["window_seconds"] + policy["max_scrape_gap_seconds"])
    if signals["latency"]["status"] in {"healthy", "unhealthy"}:
        expires.append(timestamp(baseline["generated_at"]) + policy["latency_baseline_max_age_seconds"])
    if signals["quality"]["status"] in {"healthy", "unhealthy"}:
        expires.append(timestamp(audit["generated_at"]) + policy["audit_max_age_seconds"])
    if drift_valid:
        expires.extend([timestamp(drift["current_valid_until"]), first + drift_policy["current_max_age_seconds"]])
    return {"status": status, "reasons": breaches or missing or ["all_required_signals_available_without_breach"],
            "signals": signals, "identity": identity, "deployment_id": evidence["deployment_id"],
            "generated_at": iso(now), "valid_until": iso(min(expires)),
            "action": "health_breach_recorded" if breaches else "observe", "policy": policy}


def read_optional(path):
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return unavailable("unreadable_evidence_file")


def collect(project, policy, observations=3):
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    receipt_path = project / "artifacts/serving/deployment.json"
    receipt = json.loads(receipt_path.read_text())
    identity = model_identity(receipt["model"])
    evidence = {"identity": identity, "deployment_id": receipt["container_id"], "readiness": [],
                "hardware": {"host": platform.node(), "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
                "split_sha256": hashlib.sha256((project / "data/splits.json").read_bytes()).hexdigest()}
    docker = str(Path(os.environ["LOCALAPPDATA"]) / "Programs/DockerDesktop/resources/bin/docker.exe")
    with requests.Session() as session:
        session.trust_env = False
        for index in range(observations):
            if index:
                time.sleep(policy["readiness_interval_seconds"])
            try:
                result = subprocess.run([docker, "inspect", receipt["container_id"], "--format", "{{.State.StartedAt}}"],
                                        capture_output=True, text=True, timeout=5, check=True)
                started = timestamp(result.stdout.strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                started = None
            if index and started != evidence.get("runtime_started"):
                evidence["readiness"] = []
            evidence["runtime_started"] = started
            try:
                response = session.get(f"http://127.0.0.1:{config['serving']['port']}/health", timeout=3)
                ready = response.status_code == 200 and response.json() == {"status": "ready", **receipt["model"]}
            except (requests.RequestException, ValueError):
                ready = False
            evidence["readiness"].append({"at": time.time(), "ready": ready})
        end = int(time.time() // policy["window_seconds"]) * policy["window_seconds"]
        length = policy["window_seconds"] * 2 + policy["max_scrape_gap_seconds"] * 2
        labels = f'model_version={json.dumps(identity["model_version"])},deployment_id={json.dumps(receipt["container_id"][:12])}'
        expressions = [f'up{{job="visual-analytics-api"}}[{length}s]',
                       f'{{__name__=~"va_prediction_requests_total|va_prediction_requests_created|va_prediction_duration_seconds_bucket",{labels}}}[{length}s]']
        try:
            series = []
            for expression in expressions:
                response = session.get(f"http://127.0.0.1:{config['monitoring']['prometheus_port']}/api/v1/query",
                                       params={"query": expression, "time": end}, timeout=5)
                response.raise_for_status()
                payload = response.json()
                if payload["status"] != "success" or payload["data"]["resultType"] != "matrix" or payload.get("warnings"):
                    raise ValueError("Prometheus did not return complete range vectors")
                series.extend(payload["data"]["result"])
            evidence["windows"] = traffic_windows(series, end, evidence["runtime_started"] or time.time(), policy)
            evidence["prometheus_series"] = series
        except (requests.RequestException, ValueError, KeyError) as error:
            evidence["windows"] = []
            evidence["operational_error"] = type(error).__name__
    evidence["deployment_changed"] = json.loads(receipt_path.read_text())["container_id"] != receipt["container_id"]
    try:
        from src.tracking.mlflow_utils import local_store
        client, _, registry = local_store(project)
        stable = client.get_model_version_by_alias(registry["model_name"], config["rollback"]["stable_alias"])
        evidence["stable_baseline"] = {"model_name": registry["model_name"], "model_version": str(stable.version),
                                       "checkpoint_sha256": stable.tags["checkpoint_sha256"]}
    except Exception as error:
        evidence["stable_baseline"] = None
        evidence["stable_baseline_error"] = type(error).__name__
    evidence["collected_at"] = iso(time.time())
    return evidence


def capture_latency_baseline(evidence, policy):
    windows = evidence["windows"]
    if len(windows) != 2 or not evidence["readiness"][-1]["ready"] or evidence.get("deployment_changed") or \
            any(w.get("status") != "available" or w["server_errors"] != 0 or w.get("p95_seconds") is None or
                not 0 < w["p95_seconds"] <= policy["latency_p95_floor_seconds"] for w in windows):
        raise ValueError("Latency calibration requires two qualifying error-free windows below the latency floor")
    return {"status": "available", "generated_at": evidence["collected_at"], "hardware": evidence["hardware"],
            "inference": evidence["identity"]["inference"], "source_model": evidence["identity"],
            "p95_seconds": max(w["p95_seconds"] for w in windows), "source_windows": windows,
            "note": "Initial operational calibration; does not approve model quality or stable promotion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "latency-baseline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drift", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--latency-baseline", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    evidence = collect(project, config["health"], config["health"]["readiness_failure_count"])
    if args.command == "latency-baseline":
        result = capture_latency_baseline(evidence, config["health"])
    else:
        evidence.update(drift=read_optional(args.drift), audit=read_optional(args.audit),
                        latency_baseline=read_optional(args.latency_baseline))
        try:
            result = evaluate(evidence, config["health"], config["drift"])
        except (ValueError, KeyError, TypeError, OverflowError):
            result = {"status": "unknown", "reasons": ["malformed_evidence"], "action": "observe"}
        result["evidence"] = evidence
    save(args.output, result)
    print(json.dumps({"status": result["status"], "reasons": result.get("reasons", []), "output": str(args.output)}))


if __name__ == "__main__":
    main()
