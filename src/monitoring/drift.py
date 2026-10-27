"""Collect controlled serving observations and export local Evidently PSI/KS evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
import yaml

from src.serving.api import decode_image


FEATURES = ("mean_grayscale_intensity", "log_laplacian_variance", "max_detection_confidence")
DEFINITION = {
    "version": 1,
    "image": "EXIF-oriented decoded BGR, original resolution, before resize/padding",
    "mean_grayscale_intensity": "mean(cv2.cvtColor(image, COLOR_BGR2GRAY)), range 0..255",
    "log_laplacian_variance": "ln(1 + variance(cv2.Laplacian(gray, CV_64F, ksize=1, BORDER_DEFAULT)))",
    "max_detection_confidence": "maximum confidence after serving NMS/threshold; null if no detections",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def image_features(image, detections):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return dict(zip(FEATURES, [float(gray.mean()),
                             float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())),
                             max((float(d["confidence"]) for d in detections), default=None)]))


def collect(project, pool, output):
    """Replay only a reserved monitoring pool through the actual localhost API."""
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    split_bytes = (project / "data/splits.json").read_bytes()
    splits = json.loads(split_bytes)["splits"]
    selected = splits[f"monitoring_{pool}"]
    if set(selected) & {item for name, items in splits.items()
                        if name != f"monitoring_{pool}" for item in items}:
        raise ValueError("Monitoring membership overlaps another split")
    base = f"http://127.0.0.1:{config['serving']['port']}"
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(base + "/health", timeout=15)
        response.raise_for_status()
        manifest = response.json()
        if manifest.pop("status") != "ready":
            raise ValueError("Serving model is not ready")
        identity = {key: manifest[key] for key in
                    ("model_name", "model_version", "checkpoint_sha256", "inference", "dataset_dvc_md5")}
        raw_md5 = yaml.safe_load((project / "dataset.dvc").read_text())["outs"][0]["md5"]
        # Training lineage and the replay population have separate identities.
        # Requiring equality made expanded-data checkpoints impossible to monitor.
        identity['monitoring_dataset_dvc_md5'] = raw_md5
        identity["split_sha256"] = hashlib.sha256(split_bytes).hexdigest()
        identity["feature_definition"] = DEFINITION
        identity["feature_opencv_version"] = cv2.__version__
        batch = {"schema_version": 1, "source": "controlled_monitoring_pool_replay",
                 "pool": pool, "identity": identity, "started_at": utc_now(), "observations": []}
        for index, name in enumerate(selected, 1):
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Invalid image identifier")
            body = (project / config["dataset"]["raw_path"] / "images" / f"{name}.png").read_bytes()
            image = decode_image(body, manifest["inference"]["max_image_pixels"])
            response = session.post(base + "/predict", data=body,
                                    headers={"Content-Type": "image/png"}, timeout=120)
            response.raise_for_status()  # Failed requests never become empty predictions.
            prediction = response.json()
            if any(prediction[key] != identity[key] for key in
                   ("model_name", "model_version", "checkpoint_sha256")) or \
                    prediction["inference_sha256"] != digest(identity["inference"]):
                raise ValueError("Model or inference settings changed during observation collection")
            batch["observations"].append({"image_id": name, "image_sha256": hashlib.sha256(body).hexdigest(),
                "detection_count": len(prediction["detections"]), **image_features(image, prediction["detections"])})
            if index % 25 == 0:
                print(f"{pool}: {index}/{len(selected)} successful observations", flush=True)
        batch["completed_at"] = utc_now()
        batch["manifest"] = manifest
        batch["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
        batch["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project))
        batch["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output, batch)
    return batch


def psi(reference, current, bins, pseudocount):
    """Reference-only quantile boundaries, explicit tail bins, additive count smoothing."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    # Keep the maximum inside a finite interval, including a constant reference.
    edges = np.r_[-np.inf, edges, np.nextafter(edges[-1], np.inf), np.inf]
    r, c = (np.histogram(values, bins=edges)[0] + pseudocount for values in (reference, current))
    r, c = r / r.sum(), c / c.sum()
    score = float(np.sum((c - r) * np.log(c / r)))
    return score, ["-inf" if np.isneginf(x) else "+inf" if np.isposinf(x) else float(x) for x in edges]


def validate_batch(batch):
    if batch["schema_version"] != 1 or batch["identity"]["feature_definition"] != DEFINITION:
        raise ValueError("Unsupported observation schema or feature definition")
    rows = batch["observations"]
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate observations would inflate sample counts")
    for row in rows:
        count = row["detection_count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid detection count")
        for feature in FEATURES:
            value = row[feature]
            if feature == FEATURES[2] and count == 0 and value is None:
                continue
            if value is None or not np.isfinite(value):
                raise ValueError("Missing/non-finite feature value")
            if feature == FEATURES[0] and not 0 <= value <= 255 or \
                    feature == FEATURES[1] and value < 0 or \
                    feature == FEATURES[2] and (count == 0 or not 0 <= value <= 1):
                raise ValueError("Feature value outside its valid domain")
    times = [datetime.fromisoformat(batch[key]) for key in ("started_at", "completed_at")]
    if any(t.tzinfo is None for t in times) or times[0] > times[1]:
        raise ValueError("Invalid observation timestamps")
    return times


def compare(reference, current, policy, now=None):
    """Return evidence, never a health verdict or rollback instruction."""
    now = now or datetime.now(timezone.utc)
    ref_start, ref_end = validate_batch(reference)
    cur_start, cur_end = validate_batch(current)
    if policy["ks_comparisons"] != len(FEATURES) or type(policy["psi_bins"]) is not int or policy["psi_bins"] < 2 or \
            not 0 < policy["psi_pseudocount"] < float("inf") or policy["min_samples_per_population"] < 2 or \
            not 0 < policy["ks_family_alpha"] < 1 or not 0 < policy["ks_min_statistic"] <= 1 or \
            not 0 < policy["psi_threshold"] < float("inf") or \
            min(policy["current_max_age_seconds"], policy["current_max_span_seconds"]) <= 0:
        raise ValueError("Invalid PSI/KS policy")
    reason = None
    if reference["identity"] != current["identity"]:
        reason = "incompatible_model_data_or_inference_identity"
    elif {r["image_id"] for r in reference["observations"]} & {r["image_id"] for r in current["observations"]} or \
            {r["image_sha256"] for r in reference["observations"]} & {r["image_sha256"] for r in current["observations"]}:
        reason = "overlapping_populations"
    elif ref_end > cur_start or cur_end > now:
        reason = "invalid_population_time_order"
    elif (now - cur_start).total_seconds() > policy["current_max_age_seconds"]:
        reason = "stale_current_population"
    elif (cur_end - cur_start).total_seconds() > policy["current_max_span_seconds"]:
        reason = "current_window_too_long"
    result = {"generated_at": now.isoformat(), "identity": reference["identity"],
              "current_identity": current["identity"],
              "populations": {name: {key: batch.get(key) for key in
                                      ("source", "pool", "started_at", "completed_at")}
                              for name, batch in (("reference", reference), ("current", current))},
              "policy": policy, "features": {}, "empty_prediction_rate": {},
              "current_valid_until": datetime.fromtimestamp(cur_start.timestamp() + policy["current_max_age_seconds"], timezone.utc).isoformat(),
              "purpose": "distribution evidence only; confidence is not accuracy; no rollback decision"}
    for name, batch in (("reference", reference), ("current", current)):
        rows = batch["observations"]
        empty = sum(row["detection_count"] == 0 for row in rows)
        result["empty_prediction_rate"][name] = {"images": len(rows), "empty_images": empty,
                                                "rate": empty / len(rows) if rows else None}
    for feature in FEATURES:
        r, c = (np.array([row[feature] for row in batch["observations"] if row[feature] is not None])
                for batch in (reference, current))
        item = {"reference_samples": len(r), "current_samples": len(c)}
        unavailable = reason or ("insufficient_samples" if min(len(r), len(c)) < policy["min_samples_per_population"] else None)
        if unavailable:
            item.update(status="unavailable", reason=unavailable, drift_detected=None)
        else:
            score, edges = psi(r, c, policy["psi_bins"], policy["psi_pseudocount"])
            ks = ks_2samp(r, c, alternative="two-sided", method="auto")
            psi_drift = score >= policy["psi_threshold"]
            ks_drift = bool(ks.pvalue < policy["ks_family_alpha"] / policy["ks_comparisons"] and ks.statistic >= policy["ks_min_statistic"])
            item.update(status="available", psi=score, psi_edges=edges, psi_drift=psi_drift,
                        ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue), ks_drift=ks_drift,
                        drift_detected=psi_drift or ks_drift)
        result["features"][feature] = item
    available = sum(item["status"] == "available" for item in result["features"].values())
    result["status"] = "available" if available == len(FEATURES) else "partial" if available else "unavailable"
    result["drift_detected"] = True if any(item["drift_detected"] for item in result["features"].values()) else False if result["status"] == "available" else None
    return result


def export_report(reference, current, policy, output, now=None):
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"
    from evidently import ColumnType, DataDefinition, Dataset, Report
    from evidently.metrics import ValueDrift
    from evidently.legacy.calculations.stattests import StatTest, register_stattest

    result = compare(reference, current, policy, now)
    output.mkdir(parents=True, exist_ok=False)
    result["observation_sha256"] = {"reference": digest(reference), "current": digest(current)}
    result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["packages"] = {name: importlib.metadata.version(name) for name in ("evidently", "scipy", "numpy", "pandas", "opencv-python")}
    available = [feature for feature, item in result["features"].items() if item["status"] == "available"]
    if available:
        # Evidently runs exactly the project's fixed binning/smoothing and combined KS gate.
        def psi_test(r, c, feature_type, threshold):
            score, _ = psi(r.to_numpy(), c.to_numpy(), policy["psi_bins"], policy["psi_pseudocount"])
            return score, score >= threshold

        def ks_test(r, c, feature_type, threshold):
            ks = ks_2samp(r, c, alternative="two-sided", method="auto")
            return float(ks.pvalue), bool(ks.pvalue < threshold and ks.statistic >= policy["ks_min_statistic"])

        for name, label, fn in (("va_psi", "PSI: reference quantiles, smoothed tails", psi_test),
                                ("va_ks", "KS: Bonferroni alpha and minimum effect", ks_test)):
            register_stattest(StatTest(name=name, display_name=label, allowed_feature_types=[ColumnType.Numerical]), default_impl=fn)
        definition = DataDefinition(numerical_columns=available)
        datasets = [Dataset.from_pandas(pd.DataFrame(batch["observations"])[available], data_definition=definition)
                    for batch in (reference, current)]
        metrics = [ValueDrift(column=feature, method=method, threshold=threshold)
                   for feature in available for method, threshold in
                   (("va_psi", policy["psi_threshold"]), ("va_ks", policy["ks_family_alpha"] / policy["ks_comparisons"]))]
        snapshot = Report(metrics, include_tests=False).run(current_data=datasets[1], reference_data=datasets[0])
        snapshot.save_html(str(output / "evidently.html"))
        snapshot.save_json(str(output / "evidently.json"))
        result["evidently_features"] = available
    else:
        result["evidently_features"] = []  # Never fabricate a chart for unavailable evidence.
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--pool", choices=("reference", "current"), required=True)
    capture.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--reference", type=Path, required=True)
    report.add_argument("--current", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    if args.command == "collect":
        collect(project, args.pool, args.output)
    else:
        policy = yaml.safe_load((project / "configs/config.yaml").read_text())["drift"]
        result = export_report(json.loads(args.reference.read_text()), json.loads(args.current.read_text()), policy, args.output)
        print(json.dumps({"status": result["status"], "drift_detected": result["drift_detected"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
