"""Health decision matrix and raw Prometheus sample handling; no rollback mocks."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import yaml

from src.monitoring.health import evaluate, iso, traffic_windows, histogram_p95, capture_latency_baseline
from src.monitoring.audit import labeled_audit


class HealthTests(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/config.yaml").read_text())
        self.policy, self.drift_policy = config["health"], config["drift"]
        self.now = 250.
        identity = {"model_name": "fixture", "model_version": "2", "checkpoint_sha256": "candidate",
                    "dataset_dvc_md5": "raw", "inference": {"device": "cpu", "cpu_threads": 2}}
        stable = {"model_name": "fixture", "model_version": "1", "checkpoint_sha256": "stable"}
        drift_identity = {**identity, "split_sha256": "split"}
        self.evidence = {"identity": identity, "deployment_id": "container", "runtime_started": -1000,
            "split_sha256": "split", "hardware": {"host": "fixture"}, "collected_at": iso(self.now),
            "stable_baseline": stable, "readiness": [{"at": t, "ready": True} for t in (230, 240, 250)],
            "windows": traffic_windows(self.series(), 240, -1000, self.policy),
            "latency_baseline": {"status": "available", "generated_at": iso(200), "p95_seconds": .1,
                                 "inference": identity["inference"], "hardware": {"host": "fixture"}},
            "audit": {"status": "available", "generated_at": iso(200), "candidate": identity,
                      "baseline": stable, "images": 250, "split_sha256": "split",
                      "baseline_batch_sha256": "same_batch", "candidate_batch_sha256": "same_batch",
                      "baseline_map50_95": .8, "candidate_map50_95": .79},
            "drift": {"status": "available", "generated_at": iso(240), "current_valid_until": iso(3770),
                      "policy": self.drift_policy, "identity": drift_identity, "current_identity": drift_identity,
                      "drift_detected": False, "populations": {"current": {"started_at": iso(170), "completed_at": iso(230)}}}}

    def series(self):
        times = list(range(-20, 241, 10))
        series = [{"metric": {"__name__": "up"}, "values": [[t, "1"] for t in times]}]
        for outcome, amount in (("success", 10), ("server_error", 0), ("client_error", 1000)):
            labels = {"outcome": outcome}
            series.append({"metric": {**labels, "__name__": "va_prediction_requests_total"},
                           "values": [[t, str((t + 20) / 10 * amount)] for t in times]})
            series.append({"metric": {**labels, "__name__": "va_prediction_requests_created"},
                           "values": [[t, "1"] for t in times]})
            for bound in ("0.1", "0.2", "+Inf"):
                series.append({"metric": {**labels, "__name__": "va_prediction_duration_seconds_bucket", "le": bound},
                               "values": [[t, str((t + 20) / 10 * amount)] for t in times]})
        return series

    def decision(self):
        return evaluate(self.evidence, self.policy, self.drift_policy, self.now)

    def test_complete_healthy_and_drift_alone_never_unhealthy(self):
        self.assertEqual(self.decision()["status"], "healthy")
        self.evidence["drift"]["drift_detected"] = True
        self.assertEqual(self.decision()["status"], "healthy")
        self.assertEqual(self.decision()["action"], "observe")
        self.evidence["audit"] = None
        self.assertEqual(self.decision()["status"], "unknown")

    def test_persistent_error_latency_and_combined_thresholds(self):
        for name in ("server", "latency", "combined"):
            with self.subTest(name=name):
                e = deepcopy(self.evidence)
                for window in e["windows"]:
                    if name == "latency":
                        window["p95_seconds"] = 2.01
                    else:
                        window["server_error_rate"] = .05 if name == "server" else .02
                e["drift"]["drift_detected"] = name == "combined"
                decision = evaluate(e, self.policy, self.drift_policy, self.now)
                self.assertEqual(decision["status"], "unhealthy")
                if name == "combined":
                    self.assertEqual(decision["reasons"], ["drift_and_errors"])
                e["windows"][0].update(server_error_rate=0, p95_seconds=.1)
                self.assertNotEqual(evaluate(e, self.policy, self.drift_policy, self.now)["status"], "unhealthy")
        self.evidence["windows"][0]["p95_seconds"] = 2
        self.evidence["windows"][1]["p95_seconds"] = 2
        self.assertEqual(self.decision()["signals"]["latency"]["status"], "healthy")

    def test_moderate_errors_require_fresh_overlapping_drift(self):
        for window in self.evidence["windows"]:
            window["server_error_rate"] = .03
        self.assertEqual(self.decision()["status"], "healthy")
        self.evidence["drift"]["drift_detected"] = True
        self.evidence["drift"]["populations"]["current"] = {"started_at": iso(0), "completed_at": iso(100)}
        self.assertEqual(self.decision()["status"], "healthy")
        self.evidence["drift"]["populations"]["current"] = {"started_at": iso(170), "completed_at": iso(230)}
        self.evidence["drift"]["status"] = "partial"
        self.assertEqual(self.decision()["status"], "unhealthy")

    def test_readiness_grace_spacing_and_missing_other_evidence(self):
        for probe in self.evidence["readiness"]:
            probe["ready"] = False
        self.evidence.update(audit=None, latency_baseline=None, drift=None, windows=[])
        self.assertEqual(self.decision()["reasons"], ["readiness"])
        self.evidence["runtime_started"] = 220
        self.assertEqual(self.decision()["status"], "unknown")
        self.evidence["runtime_started"] = -1000
        self.evidence["readiness"][0]["at"] = 100
        self.assertEqual(self.decision()["status"], "unknown")
        self.evidence["readiness"][0]["at"] = 239
        self.assertEqual(self.decision()["status"], "unknown")
        self.evidence["readiness"][0]["at"] = 230
        self.evidence["audit"] = {"status": "available", "generated_at": iso(200), "candidate": self.evidence["identity"]}
        self.assertEqual(self.decision()["status"], "unhealthy")

    def test_quality_drop_needs_stable_same_batch_and_minimum_images(self):
        self.evidence["audit"].update(baseline_map50_95=.6, candidate_map50_95=.55)
        self.assertEqual(self.decision()["reasons"], ["quality"])
        for key, value in (("images", 199), ("candidate_batch_sha256", "different"), ("baseline", {"model_version": "wrong"}),
                           ("generated_at", iso(-4000)), ("candidate_map50_95", float("nan"))):
            audit = deepcopy(self.evidence["audit"])
            self.evidence["audit"][key] = value
            self.assertEqual(self.decision()["status"], "unknown")
            self.evidence["audit"] = audit
        self.evidence["stable_baseline"] = None
        self.assertEqual(self.decision()["status"], "unknown")

    def test_stale_incompatible_overlapping_and_low_volume_are_unknown(self):
        for field in ("drift", "audit", "latency_baseline"):
            original = self.evidence[field]
            self.evidence[field] = None
            self.assertEqual(self.decision()["status"], "unknown")
            self.evidence[field] = original
        for update in ({"start": 100}, {"samples": 99}, {"end": 130}):
            original = deepcopy(self.evidence["windows"])
            self.evidence["windows"][1].update(update)
            self.assertEqual(self.decision()["status"], "unknown")
            self.evidence["windows"] = original
        self.evidence["latency_baseline"]["hardware"] = {"host": "other"}
        self.assertEqual(self.decision()["status"], "unknown")
        self.evidence["deployment_changed"] = True
        self.assertEqual(self.decision()["reasons"], ["deployment_changed_during_collection"])

    def test_raw_counter_windows_exclude_clients_and_reject_gaps_resets(self):
        windows = self.evidence["windows"]
        self.assertEqual([w["samples"] for w in windows], [120, 120])
        self.assertEqual(windows[0]["observed_end"], windows[1]["observed_start"])
        self.assertAlmostEqual(windows[0]["p95_seconds"], .095)
        for mode in ("gap", "outage", "reset", "restart", "missing"):
            data = self.series()
            if mode == "gap":
                data[0]["values"] = [v for v in data[0]["values"] if v[0] != 60]
            elif mode == "outage":
                data[0]["values"][8][1] = "0"
            elif mode == "reset":
                data[1]["values"][8][1] = "0"
            elif mode == "restart":
                data[2]["values"][8][1] = "2"
            else:
                data.pop(1)
            self.assertEqual(traffic_windows(data, 240, -1000, self.policy)[0]["status"], "unavailable")
        self.assertEqual(traffic_windows(self.series(), 240, 10, self.policy)[0]["status"], "unavailable")
        self.assertIsNone(histogram_p95({"1": 1, "+Inf": 100}))
        with self.assertRaises(ValueError):
            histogram_p95({".1": 100, "1": 1, "+Inf": 100})

    def test_latency_calibration_requires_measured_clean_windows(self):
        baseline = capture_latency_baseline(self.evidence, self.policy)
        self.assertAlmostEqual(baseline["p95_seconds"], .095)
        self.evidence["windows"][1]["server_errors"] = 1
        with self.assertRaises(ValueError):
            capture_latency_baseline(self.evidence, self.policy)

    def test_audit_refuses_to_invent_a_stable_model(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "artifacts/serving").mkdir(parents=True)
            (root / "configs/config.yaml").write_text("rollback: {stable_alias: stable}\n")
            (root / "artifacts/serving/deployment.json").write_text(json.dumps({"model": self.evidence["identity"]}))
            client = MagicMock()
            client.get_registered_model.return_value.aliases = {}
            with patch("src.tracking.mlflow_utils.local_store", return_value=(client, None, {"model_name": "fixture"})), \
                    patch("src.serving.deploy.stage_model") as stage:
                result = labeled_audit(root, root / "audit")
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "no_approved_stable_baseline")
            stage.assert_not_called()
            self.assertTrue((root / "audit/audit.json").exists())

    def test_audit_compares_identical_isolated_labeled_inputs(self):
        import json
        import os
        from types import SimpleNamespace
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            root = Path(temporary)
            for directory in ("configs", "artifacts/serving", "artifacts/ultralytics-settings", "data/processed/images/monitoring_current",
                              "data/processed/labels/monitoring_current", "baseline", "candidate"):
                (root / directory).mkdir(parents=True)
            os.environ["YOLO_CONFIG_DIR"] = str(root / "artifacts/ultralytics-settings")
            config = {"rollback": {"stable_alias": "stable"}, "health": {"labeled_audit_min_images": 200},
                      "dataset": {"classes": {0: "helmet"}}}
            (root / "configs/config.yaml").write_text(yaml.safe_dump(config))
            candidate = deepcopy(self.evidence["identity"])
            candidate["inference"]["image_size"] = 416
            candidate["deployment_mode"] = "experimental"
            baseline = deepcopy(candidate)
            baseline.update(model_version="1", checkpoint_sha256="stable")
            (root / "artifacts/serving/deployment.json").write_text(json.dumps({"container_id": "fixture", "model": candidate}))
            names = [f"image-{i}" for i in range(200)]
            (root / "data/splits.json").write_text(json.dumps({"splits": {"monitoring_current": names, "test": ["untouched"]}}))
            for name in names:
                Image.new("RGB", (8, 8)).save(root / f"data/processed/images/monitoring_current/{name}.png")
                (root / f"data/processed/labels/monitoring_current/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            client = MagicMock()
            client.get_registered_model.return_value.aliases = {"stable": "1"}
            with patch("src.tracking.mlflow_utils.local_store", return_value=(client, None, {"model_name": "fixture"})), \
                    patch("src.serving.deploy.stage_model", side_effect=[(root / "baseline", baseline), (root / "candidate", candidate)]), \
                    patch("src.monitoring.audit.subprocess.check_output", return_value="{}"), \
                    patch("ultralytics.YOLO") as yolo, patch("ultralytics.settings.update"), patch("torch.set_num_threads"):
                yolo.return_value.val.side_effect = [SimpleNamespace(box=SimpleNamespace(map=.8)), SimpleNamespace(box=SimpleNamespace(map=.7))]
                report = labeled_audit(root, root / "audit")
            self.assertEqual(report["images"], 200)
            self.assertEqual(report["baseline_batch_sha256"], report["candidate_batch_sha256"])
            self.assertEqual(report["candidate_map50_95"], .7)
            calls = yolo.return_value.val.call_args_list
            self.assertEqual(calls[0].kwargs["data"], calls[1].kwargs["data"])
            self.assertTrue(Path(calls[0].kwargs["data"]).is_relative_to(root / "audit/inputs"))
            self.assertFalse(list((root / "data").rglob("*.cache")))


if __name__ == "__main__":
    unittest.main()
