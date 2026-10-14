"""Distribution evidence, exclusion/freshness gates, and real Evidently export."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from src.monitoring.drift import DEFINITION, FEATURES, collect, compare, export_report, image_features, psi
from PIL import Image


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.policy = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/config.yaml").read_text())["drift"]
        self.now = datetime.now(timezone.utc)
        self.reference = self.batch("reference", 120)
        self.current = self.batch("current", 60)

    def batch(self, pool, age):
        return {"schema_version": 1, "identity": {"feature_definition": DEFINITION, "model_version": "1"},
                "started_at": (self.now - timedelta(seconds=age)).isoformat(),
                "completed_at": (self.now - timedelta(seconds=age - 20)).isoformat(),
                "observations": [{"image_id": f"{pool}-{i}", "image_sha256": f"{pool}-hash-{i}",
                    "detection_count": 1, FEATURES[0]: i / 2, FEATURES[1]: i / 100, FEATURES[2]: .5 + i / 1000}
                    for i in range(250)]}

    def test_identical_and_known_shift_with_reference_only_bins(self):
        equal = compare(self.reference, self.current, self.policy, self.now)
        self.assertEqual(equal["status"], "available")
        self.assertFalse(equal["drift_detected"])
        edges = equal["features"][FEATURES[0]]["psi_edges"]
        for row in self.current["observations"]:
            row[FEATURES[0]] += 100
        shifted = compare(self.reference, self.current, self.policy, self.now)["features"][FEATURES[0]]
        self.assertTrue(shifted["psi_drift"])
        self.assertTrue(shifted["ks_drift"])
        self.assertEqual(edges, shifted["psi_edges"])
        for other in (np.zeros(250), np.full(250, 255.)):
            score, _ = psi(np.full(250, 100.), other, 10, .5)
            self.assertTrue(np.isfinite(score))
            self.assertGreater(score, .2)

    def test_empty_predictions_are_not_zero_confidence_or_healthy(self):
        for row in self.current["observations"][:51]:
            row["detection_count"], row[FEATURES[2]] = 0, None
        result = compare(self.reference, self.current, self.policy, self.now)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["drift_detected"])
        self.assertEqual(result["features"][FEATURES[2]]["current_samples"], 199)
        self.assertEqual(result["empty_prediction_rate"]["current"]["rate"], 51 / 250)
        self.current["observations"] = []
        result = compare(self.reference, self.current, self.policy, self.now)
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["empty_prediction_rate"]["current"]["rate"])

    def test_stale_overlapping_and_incompatible_evidence_unavailable(self):
        for mutation, reason in [
            (lambda b: b["identity"].update(model_version="2"), "incompatible_model_data_or_inference_identity"),
            (lambda b: b["observations"][0].update(image_id="reference-0"), "overlapping_populations"),
            (lambda b: b["observations"][0].update(image_sha256="reference-hash-0"), "overlapping_populations"),
        ]:
            current = deepcopy(self.current)
            mutation(current)
            result = compare(self.reference, current, self.policy, self.now)
            self.assertIsNone(result["drift_detected"])
            self.assertEqual(result["features"][FEATURES[0]]["reason"], reason)
        result = compare(self.reference, self.current, self.policy, self.now + timedelta(hours=2))
        self.assertEqual(result["features"][FEATURES[0]]["reason"], "stale_current_population")
        current = deepcopy(self.current)
        current["completed_at"] = (self.now + timedelta(seconds=1)).isoformat()
        self.assertEqual(compare(self.reference, current, self.policy, self.now)["status"], "unavailable")

    def test_corrupt_observations_rejected_and_image_definitions(self):
        for value in (None, float("nan"), -1, 256):
            current = deepcopy(self.current)
            current["observations"][0][FEATURES[0]] = value
            with self.assertRaises(ValueError):
                compare(self.reference, current, self.policy, self.now)
        current = deepcopy(self.current)
        current["observations"].append(current["observations"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare(self.reference, current, self.policy, self.now)
        features = image_features(np.full((16, 16, 3), 128, dtype=np.uint8), [])
        self.assertEqual(features, dict(zip(FEATURES, (128., 0., None))))

    def test_evidently_exports_use_the_same_scores(self):
        # Include legitimate missing confidence so the real library's filtering is exercised.
        self.current["observations"][0].update(detection_count=0, max_detection_confidence=None)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "report"
            result = export_report(self.reference, self.current, self.policy, directory, self.now)
            self.assertEqual(result["status"], "available")
            payload = json.loads((directory / "evidently.json").read_text())
            metrics = payload["metrics"]
            self.assertEqual(len(metrics), 6)
            expected = [result["features"][feature][key] for feature in FEATURES for key in ("psi", "ks_pvalue")]
            for metric, score in zip(metrics, expected):
                self.assertAlmostEqual(metric["value"], score)
            self.assertGreater((directory / "evidently.html").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                export_report(self.reference, self.current, self.policy, directory, self.now)
            unavailable = Path(temporary) / "stale"
            result = export_report(self.reference, self.current, self.policy, unavailable, self.now + timedelta(hours=2))
            self.assertEqual(result["status"], "unavailable")
            self.assertFalse((unavailable / "evidently.html").exists())

    def test_collector_rejects_changed_inference_without_saving_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            for directory in ("configs", "data", "dataset/images"):
                (project / directory).mkdir(parents=True)
            (project / "configs/config.yaml").write_text(yaml.safe_dump({
                "serving": {"port": 8000}, "dataset": {"raw_path": "dataset"}}))
            (project / "dataset.dvc").write_text("outs: [{md5: fixture}]\n")
            (project / "data/splits.json").write_text(json.dumps({"splits": {"monitoring_current": ["one"], "test": ["two"]}}))
            Image.new("RGB", (4, 4)).save(project / "dataset/images/one.png")
            manifest = {"status": "ready", "model_name": "test", "model_version": "1",
                        "checkpoint_sha256": "abc", "dataset_dvc_md5": "fixture",
                        "inference": {"max_image_pixels": 100}}
            prediction = {"model_name": "test", "model_version": "1", "checkpoint_sha256": "abc",
                          "inference_sha256": "different_settings", "detections": []}
            session = MagicMock()
            session.get.return_value.json.return_value = manifest
            session.post.return_value.json.return_value = prediction
            with patch("src.monitoring.drift.requests.Session") as factory:
                factory.return_value.__enter__.return_value = session
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    collect(project, "current", project / "observations.json")
            self.assertFalse((project / "observations.json").exists())


if __name__ == "__main__":
    unittest.main()
