"""Real temporary SQLite/registry tests with tiny synthetic evidence, not model-quality tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlparse
from urllib.request import url2pathname

import yaml

from src.tracking.mlflow_utils import (file_hash, local_store, log_completed_run,
                                       read_completed_run, verify_registration)
import mlflow


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.previous_uris = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        self.temporary = tempfile.TemporaryDirectory(prefix="visual analytics tracking-")
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/config.yaml").write_text(yaml.safe_dump({"mlflow": {
            "model_name": "test-detector", "backend_store_uri": "sqlite:///artifacts/mlflow/test.db",
            "artifact_directory": "artifacts/mlflow/files", "port": 5000}}))
        (self.root / "requirements-lock.txt").write_text("test environment only\n")
        self.directory = self.make_run("baseline")

    def make_run(self, name):
        directory = self.root / "artifacts/training" / name
        for subdir in ("fit/weights", "inputs", "source"):
            (directory / subdir).mkdir(parents=True)
        (directory / "fit/weights/best.pt").write_bytes(b"synthetic checkpoint identity: " + name.encode())
        (directory / "fit/results.csv").write_text("epoch,time,train/box_loss,metrics/mAP50(B)\n1,1.5,2,0.2\n2,3,1.5,0.3\n")
        (directory / "fit/args.yaml").write_text("epochs: 2\n")
        (directory / "inputs/selection.json").write_text(json.dumps({"train": ["a"], "validation": ["b"]}))
        (directory / "inputs/dataset.yaml").write_text("train: images/train\nval: images/validation\n")
        (directory / "source/train.py").write_text("# Synthetic test fixture, not training code.\n")
        (directory / "source/dataset.dvc").write_text("outs:\n- md5: raw.dir\n")
        (directory / "source/dvc.lock").write_text(yaml.safe_dump({"stages": {
            "split": {"outs": [{"md5": "split"}]}, "preprocess": {"outs": [{"md5": "processed.dir"}]}}}))
        lineage = {"smoke_test": False, "epochs_requested": 2, "dataset_dvc_md5": "raw.dir",
                   "processed_dvc_md5": "processed.dir", "splits_dvc_md5": "split", "git_commit": "fixture",
                   "git_worktree_dirty": True, "started_at_utc": "2026-01-01T00:00:00+00:00",
                   "parameters": {"model": "yolov8n.pt", "epochs": 2}, "seed": 42,
                   "input_counts": {"train": 1, "validation": 1}, "hardware": {"device": "cpu"},
                   "ultralytics": "fixture", "file_sha256": {p.name: file_hash(p) for p in (directory / "source").iterdir()}}
        (directory / "lineage.json").write_text(json.dumps(lineage))
        report = {"evaluation_split": "validation", "stable_model": False, "smoke_test": False,
                  "epochs_completed": 2, "finite_training_losses": True, "input_counts": lineage["input_counts"],
                  "checkpoint_sha256": file_hash(directory / "fit/weights/best.pt"),
                  "metrics": {"precision": 0.5, "recall": 0.4, "map50": 0.3, "map50_95": 0.2},
                  "per_class": [{"Class": "helmet", "Box-P": 0.5, "Box-R": 0.4, "mAP50": 0.3, "mAP50-95": 0.2}]}
        (directory / "metrics.json").write_text(json.dumps(report))
        return directory

    def tearDown(self):
        # Release SQLite handles before Windows removes each isolated test directory.
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingStore
        from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
        for store in (TrackingStore, RegistryStore):
            for uri, engine in list(store._engine_map.items()):
                if self.root.as_posix() in uri:
                    engine.dispose()
                    del store._engine_map[uri]
        mlflow.set_tracking_uri(self.previous_uris[0])
        mlflow.set_registry_uri(self.previous_uris[1])
        self.temporary.cleanup()

    def test_import_retry_retrieval_and_existing_stable_alias(self):
        first = log_completed_run(self.root, self.directory)
        self.assertTrue(first["verified"])
        self.assertEqual(first["registry_aliases"], {})
        (self.directory / "mlflow.json").unlink()
        retry = log_completed_run(self.root, self.directory)
        self.assertEqual((first["run_id"], first["model_version"]), (retry["run_id"], retry["model_version"]))
        client, _, _ = local_store(self.root)
        # This alias exists only in the disposable test registry, never the project registry.
        client.set_registered_model_alias(first["model_name"], "stable", first["model_version"])
        second = log_completed_run(self.root, self.make_run("second"))
        self.assertEqual(second["registry_aliases"]["stable"], first["model_version"])
        self.assertNotEqual(second["model_version"], first["model_version"])
        self.assertEqual(len(client.search_runs([first["experiment_id"]])), 2)
        self.assertEqual(len(client.search_model_versions("name = 'test-detector'")), 2)
        artifact_root = Path(url2pathname(urlparse(client.get_run(first["run_id"]).info.artifact_uri).path))
        stored = (artifact_root / "model/best.pt").resolve()
        self.assertTrue(stored.is_relative_to(self.root))
        stored.write_bytes(b"damaged registry artifact")
        with self.assertRaisesRegex(ValueError, "checkpoint content mismatch"):
            verify_registration(client, first["model_name"], first["model_version"], self.directory)

    def test_retry_after_registration_before_receipt(self):
        with patch("src.tracking.mlflow_utils.verify_registration", side_effect=ConnectionError("interrupted retrieval")):
            with self.assertRaises(ConnectionError):
                log_completed_run(self.root, self.directory)
        self.assertFalse((self.directory / "mlflow.json").exists())
        receipt = log_completed_run(self.root, self.directory)
        client, _, _ = local_store(self.root)
        self.assertEqual(len(client.search_runs([receipt["experiment_id"]])), 1)
        self.assertEqual(len(client.search_model_versions("name = 'test-detector'")), 1)

    def test_retry_after_partial_logging_keeps_one_epoch_history(self):
        with patch("src.tracking.mlflow_utils.MlflowClient.log_artifact", side_effect=OSError("interrupted copy")):
            with self.assertRaises(OSError):
                log_completed_run(self.root, self.directory)
        receipt = log_completed_run(self.root, self.directory)
        client, _, _ = local_store(self.root)
        self.assertEqual(len(client.search_runs([receipt["experiment_id"]])), 1)
        self.assertEqual(len(client.get_metric_history(receipt["run_id"], "epoch/train/box_loss")), 2)
        self.assertEqual(client.get_run(receipt["run_id"]).info.status, "FINISHED")

    def test_invalid_evidence_rejected_before_store_creation(self):
        checkpoint = self.directory / "fit/weights/best.pt"
        original = checkpoint.read_bytes()
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash"):
            log_completed_run(self.root, self.directory)
        self.assertFalse((self.root / "artifacts/mlflow").exists())
        checkpoint.write_bytes(original)
        report = json.loads((self.directory / "metrics.json").read_text())
        report["metrics"]["map50"] = float("nan")
        (self.directory / "metrics.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "finite values"):
            read_completed_run(self.directory)

    def test_serving_stages_exact_version_and_refuses_corruption(self):
        from src.serving.deploy import stage_model

        config_path = self.root / "configs/config.yaml"
        registry_config = yaml.safe_load(config_path.read_text())["mlflow"]
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/config.yaml").read_text())
        config["mlflow"] = registry_config
        config_path.write_text(yaml.safe_dump(config))
        parameters = self.directory / "source/params.yaml"
        parameters.write_text("preprocessing:\n  pad_value: 114\n")
        lineage_path = self.directory / "lineage.json"
        lineage = json.loads(lineage_path.read_text())
        lineage["parameters"]["image_size"] = 416
        lineage["file_sha256"]["params.yaml"] = file_hash(parameters)
        lineage_path.write_text(json.dumps(lineage))
        registered = log_completed_run(self.root, self.directory)
        staged, manifest = stage_model(self.root, self.root, registered["model_version"], experimental=True)
        self.assertEqual(file_hash(staged / "best.pt"), registered["checkpoint_sha256"])
        self.assertEqual(manifest["deployment_mode"], "experimental")
        client, _, _ = local_store(self.root)
        self.assertEqual(client.get_registered_model(registered["model_name"]).aliases, {})
        client.set_registered_model_alias(registered["model_name"], "stable", registered["model_version"])
        _, approved = stage_model(self.root, self.root)
        self.assertEqual(approved["model_version"], registered["model_version"])
        self.assertEqual(approved["deployment_mode"], "approved")
        (staged / "best.pt").write_bytes(b"corrupt local copy")
        with self.assertRaisesRegex(ValueError, "staged checkpoint is corrupt"):
            stage_model(self.root, self.root, registered["model_version"], experimental=True)


if __name__ == "__main__":
    unittest.main()
