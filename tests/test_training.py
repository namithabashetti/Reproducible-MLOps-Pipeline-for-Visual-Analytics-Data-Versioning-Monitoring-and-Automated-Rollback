"""Check held-out isolation and metric recording without claiming mocked training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml
import numpy as np

from src.training.train import prepare_data
from src.training.evaluate import evaluate


class TrainingTests(unittest.TestCase):
    def test_smoke_inputs_cover_classes_and_exclude_held_out_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/processed"
            manifest = {"seed": 42, "splits": {}}
            for split in ("train", "validation", "test", "monitoring_reference", "monitoring_current"):
                names = [f"{split}{i}" for i in range(70)]
                manifest["splits"][split] = names
                for category, suffix in (("images", ".png"), ("labels", ".txt")):
                    folder = data / category / split
                    folder.mkdir(parents=True)
                    for i, name in enumerate(names):
                        (folder / f"{name}{suffix}").write_text(f"{i % 3} 0.5 0.5 0.2 0.2\n")
            (root / "data/splits.json").write_text(json.dumps(manifest))
            (data / "dataset.yaml").write_text(yaml.safe_dump({
                "train": "images/train", "val": "images/validation",
                "names": {0: "helmet", 1: "head", 2: "person"}}))
            staged = root / "staged"
            config, counts = prepare_data(root, staged, smoke=True)
            self.assertEqual(counts, {"train": 64, "validation": 32})
            self.assertEqual(set(p.name for p in (staged / "images").iterdir()), {"train", "validation"})
            self.assertNotIn("test", yaml.safe_load(config.read_text()))
            for split in counts:
                observed = {int(p.read_text().split()[0]) for p in (staged / "labels" / split).glob("*.txt")}
                self.assertEqual(observed, {0, 1, 2})
            second = root / "second"
            prepare_data(root, second, smoke=True)
            self.assertEqual((staged / "selection.json").read_bytes(), (second / "selection.json").read_bytes())
            label = next((staged / "labels/train").glob('*.txt'))
            original = data / "labels/train" / label.name
            before = original.read_bytes()
            label.write_text("changed locally")
            self.assertEqual(original.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                prepare_data(root, staged, smoke=True)

    def test_evaluation_records_validation_and_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            weights = output / 'test.pt'
            weights.write_bytes(b'test checkpoint identity only')
            result = Mock()
            result.box.mp, result.box.mr, result.box.map50, result.box.map = 0.5, 0.4, 0.3, 0.2
            result.summary.return_value = [{"Class": "helmet", "Images": np.int64(4), "mAP50": np.float64(0.3)}]
            with patch.dict('sys.modules', {'ultralytics': Mock()}) as modules:
                model = modules['ultralytics'].YOLO.return_value
                model.val.return_value = result
                report = evaluate(weights, output/'data.yaml', output, 'cpu', 416, 4)
                self.assertEqual(model.val.call_args.kwargs['split'], 'val')
                self.assertEqual(report['evaluation_split'], 'validation')
                self.assertFalse(report['stable_model'])
                self.assertEqual(len(report['checkpoint_sha256']), 64)
                result.box.map = float('nan')
                with self.assertRaisesRegex(ValueError, 'invalid detection metrics'):
                    evaluate(weights, output/'data.yaml', output, 'cpu', 416, 4)


if __name__ == '__main__':
    unittest.main()
