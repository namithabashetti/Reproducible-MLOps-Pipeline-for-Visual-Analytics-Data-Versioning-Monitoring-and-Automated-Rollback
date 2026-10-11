"""Exercise grouping, split isolation, and real image/annotation conversion."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from src.data.preprocess import letterbox, preprocess_dataset, yolo_labels
from src.data.split import assign_splits, find_groups
from src.data.validate import validate_dataset


class PreprocessingTests(unittest.TestCase):
    def test_padding_and_rounding_preserve_box_geometry_and_color(self):
        image = np.zeros((5, 8, 3), np.uint8)
        image[:] = (10, 40, 200)
        padded, transform = letterbox(image, 10, 114)
        self.assertEqual(padded.shape, (10, 10, 3))
        np.testing.assert_array_equal(padded[2:7, 1:9], image)
        self.assertTrue(np.all(padded[0] == 114))
        self.assertEqual(transform, (1, 1, 1, 2))
        row = yolo_labels([("helmet", (0, 0, 8, 5))], {"helmet": 0}, transform, 10).split()
        np.testing.assert_allclose([float(x) for x in row[1:]], [0.5, 0.45, 0.8, 0.5])
        _, transform = letterbox(np.zeros((7, 11, 3), np.uint8), 5, 114)
        row = yolo_labels([("helmet", (0, 0, 11, 7))], {"helmet": 0}, transform, 5).split()
        np.testing.assert_allclose([float(x) for x in row[1:]], [0.5, 0.5, 1, 0.6])
        self.assertEqual(yolo_labels([], {"helmet": 0}, transform, 5), "")

    def test_group_assignment_is_repeatable_complete_and_disjoint(self):
        groups = [[f"n{i}", f"n{i+1}"] for i in range(0, 10, 2)]
        groups += [[f"n{i}"] for i in range(10, 50)]
        labels = {f"n{i}": {"helmet": 1, "person": 1} for i in range(50)}
        targets = {"train": 30, "validation": 8, "test": 6, "monitoring_reference": 3,
                   "monitoring_current": 3}
        splits = assign_splits(groups, labels, targets, labels["n0"], 42)
        self.assertEqual(splits, assign_splits(list(reversed(groups)), labels, targets, labels["n0"], 42))
        self.assertEqual({s: len(n) for s, n in splits.items()}, targets)
        membership = {n: s for s, names in splits.items() for n in names}
        self.assertEqual(set(membership), set(labels))
        for group in groups:
            self.assertEqual(len({membership[n] for n in group}), 1)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            assign_splits([list(labels)], labels, targets, labels["n0"], 42)

    def test_real_outputs_raw_preservation_and_invalid_manifest_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "raw", root / "processed"
            (raw / "images").mkdir(parents=True)
            (raw / "annotations").mkdir()
            image = np.random.default_rng(42).integers(0, 256, (15, 16, 3), dtype=np.uint8)
            for i in range(5):
                sample = cv2.flip(image, 1) if i == 1 else image
                cv2.imwrite(str(raw / "images" / f"n{i}.png"), sample)
                (raw / "annotations" / f"n{i}.xml").write_text(
                    f'<annotation><filename>n{i}.png</filename><size><width>16</width>'
                    '<height>15</height></size><object><name>helmet</name><bndbox>'
                    '<xmin>0</xmin><ymin>0</ymin><xmax>16</xmax><ymax>15</ymax>'
                    '</bndbox></object></annotation>')
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.rglob('*') if p.is_file()}
            groups, _, pairs = find_groups(raw, {"helmet"}, 0)
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(pairs), 10)
            # Geometry fixture assigns independently; grouping is tested above.
            splits = dict(zip(("train", "validation", "test", "monitoring_reference", "monitoring_current"),
                              [[f"n{i}"] for i in range(5)]))
            manifest = {"splits": splits, "related_groups": [], "raw_content_manifest_sha256":
                        validate_dataset(raw, {"helmet"})["content_manifest_sha256"]}
            with self.assertRaisesRegex(ValueError, "separate"):
                preprocess_dataset(raw, raw / "derived", manifest, {0: "helmet"}, 16, 114)
            invalid = json.loads(json.dumps(manifest))
            invalid["related_groups"] = [["n0", "n1"]]
            with self.assertRaisesRegex(ValueError, "crosses"):
                preprocess_dataset(raw, output, invalid, {0: "helmet"}, 16, 114)
            invalid["related_groups"] = []
            invalid["splits"]["test"] = ["n0"]
            with self.assertRaisesRegex(ValueError, "exactly once"):
                preprocess_dataset(raw, output, invalid, {0: "helmet"}, 16, 114)
            invalid = {**manifest, "raw_content_manifest_sha256": "stale"}
            with self.assertRaisesRegex(ValueError, "Raw content changed"):
                preprocess_dataset(raw, output, invalid, {0: "helmet"}, 16, 114)
            report = preprocess_dataset(raw, output, manifest, {0: "helmet"}, 16, 114)
            self.assertEqual(sum(v["images"] for v in report["statistics"].values()), 5)
            converted = cv2.imread(str(output / "images/train/n0.png"))
            np.testing.assert_array_equal(converted[:15], image)
            self.assertTrue(np.all(converted[15] == 114))
            self.assertEqual((output / "labels/train/n0.txt").read_text(),
                             '0 0.500000000 0.468750000 1.000000000 0.937500000\n')
            config = yaml.safe_load((output / "dataset.yaml").read_text())
            self.assertEqual(set(config), {"train", "val", "names"})
            repeated = root / "repeated"
            self.assertEqual(report, preprocess_dataset(raw, repeated, manifest, {0: "helmet"}, 16, 114))
            self.assertEqual({p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()},
                             {p.relative_to(repeated): p.read_bytes() for p in repeated.rglob('*') if p.is_file()})
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
            with self.assertRaisesRegex(ValueError, "not empty"):
                preprocess_dataset(raw, output, manifest, {0: "helmet"}, 16, 114)


if __name__ == "__main__":
    unittest.main()
