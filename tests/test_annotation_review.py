"""Check correction provenance, complete review, and isolated YOLO export."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from scripts.review_annotations import check_review, digest, export_review, source_records


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        for folder in ("dataset/images", "dataset/annotations", "data", "docs"):
            (self.project / folder).mkdir(parents=True)
        for name in ("train1", "val1", "heldout1"):
            cv2.imwrite(str(self.project / "dataset/images" / f"{name}.png"), np.zeros((12, 16, 3), np.uint8))
            (self.project / "dataset/annotations" / f"{name}.xml").write_text(
                f'<annotation><filename>{name}.png</filename><size><width>16</width><height>12</height>'
                '</size><object><name>helmet</name><bndbox><xmin>2</xmin><ymin>1</ymin>'
                '<xmax>6</xmax><ymax>3</ymax></bndbox></object></annotation>')
        (self.project / "data/splits.json").write_text(json.dumps({"splits": {
            "train": ["train1"], "validation": ["val1"], "test": ["heldout1"],
            "monitoring_reference": [], "monitoring_current": []}}))
        (self.project / "docs/annotation-policy.md").write_text("fixture policy")
        (self.project / "params.yaml").write_text("training:\n  image_size: 16\npreprocessing:\n  pad_value: 114\n")
        self.review = {"schema_version": 1, "split_sha256": digest(self.project / "data/splits.json"),
                       "policy_sha256": digest(self.project / "docs/annotation-policy.md"),
                       "records": source_records(self.project)}
        self.output = self.project / "data/person-corrected-test"

    def reviewed(self):
        for row in self.review["records"]:
            row.update(status="reviewed", reviewer="fixture reviewer", reason="visible person added",
                       reviewed_at=datetime.now(timezone.utc).isoformat(), person_boxes=[[1, 0, 15, 12]])

    def test_unreviewed_or_unresolved_records_cannot_export(self):
        self.assertFalse(check_review(self.project, self.review)["ready_for_export"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_review(self.project, self.review, self.output)
        self.assertFalse(self.output.exists())
        self.reviewed()
        self.review["records"][0]["issues"] = ["unclear reflected person"]
        with self.assertRaisesRegex(ValueError, "no open issues"):
            export_review(self.project, self.review, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_duplicate_and_heldout_records_are_rejected(self):
        for records in (self.review["records"][:1], [self.review["records"][0]] * 2,
                        [{**self.review["records"][0], "image_id": "heldout1"}, self.review["records"][1]]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                check_review(self.project, {**self.review, "records": records})

    def test_changed_source_policy_and_split_invalidate_review(self):
        self.reviewed()
        for relative in ("dataset/images/train1.png", "dataset/annotations/train1.xml",
                         "docs/annotation-policy.md", "data/splits.json"):
            path = self.project / relative
            previous = path.read_bytes()
            path.write_bytes(previous + b"\n")
            with self.subTest(path=relative), self.assertRaises(ValueError):
                check_review(self.project, self.review)
            path.write_bytes(previous)

    def test_invalid_boxes_and_review_attribution_rejected(self):
        self.reviewed()
        invalid = ([[0, 0, 17, 12]], [[1, 1, 1, 2]], [[0, 0, float("nan"), 2]],
                   [[False, 0, 2, 2]], [[0, 0, 2, 2], [0, 0, 2, 2]])
        for boxes in invalid:
            copy = deepcopy(self.review)
            copy["records"][0]["person_boxes"] = boxes
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                check_review(self.project, copy)
        for field, value in (("reviewer", ""), ("reason", ""), ("reviewed_at", "2026-01-01T00:00:00"),
                             ("reviewed_at", "2999-01-01T00:00:00+00:00")):
            copy = deepcopy(self.review)
            copy["records"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check_review(self.project, copy)

    def test_export_preserves_raw_and_helmet_labels_with_correct_geometry(self):
        self.reviewed()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.project / "dataset").rglob("*") if p.is_file()}
        result = export_review(self.project, self.review, self.output)
        self.assertTrue(result["ready_for_export"])
        labels = (self.output / "labels/train/train1.txt").read_text().splitlines()
        self.assertEqual(labels, ["0 0.250000000 0.250000000 0.250000000 0.125000000",
                                  "2 0.500000000 0.500000000 0.875000000 0.750000000"])
        padded = cv2.imread(str(self.output / "images/train/train1.png"))
        self.assertTrue(np.all(padded[:2] == 114))
        self.assertTrue(np.all(padded[2:14] == 0))
        self.assertFalse((self.output / "images/test").exists())
        self.assertEqual(yaml.safe_load((self.output / "dataset.yaml").read_text())["names"],
                         {0: "helmet", 1: "head", 2: "person"})
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        with self.assertRaises(FileExistsError):
            export_review(self.project, self.review, self.output)

    def test_raw_or_baseline_export_destination_is_rejected(self):
        self.reviewed()
        for target in (self.project / "dataset/new", self.project / "data/processed",
                       self.project / "data/person-corrected-test/child"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                export_review(self.project, self.review, target)


if __name__ == "__main__":
    unittest.main()
