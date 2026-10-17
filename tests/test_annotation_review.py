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

from scripts.review_annotations import check_review, digest, export_review, make_cohort, policy_digest, source_records


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
        self.review = {"schema_version": 2, "split_sha256": digest(self.project / "data/splits.json"),
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

    def test_explicit_head_correction_exports_without_changing_source_xml(self):
        self.reviewed()
        row = self.review["records"][0]
        row["head_boxes"] = [[10, 1, 14, 5]]
        row["reason"] += "; missing uncovered head added after image review"
        xml = self.project / "dataset/annotations/train1.xml"
        original = xml.read_bytes()
        export_review(self.project, self.review, self.output)
        labels = (self.output / "labels/train/train1.txt").read_text().splitlines()
        self.assertEqual(labels[1], "1 0.750000000 0.312500000 0.250000000 0.250000000")
        self.assertEqual(xml.read_bytes(), original)
        self.assertEqual(check_review(self.project, self.review)["changed_images"], 2)

    def test_all_classes_are_validated_and_legacy_review_requires_reconciliation(self):
        self.reviewed()
        for field in ("helmet_boxes", "head_boxes", "person_boxes"):
            copy = deepcopy(self.review)
            copy["records"][0][field] = [[0, 0, 17, 12]]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid"):
                check_review(self.project, copy)
            del copy["records"][0][field]
            with self.assertRaisesRegex(ValueError, "must be a list"):
                check_review(self.project, copy)
        with self.assertRaisesRegex(ValueError, "schema"):
            check_review(self.project, {**self.review, "schema_version": 1})

    def test_policy_identity_survives_windows_git_newline_conversion(self):
        self.reviewed()
        path = self.project / "docs/annotation-policy.md"
        path.write_bytes(b"first line\nsecond line\n")
        self.review["policy_sha256"] = policy_digest(path)
        path.write_bytes(b"first line\r\nsecond line\r\n")
        self.assertTrue(check_review(self.project, self.review)["ready_for_export"])
        path.write_bytes(b"changed line\r\nsecond line\r\n")
        with self.assertRaisesRegex(ValueError, "policy_sha256"):
            check_review(self.project, self.review)

    def freeze_cohort(self, selection):
        path = self.project / "data/annotation-cohort.json"
        path.write_text(json.dumps({"schema_version": 1, "split_sha256": self.review["split_sha256"],
                                    "selection": selection}))
        self.review["cohort_sha256"] = digest(path)
        return path

    def test_subset_export_excludes_pending_unselected_images(self):
        manifest_path = self.project / "data/splits.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"]["train"].append("extra")
        manifest_path.write_text(json.dumps(manifest))
        for directory, suffix in (("images", ".png"), ("annotations", ".xml")):
            source = self.project / "dataset" / directory / ("train1" + suffix)
            target = source.with_name("extra" + suffix)
            target.write_bytes(source.read_bytes().replace(b"train1", b"extra") if suffix == ".xml" else source.read_bytes())
        self.review["split_sha256"] = digest(manifest_path)
        self.review["records"] = source_records(self.project)
        self.reviewed()
        extra = next(r for r in self.review["records"] if r["image_id"] == "extra")
        extra.update(status="pending", reviewer="", reason="", reviewed_at="")
        self.freeze_cohort({"train": ["train1"], "validation": ["val1"]})
        result = export_review(self.project, self.review, self.output)
        self.assertTrue(result["ready_for_export"])
        self.assertEqual(result["selected_images"], 2)
        self.assertEqual(result["status_counts"]["pending"], 1)
        self.assertEqual([p.stem for p in (self.output / "images/train").glob("*.png")], ["train1"])
        self.assertEqual(json.loads((self.output / "selection.json").read_text()),
                         {"train": ["train1"], "validation": ["val1"]})

    def test_selected_unresolved_image_still_blocks_export(self):
        self.reviewed()
        self.freeze_cohort({"train": ["train1"], "validation": ["val1"]})
        self.review["records"][1].update(status="needs_adjudication", issues=["occluded head"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_review(self.project, self.review, self.output)
        self.assertFalse(self.output.exists())

    def test_cohort_rejects_changed_missing_duplicate_and_heldout_membership(self):
        self.reviewed()
        path = self.freeze_cohort({"train": ["train1"], "validation": ["val1"]})
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Changed cohort"):
            check_review(self.project, self.review)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            check_review(self.project, self.review)
        for names in (["heldout1"], ["train1", "train1"], [], ["val1"]):
            self.freeze_cohort({"train": names, "validation": ["val1"]})
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "membership"):
                check_review(self.project, self.review)

    def test_sampling_is_reproducible_covers_strata_and_ignores_review_status(self):
        records = []
        for split in ("train", "validation"):
            for i in range(20):
                records.append({"image_id": f"{split}{i}", "split": split,
                                "helmet_boxes": [[0, 0, 1, 1]] if i < 19 else [],
                                "head_boxes": [] if i < 19 else [[0, 0, 1, 1]], "person_boxes": []})
        first = make_cohort(records, "split-hash", 5, 5)
        altered = [{**r, "status": "reviewed", "reason": "different outcomes"} for r in reversed(records)]
        self.assertEqual(first, make_cohort(altered, "split-hash", 5, 5))
        for split in ("train", "validation"):
            self.assertEqual(len(first["selection"][split]), 5)
            self.assertIn(f"{split}19", first["selection"][split])
        with self.assertRaises(ValueError):
            make_cohort(records, "split-hash", 1, 5)

    def test_reviewed_candidate_stages_copies_and_rejects_changed_selection(self):
        from src.training.train import prepare_data

        self.reviewed()
        self.freeze_cohort({"train": ["train1"], "validation": ["val1"]})
        export_review(self.project, self.review, self.output)
        staged = self.project / "staged"
        _, counts = prepare_data(self.project, staged, False, self.output)
        self.assertEqual(counts, {"train": 1, "validation": 1})
        hashes = json.loads((staged / "input-sha256.json").read_text())
        self.assertEqual(len(hashes), 4)
        for name, expected in hashes.items():
            self.assertEqual(digest(staged / name), expected)
        label = staged / "labels/train/train1.txt"
        original = (self.output / "labels/train/train1.txt").read_bytes()
        label.write_text("modified staging copy")
        self.assertEqual((self.output / "labels/train/train1.txt").read_bytes(), original)
        (self.output / "selection.json").write_text(json.dumps({"train": ["heldout1"], "validation": ["val1"]}))
        with self.assertRaisesRegex(ValueError, "selection"):
            prepare_data(self.project, self.project / "bad-staging", False, self.output)
        self.assertFalse((self.project / "bad-staging").exists())

    def test_as_is_training_requires_attribution_and_never_bypasses_validation_review(self):
        self.reviewed()
        train, val = self.review['records']
        train.update(status='accepted_as_is', reason='User explicitly accepted existing training labels',
                     issues=['Known incomplete person labels retained'])
        self.assertTrue(check_review(self.project, self.review)['contains_as_is_training'])
        val['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            export_review(self.project, self.review, self.output)
        val['status'] = 'accepted_as_is'
        with self.assertRaisesRegex(ValueError, 'limited to training'):
            check_review(self.project, self.review)
        val['status'] = 'reviewed'
        train['reviewer'] = ''
        with self.assertRaises(ValueError):
            check_review(self.project, self.review)
        train['reviewer'] = 'User acceptance'
        self.assertTrue(export_review(self.project, self.review, self.output)['ready_for_export'])
        self.assertEqual(train['issues'], ['Known incomplete person labels retained'])


if __name__ == "__main__":
    unittest.main()
