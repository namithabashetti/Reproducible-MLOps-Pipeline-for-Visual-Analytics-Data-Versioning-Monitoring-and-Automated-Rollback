"""Automation must prioritize review without silently accepting model labels."""

from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from scripts.annotation_triage import fingerprint, make_server, rank_record, save_decision, scan
from scripts.review_annotations import check_review, digest, export_review, policy_digest, source_records


class AnnotationTriageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name)
        for directory in ("dataset/images", "dataset/annotations", "docs", "data", "reports/review"):
            (self.project / directory).mkdir(parents=True)
        for name in ("a", "b", "test"):
            Image.new("RGB", (20, 20)).save(self.project / "dataset/images" / (name + ".png"))
            (self.project / "dataset/annotations" / (name + ".xml")).write_text(
                f'<annotation><filename>{name}.png</filename><size><width>20</width><height>20</height>'
                '</size><object><name>helmet</name><bndbox><xmin>4</xmin><ymin>3</ymin>'
                '<xmax>10</xmax><ymax>7</ymax></bndbox></object></annotation>')
        (self.project / "data/splits.json").write_text(json.dumps({"splits": {
            "train": ["a"], "validation": ["b"], "test": ["test"]}}))
        (self.project / "docs/annotation-policy.md").write_text("fixture policy\n")
        self.folder = self.project / "reports/review"
        self.review = {"schema_version": 2, "split_sha256": digest(self.project / "data/splits.json"),
                       "policy_sha256": policy_digest(self.project / "docs/annotation-policy.md"),
                       "records": source_records(self.project)}
        (self.folder / "review.json").write_text(json.dumps(self.review))
        self.suggestions = {"status": "unreviewed_model_suggestions", "records": {
            r["image_id"]: {"image_sha256": r["image_sha256"], "proposed_boxes": [[2, 2, 14, 19]], "confidence": [.9]}
            for r in self.review["records"]}}
        (self.folder / "model-suggestions.json").write_text(json.dumps(self.suggestions))
        self.identity = {**self.review, "records": {r["image_id"]: r for r in self.review["records"]}}

    def decision(self):
        row = self.review["records"][0]
        return {"image_id": row["image_id"], "expected_record": fingerprint(row), "reviewer": "Fixture reviewer",
                "reason": "Checked all classes and added visible person", "status": "reviewed", "issues": [],
                "attested": True, "helmet_boxes": row["helmet_boxes"], "head_boxes": [],
                "person_boxes": [[2, 2, 14, 19]]}

    def test_scan_prioritizes_validation_without_changing_ground_truth(self):
        before = (self.folder / "review.json").read_bytes()
        _, _, report = scan(self.project, self.folder)
        self.assertEqual(report["queue"][0]["split"], "validation")
        self.assertEqual(report["images_per_flag"]["unmatched_person_suggestions"], 2)
        self.assertEqual(report["images_per_flag"]["head_or_helmet_without_person_coverage"], 2)
        self.assertEqual(before, (self.folder / "review.json").read_bytes())
        self.assertFalse((self.folder / "review-journal.jsonl").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_review(self.project, self.review, self.project / "data/person-corrected-test")

    def test_stale_heldout_or_invalid_suggestions_rejected(self):
        for mode in ("stale", "heldout", "nan", "outside", "missing"):
            proposal = deepcopy(self.suggestions)
            if mode == "stale": proposal["records"]["a"]["image_sha256"] = "wrong"
            if mode == "heldout": proposal["records"]["test"] = proposal["records"]["a"]
            if mode == "nan": proposal["records"]["a"]["confidence"] = [float("nan")]
            if mode == "outside": proposal["records"]["a"]["proposed_boxes"] = [[0, 0, 21, 20]]
            if mode == "missing": del proposal["records"]["b"]
            (self.folder / "model-suggestions.json").write_text(json.dumps(proposal))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                scan(self.project, self.folder)

    def test_explicit_save_preserves_raw_and_journals_before_after(self):
        raw = {p: p.read_bytes() for p in (self.project / "dataset").rglob("*") if p.is_file()}
        after = save_decision(self.project, self.folder, self.decision(), self.identity)
        self.assertEqual(after["person_boxes"], [[2, 2, 14, 19]])
        review = json.loads((self.folder / "review.json").read_text())
        self.assertEqual(check_review(self.project, review)["status_counts"], {"reviewed": 1, "pending": 1})
        entry = json.loads((self.folder / "review-journal.jsonl").read_text())
        self.assertEqual(entry["before"]["a"]["status"], "pending")
        self.assertEqual(entry["after"]["a"], after)
        self.assertEqual(raw, {p: p.read_bytes() for p in raw})
        with self.assertRaisesRegex(ValueError, "Record changed"):
            save_decision(self.project, self.folder, self.decision(), self.identity)
        self.assertFalse((self.folder / "review-write.lock").exists())

    def test_unattested_invalid_and_foreign_records_cannot_save(self):
        before = (self.folder / "review.json").read_bytes()
        for field, value in (("attested", False), ("person_boxes", [[0, 0, 30, 20]]),
                             ("image_id", "test"), ("issues", ["still unsure"]), ("reviewer", "")):
            request = self.decision(); request[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                save_decision(self.project, self.folder, request, self.identity)
            self.assertEqual(before, (self.folder / "review.json").read_bytes())
        self.assertFalse((self.folder / "review-journal.jsonl").exists())
        request = self.decision(); request.update(status="needs_adjudication", attested=False, issues=["unclear body"])
        self.assertEqual(save_decision(self.project, self.folder, request, self.identity)["status"], "needs_adjudication")

    def test_changed_source_and_policy_prevent_save(self):
        for relative in ("dataset/images/a.png", "dataset/annotations/a.xml", "docs/annotation-policy.md", "data/splits.json"):
            path = self.project / relative; before = path.read_bytes(); path.write_bytes(before + b"\n")
            with self.subTest(path=relative), self.assertRaises(ValueError):
                save_decision(self.project, self.folder, self.decision(), self.identity)
            path.write_bytes(before)
        self.assertFalse((self.folder / "review-journal.jsonl").exists())

    def test_rank_flags_overlap_but_does_not_relabel(self):
        row = deepcopy(self.review["records"][0])
        row["head_boxes"] = row["helmet_boxes"].copy()
        row["person_boxes"] = [[2, 2, 14, 19], [2, 2, 14, 18.5]]
        ranked = rank_record(row, self.suggestions["records"]["a"])
        self.assertEqual(ranked["flags"]["head_helmet_overlap"], 1)
        self.assertEqual(ranked["flags"]["near_duplicate_same_class"], 1)
        self.assertEqual(row["status"], "pending")

    def test_http_routes_are_local_scoped_and_writes_require_token(self):
        server = make_server(self.project, self.folder, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response: page = response.read().decode()
        token = re.search(r'const TOKEN="([^"]+)"', page).group(1)
        for path in ("/image/test", "/image/../docs/annotation-policy.md", "/docs/annotation-policy.md"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as result:
                urlopen(base + path)
            self.assertEqual(result.exception.code, 404)
        with self.assertRaises(HTTPError):
            urlopen(Request(base + "/api/queue", headers={"Host": "external.example"}))
        data = json.dumps(self.decision()).encode()
        with self.assertRaises(HTTPError) as result:
            urlopen(Request(base + "/api/decision", data=data, headers={"Content-Type": "application/json"}))
        self.assertEqual(result.exception.code, 403)
        headers = {"Content-Type": "application/json", "X-Review-Token": token, "Origin": base}
        with urlopen(Request(base + "/api/decision", data=data, headers=headers)) as response:
            self.assertEqual(json.load(response)["status"], "reviewed")

    def test_frozen_cohort_limits_scan_and_browser_queue(self):
        manifest_path = self.project / "data/splits.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["splits"]["train"].append("extra")
        manifest_path.write_text(json.dumps(manifest))
        for directory, suffix in (("images", ".png"), ("annotations", ".xml")):
            original = self.project / "dataset" / directory / ("a" + suffix)
            target = original.with_name("extra" + suffix)
            content = original.read_bytes()
            target.write_bytes(content.replace(b"a.png", b"extra.png") if suffix == ".xml" else content)
        self.review.update(split_sha256=digest(manifest_path), records=source_records(self.project))
        cohort = self.project / "data/annotation-cohort.json"
        cohort.write_text(json.dumps({"schema_version": 1, "split_sha256": self.review["split_sha256"],
                                      "selection": {"train": ["a"], "validation": ["b"]}}))
        self.review["cohort_sha256"] = digest(cohort)
        (self.folder / "review.json").write_text(json.dumps(self.review))
        extra = next(r for r in self.review["records"] if r["image_id"] == "extra")
        self.suggestions["records"]["extra"] = {"image_sha256": extra["image_sha256"],
                                                  "proposed_boxes": [], "confidence": []}
        (self.folder / "model-suggestions.json").write_text(json.dumps(self.suggestions))
        _, _, report = scan(self.project, self.folder)
        self.assertEqual({r["image_id"] for r in report["queue"]}, {"a", "b"})
        server = make_server(self.project, self.folder, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/queue") as response:
            self.assertEqual({r["image_id"] for r in json.load(response)["records"]}, {"a", "b"})
        self.assertEqual(len(json.loads((self.folder / "review.json").read_text())["records"]), 3)


if __name__ == "__main__":
    unittest.main()
