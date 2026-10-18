"""Local annotation checks, priority queue, and browser editor; no model inference."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from scripts.review_annotations import check_review, cohort_selection, digest, policy_digest, validate_record


FIELDS = ("helmet_boxes", "head_boxes", "person_boxes")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / (area(a) + area(b) - overlap)


def validate_suggestions(review, suggestions):
    rows = {r["image_id"]: r for r in review["records"]}
    if suggestions.get("status") != "unreviewed_model_suggestions":
        raise ValueError("Expected unreviewed model suggestions")
    if set(suggestions["records"]) != set(rows):
        raise ValueError("Suggestions must cover exactly the training/validation review set")
    for name, proposal in suggestions["records"].items():
        row = rows[name]
        if proposal["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"Stale suggestion image: {name}")
        boxes, scores = proposal["proposed_boxes"], proposal["confidence"]
        if len(boxes) != len(scores) or any(type(s) not in (int, float) or not math.isfinite(s)
                                          or not 0 <= s <= 1 for s in scores):
            raise ValueError(f"Invalid suggestion confidence: {name}")
        validate_record({**row, "person_boxes": boxes}, row)


def rank_record(row, proposal):
    """Heuristic warnings, not estimated annotation accuracy or automatic corrections."""
    people = row["person_boxes"]
    heads = row["helmet_boxes"] + row["head_boxes"]
    strong = [b for b, c in zip(proposal["proposed_boxes"], proposal["confidence"]) if c >= .5]
    flags = {}
    if row["issues"]:
        flags["open_review_issues"] = len(row["issues"])
    missing = sum(not any(iou(b, p) >= .5 for p in people) for b in strong)
    if missing:
        flags["unmatched_person_suggestions"] = missing
    unsupported = sum(not any(iou(p, b) >= .5 for b in strong) for p in people)
    if unsupported:
        flags["person_boxes_without_strong_model_match"] = unsupported
    uncovered = sum(not any(intersection(h, p) / area(h) >= .8 for p in people) for h in heads)
    if uncovered:
        flags["head_or_helmet_without_person_coverage"] = uncovered
    duplicates = sum(iou(a, b) >= .9 for f in FIELDS for j, a in enumerate(row[f]) for b in row[f][j + 1:])
    if duplicates:
        flags["near_duplicate_same_class"] = duplicates
    conflicts = sum(iou(h, b) >= .7 for h in row["head_boxes"] for b in row["helmet_boxes"])
    if conflicts:
        flags["head_helmet_overlap"] = conflicts
    width, height = row["size"]
    edge = sum(b[0] <= 2 or b[1] <= 2 or b[2] >= width - 2 or b[3] >= height - 2
               for f in FIELDS for b in row[f])
    if edge:
        flags["cropped_or_border_boxes"] = edge
    if len(heads) >= 6:
        flags["crowded_image"] = len(heads)
    weights = {"open_review_issues": 20, "unmatched_person_suggestions": 5,
               "head_or_helmet_without_person_coverage": 3, "head_helmet_overlap": 10,
               "near_duplicate_same_class": 10}
    score = sum(weights.get(k, 1) * min(v, 10) for k, v in flags.items())
    return {"image_id": row["image_id"], "split": row["split"], "status": row["status"],
            "priority_score": score, "flags": flags}


def scan(project, folder):
    review = json.loads((folder / "review.json").read_text())
    result = check_review(project, review)
    suggestions = json.loads((folder / "model-suggestions.json").read_text())
    validate_suggestions(review, suggestions)
    selected = {name for names in cohort_selection(project, review).values() for name in names}
    queue = [rank_record(row, suggestions["records"][row["image_id"]]) for row in review["records"]
             if row["image_id"] in selected]
    queue.sort(key=lambda r: (r["status"] == "reviewed", r["split"] != "validation",
                              -r["priority_score"], r["image_id"]))
    counts = Counter(flag for row in queue for flag in row["flags"])
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "review_sha256": digest(folder / "review.json"),
              "suggestions_sha256": digest(folder / "model-suggestions.json"), "review": result,
              "flagged_images": sum(bool(r["flags"]) for r in queue), "images_per_flag": dict(counts),
              "note": "Flags are heuristics, not confirmed errors. No flags does not mean verified. "
                      "Model agreement cannot establish complete labels. No review status was changed.",
              "parameters": {"strong_confidence": .5, "match_iou": .5, "head_coverage": .8,
                             "duplicate_iou": .9, "head_helmet_iou": .7, "border_pixels": 2},
              "queue": queue}
    (folder / "triage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return review, suggestions, report


def training_draft(review, suggestions):
    """Prepare unverified training additions; never relabel validation or accept a review."""
    validate_suggestions(review, suggestions)
    records, totals = [], Counter()
    for row in review["records"]:
        if row["split"] != "train":
            continue
        heads = row["helmet_boxes"] + row["head_boxes"]
        proposal = suggestions["records"][row["image_id"]]
        additions, scores = [], []
        for box, confidence in sorted(zip(proposal["proposed_boxes"], proposal["confidence"]),
                                      key=lambda pair: -pair[1]):
            if confidence < .7:
                continue
            if any(iou(box, existing) >= .5 for existing in row["person_boxes"] + additions):
                continue
            # Reject proposals with no head support or potentially merged people.
            if sum(intersection(head, box) / area(head) >= .8 for head in heads) != 1:
                continue
            additions.append(box)
            scores.append(confidence)
        combined = row["person_boxes"] + additions
        uncovered = sum(not any(intersection(head, box) / area(head) >= .8 for box in combined)
                        for head in heads)
        records.append({"image_id": row["image_id"], "split": "train",
                        "image_sha256": row["image_sha256"], "xml_sha256": row["xml_sha256"],
                        "review_record_sha256": fingerprint(row), "added_person_boxes": additions,
                        "confidence": scores, "uncovered_head_or_helmet_boxes_after": uncovered})
        totals.update(images_with_additions=int(bool(additions)), added_person_boxes=len(additions),
                      existing_person_boxes=len(row["person_boxes"]),
                      images_with_person_after=int(bool(combined)), uncovered_heads_after=uncovered)
    return {"status": "unverified_training_proposal_not_ground_truth", "training_images": len(records),
            "settings": {"confidence_min": .7, "duplicate_iou": .5, "head_coverage_min": .8,
                         "supported_heads_per_proposal": 1},
            "summary": dict(totals), "records": records,
            "limitations": "Filters are heuristics; false positives, missed people and merged/cropped copies remain possible. "
                           "No review status or training labels were changed. No validation, test or monitoring proposals are included."}


def apply_training_draft(review, draft, draft_sha256, authorization):
    """Derive experimental training labels, preserving the source review and validation."""
    if draft.get("status") != "unverified_training_proposal_not_ground_truth" or not authorization.strip():
        raise ValueError("An unverified draft and explicit user authorization are required")
    if not isinstance(draft_sha256, str) or len(draft_sha256) != 64 or any(c not in '0123456789abcdef' for c in draft_sha256):
        raise ValueError("Invalid draft fingerprint")
    proposals = draft["records"]
    training = {r["image_id"]: r for r in review["records"] if r["split"] == "train"}
    if (len(proposals) != len(training) or len({r["image_id"] for r in proposals}) != len(proposals)
            or {r["image_id"] for r in proposals} != set(training)):
        raise ValueError("Draft must contain each training image exactly once and no validation images")
    result = deepcopy(review)
    rows = {r["image_id"]: r for r in result["records"]}
    stamp = datetime.now(timezone.utc).isoformat()
    additions = changed = 0
    for proposal in proposals:
        name = proposal["image_id"]
        original, row = training[name], rows[name]
        if (proposal["split"] != "train" or proposal["review_record_sha256"] != fingerprint(original)
                or any(proposal[k] != original[k] for k in ("image_sha256", "xml_sha256"))):
            raise ValueError("Draft source/review identity changed")
        boxes, scores = proposal["added_person_boxes"], proposal["confidence"]
        if len(boxes) != len(scores) or any(type(c) not in (float, int) or not math.isfinite(c) or not .7 <= c <= 1 for c in scores):
            raise ValueError("Invalid draft confidence")
        if not boxes:
            continue
        old_count = len(row["person_boxes"])
        row["person_boxes"] += deepcopy(boxes)
        row.update(status="accepted_as_is", reviewer="User-authorized machine-generated training labels; not visual review",
                   reviewed_at=stamp, reason=authorization + " Added person boxes remain machine-generated and unverified.")
        row["issues"].append("Machine-generated person additions are unverified; missed/incorrect boxes may remain.")
        row["person_box_provenance"] = {"existing_box_count": old_count, "added_box_count": len(boxes),
                                        "added_box_confidence": list(scores), "draft_sha256": draft_sha256,
                                        "status": "model_generated_unverified"}
        validate_record(row, original)
        additions += len(boxes)
        changed += 1
    if additions != draft["summary"]["added_person_boxes"] or changed != draft["summary"]["images_with_additions"]:
        raise ValueError("Draft counts disagree with its records")
    result["training_proposal"] = {"status": "model_generated_unverified", "draft_sha256": draft_sha256,
                                   "base_review_sha256": draft["review_sha256"],
                                   "suggestions_sha256": draft["suggestions_sha256"],
                                   "settings": draft["settings"], "added_boxes": additions,
                                   "changed_training_images": changed, "authorization": authorization,
                                   "authorized_at": stamp}
    return result


def save_decision(project, folder, request, identity):
    """Apply one explicit local review, with conflict detection and retained history."""
    required = {"image_id", "expected_record", "reviewer", "reason", "status", "issues", "attested", *FIELDS}
    if set(request) != required:
        raise ValueError("Unexpected or missing decision fields")
    if request["status"] not in {"reviewed", "needs_adjudication"}:
        raise ValueError("Choose reviewed or needs_adjudication")
    if request["status"] == "reviewed" and request["attested"] is not True:
        raise ValueError("Confirm visual review of all three classes before accepting")
    if any(not isinstance(request[k], str) or not request[k].strip() for k in ("reviewer", "reason")):
        raise ValueError("Reviewer name and reason are required")
    if request["status"] == "needs_adjudication" and not request["issues"]:
        raise ValueError("Record the unresolved issue")
    lock = folder / "review-write.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    try:
        path = folder / "review.json"
        previous_hash = digest(path)
        review = json.loads(path.read_text())
        if review.get("cohort_sha256") != identity.get("cohort_sha256"):
            raise ValueError("Review cohort changed; restart and reload")
        cohort_selection(project, review)
        for key, relative in (("split_sha256", "data/splits.json"), ("policy_sha256", "docs/annotation-policy.md")):
            actual = policy_digest(project / relative) if key == "policy_sha256" else digest(project / relative)
            if actual != identity[key] or review[key] != identity[key]:
                raise ValueError("Source policy/split changed; restart and reconcile review")
        rows = {r["image_id"]: r for r in review["records"]}
        name = request["image_id"]
        if name not in identity["records"] or name not in rows:
            raise ValueError("Image is not in this training/validation review")
        before = rows[name]
        if fingerprint(before) != request["expected_record"]:
            raise ValueError("Record changed; reload before saving")
        source = identity["records"][name]
        for directory, suffix, field in (("images", ".png", "image_sha256"), ("annotations", ".xml", "xml_sha256")):
            if digest(project / "dataset" / directory / (name + suffix)) != source[field]:
                raise ValueError("Original image/XML changed; reconcile review")
        after = {**before, **{k: request[k] for k in (*FIELDS, "reviewer", "reason", "status", "issues")},
                 "reviewed_at": datetime.now(timezone.utc).isoformat()}
        validate_record(after, source)
        entry = {"at": after["reviewed_at"], "method": "local_browser_review", "attested": request["attested"],
                 "review_before_sha256": previous_hash, "before": {name: before}, "after": {name: after}}
        candidate = deepcopy(review)
        candidate["records"][next(i for i, r in enumerate(candidate["records"]) if r["image_id"] == name)] = after
        temporary = folder / "review-write.tmp"
        temporary.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        if digest(path) != previous_hash:
            raise ValueError("Review changed during save; reload")
        # Journal first: a crash may leave an unapplied entry, never an unrecorded edit.
        with (folder / "review-journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        temporary.replace(path)
        return after
    finally:
        lock.unlink()


def make_server(project, folder, port=8766):
    review, suggestions, report = scan(project, folder)
    identity = {"split_sha256": review["split_sha256"], "policy_sha256": review["policy_sha256"],
                "cohort_sha256": review.get("cohort_sha256"),
                "records": {r["image_id"]: r for r in review["records"]}}
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return self.headers.get("Host") == expected and self.headers.get("Origin", "http://" + expected) == "http://" + expected

        def do_GET(self):
            if not self.local_request():
                return self.send(403, b'{}')
            route = urlsplit(self.path).path
            if route == "/":
                page = Path(__file__).with_name("annotation_review.html").read_text(encoding="utf-8")
                return self.send(200, page.replace("__SESSION_TOKEN__", token).encode(), "text/html; charset=utf-8")
            if route == "/api/queue":
                current = json.loads((folder / "review.json").read_text())
                selected = {name for names in cohort_selection(project, current).values() for name in names}
                payload = {"records": [{**r, "revision": fingerprint(r),
                                        "triage": rank_record(r, suggestions["records"][r["image_id"]]),
                                        "suggestions": suggestions["records"][r["image_id"]]}
                                       for r in current["records"] if r["image_id"] in selected],
                           "scan_note": report["note"]}
                return self.send(200, json.dumps(payload).encode())
            if route.startswith("/image/"):
                name = route[len("/image/"):]
                if name in identity["records"]:
                    return self.send(200, (project / "dataset/images" / (name + ".png")).read_bytes(), "image/png")
            self.send(404, b'{}')

        def do_POST(self):
            if not self.local_request() or self.headers.get("X-Review-Token") != token:
                return self.send(403, b'{}')
            if self.path != "/api/decision":
                return self.send(404, b'{}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100000:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                after = save_decision(project, folder, request, identity)
                response = {**after, "revision": fingerprint(after),
                            "triage": rank_record(after, suggestions["records"][after["image_id"]]),
                            "suggestions": suggestions["records"][after["image_id"]]}
                self.send(200, json.dumps(response).encode())
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.send(409, json.dumps({"error": str(exc)}).encode())

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "serve", "draft"))
    parser.add_argument("--review", type=Path, default=Path("reports/person-review-v1"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    folder = args.review.resolve()
    if not folder.is_relative_to(project / "reports") or folder == project / "reports":
        parser.error("Review must be a folder under reports")
    if args.command == "draft":
        review = json.loads((folder / "review.json").read_text())
        check_review(project, review)
        suggestions = json.loads((folder / "model-suggestions.json").read_text())
        report = training_draft(review, suggestions)
        report.update(created_at=datetime.now(timezone.utc).isoformat(),
                      review_sha256=digest(folder / "review.json"),
                      suggestions_sha256=digest(folder / "model-suggestions.json"),
                      generator_sha256=digest(Path(__file__)))
        with (folder / "training-draft.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    elif args.command == "scan":
        _, _, report = scan(project, folder)
        print(json.dumps({k: v for k, v in report.items() if k != "queue"}, indent=2))
    else:
        server = make_server(project, folder, args.port)
        print(f"Local annotation review: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
