"""Review three-class corrections separately from immutable source annotations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image
import yaml

from src.data.preprocess import letterbox, yolo_labels
from src.data.validate import read_annotation


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def policy_digest(path):
    # Git may check out Markdown with CRLF on Windows; the policy text is identical.
    return sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def source_records(project):
    manifest = json.loads((project / "data/splits.json").read_text())
    all_names = [name for names in manifest["splits"].values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Split membership overlaps")
    records = []
    for split in ("train", "validation"):
        for name in sorted(manifest["splits"][split]):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("Unsafe image name")
            image_path = project / "dataset/images" / f"{name}.png"
            xml_path = project / "dataset/annotations" / f"{name}.xml"
            with Image.open(image_path) as image:
                size = image.size
            objects = read_annotation(xml_path, size, {"helmet", "head", "person"})
            records.append({
                "image_id": name, "split": split, "size": list(size),
                "image_sha256": digest(image_path), "xml_sha256": digest(xml_path),
                **{f"{kind}_boxes": [list(box) for label, box in objects if label == kind]
                   for kind in ("helmet", "head", "person")},
                "status": "pending", "reviewer": "", "reviewed_at": "",
                "reason": "", "issues": [],
            })
    return records


def validate_record(row, source):
    """Validate one correction against its immutable source identity."""
    if row is None or any(row.get(key) != source[key] for key in
                          ("image_id", "split", "size", "image_sha256", "xml_sha256")):
        raise ValueError(f"Source identity mismatch: {source['image_id']}")
    status = row.get("status")
    if status not in {"pending", "needs_adjudication", "reviewed", "accepted_as_is"}:
        raise ValueError(f"Invalid review status: {source['image_id']}")
    if not isinstance(row.get("issues"), list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in row["issues"]):
        raise ValueError("Issues must be a list of nonempty strings")
    if status == "accepted_as_is" and source["split"] != "train":
        raise ValueError("As-is acceptance is limited to training; validation requires review")
    if status in {"reviewed", "accepted_as_is"}:
        if (status == "reviewed" and row["issues"]) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                for key in ("reviewer", "reason", "reviewed_at")):
            raise ValueError("Reviewed records require reviewer, time, reason and no open issues")
        stamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp) or stamp > datetime.now(timezone.utc):
            raise ValueError("Review time must be UTC and cannot be in the future")
    width, height = source["size"]
    fields = ("helmet_boxes", "head_boxes", "person_boxes")
    for field in fields:
        boxes = row.get(field)
        if not isinstance(boxes, list):
            raise ValueError(f"{source['image_id']}: {field} must be a list")
        seen = set()
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4 or
                any(type(value) not in (float, int) or not math.isfinite(value) for value in box)):
                raise ValueError(f"{source['image_id']}: box must contain four finite coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or tuple(box) in seen:
                raise ValueError(f"{source['image_id']}: invalid or duplicate {field}")
            seen.add(tuple(box))


def make_cohort(records, split_sha256, train_count=200, validation_count=100, seed=42):
    """Freeze a stratified sample using source labels, never predictions or review outcomes."""
    selection, strata = {}, {}
    for split, count in (("train", train_count), ("validation", validation_count)):
        groups = {}
        for row in records:
            if row["split"] != split:
                continue
            key = "".join(str(int(bool(row[f"{kind}_boxes"]))) for kind in ("helmet", "head", "person"))
            key += "-crowded" if len(row["helmet_boxes"]) + len(row["head_boxes"]) >= 6 else "-sparse"
            groups.setdefault(key, []).append(row["image_id"])
        total = sum(map(len, groups.values()))
        if type(count) is not int or not len(groups) <= count <= total:
            raise ValueError("Cohort size must cover each source stratum and fit the split")
        quotas = {key: 1 for key in groups}
        while sum(quotas.values()) < count:
            eligible = [key for key in sorted(groups) if quotas[key] < len(groups[key])]
            key = max(eligible, key=lambda k: count * len(groups[k]) / total - quotas[k])
            quotas[key] += 1
        selected = []
        for key in sorted(groups):
            ranked = sorted(groups[key], key=lambda name: sha256(f"{seed}:{split}:{name}".encode()).hexdigest())
            selected.extend(ranked[:quotas[key]])
        selection[split] = sorted(selected)
        strata[split] = {key: {"population": len(groups[key]), "selected": quotas[key]} for key in sorted(groups)}
    return {"schema_version": 1, "seed": seed, "split_sha256": split_sha256,
            "method": "source_label_presence_and_crowding_stratified_sha256",
            "selection": selection, "strata": strata,
            "purpose": "Incremental development; subset metrics are not full-dataset or final-test acceptance"}


def cohort_selection(project, review):
    path = project / "data/annotation-cohort.json"
    if not path.exists():
        if review.get("cohort_sha256"):
            raise ValueError("Frozen cohort is missing")
        return {split: [r["image_id"] for r in review["records"] if r["split"] == split]
                for split in ("train", "validation")}
    if review.get("cohort_sha256") != digest(path):
        raise ValueError("Changed cohort; reconcile explicitly instead of silently resampling")
    cohort = json.loads(path.read_text(encoding="utf-8"))
    if cohort.get("schema_version") != 1 or cohort.get("split_sha256") != review["split_sha256"]:
        raise ValueError("Invalid cohort identity")
    selection = cohort["selection"]
    if set(selection) != {"train", "validation"}:
        raise ValueError("Cohort must contain only train and validation")
    for split, names in selection.items():
        allowed = {r["image_id"] for r in review["records"] if r["split"] == split}
        if (not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names)
                or len(names) != len(set(names)) or not set(names) <= allowed):
            raise ValueError("Invalid cohort membership")
    return selection


def check_review(project, review):
    if review.get("schema_version") != 2:
        raise ValueError("Unsupported review schema")
    for field, path in (("split_sha256", "data/splits.json"),
                        ("policy_sha256", "docs/annotation-policy.md")):
        actual = policy_digest(project / path) if field == "policy_sha256" else digest(project / path)
        if review.get(field) != actual:
            raise ValueError(f"Changed {field}; review must be reconciled")
    expected = source_records(project)
    rows = review["records"]
    if len(rows) != len(expected) or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Review must contain every training/validation image exactly once")
    by_name = {r["image_id"]: r for r in rows}
    counts = Counter()
    changed = 0
    for source in expected:
        row = by_name.get(source["image_id"])
        validate_record(row, source)
        counts[row["status"]] += 1
        changed += any(row[field] != source[field] for field in ("helmet_boxes", "head_boxes", "person_boxes"))
    selection = cohort_selection(project, review)
    selected = [by_name[name] for names in selection.values() for name in names]
    selected_counts = Counter(r["status"] for r in selected)
    return {"images": len(rows), "status_counts": dict(counts), "changed_images": changed,
            "selected_images": len(selected), "selected_status_counts": dict(selected_counts),
            "selected_split_counts": {s: len(names) for s, names in selection.items()},
            "contains_as_is_training": bool(selected_counts["accepted_as_is"]),
            "ready_for_export": bool(selected) and all(r["status"] == "reviewed" or
                (r["split"] == "train" and r["status"] == "accepted_as_is") for r in selected)}


def export_review(project, review, output, proposal_path=None):
    output = output.resolve()
    if output.parent != (project / "data").resolve() or not re.fullmatch(
            r"person-corrected-[A-Za-z0-9_-]+", output.name):
        raise ValueError("Output must be a new data/person-corrected-* directory")
    if output.exists():
        raise FileExistsError("Export destination already exists")
    proposal = review.get("training_proposal")
    if proposal:
        if proposal_path is None or digest(proposal_path) != proposal["draft_sha256"]:
            raise ValueError("Model-assisted export requires the exact approved proposal file")
    elif proposal_path is not None:
        raise ValueError("Unexpected training proposal file")
    result = check_review(project, review)
    if not result["ready_for_export"]:
        raise ValueError("Review is incomplete; no candidate dataset was exported")
    params = yaml.safe_load((project / "params.yaml").read_text())
    size, padding = params["training"]["image_size"], params["preprocessing"]["pad_value"]
    classes = {"helmet": 0, "head": 1, "person": 2}
    # Recheck the same bytes consumed by conversion, not just the earlier scan.
    output.mkdir()
    selection = cohort_selection(project, review)
    selected_names = {name for names in selection.values() for name in names}
    for row in review["records"]:
        if row["image_id"] not in selected_names:
            continue
        name, split = row["image_id"], row["split"]
        image_path = project / "dataset/images" / f"{name}.png"
        xml_path = project / "dataset/annotations" / f"{name}.xml"
        image_bytes, xml_bytes = image_path.read_bytes(), xml_path.read_bytes()
        if sha256(image_bytes).hexdigest() != row["image_sha256"] or sha256(xml_bytes).hexdigest() != row["xml_sha256"]:
            raise ValueError("Source changed during export; partial output is not ready")
        objects = [(label, box) for label in classes for box in row[f"{label}_boxes"]]
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or (image.shape[1], image.shape[0]) != tuple(row["size"]):
            raise ValueError("Image cannot be decoded at its recorded dimensions")
        padded, transform = letterbox(image, size, padding)
        for folder in ("images", "labels"):
            (output / folder / split).mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output / "images" / split / f"{name}.png"), padded,
                           [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError("Cannot write candidate image")
        (output / "labels" / split / f"{name}.txt").write_text(
            yolo_labels(objects, classes, transform, size), encoding="utf-8")
    # This file is written last; interrupted exports cannot be selected as datasets.
    (output / "review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if review.get("cohort_sha256"):
        shutil.copy2(project / "data/annotation-cohort.json", output / "annotation-cohort.json")
    if proposal:
        shutil.copy2(proposal_path, output / "training-proposal.json")
    (output / "dataset.yaml").write_text(yaml.safe_dump({
        "train": "images/train", "val": "images/validation", "names": {v: k for k, v in classes.items()}
    }, sort_keys=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check", "export"))
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proposal", type=Path, help="Exact approved draft for model-assisted training exports")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    folder = args.review.resolve()
    if not folder.is_relative_to(project / "reports") or folder == project / "reports":
        parser.error("Review must be in a separate directory under reports")
    if args.command == "export" and args.output is None:
        parser.error("export requires --output")
    try:
        path = folder / "review.json"
        if args.command == "init":
            if folder.exists():
                raise FileExistsError("Review destination already exists")
            review = {"schema_version": 2, "split_sha256": digest(project / "data/splits.json"),
                      "policy_sha256": policy_digest(project / "docs/annotation-policy.md"),
                      "records": source_records(project)}
            folder.mkdir(parents=True)
            path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
            shutil.copy2(project / "docs/annotation-policy.md", folder / "policy.md")
            print(json.dumps({"images": len(review["records"]), "status": "pending"}))
            return
        review = json.loads(path.read_text(encoding="utf-8"))
        result = (export_review(project, review, args.output, args.proposal) if args.command == "export"
                  else check_review(project, review))
        print(json.dumps(result, indent=2))
        if not result["ready_for_export"]:
            raise SystemExit(2)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Annotation review failed: {exc}\n")


if __name__ == "__main__":
    main()
