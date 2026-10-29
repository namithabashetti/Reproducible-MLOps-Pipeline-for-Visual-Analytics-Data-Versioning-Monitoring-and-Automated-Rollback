"""Audit publisher SHEL5K v4 labels and export a separate three-class candidate.

Consumes the versioned Mendeley file manifests and annotation-only downloads.
Never changes raw images, earlier labels, review records, or split membership.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
import xml.etree.ElementTree as ET

import cv2
from PIL import Image

from src.data.preprocess import letterbox, yolo_labels


SOURCE = "https://data.mendeley.com/datasets/9rcv8mm682/4"
MAPPING = {"helmet": "helmet", "head": "head", "person_with_helmet": "person",
           "person_no_helmet": "person", "person": "person"}
IGNORED = {"head_with_helmet", "face"}
CLASSES = {"helmet": 0, "head": 1, "person": 2}


def fetch(evidence):
    """Fetch public metadata and XMLs only; resume files matching publisher SHA256."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    evidence.mkdir(parents=True, exist_ok=True)
    base = "https://data.mendeley.com/public-api/datasets/9rcv8mm682/"
    response = requests.get(base + "snapshot/4", timeout=45)
    response.raise_for_status()
    (evidence / "mendeley-v4-snapshot.json").write_text(response.text)
    manifests = {}
    for kind, folder in (("annotations", "89498925-e287-4485-baee-bf34822d49e7"),
                         ("images", "de4001f6-9000-4956-bdc1-80498271b0a5")):
        entries = []
        for start in range(0, 6000, 1000):
            response = requests.get(base + "files", params={"folder_id": folder, "version": 4,
                                    "$start": start, "$limit": 1000}, timeout=45)
            response.raise_for_status()
            page = response.json()
            entries.extend(page)
            if len(page) < 1000:
                break
        (evidence / f"shel5k-v4-{kind}-manifest.json").write_text(json.dumps(entries, indent=2))
        manifests[kind] = entries
    output = evidence / "shel5k-v4-annotations"
    output.mkdir(exist_ok=True)
    local = threading.local()

    def download(row):
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".xml") or "\\" in name:
            raise ValueError("Unexpected publisher annotation filename")
        target, details = output / name, row["content_details"]
        if target.exists() and target.stat().st_size == row["size"] and sha(target) == details["sha256_hash"]:
            return
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for attempt in range(4):
            try:
                response = local.session.get(details["download_url"], timeout=45)
                response.raise_for_status()
                content = response.content
                if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != details["sha256_hash"]:
                    raise ValueError("Publisher annotation digest mismatch: " + name)
                target.write_bytes(content)
                return
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, row): row["filename"] for row in manifests["annotations"]}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append({"file": futures[future], "error": str(exc)})
            if count % 250 == 0:
                print(f"Checked {count}/5000 annotations; failures: {len(errors)}", flush=True)
    if errors:
        (evidence / "download-errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"{len(errors)} annotation downloads failed; rerun --fetch to resume")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_labels(path, size):
    root = ET.parse(path).getroot()
    if root.tag != "annotation" or root.findtext("filename") != path.stem + ".png":
        raise ValueError(f"Wrong image identity in {path.name}")
    if tuple(int(root.findtext("size/" + a, "0")) for a in ("width", "height")) != size:
        raise ValueError(f"Wrong image dimensions in {path.name}")
    objects, counts = [], Counter()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in MAPPING and name not in IGNORED:
            raise ValueError(f"Unknown source class {name!r} in {path.name}")
        box = tuple(float(obj.findtext("bndbox/" + a, "nan"))
                    for a in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1, x2, y2 = box
        if not all(math.isfinite(x) for x in box) or not (
                0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError(f"Invalid box in {path.name}: {name} {box}")
        counts[name] += 1
        if name in MAPPING:
            mapped = (MAPPING[name], box)
            if mapped in objects:
                raise ValueError(f"Duplicate mapped box in {path.name}: {mapped}")
            objects.append(mapped)
    return objects, counts


def audit(project, evidence):
    snapshot = json.loads((evidence / "mendeley-v4-snapshot.json").read_text())
    if (snapshot["id"], snapshot["version"], snapshot["licence"]["short_name"]) != (
            "9rcv8mm682", 4, "CC BY 4.0"):
        raise ValueError("Expected the pinned SHEL5K version 4 release")
    manifests = {kind: json.loads((evidence / f"shel5k-v4-{kind}-manifest.json").read_text())
                 for kind in ("images", "annotations")}
    local_ids = {p.stem for p in (project / "dataset/images").glob("*.png")}
    if len(local_ids) != 5000:
        raise ValueError("Expected the verified 5,000-image original release")
    for kind, entries in manifests.items():
        suffix = ".png" if kind == "images" else ".xml"
        names = [e["filename"] for e in entries]
        if len(names) != len(set(names)) or set(names) != {n + suffix for n in local_ids}:
            raise ValueError(f"Publisher {kind} membership differs from local images")
    annotation_entries = {e["filename"]: e for e in manifests["annotations"]}
    counts, mapped_counts, fingerprints, errors = Counter(), Counter(), {}, []
    for entry in manifests["images"]:
        image = project / "dataset/images" / entry["filename"]
        annotation = evidence / "shel5k-v4-annotations" / (image.stem + ".xml")
        try:
            for file, record in ((image, entry), (annotation, annotation_entries[annotation.name])):
                if file.stat().st_size != record["size"] or sha(file) != record["content_details"]["sha256_hash"]:
                    raise ValueError(f"Publisher SHA256/length mismatch: {file.name}")
            with Image.open(image) as im:
                objects, source_counts = read_labels(annotation, im.size)
            counts.update(source_counts)
            mapped_counts.update(n for n, _ in objects)
            fingerprints[image.stem] = {"image": entry["content_details"]["sha256_hash"],
                                       "annotation": sha(annotation)}
        except (ValueError, OSError, ET.ParseError) as exc:
            errors.append({"image": image.stem, "error": str(exc)})
    report = {"source": SOURCE, "doi": snapshot["doi"], "license": "CC BY 4.0",
              "status": "publisher_labels_structurally_validated" if not errors else "blocked",
              "images": len(local_ids), "source_class_boxes": dict(counts),
              "mapped_class_boxes": dict(mapped_counts), "errors": errors,
              "mapping": MAPPING, "dropped_classes": sorted(IGNORED),
              "fingerprints": fingerprints,
              "limitations": ["Publisher annotations, not 5,000 independently reviewed images.",
                              "Structural checks do not establish visual completeness.",
                              "Helmet denotes the helmet object, unlike earlier mixed head/helmet boxes."]}
    (evidence / "shel5k-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export(project, evidence, output, report):
    if report["errors"]:
        raise ValueError("Resolve the annotation audit errors before exporting")
    if output.parent != (project / "data").resolve() or not output.name.startswith("expanded-shel5k-"):
        raise ValueError("Use a new data/expanded-shel5k-* output")
    if output.exists():
        raise FileExistsError("Preserve previous datasets; output already exists")
    base = project / "data/expanded-protocol-v2"
    manifest = json.loads((base / "manifest.json").read_text())
    membership = {n: split for split, names in manifest["selection"].items() for n in names}
    if len(membership) != sum(map(len, manifest["selection"].values())):
        raise ValueError("Base split overlap")
    for group in manifest["related_groups"]:
        if len({membership[n] for n in group}) != 1:
            raise ValueError("Related images cross splits")
    # Verify frozen base bytes before copying. Test data stay in their original split.
    base_hashes = json.loads((base / "input-sha256.json").read_text())
    expected = {f"{kind}/{split}/{n}{suffix}" for split, names in manifest["selection"].items()
                for n in names for kind, suffix in (("images", ".png"), ("labels", ".txt"))}
    if set(base_hashes) != expected or any(sha(base / n) != h for n, h in base_hashes.items()):
        raise ValueError("Frozen base dataset changed")
    output.mkdir()
    fingerprints, statistics = {}, {}
    for split, names in manifest["selection"].items():
        counts = Counter()
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)
        for name in names:
            image_dst = output / "images" / split / (name + ".png")
            label_dst = output / "labels" / split / (name + ".txt")
            shutil.copy2(base / "images" / split / image_dst.name, image_dst)
            if name in report["fingerprints"]:
                raw = project / "dataset/images" / (name + ".png")
                annotation = evidence / "shel5k-v4-annotations" / (name + ".xml")
                frozen = report["fingerprints"][name]
                if sha(raw) != frozen["image"] or sha(annotation) != frozen["annotation"]:
                    raise ValueError("Source changed after audit: " + name)
                image = cv2.imread(str(raw))
                objects, _ = read_labels(annotation, (image.shape[1], image.shape[0]))
                padded, transform = letterbox(image, 416, 114)
                if not (padded == cv2.imread(str(image_dst))).all():
                    raise ValueError("Base preprocessing geometry differs: " + name)
                label_dst.write_text(yolo_labels(objects, CLASSES, transform, 416))
                counts.update(n for n, _ in objects)
            else:
                shutil.copy2(base / "labels" / split / label_dst.name, label_dst)
                for line in label_dst.read_text().splitlines():
                    counts[list(CLASSES)[int(line.split()[0])]] += 1
            for file in (image_dst, label_dst):
                fingerprints[file.relative_to(output).as_posix()] = sha(file)
        statistics[split] = {"images": len(names), "class_boxes": dict(counts)}
    for name in ("dataset.yaml", "selection.json", "audit.json", "README.dataset.txt", "README.roboflow.txt"):
        shutil.copy2(base / name, output / name)
    manifest.update(label_status="SHEL5K v4 publisher labels for original images; CSS labels as supplied",
                    label_upgrade={"source": SOURCE, "license": "CC BY 4.0", "mapping": MAPPING,
                                   "dropped_classes": sorted(IGNORED),
                                   "audit_sha256": sha(evidence / "shel5k-audit.json"),
                                   "builder": "scripts/prepare_shel5k.py"},
                    statistics=statistics,
                    final_test_status="previously_consumed_by_model_5",
                    previous_manifest_sha256=sha(base / "manifest.json"),
                    evaluation_restriction="The existing test was consumed by model 5. Further scores are reused-test research evidence, not a fresh final acceptance test.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "input-sha256.json").write_text(json.dumps(fingerprints, indent=2) + "\n")
    shutil.copy2(evidence / "shel5k-audit.json", output / "annotation-provenance.json")
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true", help="Resume the public annotation-only download")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.fetch:
        fetch(args.evidence.resolve())
    report = audit(project, args.evidence.resolve())
    print(json.dumps({k: v for k, v in report.items() if k != "fingerprints"}, indent=2), flush=True)
    if report["errors"]:
        raise SystemExit(2)
    if args.output:
        print(json.dumps(export(project, args.evidence.resolve(), args.output.resolve(), report), indent=2))


if __name__ == "__main__":
    main()
