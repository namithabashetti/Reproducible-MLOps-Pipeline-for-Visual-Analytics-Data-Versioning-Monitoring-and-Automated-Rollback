"""Read-only YOLO dataset audit and leakage screening against the original photos."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import yaml

from src.data.split import image_hash


def read_labels(path, class_count):
    rows, errors = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("expected class and four coordinates")
            label = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if not 0 <= label < class_count or not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError("unknown class or non-finite coordinate")
            if not (w > 0 and h > 0 and x-w/2 >= -1e-6 and y-h/2 >= -1e-6
                    and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6):
                raise ValueError("box extends outside the image or has no area")
            row = [label, x, y, w, h]
            if row in rows:
                raise ValueError("duplicate box")
            rows.append(row)
        except ValueError as exc:
            errors.append({"line": number, "reason": str(exc)})
    return rows, errors


def fingerprint(path):
    content = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        gray = np.asarray(rgb.convert("L"))
        return {"sha256": hashlib.sha256(content).hexdigest(),
                "pixels_sha256": hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
                "size": list(rgb.size), "phash": image_hash(gray),
                "mirror_phash": image_hash(cv2.flip(gray, 1))}


def audit(project, source, names, distance=10):
    splits = json.loads((project / "data/splits.json").read_text())["splits"]
    original_membership = {name: split for split, ids in splits.items() for name in ids}
    originals = sorted((project / "dataset/images").glob("*.png"))
    additions, errors, statistics = [], [], {}
    for split in ("train", "valid", "test"):
        images = sorted((source / split / "images").glob("*"))
        labels = sorted((source / split / "labels").glob("*.txt"))
        if not images:
            raise ValueError(f"Empty {split} split")
        missing = {p.stem for p in images} - {p.stem for p in labels}
        orphaned = {p.stem for p in labels} - {p.stem for p in images}
        if missing or orphaned:
            raise ValueError(f"Unpaired {split} files: {sorted(missing | orphaned)}")
        counts, empty = Counter(), 0
        for path in images:
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                raise ValueError(f"Unexpected image: {path}")
            label = source / split / "labels" / (path.stem + ".txt")
            rows, invalid = read_labels(label, len(names))
            errors.extend({"file": label.relative_to(source).as_posix(), **e} for e in invalid)
            counts.update(row[0] for row in rows)
            empty += not rows
            additions.append({"path": path.relative_to(source).as_posix(), "split": split,
                              "source_family": re.sub(r"\.rf\.[^.]+$", "", path.stem),
                              "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                              "boxes": len(rows), "classes": sorted({r[0] for r in rows})})
        statistics[split] = {"images": len(images), "labels": len(labels), "empty_labels": empty,
                             "class_boxes": {names[k]: v for k, v in sorted(counts.items())}}
    print(f"Validated label structure for {len(additions)} images; decoding both collections.", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        original = [{"path": p.relative_to(project).as_posix(), "split": original_membership[p.stem], **f}
                    for p, f in zip(originals, pool.map(fingerprint, originals))]
        for row, result in zip(additions, pool.map(fingerprint, [source / r["path"] for r in additions])):
            row.update(result)
    print("Image decoding complete; screening original, mirrored and augmented relatives.", flush=True)
    related_original, related_additional = [], []
    for i, row in enumerate(additions):
        for other in original:
            d = min((row["phash"] ^ other["phash"]).bit_count(),
                    (row["phash"] ^ other["mirror_phash"]).bit_count())
            exact = row["pixels_sha256"] == other["pixels_sha256"]
            if d <= distance or exact:
                related_original.append({"additional": row["path"], "original": other["path"],
                                         "original_split": other["split"], "distance": d, "exact_pixels": exact})
        for other in additions[:i]:
            d = min((row["phash"] ^ other["phash"]).bit_count(),
                    (row["phash"] ^ other["mirror_phash"]).bit_count())
            family = row["source_family"] == other["source_family"]
            exact = row["pixels_sha256"] == other["pixels_sha256"]
            if d <= distance or family or exact:
                related_additional.append({"a": other["path"], "b": row["path"], "distance": d,
                                           "same_source_family": family, "exact_pixels": exact,
                                           "cross_split": row["split"] != other["split"]})
    families = defaultdict(set)
    for row in additions:
        families[row["source_family"]].add(row["split"])
    summary = {"images": len(additions), "original_images": len(original), "invalid_label_rows": len(errors),
               "source_families": len(families), "cross_split_source_families": sum(len(s)>1 for s in families.values()),
               "original_related_pairs": len(related_original),
               "original_held_out_related_images": len({p["additional"] for p in related_original if p["original_split"] != "train"}),
               "additional_cross_split_pairs": sum(p["cross_split"] for p in related_additional)}
    return {"schema": 1, "source": str(source), "class_names": names, "hash_distance": distance,
            "summary": summary, "statistics": statistics, "label_errors": errors,
            "additional_images": additions, "original_images": original,
            "original_related_pairs": related_original, "additional_related_pairs": related_additional,
            "limitations": ["Perceptual similarity is a conservative candidate screen, not proof of common scenes.",
                            "Crops, video neighbors and other undetected relationships may remain.",
                            "Structural checks do not establish label completeness or semantic equivalence.",
                            "Source-family counts are filename groups, not verified counts of distinct scenes."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--class-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source) or output.exists():
        parser.error("Use a new output file outside the source dataset")
    config = yaml.safe_load(args.class_config.read_text())
    names = config["names"]
    if not isinstance(names, list) or config["nc"] != len(names) or len(set(names)) != len(names):
        parser.error("Expected a unique ordered class list matching nc")
    result = audit(project, source, names)
    result["class_config_sha256"] = hashlib.sha256(args.class_config.read_bytes()).hexdigest()
    result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
