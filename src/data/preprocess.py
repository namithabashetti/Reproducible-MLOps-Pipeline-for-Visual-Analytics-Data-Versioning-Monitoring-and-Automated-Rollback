"""Create lossless padded PNGs and YOLO labels without touching raw inputs."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import yaml

from src.data.validate import read_annotation, validate_dataset


def letterbox(image, size, pad_value):
    height, width = image.shape[:2]
    scale = min(size / width, size / height, 1.0)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    if (new_width, new_height) != (width, height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    padded = cv2.copyMakeBorder(image, top, size - new_height - top,
                                left, size - new_width - left, cv2.BORDER_CONSTANT,
                                value=(pad_value,) * 3)
    return padded, (new_width / width, new_height / height, left, top)


def yolo_labels(objects, class_ids, transform, size):
    sx, sy, left, top = transform
    lines = []
    for name, (x1, y1, x2, y2) in objects:
        # Preserve the validated zero-origin coordinates; do not apply a VOC -1 shift.
        values = (((x1 + x2) * sx / 2 + left) / size,
                  ((y1 + y2) * sy / 2 + top) / size,
                  (x2 - x1) * sx / size, (y2 - y1) * sy / size)
        if not all(0 <= value <= 1 for value in values) or min(values[2:]) <= 0:
            raise ValueError("Transformed box is invalid")
        lines.append(f"{class_ids[name]} " + " ".join(f"{value:.9f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def check_manifest(raw, output, manifest, classes):
    raw, output = raw.resolve(), output.resolve()
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError("Processed output must be separate from raw data")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Processed output is not empty; use dvc repro to regenerate tracked outputs")
    expected = {p.stem for p in (raw / "images").glob("*.png")}
    names = [n for split in manifest["splits"].values() for n in split]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError("Split manifest must cover every raw image exactly once")
    required = {"train", "validation", "test", "monitoring_reference", "monitoring_current"}
    if set(manifest["splits"]) != required:
        raise ValueError("Unexpected split names")
    membership = {n: s for s, members in manifest["splits"].items() for n in members}
    for group in manifest["related_groups"]:
        if not set(group) <= expected or len({membership[n] for n in group}) != 1:
            raise ValueError("Related image group crosses split boundaries")
    validation = validate_dataset(raw, set(classes.values()))
    if validation["content_manifest_sha256"] != manifest["raw_content_manifest_sha256"]:
        raise ValueError("Raw content changed since split generation; regenerate the manifest")
    return validation


def preprocess_dataset(raw, output, manifest, classes, size, pad_value):
    if not isinstance(size, int) or size <= 0 or not isinstance(pad_value, int) or not 0 <= pad_value <= 255:
        raise ValueError("Expected positive integer image size and integer padding in [0,255]")
    if sorted(classes) != list(range(len(classes))) or len(set(classes.values())) != len(classes):
        raise ValueError("Classes must have unique names and consecutive IDs starting at zero")
    validation = check_manifest(raw, output, manifest, classes)
    class_ids = {name: index for index, name in classes.items()}
    counts = {}
    fingerprints = {}
    for split, names in manifest["splits"].items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        boxes = Counter({c: 0 for c in classes.values()})
        for name in names:
            image = cv2.imread(str(raw / "images" / f"{name}.png"))
            if image is None:
                raise ValueError(f"Cannot decode {name}.png")
            objects = read_annotation(raw / "annotations" / f"{name}.xml",
                                      (image.shape[1], image.shape[0]), set(class_ids))
            image, transform = letterbox(image, size, pad_value)
            image_path, label_path = image_dir / f"{name}.png", label_dir / f"{name}.txt"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"Could not write {image_path}")
            label_path.write_text(yolo_labels(objects, class_ids, transform, size), encoding="utf-8")
            boxes.update(c for c, _ in objects)
            for path in (image_path, label_path):
                fingerprints[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = {"images": len(names), "class_boxes": dict(boxes)}
    # Omitting path makes YOLO resolve paths against this YAML's directory.
    # Test/monitoring pools are deliberately absent from the training configuration.
    (output / "dataset.yaml").write_text(yaml.safe_dump({
        "train": "images/train", "val": "images/validation", "names": classes,
    }, sort_keys=False), encoding="utf-8")
    content = "".join(f"{name}\0{digest}\n" for name, digest in sorted(fingerprints.items()))
    report = {
        "raw_content_manifest_sha256": validation["content_manifest_sha256"],
        "processed_content_manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "image_size": size, "pad_value": pad_value, "upscale": False,
        "coordinate_policy": "Preserve raw zero-origin boundaries; normalize transformed xywh",
        "pixel_policy": "OpenCV BGR arrays, lossless uint8 PNG; YOLO loader handles RGB/tensor normalization",
        "opencv_version": cv2.__version__, "statistics": counts,
    }
    (output / "preprocessing.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return report


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((project / "configs/config.yaml").read_text())["dataset"]
    params = yaml.safe_load((project / "params.yaml").read_text())
    manifest = json.loads((project / "data/splits.json").read_text())
    report = preprocess_dataset(project / config["raw_path"], project / config["processed_path"],
                                manifest, config["classes"], params["training"]["image_size"],
                                params["preprocessing"]["pad_value"])
    print(json.dumps(report, indent=2))
