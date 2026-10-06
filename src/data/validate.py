"""Read-only validation of the supplied PNG/Pascal VOC dataset."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


def read_annotation(path, image_size, class_names):
    """Return class names and pixel boxes, rejecting inconsistent annotations."""
    root = ET.parse(path).getroot()
    if root.tag != "annotation":
        raise ValueError(f"{path.name}: expected an annotation root")
    if root.findtext("filename") != path.stem + ".png":
        raise ValueError(f"{path.name}: filename does not match its PNG")
    size = tuple(int(root.findtext(f"size/{axis}", "0")) for axis in ("width", "height"))
    if size != image_size:
        raise ValueError(f"{path.name}: XML dimensions {size} != image {image_size}")
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in class_names:
            raise ValueError(f"{path.name}: unknown class {name!r}")
        box = tuple(float(obj.findtext(f"bndbox/{axis}", "nan"))
                    for axis in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1, x2, y2 = box
        width, height = size
        if not all(math.isfinite(value) for value in box):
            raise ValueError(f"{path.name}: missing or non-finite box coordinate")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"{path.name}: invalid box {box} for image {size}")
        if (name, box) in objects:
            raise ValueError(f"{path.name}: duplicate box {name}, {box}")
        objects.append((name, box))
    return objects


def validate_dataset(raw_path, class_names):
    """Decode all images, validate paired labels, and fingerprint every raw file."""
    raw_path = Path(raw_path)
    for folder in ("images", "annotations"):
        if not (raw_path / folder).is_dir():
            raise ValueError(f"Missing dataset directory: {raw_path / folder}")
    images = sorted((raw_path / "images").iterdir())
    annotations = sorted((raw_path / "annotations").iterdir())
    if not images or any(not p.is_file() or p.suffix != ".png" for p in images):
        raise ValueError("images/ must contain PNG files only and must not be empty")
    if any(not p.is_file() or p.suffix != ".xml" for p in annotations):
        raise ValueError("annotations/ must contain XML files only")
    missing = {p.stem for p in images} - {p.stem for p in annotations}
    orphaned = {p.stem for p in annotations} - {p.stem for p in images}
    if missing or orphaned:
        raise ValueError(f"Unpaired files: missing XML={sorted(missing)}, orphan XML={sorted(orphaned)}")

    boxes = Counter({name: 0 for name in class_names})
    class_images = Counter({name: 0 for name in class_names})
    dimensions = Counter()
    duplicates = defaultdict(list)
    fingerprints = {}
    image_bytes = empty_annotations = small_boxes = zero_origin_boxes = 0
    for path in images:
        content = path.read_bytes()
        image_bytes += len(content)
        fingerprints[path.relative_to(raw_path).as_posix()] = sha256(content).hexdigest()
        with Image.open(BytesIO(content)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"{path.name}: expected RGB PNG, got {image.format}/{image.mode}")
            dimensions[f"{image.width}x{image.height}"] += 1
            pixel_hash = sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            duplicates[pixel_hash].append(path.name)
            objects = read_annotation(raw_path / "annotations" / (path.stem + ".xml"),
                                      image.size, class_names)
        empty_annotations += not objects
        class_images.update({name for name, _ in objects})
        for name, (x1, y1, x2, y2) in objects:
            boxes[name] += 1
            small_boxes += min(x2 - x1, y2 - y1) < 16
            zero_origin_boxes += x1 == 0 or y1 == 0

    total_bytes = 0
    for path in sorted(raw_path.rglob("*")):
        if path.is_file():
            relative = path.relative_to(raw_path).as_posix()
            total_bytes += path.stat().st_size
            if relative not in fingerprints:
                fingerprints[relative] = sha256(path.read_bytes()).hexdigest()
    # Include relative paths as well as bytes, so a rename changes dataset identity.
    manifest = "".join(f"{name}\0{digest}\n" for name, digest in sorted(fingerprints.items()))
    return {
        "images": len(images), "annotations": len(annotations),
        "files": len(fingerprints), "total_bytes": total_bytes,
        "mean_image_bytes": image_bytes / len(images), "dimensions": dict(dimensions),
        "class_boxes": dict(boxes), "class_images": dict(class_images),
        "empty_annotations": empty_annotations,
        "boxes_with_side_under_16_pixels": small_boxes,
        "boxes_touching_zero": zero_origin_boxes,
        "exact_pixel_duplicate_groups": [names for names in duplicates.values() if len(names) > 1],
        "content_manifest_sha256": sha256(manifest.encode()).hexdigest(),
        "provenance_status": "unverified",
        "limitations": ["Near-duplicates and scene groups are not checked.",
                        "Structural validity does not establish annotation completeness."],
    }


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project / "configs/config.yaml")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["dataset"]
    raw_path = (project / config["raw_path"]).resolve()
    if args.report and args.report.resolve().is_relative_to(raw_path):
        parser.error("Report output must be outside the original dataset")
    try:
        result = validate_dataset(raw_path, set(config["classes"].values()))
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
