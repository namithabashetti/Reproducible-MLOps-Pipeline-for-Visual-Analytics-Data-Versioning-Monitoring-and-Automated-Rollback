"""Keep visually similar inputs together in deterministic, class-balanced splits."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import cv2
import numpy as np
import yaml

from src.data.validate import read_annotation, validate_dataset


def image_hash(image):
    """63 low-frequency DCT bits, excluding overall brightness (the DC term)."""
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))[:8, :8].ravel()[1:]
    return int.from_bytes(np.packbits(coefficients > np.median(coefficients)).tobytes(), "big")


def find_groups(raw, classes, max_distance):
    if not 0 <= max_distance <= 63:
        raise ValueError("group_hash_distance must be between 0 and 63")
    names, hashes, labels = [], [], {}
    for path in sorted((raw / "images").glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode {path.name}")
        names.append(path.stem)
        hashes.append((image_hash(image), image_hash(cv2.flip(image, 1))))
        objects = read_annotation(raw / "annotations" / f"{path.stem}.xml",
                                  (image.shape[1], image.shape[0]), classes)
        labels[path.stem] = Counter(name for name, _ in objects)
    parent = list(range(len(names)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = []
    for i, (normal, _) in enumerate(hashes):
        for j in range(i):
            distance = min((normal ^ hashes[j][0]).bit_count(),
                           (normal ^ hashes[j][1]).bit_count())
            if distance <= max_distance:
                parent[root(i)] = root(j)
                pairs.append([names[j], names[i], distance])
    groups = defaultdict(list)
    for index, name in enumerate(names):
        groups[root(index)].append(name)
    return list(groups.values()), labels, pairs


def assign_splits(groups, labels, targets, classes, seed):
    """Place whole groups; balance class presence while honoring exact capacities."""
    total = sum(map(len, groups))
    if sum(targets.values()) != total or any(n <= 0 for n in targets.values()):
        raise ValueError("Positive split counts must sum to the dataset size")
    if len({name for group in groups for name in group}) != total:
        raise ValueError("Groups must contain each image exactly once")
    classes = sorted(classes)
    totals = np.array([total] + [sum(c in labels[n] for n in labels) for c in classes])
    desired = {s: totals * size / total for s, size in targets.items()}
    actual = {s: np.zeros(len(totals)) for s in targets}
    result = {s: [] for s in targets}
    ordered = sorted(groups, key=lambda group: tuple(group))
    random.Random(seed).shuffle(ordered)
    # Large groups first avoid fragmenting the remaining capacity; rare classes first
    # within each size leave abundant classes available to fill the final slots.
    ordered.sort(key=lambda g: (-len(g), -sum(
        1 / max(totals[i + 1], 1) for n in g for i, c in enumerate(classes) if c in labels[n])))
    for group in ordered:
        vector = np.array([len(group)] + [sum(c in labels[n] for n in group) for c in classes])
        eligible = [s for s in targets if len(result[s]) + len(group) <= targets[s]]
        if not eligible:
            raise ValueError("Whole groups cannot fit the requested counts; review split sizes")
        scores = {s: np.sum(((actual[s] + vector - desired[s]) ** 2
                             - (actual[s] - desired[s]) ** 2) / np.maximum(desired[s], 1))
                  for s in eligible}
        selected = min(eligible, key=lambda s: (scores[s], s))
        result[selected].extend(group)
        actual[selected] += vector
    return {s: sorted(names) for s, names in result.items()}


def create_manifest(raw, classes, params):
    validation = validate_dataset(raw, set(classes))
    groups, labels, pairs = find_groups(raw, set(classes), params["group_hash_distance"])
    targets = dict(params["split_counts"])
    monitoring = targets.pop("monitoring")
    targets["monitoring_reference"] = params["monitoring_reference_count"]
    targets["monitoring_current"] = params["monitoring_current_count"]
    if targets["monitoring_reference"] + targets["monitoring_current"] != monitoring:
        raise ValueError("Reference/current counts must sum to the monitoring count")
    splits = assign_splits(groups, labels, targets, classes, params["seed"])
    statistics = {}
    for split, names in splits.items():
        boxes = sum((labels[n] for n in names), Counter())
        images = {c: sum(c in labels[n] for n in names) for c in sorted(classes)}
        if not all(images.values()):
            raise ValueError(f"{split} lacks a class; review grouping and split sizes")
        statistics[split] = {"images": len(names), "class_images": images,
                             "class_boxes": {c: boxes[c] for c in sorted(classes)}}
    return {
        "raw_content_manifest_sha256": validation["content_manifest_sha256"],
        "seed": params["seed"], "group_hash_distance": params["group_hash_distance"],
        "group_method": "63-bit DCT hash; normal/mirrored distance; connected components",
        "limitation": "Conservative candidates, not verified source-scene IDs; undetected relations remain possible.",
        "related_groups": sorted([g for g in groups if len(g) > 1]),
        "candidate_pairs": pairs, "statistics": statistics, "splits": splits,
    }


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((project / "configs/config.yaml").read_text())["dataset"]
    params = yaml.safe_load((project / "params.yaml").read_text())["data"]
    manifest = create_manifest(project / config["raw_path"], config["classes"].values(), params)
    output = project / "data/splits.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["statistics"], indent=2))
