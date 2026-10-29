"""Small real-image fixtures exercise the raw-data validation contract."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.data.validate import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "annotations").mkdir()
        Image.new("RGB", (16, 12), "red").save(self.root / "images/sample.png")
        self.xml = self.root / "annotations/sample.xml"
        self.annotation = (
            '<annotation><filename>sample.png</filename>'
            '<size><width>16</width><height>12</height></size>'
            '<object><name>helmet</name><bndbox><xmin>0</xmin><ymin>0</ymin>'
            '<xmax>16</xmax><ymax>12</ymax></bndbox></object></annotation>'
        )
        self.xml.write_text(self.annotation)

    def test_valid_edge_box_and_read_only_fingerprint(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = validate_dataset(self.root, {"helmet", "head", "person"})
        self.assertEqual(result["class_boxes"], {"helmet": 1, "head": 0, "person": 0})
        self.assertEqual(result["boxes_touching_zero"], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.xml.write_text(self.annotation.replace("helmet", "head"))
        changed = validate_dataset(self.root, {"helmet", "head", "person"})
        self.assertNotEqual(result["content_manifest_sha256"], changed["content_manifest_sha256"])

    def test_invalid_annotations(self):
        cases = {
            "unknown class": self.annotation.replace("helmet", "hat"),
            "non-finite": self.annotation.replace("<xmin>0", "<xmin>nan"),
            "invalid box": self.annotation.replace("<xmax>16", "<xmax>17"),
            "zero area": self.annotation.replace("<xmax>16", "<xmax>0"),
            "dimensions": self.annotation.replace("<width>16", "<width>15"),
            "filename": self.annotation.replace("sample.png", "other.png"),
        }
        for description, annotation in cases.items():
            with self.subTest(description=description):
                self.xml.write_text(annotation)
                with self.assertRaises(ValueError):
                    validate_dataset(self.root, {"helmet"})

    def test_provenance_is_bound_to_current_bytes(self):
        original = validate_dataset(self.root, {"helmet", "head"})
        record = {"status": "release_identity_verified",
                  "raw_content_manifest_sha256": original["content_manifest_sha256"]}
        self.assertEqual(validate_dataset(self.root, {"helmet", "head"}, record)["provenance_status"],
                         "release_identity_verified")
        self.xml.write_text(self.annotation.replace("helmet", "head"))
        with self.assertRaisesRegex(ValueError, "no longer match"):
            validate_dataset(self.root, {"helmet", "head"}, record)

    def test_missing_and_orphan_labels(self):
        self.xml.rename(self.root / "annotations/other.xml")
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            validate_dataset(self.root, {"helmet"})

    def test_corrupt_image(self):
        (self.root / "images/sample.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            validate_dataset(self.root, {"helmet"})

    def test_exact_duplicate_images_are_reported(self):
        image = self.root / "images/sample.png"
        (self.root / "images/copy.png").write_bytes(image.read_bytes())
        (self.root / "annotations/copy.xml").write_text(self.annotation.replace("sample.png", "copy.png"))
        result = validate_dataset(self.root, {"helmet"})
        self.assertEqual(result["exact_pixel_duplicate_groups"], [["copy.png", "sample.png"]])


if __name__ == "__main__":
    unittest.main()
