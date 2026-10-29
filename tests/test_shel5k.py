"""Protect source-label geometry and three-class mapping during the research upgrade."""

from pathlib import Path
import tempfile
import unittest

from scripts.prepare_shel5k import read_labels


class Shel5kTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.xml"

    def write(self, objects, filename="sample.png", width=100):
        text = f"<annotation><filename>{filename}</filename><size><width>{width}</width><height>80</height></size>"
        for name, box in objects:
            text += f"<object><name>{name}</name><bndbox>"
            text += "".join(f"<{a}>{v}</{a}>" for a, v in zip(("xmin", "ymin", "xmax", "ymax"), box))
            text += "</bndbox></object>"
        self.path.write_text(text + "</annotation>")

    def test_person_merge_and_no_duplicate_helmeted_head(self):
        box = (1, 2, 15, 20)
        self.write([("helmet", box), ("head_with_helmet", (1, 2, 15, 30)),
                    ("face", (2, 21, 14, 29)), ("person_with_helmet", (0, 0, 40, 80)),
                    ("person_no_helmet", (50, 0, 100, 80)), ("head", (60, 1, 75, 20))])
        objects, counts = read_labels(self.path, (100, 80))
        self.assertEqual([n for n, _ in objects], ["helmet", "person", "person", "head"])
        self.assertEqual(objects[0][1], box)
        self.assertEqual(sum(counts.values()), 6)

    def test_reject_invalid_geometry_unknown_class_and_identity(self):
        for name, box in [("person", (0, 0, 101, 80)), ("person", (1, 0, 1, 80)),
                          ("person", (float("nan"), 0, 2, 3)), ("unknown", (0, 0, 2, 3))]:
            with self.subTest(name=name, box=box):
                self.write([(name, box)])
                with self.assertRaises(ValueError):
                    read_labels(self.path, (100, 80))
        for kwargs in ({"filename": "other.png"}, {"width": 99}):
            self.write([], **kwargs)
            with self.assertRaises(ValueError):
                read_labels(self.path, (100, 80))

    def test_colliding_person_classes_are_not_silently_deduplicated(self):
        box = (0, 0, 100, 80)
        self.write([("person_with_helmet", box), ("person_no_helmet", box)])
        with self.assertRaisesRegex(ValueError, "Duplicate mapped"):
            read_labels(self.path, (100, 80))


if __name__ == "__main__":
    unittest.main()
