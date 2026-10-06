"""Exercise real DVC recovery below a Git-ignored parent directory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from src.data.validate import validate_dataset
from src.data.verify_restore import verify_restore


class RestoreTests(unittest.TestCase):
    def test_fresh_cache_restore_inside_ignored_directory(self):
        with tempfile.TemporaryDirectory(prefix="visual-analytics-test-") as directory:
            project = Path(directory).resolve()
            self.assertTrue(project.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            (project / ".gitignore").write_text("/reports/\n")
            for folder in ("dataset/images", "dataset/annotations", "configs", "reports"):
                (project / folder).mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(project / "dataset/images/sample.png")
            (project / "dataset/annotations/sample.xml").write_text(
                '<annotation><filename>sample.png</filename>'
                '<size><width>8</width><height>8</height></size></annotation>'
            )
            (project / "configs/config.yaml").write_text("dataset:\n  classes:\n    0: helmet\n")
            original = validate_dataset(project / "dataset", {"helmet"})
            (project / "reports/dataset-validation.json").write_text(json.dumps(original))
            commands = [
                ["init", "-q"], ["config", "core.analytics", "false"],
                ["config", "cache.type", "copy"],
                ["remote", "add", "-d", "b2", str(project / "unused-remote")],
                ["add", "--no-relink", "dataset"],
            ]
            for arguments in commands:
                subprocess.run([sys.executable, "-m", "dvc", *arguments], cwd=project,
                               check=True, capture_output=True, text=True)
            result = verify_restore(project, local_cache=True)
            self.assertEqual(result["source"], "local_dvc_cache")
            self.assertTrue(result["fresh_cache"] and result["verified"])
            self.assertEqual(result["content_manifest_sha256"], original["content_manifest_sha256"])
            self.assertEqual(validate_dataset(project / "dataset", {"helmet"}), original)
            self.assertEqual(list((project / "reports").glob("restore-*")), [])
            self.assertFalse((project / "reports/b2-restore.json").exists())


if __name__ == "__main__":
    unittest.main()
