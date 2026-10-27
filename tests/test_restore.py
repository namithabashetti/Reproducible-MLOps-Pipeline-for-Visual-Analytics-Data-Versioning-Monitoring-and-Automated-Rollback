"""Exercise real DVC recovery below a Git-ignored parent directory."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.data.validate import validate_dataset
from src.data.verify_restore import verify_restore
from scripts import b2_session, gdrive_session


class RestoreTests(unittest.TestCase):
    def test_b2_helper_targets_b2_when_default_remote_changes(self):
        for action, expected_calls in (("status", 1), ("push-verify", 2)):
            with self.subTest(action=action), \
                 patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test-id",
                                           "AWS_SECRET_ACCESS_KEY": "test-secret"}), \
                 patch.object(sys, "argv", ["b2_session.py", action, "--from-environment"]), \
                 patch.object(b2_session.subprocess, "run") as run:
                b2_session.main()
                self.assertEqual(run.call_count, expected_calls)
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command[command.index("--remote") + 1], "b2")

    def test_fresh_local_restore_and_resume_after_partial_remote_failure(self):
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

            # Use a filesystem DVC remote to exercise interruption/resume without
            # cloud requests. Remove one remote object to force a partial failure.
            subprocess.run([sys.executable, "-m", "dvc", "push", "dataset.dvc"],
                           cwd=project, check=True, capture_output=True, text=True)
            image_hash = hashlib.md5((project / "dataset/images/sample.png").read_bytes()).hexdigest()
            xml_bytes = (project / "dataset/annotations/sample.xml").read_bytes()
            xml_hash = hashlib.md5(xml_bytes).hexdigest()
            remote = project / "unused-remote/files/md5"
            remote_xml = remote / xml_hash[:2] / xml_hash[2:]
            remote_xml.chmod(0o600)
            remote_xml.unlink()
            with self.assertRaises(subprocess.CalledProcessError):
                verify_restore(project, remote="b2")
            self.assertFalse((project / "reports/b2-restore.json").exists())
            recovery_caches = list((project / "reports/b2-restore-cache").glob("*/*"))
            self.assertEqual(len(recovery_caches), 1)
            cached_image = recovery_caches[0] / "files/md5" / image_hash[:2] / image_hash[2:]
            self.assertTrue(cached_image.is_file())
            # Retrying must use the preserved image; it is now absent remotely.
            remote_image = remote / image_hash[:2] / image_hash[2:]
            remote_image.chmod(0o600)
            remote_image.unlink()
            remote_xml.write_bytes(xml_bytes)
            result = verify_restore(project, remote="b2")
            self.assertTrue(result["verified"] and result["resumed"] and result["isolated_cache"])
            self.assertFalse(result["fresh_cache"])
            self.assertEqual(result["content_manifest_sha256"], original["content_manifest_sha256"])
            self.assertEqual(validate_dataset(project / "dataset", {"helmet"}), original)

            # A second provider must start with its own cache and separate evidence.
            remote_image.write_bytes((project / "dataset/images/sample.png").read_bytes())
            subprocess.run([sys.executable, "-m", "dvc", "remote", "add", "gdrive",
                            str(project / "unused-remote")], cwd=project, check=True)
            drive_result = verify_restore(project)
            self.assertEqual(drive_result["source"], "google_drive")
            self.assertTrue(drive_result["fresh_cache"])
            self.assertFalse(drive_result["resumed"])
            self.assertTrue((project / "reports/gdrive-restore.json").exists())
            self.assertTrue((project / "reports/b2-restore.json").exists())

    def test_google_credentials_stay_out_of_arguments_and_parent_environment(self):
        with tempfile.TemporaryDirectory(prefix="drive-auth-test-") as directory:
            private = Path(directory) / "visual-analytics-dvc"
            private.mkdir()
            (private / "google-oauth-client.json").write_text('{"installed": {}}')
            auth = Mock()
            auth.credentials.to_json.return_value = "test-only-oauth-token"
            calls = []

            def run(command, **kwargs):
                calls.append((list(command), dict(kwargs["env"])))

            with patch.dict("os.environ", {"LOCALAPPDATA": directory}, clear=True), \
                 patch.object(gdrive_session, "Config", return_value={
                     "remote": {"gdrive": {"url": "gdrive://test-folder"}}}), \
                 patch.object(gdrive_session, "GoogleAuth", return_value=auth), \
                 patch.object(gdrive_session, "GoogleDrive") as drive, \
                 patch.object(gdrive_session.subprocess, "run", side_effect=run), \
                 patch.object(sys, "argv", ["gdrive_session.py", "push-verify"]):
                drive.return_value.GetAbout.return_value = {}
                self.assertEqual(gdrive_session.main(), 0)
                self.assertNotIn("GDRIVE_CREDENTIALS_DATA", gdrive_session.os.environ)
            self.assertEqual(len(calls), 2)
            for command, environment in calls:
                self.assertNotIn("test-only-oauth-token", " ".join(command))
                self.assertEqual(environment["GDRIVE_CREDENTIALS_DATA"], "test-only-oauth-token")
            self.assertEqual(list(private.iterdir()), [private / "google-oauth-client.json"])

if __name__ == "__main__":
    unittest.main()
