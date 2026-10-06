"""Restore dataset.dvc into an isolated workspace/cache and compare content."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from src.data.validate import validate_dataset


def verify_restore(project, local_cache=False):
    project = Path(project).resolve()
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    expected = json.loads((project / "reports/dataset-validation.json").read_text())
    pointer = yaml.safe_load((project / "dataset.dvc").read_text())["outs"][0]
    if pointer["nfiles"] != expected["files"] or pointer["size"] != expected["total_bytes"]:
        raise ValueError("Dataset pointer and validation report disagree; validate and track again")
    reports = project / "reports"
    command = [sys.executable, "-m", "dvc"]
    with tempfile.TemporaryDirectory(prefix="restore-", dir=reports) as temporary:
        workspace = Path(temporary).resolve()
        if not workspace.is_relative_to(reports.resolve()):
            raise ValueError("Restore workspace must remain inside reports/")
        subprocess.run(command + ["init", "--no-scm", "-q"], cwd=workspace, check=True)
        shutil.copyfile(project / "dataset.dvc", workspace / "dataset.dvc")
        shutil.copyfile(project / ".dvc/config", workspace / ".dvc/config")
        # This workspace is nested below the parent Git repository's ignored reports/.
        subprocess.run(command + ["config", "core.no_scm", "true"], cwd=workspace, check=True)
        if local_cache:
            subprocess.run(command + ["remote", "remove", "b2"], cwd=workspace, check=True)
            subprocess.run(command + ["remote", "add", "-d", "local-cache",
                                      str(project / ".dvc/cache")], cwd=workspace, check=True)
        subprocess.run(command + ["pull", "dataset.dvc", "--jobs", "4"], cwd=workspace, check=True)
        restored = validate_dataset(workspace / "dataset", set(config["dataset"]["classes"].values()))
        if restored["content_manifest_sha256"] != expected["content_manifest_sha256"]:
            raise ValueError("Restored file paths/content differ from the validated dataset")
    result = {
        "source": "local_dvc_cache" if local_cache else "backblaze_b2",
        "dvc_md5": pointer["md5"],
        "content_manifest_sha256": restored["content_manifest_sha256"],
        "files": restored["files"], "total_bytes": restored["total_bytes"],
        "fresh_cache": True, "verified": True,
    }
    output = reports / ("local-restore.json" if local_cache else "b2-restore.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cache", action="store_true",
                        help="Verify local cache recovery only; this does not test Backblaze")
    args = parser.parse_args()
    print(json.dumps(verify_restore(Path(__file__).resolve().parents[2], args.local_cache), indent=2))
