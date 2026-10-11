"""Restore dataset.dvc into an isolated workspace/cache and compare content."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from src.data.validate import validate_dataset


def verify_restore(project, local_cache=False, remote="b2"):
    if remote not in ("b2", "gdrive"):
        raise ValueError("Supported remotes are b2 and gdrive")
    label = "Google Drive" if remote == "gdrive" else "B2"
    project = Path(project).resolve()
    config = yaml.safe_load((project / "configs/config.yaml").read_text())
    expected = json.loads((project / "reports/dataset-validation.json").read_text())
    pointer = yaml.safe_load((project / "dataset.dvc").read_text())["outs"][0]
    if pointer["nfiles"] != expected["files"] or pointer["size"] != expected["total_bytes"]:
        raise ValueError("Dataset pointer and validation report disagree; validate and track again")
    reports = project / "reports"
    command = [sys.executable, "-m", "dvc"]
    resumed = False
    recovery_cache = None
    if not local_cache:
        # Keep remote downloads across failed attempts, separate from the original cache.
        # A changed dataset or remote configuration gets a different cache.
        remote_id = hashlib.sha256((project / ".dvc/config").read_bytes()).hexdigest()[:16]
        cache_root = reports / f"{remote}-restore-cache"
        recovery_cache = (cache_root / pointer["md5"] / remote_id).resolve()
        if not recovery_cache.is_relative_to(cache_root.resolve()):
            raise ValueError("Recovery cache must remain inside its reports directory")
        resumed = any((recovery_cache / "files/md5").glob("*/*"))
        recovery_cache.mkdir(parents=True, exist_ok=True)
        print(f"Resuming {label} verification from retained downloads." if resumed
              else f"Starting {label} verification with an empty, isolated download cache.", flush=True)
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
            subprocess.run(command + ["remote", "add", "-d", "local-cache",
                                      str(project / ".dvc/cache")], cwd=workspace, check=True)
        else:
            subprocess.run(command + ["config", "cache.dir", str(recovery_cache)],
                           cwd=workspace, check=True)
        pull_remote = "local-cache" if local_cache else remote
        jobs = "16" if not local_cache and remote == "gdrive" else "4"
        try:
            subprocess.run(command + ["pull", "dataset.dvc", "--remote", pull_remote, "--jobs", jobs],
                           cwd=workspace, check=True)
        except (subprocess.CalledProcessError, KeyboardInterrupt):
            if recovery_cache:
                print(f"Verification incomplete. Completed {label} downloads are retained. "
                      "After resolving the error or waiting for the daily cap reset, "
                      "run verification again to fetch missing files.", flush=True)
            raise
        restored = validate_dataset(workspace / "dataset", set(config["dataset"]["classes"].values()))
        if restored["content_manifest_sha256"] != expected["content_manifest_sha256"]:
            raise ValueError("Restored file paths/content differ from the validated dataset")
    result = {
        "source": "local_dvc_cache" if local_cache else (
            "google_drive" if remote == "gdrive" else "backblaze_b2"),
        "dvc_md5": pointer["md5"],
        "content_manifest_sha256": restored["content_manifest_sha256"],
        "files": restored["files"], "total_bytes": restored["total_bytes"],
        "fresh_cache": not resumed, "isolated_cache": True,
        "resumed": resumed, "verified": True,
    }
    output = reports / ("local-restore.json" if local_cache else f"{remote}-restore.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--local-cache", action="store_true",
                        help="Verify local cache recovery only; this does not test cloud storage")
    source.add_argument("--remote", choices=("b2", "gdrive"), default="b2")
    args = parser.parse_args()
    print(json.dumps(verify_restore(Path(__file__).resolve().parents[2], args.local_cache,
                                    args.remote), indent=2))
