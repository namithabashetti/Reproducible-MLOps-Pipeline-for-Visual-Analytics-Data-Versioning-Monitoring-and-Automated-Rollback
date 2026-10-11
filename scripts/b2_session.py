"""Prompt locally for replacement B2 credentials without saving or echoing them."""

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "push", "verify", "push-verify"))
    parser.add_argument("--from-environment", action="store_true",
                        help="Use credentials supplied by the local credential window")
    args = parser.parse_args()
    if not args.from_environment and not sys.stdin.isatty():
        parser.error("Run this script yourself in an interactive local terminal")
    warnings.simplefilter("error", getpass.GetPassWarning)
    environment = os.environ.copy()
    for name in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        environment.pop(name, None)
    if not args.from_environment:
        print("Use the replacement application key only. Input is hidden and is not saved.")
        environment["AWS_ACCESS_KEY_ID"] = getpass.getpass("Replacement B2 key ID: ").strip()
        environment["AWS_SECRET_ACCESS_KEY"] = getpass.getpass("Replacement B2 application key: ").strip()
    environment["AWS_EC2_METADATA_DISABLED"] = "true"
    if not environment.get("AWS_ACCESS_KEY_ID") or not environment.get("AWS_SECRET_ACCESS_KEY"):
        parser.error("Both credential fields are required")
    project = Path(__file__).resolve().parents[1]
    dvc = [sys.executable, "-m", "dvc"]
    try:
        if args.action == "status":
            subprocess.run(dvc + ["status", "-c", "dataset.dvc", "--remote", "b2"], cwd=project,
                           env=environment, check=True)
        if args.action in ("push", "push-verify"):
            subprocess.run(dvc + ["push", "dataset.dvc", "--remote", "b2", "--jobs", "4"], cwd=project,
                           env=environment, check=True)
        if args.action in ("verify", "push-verify"):
            subprocess.run([sys.executable, "-m", "src.data.verify_restore", "--remote", "b2"], cwd=project,
                           env=environment, check=True)
    finally:
        environment.pop("AWS_ACCESS_KEY_ID", None)
        environment.pop("AWS_SECRET_ACCESS_KEY", None)


if __name__ == "__main__":
    main()
