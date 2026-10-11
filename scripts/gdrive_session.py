"""Sign in locally with Google, then run DVC without exposing credentials."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dvc.config import Config
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("auth", "push-verify", "verify"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = Config(dvc_dir=str(project / ".dvc"))
    if not config.get("remote", {}).get("gdrive", {}).get("url", "").startswith("gdrive://"):
        parser.error("Configure the private Google Drive folder as remote 'gdrive' first")
    private = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "visual-analytics-dvc"
    if private.resolve().is_relative_to(project):
        parser.error("Google credentials must be stored outside the repository")
    client_file = private / "google-oauth-client.json"
    token_file = private / "google-user-token.json"
    if not client_file.exists():
        from tkinter import Tk, filedialog

        print("Select the downloaded Google OAuth Desktop client JSON. Do not paste its contents.")
        window = Tk()
        window.withdraw()
        try:
            selected = filedialog.askopenfilename(
                title="Select Google OAuth Desktop client JSON", filetypes=[("JSON files", "*.json")])
        finally:
            window.destroy()
        if not selected:
            parser.error("No OAuth client selected; nothing was uploaded")
        client = Path(selected).resolve()
        if client.is_relative_to(project):
            parser.error("Move the OAuth client JSON outside the repository before selecting it")
        if "installed" not in json.loads(client.read_text(encoding="utf-8-sig")):
            parser.error("Select an OAuth Desktop client JSON, not a service-account key")
        private.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(client, client_file)
    auth = GoogleAuth(settings={
        "client_config_backend": "file", "client_config_file": str(client_file),
        "save_credentials": True, "save_credentials_backend": "file",
        "save_credentials_file": str(token_file), "get_refresh_token": True,
        "oauth_scope": ["https://www.googleapis.com/auth/drive",
                        "https://www.googleapis.com/auth/drive.appdata"],
    })
    try:
        auth.LocalWebserverAuth()
        about = GoogleDrive(auth).GetAbout()
    except Exception as error:
        print(f"Google sign-in/account check failed ({type(error).__name__}). "
              "Check the OAuth app setup or reauthorize; do not share token files.", file=sys.stderr)
        return 1
    if about.get("quotaBytesTotal"):
        remaining = int(about["quotaBytesTotal"]) - int(about.get("quotaBytesUsedAggregate", 0))
        print(f"Google account storage remaining: {remaining / 1e9:.2f} GB.")
    if args.action == "auth":
        print("Google authentication succeeded. No dataset transfer performed.")
        return 0
    environment = os.environ.copy()
    environment["GDRIVE_CREDENTIALS_DATA"] = auth.credentials.to_json()
    try:
        if args.action == "push-verify":
            subprocess.run([sys.executable, "-m", "dvc", "push", "dataset.dvc",
                            "--remote", "gdrive", "--jobs", "16"],
                           cwd=project, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "src.data.verify_restore", "--remote", "gdrive"],
                       cwd=project, env=environment, check=True)
    finally:
        environment.pop("GDRIVE_CREDENTIALS_DATA", None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
