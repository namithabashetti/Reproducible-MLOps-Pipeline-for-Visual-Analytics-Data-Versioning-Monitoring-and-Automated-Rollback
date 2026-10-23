# Run and explain the college demo in Visual Studio Code

This guide uses the existing Windows environment, trained checkpoints and local
Docker deployment. No installation, training or dataset transfer is needed for
this presentation. VS Code is the editor; Docker runs the inference and monitoring
services. There is no single Run button that starts the entire MLOps pipeline.

## 1. Open the folder and terminal

In VS Code, use **File > Open Folder** and select
`F:\major project implementation`. Use **Terminal > New Terminal**, with
PowerShell selected. Commands below use the existing environment explicitly,
so environment activation is unnecessary.

```powershell
Set-Location -LiteralPath 'F:\major project implementation'
.\.venv\Scripts\python.exe --version
```

Expected Python: 3.11.9. If using the Python extension, select this folder's
`.venv\Scripts\python.exe` interpreter. Do not recreate the existing environment.

## 2. Start the existing services after reboot

Open Docker Desktop and wait for its engine to be ready. Then run:

```powershell
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
& $dockerExe compose -f compose.monitoring.yaml up -d
& $dockerExe start visual-analytics-api
& $dockerExe ps
Invoke-RestMethod 'http://127.0.0.1:8000/health' | Select-Object status, model_name, model_version, deployment_mode
```

Allow the checkpoint to load before checking health again. Expect `ready` and an
experimental model. These commands restart the existing laptop installation;
they are not clean-machine provisioning commands. If the API container has been
deleted, `docker start` cannot recreate it.

In a second terminal, start MLflow only if port 5000 is not already serving it:

```powershell
.\.venv\Scripts\python.exe -m src.tracking.mlflow_utils serve
```

Keep that terminal open; Ctrl+C stops MLflow's UI server. The registry is stored
locally and persists independently of this UI process.

| Page | What to show |
| --- | --- |
| http://127.0.0.1:8767 | Main presentation page: upload, detection boxes, camera controls, controlled quality failure and automatic recovery |
| http://127.0.0.1:5000 | Experiments, metrics, parameters and the three experimental `hard-hat-detector` model versions |
| http://127.0.0.1:8000/health | Loaded model version, checkpoint identity and readiness |
| http://127.0.0.1:3000 | Grafana operations dashboard: requests, errors, latency and model identity |
| http://127.0.0.1:9090 | Prometheus targets and the metrics behind Grafana |

The API intentionally disables `/docs`. Prediction requests use raw image bytes.

In a third terminal, start the presentation page:

```powershell
.\.venv\Scripts\python.exe -m src.serving.console
```

Open **http://127.0.0.1:8767/**. Choose **Use sample image**, or select a PNG/JPEG
under 4 MB and click **Detect image**. The boxes and confidence scores come from
the actual Docker API. **Start camera** requests browser camera permission and
submits frames locally. Camera capture, 640x480 API inference and stopping were
verified on this laptop. Test your actual hard-hat scene beforehand; live-scene
accuracy is not guaranteed. Image upload remains a repeatable presentation option. Click **Stop camera** before
running exact-count monitoring validation. Keep the console terminal open.

If the API container is missing, recreate it from the verified local image:

```powershell
.\.venv\Scripts\python.exe -m src.serving.deploy --version 1 --experimental --image visual-analytics-api:security-20261022
```

If that image is absent, build it first with
`& $dockerExe build -t visual-analytics-api:security-20261022 .`.

## 3. Make a real prediction

In the first terminal:

```powershell
$sample = 'F:\major project implementation\dataset\images\hard_hat_workers1015.png'
$prediction = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'image/png' -InFile $sample
$prediction | ConvertTo-Json -Depth 6
```

Explain: the API validates the image, loads the deployed checkpoint, applies
416-pixel preprocessing, runs YOLO inference and returns class, confidence and
box coordinates in the original image. It also records request outcome and latency.

Open `reports/presentation-preview/prediction.png` for the saved annotated result
from an actual API request, and `prediction.json` beside it for the exact response.
This is a prediction illustration, not ground truth or an accuracy measurement.
The original dataset image was not changed. Grafana scrapes periodically; select
a recent time range and allow time for fresh requests to appear. Rate panels may
need multiple samples and more than one request.

## 4. Demonstrate real automatic rollback

For the presentation, click **Start controlled quality demo** on the live page.
It measures the predecessor on 250 labeled monitoring images, then deploys a
candidate with its confidence threshold visibly set to **1.0**. Click **Detect
image** again to show missing detections. Click **Check quality now** to request
the labeled batch audit, or wait for the automatic scheduled audit. Startup grace
still applies. Three empty live observations can request this audit too.

The existing watcher independently checks the measured AP50–95 drop of at least
0.05, evidence freshness and identities, then restores the recorded predecessor
through the normal deployment path. The page reports progress and all three
measured AP scores. Recovery includes three readiness checks, twenty predictions,
fresh Prometheus identity and another labeled quality audit. A miss on a camera
frame alone is insufficient: without labels the system cannot know what it missed.

This is a **controlled configuration fault** in an isolated registry. It does not
claim natural model deterioration or grant production quality approval. The demo
temporarily occupies port 8000; do not run concurrent deployments or validation.
The original experimental API is restored automatically. Do not close the console
or stop Docker while it runs. Allow roughly 4–8 minutes, depending on laptop load.

The equivalent terminal quality demo is:

```powershell
$qualityDemo = 'reports/quality-demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\.venv\Scripts\python.exe scripts/demo_rollback.py --mode quality --output $qualityDemo
```

The separate **Run service-outage demo** button exercises readiness failure.
The terminal command below runs that outage scenario (the default mode):

Leave Docker and monitoring running. Use the terminal, not Run Python File:

```powershell
$demo = 'reports/rollback-demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\.venv\Scripts\python.exe scripts/demo_rollback.py --output $demo
if ($LASTEXITCODE -eq 0) {
    $result = Get-Content -LiteralPath "$demo/result.json" -Raw | ConvertFrom-Json
    $result | Select-Object status, quality_approval
    $result.rollback.recovery | Select-Object monitoring_resumed
}
```

Allow several minutes and keep the terminal open. The script temporarily replaces
the local API. It creates an isolated registry copy with demo-only predecessor
aliases, verifies a healthy candidate, then deliberately stops its container.
The watcher detects persistent readiness failures and automatically restores the
recorded predecessor. It verifies three readiness responses, twenty real image
predictions and resumed monitoring, then restores the original experimental API.
Each run requires a fresh report directory; the timestamp provides one.

Expected success: `status: passed`, `quality_approval: false`, and
`monitoring_resumed: true`. Read the final `result.json` for actual evidence.
Do not claim success if the command fails. Prior successful evidence is available
at `reports/rollback-demo-20261021/result.json` and must be identified as a prior run.

The normal main registry has no quality-approved predecessor. Its controller
therefore refuses a main-registry switch. The isolated demo proves operational
automatic recovery without claiming that its model fixtures passed quality gates.
The demo starts/stops its own watcher; it is not a Windows startup service.

## 5. Explain the implementation in lifecycle order

| Stage | Source to open | Explanation |
| --- | --- | --- |
| Version data | `dataset.dvc`, `dvc.yaml`, `dvc.lock` | Git tracks data identities and pipeline definitions; DVC manages the bytes with Google Drive as the approved remote. |
| Validate and prepare | `src/data/validate.py`, `src/data/split.py`, `src/data/preprocess.py` | Check original inputs, preserve fixed splits, resize/pad images and convert XML boxes into YOLO labels. |
| Train and evaluate | `src/training/train.py`, `src/training/evaluate.py` | Train YOLOv8n with recorded parameters and evaluate checkpoints; demonstration reuses completed runs. |
| Track and register | `src/tracking/mlflow_utils.py` | Record experiment evidence and retrieve verified experimental model versions. Registration is not quality approval. |
| Build and deploy | `.github/workflows/ci.yml`, `Dockerfile`, `src/serving/deploy.py` | CI validates/builds; deployment verifies model bytes, starts Docker and checks readiness. CI need not run during the presentation. |
| Serve | `src/serving/api.py` | Receive PNG/JPEG bytes and return detections with model identity. |
| Monitor | `compose.monitoring.yaml`, `src/monitoring/` | Prometheus/Grafana expose operational signals; drift evidence contributes to deterministic health decisions. Drift alone cannot trigger rollback. |
| Recover | `src/monitoring/rollback.py`, `scripts/demo_rollback.py` | The watcher refreshes health evidence, checks recovery gates, restores a recorded predecessor and verifies recovery. |

Suggested spoken explanation:

> This project connects data versioning, repeatable model training, experiment
> tracking, container deployment, monitoring and automatic recovery. Every served
> model has a recorded identity. The live page shows actual detections. In the
> quality demonstration, a deliberately faulty confidence configuration suppresses
> detections. A labeled batch measures the regression; the watcher restores the
> predecessor and verifies that predictions, quality and monitoring recover.

## 6. State the accepted scope accurately

This is a working local college MLOps demonstration. Helmet/head detection is
stronger than person detection. Person detection and validation-label completeness
remain limitations; full three-class quality and production readiness are not
established. Main-registry stable aliases remain unassigned. Annotation work is
paused under the accepted handoff. No training or annotation is required to show
the completed operational demonstration.

Use [the final audit](final-audit.md) for current tests, exact evidence locations,
model comparisons, the demonstration gallery and unresolved security limits.
The original Kaggle source/version remains unknown.

Official VS Code references: [terminal basics](https://code.visualstudio.com/docs/terminal/basics)
and [opening folders from the command line](https://code.visualstudio.com/docs/configure/command-line).
