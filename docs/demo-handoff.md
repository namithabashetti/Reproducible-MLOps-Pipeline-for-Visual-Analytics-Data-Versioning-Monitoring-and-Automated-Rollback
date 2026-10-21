# College demonstration handoff — 21 October 2026

The user accepted a working college MLOps demonstration with model-quality
limitations documented. This replaces full three-class model quality as the
current handoff requirement. It does not certify model quality, approve a
production model, or promote main-registry stable aliases.

## What works

| Component | Evidence |
| --- | --- |
| DVC data versioning and Google Drive storage | Original snapshot recovery previously verified; current local DVC status clean. |
| Preprocessing and reproducible GPU training | Original data preserved; three completed model experiments with recorded inputs and source. |
| MLflow tracking and registry | Three experimental versions; checkpoint and artifact retrieval verified. |
| Docker and FastAPI inference | Local API restored and a real validation-image prediction passed. |
| Prometheus and Grafana | Live API scrape and provisioned seven-panel dashboard verified. |
| Drift and health decisions | Existing PSI/KS reporting and combined health rules; drift alone cannot trigger rollback. |
| Automatic rollback | Real container outage caused the continuous watcher to restore the predecessor automatically in an isolated demo registry. |
| Recovery validation | Three readiness checks, twenty real-image predictions and fresh Prometheus identity passed. |
| GitHub Actions | Existing manual validation/build/deployment workflow; latest change tested locally, no new Actions run claimed. |
| Local regression tests | All 89 tests passed. |

## Open the project

Run commands from `F:\major project implementation` in PowerShell.

- **API health:** <http://127.0.0.1:8000/health>. The prediction endpoint accepts
  raw PNG/JPEG bytes, not multipart uploads. Use the PowerShell example below;
  the serving application disables interactive API documentation.
- **Dashboard:** <http://127.0.0.1:3000>.
- **Metrics:** <http://127.0.0.1:9090>.
- **MLflow:** run the following in a separate terminal, then open
  <http://127.0.0.1:5000>. Its UI is optional; the registry already exists locally.

```powershell
.\.venv\Scripts\python.exe -m src.tracking.mlflow_utils serve
```

For a real image prediction:

```powershell
$sample = 'F:\major project implementation\dataset\images\hard_hat_workers1015.png'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'image/png' -InFile $sample
```

## Demonstrate automatic rollback

With Docker Desktop and the monitoring stack running, choose a new report folder:

```powershell
.\.venv\Scripts\python.exe scripts/demo_rollback.py --output reports/rollback-demo-presentation
```

Allow several minutes. The demo retains the configured startup grace period and
failure-persistence checks. It temporarily replaces the local API, verifies a
healthy candidate, stops its container, and lets the running watcher detect and
recover from the outage. It uses actual Docker deployments, checkpoints and
Prometheus samples, not simulated success results. The original experimental API
is restored at the end. Do not close the terminal during this sequence.

Open the generated `result.json`: expect `status: passed`, a recovered rollback,
three readiness responses, twenty predictions and `monitoring_resumed: true`.
The demo creates its own registry copy and demo-only model aliases; the main
registry remains experimental. Reuse is prevented: use a fresh output directory
for each presentation. No training, dataset download or cloud transfer is needed.

## Restart after reboot

Start Docker Desktop and wait for its engine, then:

```powershell
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe'
& $dockerExe compose -f compose.monitoring.yaml up -d
& $dockerExe start visual-analytics-api
```

Wait for <http://127.0.0.1:8000/health> to return `ready`. The continuous controller
is available through `python -m src.monitoring.rollback --watch`; it is not a
Windows startup service. Outside the isolated demonstration, it needs a genuinely
approved, recorded predecessor before it will switch models. The demo starts and
stops its watcher automatically.

## Limitations to state during assessment

- Person detection remains weak. The latest experiment used unverified assisted
  training labels. Its 100-image validation subset contains known omissions, so
  the reported metrics do not establish reliable three-class detection.
- Validation correction is preserved at 43 reviewed, 51 pending and 6 unresolved
  images. Annotation work is paused under this accepted demonstration handoff.
- Registry registration is not quality approval. Main `stable` and
  `previous_stable` aliases remain unassigned. Demo-only aliases are test fixtures.
- Original Kaggle URL/version is unknown. Do not claim verified dataset provenance.
- Hosting is on this laptop. No public-cloud deployment or production-readiness
  claim is included. Service recovery does not prove detection accuracy.

## Evidence and source

- `reports/rollback-demo-20261021/result.json`: actual automatic recovery and
  restoration of the original API.
- `reports/rollback-demo-20261021/source/sha256.json`: executed source snapshot.
- `reports/closure-services.json`: API prediction, monitoring and dashboard checks.
- `reports/closure-tests.log`: 89 passing tests.
- `reports/assisted-fulltrain-v1-comparison.json`: latest model comparison.
- `scripts/demo_rollback.py`: repeatable demonstration.
- `src/monitoring/rollback.py`: continuous watcher and bounded recovery controller.

The main registry's model approvals were not changed by the demonstration or this
handoff. Keep the local artifacts and DVC cache: Git stores source and pointers,
not all model weights, report evidence or derived dataset bytes.
