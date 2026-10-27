# Faculty demonstration guide

This is a working local MLOps demonstration, with documented model-quality limits.
It runs real inference and a real automatic Docker rollback. The failure in Demo
Mode is deliberately injected; a missed webcam frame is not an accuracy measure.

## Start after reboot

In Visual Studio Code, open `F:\major project implementation`, then open a
PowerShell terminal. Keep the laptop connected to power during training.

```powershell
Set-Location -LiteralPath 'F:\major project implementation'
.\scripts\start_demo.ps1
```

This starts the existing Docker services, saved detector, MLflow and console.
It does not train, upload data or approve models. The expected normal detector
is experimental model 4. If a deployment lock exists, finish the active test or
inspect its report before restarting. Never delete a lock just to get past it.
If Windows execution policy blocks the script, the equivalent service commands
are in the script; do not change machine-wide security settings.

| Open | What to explain |
|---|---|
| [Main console](http://127.0.0.1:8767/) | Uploads, camera, bounding boxes, model identity, rollback stages, model/data comparison and event history |
| [Grafana](http://127.0.0.1:3000/d/visual-analytics-operations) | Requests, errors, response time, confidence, detected classes, loaded model, saved evaluations and measured data change |
| [MLflow](http://127.0.0.1:5000/) | Actual runs, parameters, datasets, checkpoint hashes and registered versions |
| [FastAPI docs](http://127.0.0.1:8000/docs) | The inference API behind the console |
| [API health](http://127.0.0.1:8000/health) | Ready state and exact loaded checkpoint/settings |

## A five-minute presentation

1. **Detection.** Upload `demo-images/test_normal_01.png`, or click the first
   gallery image. Point to helmet/head/person boxes, the model label and measured
   inference time. A percentage beside a box is confidence, not measured accuracy.
   For webcam inference, allow camera access to this local page and press Stop
   camera afterward. Frames are not retained by the console.
2. **Different data.** Try `test_changed_data_01.png` and either ladder image.
   Dataset 2 adds construction scenes, but also contains augmented variants.
   These four images are selected validation examples; they do not prove overall
   performance. Their source mapping, hashes and supplied license are in the
   same folder. Use **Download faculty images** to obtain the ZIP with its guide.
3. **Model history.** Open Models & data. Models 1–3 use the original image set
   with different label versions; model 4 adds images and trained for 60 epochs.
   Compare the same 196-image evaluation, not scores from different datasets or
   evaluation settings. Model 5 completed 30 epochs on the fresh 70/15/15
   protocol; its comparison and reserved-test status are shown separately.
4. **Failure.** In Demo Mode select trained model 4 and press **Start rollback
   demo**. The system deploys and measures model 1, then deploys the candidate
   with confidence threshold 1.0. Test the same image during the candidate stage:
   detections disappear because of that explicit configuration fault.
5. **Automatic rollback.** The scheduled audit compares both models on the same
   250 labeled images. A drop of at least five percentage points in AP50–95 must
   pass freshness and identity checks. The watcher independently restores the
   captured predecessor, then verifies three readiness checks, 20 predictions,
   fresh monitoring and recovered detection quality. The normal model 4 is
   restored after the demo. Wait for **Passed** and an online normal detector.
6. **Monitoring.** Open Grafana during the run. Show the loaded model change,
   real requests and latency. Return to the main page for the before/fault/after
   scores and saved event history. Previous events are explicitly historical.

The copied demo registry uses its own version 2 for the candidate. The main page
also shows the trained source version, so this does not overwrite trained model 2.
During a model switch the detector may briefly be unavailable; old boxes are
cleared to avoid displaying a previous model's result under a new model label.

## Optional outage demonstration

Open **Advanced controls and test log** and start the outage test only after the
quality test finishes. It actually stops the candidate container. Consecutive
readiness failures cause the watcher to redeploy the predecessor. No fabricated
error metrics or edited timestamps are used. Do not run the two tests together.

## Answers to likely questions

**Where is detection?** The console sends the uploaded image or webcam frame to
FastAPI `/predict`. The Docker service loads the verified checkpoint and returns
real boxes. The browser draws them over that same image.

**Does every missed object trigger rollback?** No. Without labels, an empty frame
can simply contain no relevant object. The demonstration uses an independently
measured labeled batch to establish a quality failure. Outages and sustained
operational faults have separate health rules.

**What is data drift?** PSI and KS compare brightness, sharpness and output
confidence distributions. Drift means the data changed; it does not establish
that the model is wrong. Drift alone cannot trigger rollback. Expired or
incompatible evidence is shown as unavailable, not healthy.

**Are all models progressively better?** Shared AP50–95 is 32.44%, 32.63%, 33.90%
and 44.91% for models 1–4. Early gains are small, and not every metric improves.
More epochs do not guarantee better accuracy or faster inference.

**Why new training?** The old split was not 70/15/15. The corrected protocol has
5,385 train, 1,154 validation and 1,154 test images, keeping related-image groups
together. Old checkpoints saw some images reassigned to the new test split,
so only fresh protocol training can legitimately use that test set.

**Is this production ready?** No. Person-label completeness and person detection
remain limitations. The original dataset URL/version is unverified. No main
registry stable alias is approved. The college demo exercises the real mechanism
in a separate demo registry; the production path correctly refuses an unapproved
rollback target.

**Where are data versions?** DVC stores dataset identities and Google Drive is
the approved private remote. Git stores source/configuration; MLflow stores runs
and checkpoints. The expanded v1, corrected dataset and faculty images were
uploaded to Google Drive, restored using a separate cache, and verified against
the original file hashes. See the cloud recovery receipt in the final audit folder.

## Troubleshooting and evidence

If the detector is offline, inspect `reports/startup`, Docker logs and
`artifacts/serving/deployment.json`. Do not retrain just to restart the demo.
Each test writes `reports/live-demo-<id>/result.json` and its detailed measurements.
If restoration fails, the result remains failed even if the preceding rollback
recovered. Preserve that evidence and repair the normal deployment before presenting.

The current verification matrix and remaining work are in [final-audit.md](final-audit.md).
Source and label constraints are in [additional-data.md](additional-data.md) and
[annotation-policy.md](annotation-policy.md). Google Drive credentials remain
local and must never be included in screenshots, reports or Git.
