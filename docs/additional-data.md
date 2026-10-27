# Expanded construction-safety experiment — 24 October 2026

The user selected a 60-epoch experiment using additional local images. This is a
new experiment; it does not overwrite the three existing trained models or
automatically replace the detector used by the presentation interface.

## Source and inspection

`archive (2)/css-data` contains 2,801 JPEG images and paired YOLO label files:
2,605 training, 114 validation and 82 test images. The supplied README identifies
Construction Site Safety v28, exported on 1 February 2023, and states CC BY 4.0.
It describes five augmented versions per source. The audit finds 700 filename
families; this is not a verified count of independent scenes. The supplied
`results_yolov8n_100e` files are external results, not this project's training.

The read-only audit decoded every additional and original image, hashed bytes
and decoded pixels, checked label classes and box geometry, and compared normal
and mirrored perceptual hashes at distance 10. It also grouped filename-based
augmentation relatives. Evidence: `reports/additional-data-audit-20261024.json`.

The audit found two duplicate label rows, eight filename families crossing
supplied splits, 43 cross-split related pairs and 16 potential original-data
matches. Similarity matches are conservative candidates, not confirmed copies.
Whole connected groups with any invalid label, split conflict or possible
original overlap were excluded: **108 images in 28 groups**. No supplied
validation/test image was moved into training. Undetected video neighbors,
crops and other scene relationships may remain.

## Class mapping and candidate

| Supplied class | Project class |
| --- | --- |
| 0: Hardhat | 0: helmet |
| 2: NO-Hardhat | 1: head |
| 5: Person | 2: person |

The other seven classes are omitted explicitly. Four validation examples were
inspected to check mapping semantics: NO-Hardhat boxes surround uncovered heads.
This does not establish complete/correct annotations, and source box conventions
differ. Geometry is retained; labels are not marked human-reviewed.

`data/expanded-css-v1` preserves the frozen model-2 inputs: all 3,500 training
images and the fixed 100 validation images. It adds 2,520 training and 96
validation images. Its 77 screened external test images stay reserved.

| Pool | Existing | Added | Candidate total |
| --- | ---: | ---: | ---: |
| Training | 3,500 | 2,520 | 6,020 |
| Validation | 100 | 96 | 196 |
| Reserved external test | 0 | 77 | 77 |

The original test/monitoring pools also remain reserved. The source images are
unchanged. Derived images use the existing 416-pixel letterbox and padding 114;
the training YAML names only training and validation pools. Per-file hashes,
source mappings, exclusions, the audit and attribution are retained with the
candidate. Both raw additions and the candidate are versioned locally with DVC;
this phase does not perform a remote transfer.

## Training and fair comparison

The authorized run uses YOLOv8n, 60 epochs, batch 4, 416 pixels, CUDA, AdamW and
the existing deterministic settings. AMP stays disabled as in the established
training configuration. The epoch override is recorded in the run's lineage and
effective training parameters. The completed 60-epoch fit took about 160 minutes on this laptop.

```powershell
.\.venv\Scripts\python.exe -u -m src.training.train --expanded-data data/expanded-css-v1 --run-name expanded-css-v1-60e --epochs 60
```

After completion, the training command automatically compares models 1, 2 and
the new candidate on the **same** 196-image validation set, and reports old/new
cohorts separately in the run's `comparison/comparison.json`. Comparing the
candidate's expanded-validation score with an older 100-image score is invalid.
Report all three classes and measured inference latency. Do not claim improved
accuracy, efficiency, production approval or a newly deployed model before those
checks. The 77 external test images must not choose epochs, thresholds or weights.

The candidate uses source annotations as supplied. The old validation review
limitations and external label uncertainty remain explicit. More images and
epochs permit an improvement experiment; they do not guarantee improvement.

## Execution record

All 103 local regression tests passed in 76.900 seconds before launch
(`reports/expanded-final-tests-20261024.log`). DVC reported no changed outputs.
The run launched on 24 October 2026 at 20:34 IST and completed all 60 epochs.
Training losses were finite. Registration and the fair comparison completed successfully. Logs are
`reports/expanded-css-v1-60e.log` and `reports/expanded-css-v1-60e.err`.
Completed epochs appear in
`artifacts/training/expanded-css-v1-60e/fit/results.csv`; final metrics,
registration and comparison receipts are written only after their checks pass.

## Completed results and deployment

The new checkpoint is registered as **trained model 4**, preserving models 1–3.
SHA-256: `0dd20f006373ef0fe7dc706e69c9105a39b03ceb3ea0acfd7a6a6336b1fea7bc`.
Registration: `artifacts/training/expanded-css-v1-60e/mlflow.json`.
Comparison: `artifacts/training/expanded-css-v1-60e/comparison/comparison.json`.

| Trained model | Combined 196 AP50–95 | Original 100 | Additional 96 |
| --- | ---: | ---: | ---: |
| 1 | 32.20% | 41.41% | 3.79% |
| 2 | 32.67% | 41.13% | 5.76% |
| 4 | 46.14% | 40.43% | 33.17% |

Model 4 improves combined AP by 13.94 percentage points versus model 1, while
original-image AP drops 0.98 points. These are validation results on the same
cohorts and settings, not independent test-set acceptance. Its person recall is
45.68%; incomplete person detection and source-label uncertainty remain.

Local CPU benchmark: model 1 median 22.65 ms / p95 24.94 ms; model 4 median
22.37 ms / p95 24.78 ms. Forty images, two warmups, two CPU threads, 416 pixels,
confidence 0.25. This is engine inference time, not HTTP latency. Speed is similar;
there is no established efficiency gain from extra epochs.

Model 4 was deployed through the existing experimental deployment path, with
readiness and prediction identity verified (`reports/model4-deploy-20261024.log`).
Main-registry stable aliases remain unchanged. No remote transfer was performed.

The interface includes four additional validation examples selected after comparing
predictions. They illustrate behavior and must not be used as an unbiased benchmark.
`reports/model1-vs-model4-test-pack-20261024.zip` includes raw input PNGs, actual
side-by-side prediction sheets, prediction records and instructions. The raw source
and reserved external test images remain unchanged. Upload raw PNGs, not sheets.

The complete model 1 → 4 → 1 controlled rollback passed on 25 October 2026:
`reports/live-demo-9d6d6df4e8e5/result.json`. Normal model 4 was restored afterward.
See [faculty-demo.md](faculty-demo.md) for the recovery evidence and interrupted prior run.

## 27 October 2026: corrected protocol and current comparison

The explicit shared evaluation in `reports/final-audit-20261026/comparison`
supersedes earlier scores when presenting models together. AP50-95 on the same
196 validation images is 32.44%, 32.63%, 33.90% and 44.91% for versions 1-4.
It explicitly sets `rect=False`; earlier implicit-setting evaluations remain
historical evidence and must not be mixed into this table.

The source export contains 2,801 images; 108 excluded images leave 2,693 eligible
additional images. These include related augmented variants, not 2,693 proven
independent scenes. `data/expanded-protocol-v2` combines them with all 5,000
original images and preserves related groups. Counts are 5,385 training,
1,154 validation and 1,154 test (approximately 70/15/15). The 500 original
monitoring images are validation replay subsets in this new protocol.

Old protected test images remain test; old validation and monitoring remain
validation. Some previously trained-on images move to the new test set.
Consequently only fresh protocol training can use this final test. Versions 1-4
cannot claim unbiased final-test scores on it. Labels still have documented
completeness limitations; changing splits does not repair annotations.

`protocol-v2-30e` is a new experiment initialized from the pretrained YOLOv8n,
not from model 4. It was interrupted during epoch 12 and resumed on 27 October
from the completed epoch-11 checkpoint with optimizer state and input checks.
The pre-resume checkpoints/history and resume source are preserved in its
`resumes` directory and included in registry evidence. Resumed training is not
claimed to be bit-identical to an uninterrupted run. All 30 epochs completed and
the verified checkpoint registered as experimental model 5 on 27 October.
The five-model shared comparison and one-time 1,154-image final test completed.
Model 5 scored 42.25% shared-validation AP50–95 versus model 4's 44.91%, so it
was not promoted. Its separate held-out test scored 50.25% AP50–95; these cohorts
must not be mixed. Model 4 remains live and no stable quality approval was given.
