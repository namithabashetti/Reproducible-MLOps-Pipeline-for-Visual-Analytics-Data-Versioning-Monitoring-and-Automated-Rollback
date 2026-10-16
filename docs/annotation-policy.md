# Person annotation correction policy, version 1

This policy governs a separate derived correction set. The original annotation
guidelines are unknown. It defines the intended visible-person task without
claiming to recover the original annotator's intent.

- Preserve the original PNG/XML files, all three classes, and split membership.
- Review every training and validation image, including images with no supplied
  person labels. Do not select only images where the model appears correct.
- Annotate each clearly identifiable visible person, including partial people.
  Use a tight rectangle enclosing visible body parts, including the head and worn
  helmet. Do not extend the rectangle to an imagined hidden body or outside the
  image. Independently visible people receive separate boxes even when occluded.
- Reflections and artificial mirrored borders already exist in these source
  images. A separately visible reflected person can need a separate box. Do not
  merge two copies into one person. Mark uncertain fragments for adjudication;
  an unresolved image cannot count as reviewed or enter the corrected export.
- Keep existing helmet/head labels unchanged in this person-only correction.
  Flag visibly incorrect helmet/head labels for adjudication rather than claim
  the resulting three-class ground truth is complete.
- Review proposed boxes against the actual image. Model outputs may prioritize
  inspection, but cannot automatically become reviewed ground truth. Record the
  reviewer, UTC review time, and reason, including for an unchanged/empty result.
- Coordinates are zero-origin pixel boundaries in the **raw** PNG: `[xmin, ymin,
  xmax, ymax]`, with positive width/height and boundaries within image dimensions.
  The export uses the existing letterbox transformation for YOLO labels.
- Preserve image hashes, XML hashes, split hash, and this policy's hash. Any input
  or policy change invalidates the review instead of silently carrying it forward.
- All 3,500 training and 500 validation records must be reviewed with no open
  issues before exporting a candidate for training. A successful structural check
  cannot prove that a reviewer actually looked at an image or judged it correctly.

The final-test and monitoring images are absent from this review set. Freeze the
policy before a separate, prediction-independent integrity review of those pools.
Use final-test predictions only after selecting the model. Corrected training and
validation labels do not by themselves repair the labeled monitoring comparison.

An export is a new candidate dataset, not a model-quality approval. Track the
finished candidate with DVC, retain the original baseline, train a new isolated
run, and compare all three validation classes on the same corrected labels.
Only a model with reviewed quality evidence can be proposed for a stable alias.
Actual approved-model rollback and post-rollback monitoring remain required.

## Local review commands

```powershell
.\.venv\Scripts\python.exe -m scripts.review_annotations init --review reports/person-review-v1
.\.venv\Scripts\python.exe -m scripts.review_annotations check --review reports/person-review-v1
# Only after every record has been reviewed and all issues resolved:
.\.venv\Scripts\python.exe -m scripts.review_annotations export --review reports/person-review-v1 --output data/person-corrected-v1
```

`review.json` starts with the supplied person boxes as **pending**, not verified.
Edit `person_boxes`, `status` (`pending`, `needs_adjudication`, or `reviewed`),
`reviewer`, `reviewed_at`, `reason`, and `issues` for each image. Reviewed records
require a UTC ISO timestamp and an empty issues list. Do not edit source identity
fields. `check` returns exit code 2 for an incomplete review and 1 for invalid
records. Export rejects incomplete reviews, existing destinations, and paths
outside a new `data/person-corrected-*` directory.
