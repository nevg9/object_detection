# Project context

- Python image classification, object detection, data processing, and Flask utilities live in `image_classifier/`, `object_detection/`, `process_data/`, and `flask_web/`. The root `README.md` identifies Ultralytics as a dependency.

# Pbiomon annotation policy

- Image and video downloads must archive all annotation information returned by the API by default: full `finalIdentification` (including every `attributes[]` item, remarks, boxes, and identification metadata), full `ai`, and resource metadata. Preserve raw responses and unknown nested fields; do not silently reduce annotations to species and boxes.
- The v1.4.0 attribute fields are `gender`, `age`, `postureBehavior`, `generalBehavior`, `breedingBehavior`, `groupingBehavior`, and `photoQuality`. Preserve item order, duplicates, and null values. Do not infer a one-to-one mapping between attributes and boxes or animal individuals.
- For annotation backfills, use the existing archive root and ledger with `--annotations-only`. Preserve existing media files, receipts, hashes, and media queue states. Historical attribute updates require rescanning the original identification-date range; the normal overlap window cannot cover arbitrary old edits. See `docs/pbiomon-download.md` for commands and acceptance checks.
- Full downloads and annotation backfills must use the same per-resource bundle schema, paths, and exporter; do not create a parallel backfill-only annotation format. Keep complete source annotations in the archive and derive versioned YOLO datasets separately. YOLO detection labels contain class and normalized boxes; keep gender, age, other attributes, source/frame identities, and annotation versions in companion metadata. Resolve coordinate units before box conversion and box-to-attribute associations before producing per-object attribute supervision; do not guess either.
- The user has authorized persisting the production API secret on `image0_processing` outside the Git checkout, at `/mnt/data2/pbiomon_yolo/state/api-credentials.json`. Keep it owned by the running user with mode `0600` and its containing directory `0700`. The downloader automatically reads this file; preserve it across code updates and never include its contents in Git, logs, command-line arguments, status, or chat replies.

# Validation boundaries

- No unified test command is currently documented in the root `README.md`. Select checks for the affected component; a filename containing `test` does not establish that it is a test suite.
- `image_classifier/test.py` scans a hard-coded dataset path and writes images back in place. `process_data/verify_image_and_remove_image_and_label.py` and `process_data/remove_dir_invalid_image.py` can delete images. Do not run these as routine validation; use disposable copies for checks, or the explicitly authorized dataset and operation for a requested data mutation. Reuse authorization already given for that action and target.
- For instruction or documentation edits, inspect the diff and affected links or syntax. Do not start training, download datasets/models, or run data-cleaning scripts solely to validate prose. For behavior changes, preserve the task's acceptance criteria and test the affected behavior in its required environment.

# Git workflow

- This project's default and integration branch is `main`, matching canonical `origin/HEAD` and the user's corrected instruction. Completed and validated changes must be integrated and pushed to `origin/main`; do not create or use `master` as a substitute for the existing main branch.
- Keep the local checkout and the deployment checkout on `image0_processing` (`/mnt/data2/pbiomon_yolo/code/repository`) on `main`, tracking `origin/main`. Publish local changes through canonical `origin`, then synchronize the server with a fast-forward-only pull and verify commit and runtime-source identity.
- Preserve unrelated user changes and only include files belonging to the completed task when committing or merging.
