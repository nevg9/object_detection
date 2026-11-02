# Project context

- Python image classification, object detection, data processing, and Flask utilities live in `image_classifier/`, `object_detection/`, `process_data/`, and `flask_web/`. The root `README.md` identifies Ultralytics as a dependency.

# Validation boundaries

- No unified test command is currently documented in the root `README.md`. Select checks for the affected component; a filename containing `test` does not establish that it is a test suite.
- `image_classifier/test.py` scans a hard-coded dataset path and writes images back in place. `process_data/verify_image_and_remove_image_and_label.py` and `process_data/remove_dir_invalid_image.py` can delete images. Do not run these as routine validation; use disposable copies for checks, or the explicitly authorized dataset and operation for a requested data mutation. Reuse authorization already given for that action and target.
- For instruction or documentation edits, inspect the diff and affected links or syntax. Do not start training, download datasets/models, or run data-cleaning scripts solely to validate prose. For behavior changes, preserve the task's acceptance criteria and test the affected behavior in its required environment.

# Git workflow

- This project's integration branch is `master`, as explicitly requested by the user. Completed and validated code changes must be committed and integrated into `master`; do not leave confirmed work only on a feature branch.
- Preserve unrelated user changes and only include files belonging to the completed task when committing or merging.
