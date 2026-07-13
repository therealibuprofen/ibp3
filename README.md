# PPC Within-Session Direction Decoding (Python)

Python implementation of within-session single-trial intended movement
direction decoding for the Griggs et al. PPC fUSI dataset.

This project is kept separate from the MATLAB paper repository
`../PPC_directional_tuning` so the Python port is easy to find and run.

## Run

The virtual environment has already been created in `.venv` and the required
packages have been installed.

```bash
cd /Users/ibuprofen/Desktop/griggs3/PPC_direction_decoding_python
MPLCONFIGDIR=/private/tmp/ppc_matplotlib_cache .venv/bin/python run_within_session_decoding.py \
  ../data/doppler_S102_R1+normcorre.mat \
  --cv-scheme kfold \
  --n-permutations 100000 \
  --output-dir ../data/output/python_within_session/S102_R1
```

By default this runs `mode=fixed_memory_3frames`: each trial contributes one
feature vector built from the 3 fUSI frames immediately before memory end, and
the decoder is evaluated once with 10-fold cross-validation.

To reproduce the older per-timepoint dynamic-window analysis, explicitly pass:

```bash
--mode dynamic_time_window
```

Fast smoke test:

```bash
cd /Users/ibuprofen/Desktop/griggs3/PPC_direction_decoding_python
MPLCONFIGDIR=/private/tmp/ppc_matplotlib_cache .venv/bin/python run_within_session_decoding.py \
  ../data/doppler_S27_R99+normcorre.mat \
  --cv-scheme kfold \
  --n-permutations 10 \
  --output-dir ../data/output/python_within_session_smoke/S27_R99_full_preprocess
```

## Batch Run On Server

The batch runner reads `ProjectRecord` metadata, looks for matching
`doppler_S*_R*+normcorre.mat` files, runs each decodable session, and writes
per-session outputs plus a combined `batch_summary.csv/json`.

```bash
cd /path/to/griggs3/PPC_direction_decoding_python
MPLCONFIGDIR=/tmp/ppc_matplotlib_cache .venv/bin/python run_batch_within_session_decoding.py \
  --data-root /data2/yuq1ngr/dataset/data2 \
  --doppler-dir /data2/yuq1ngr/dataset/data2/doppler \
  --project-record /data2/yuq1ngr/dataset/data2/projectrecord \
  --output-dir /data2/yuq1ngr/dataset/data2/output/python_within_session_batch \
  --cv-scheme kfold \
  --n-permutations 100000
```

If `--project-record` is omitted, the script searches common names under
`--data-root` and then falls back to `../data/ProjectRecord_paper.json`.
Use `--dry-run` first to verify which sessions will run:

```bash
.venv/bin/python run_batch_within_session_decoding.py \
  --data-root /data2/yuq1ngr/dataset/data2 \
  --dry-run
```

Optional filters:

```bash
--sessions S27_R99 S76_R1
--sessions 27:99 76:1
--sessions S27
```

To rerun only sessions that failed in a previous batch summary:

```bash
MPLCONFIGDIR=/tmp/ppc_matplotlib_cache .venv/bin/python run_batch_within_session_decoding.py \
  --data-root /data2/yuq1ngr/dataset/data2 \
  --doppler-dir /data2/yuq1ngr/dataset/data2/doppler \
  --project-record /data2/yuq1ngr/dataset/data2/projectrecord \
  --output-dir /data2/yuq1ngr/ibp3/output/python_within_session_batch_rerun_failed \
  --rerun-failed-from-summary /data2/yuq1ngr/ibp3/output/python_within_session_batch/batch_summary.csv \
  --cv-scheme kfold \
  --n-permutations 100000
```

## Notes

- Input files are MATLAB v7.3 task-aligned `doppler_S*_R*+normcorre.mat`
  sessions.
- The distributed `+normcorre` files are treated as already rigid
  motion-corrected by the original MATLAB NoRMCorre pipeline.
- Labels and timing are derived from the existing behavior fields,
  especially `behavior.targetPos`, `behavior.memory`, and
  `coreParams.framerate`.
- Features use the whole finite Power Doppler image, not user-drawn
  anatomical ROIs.
- Default features are fixed to the 3 frames before memory end. Dynamic
  time-window decoding only runs when `--mode dynamic_time_window` is supplied.
- z-score, PCA, and LDA are fit strictly inside each training fold.
- Outputs include JSON, NPZ arrays, plots, and `summary.csv` with accuracy,
  balanced accuracy, and the final/session confusion matrix.
- The multicoder output keeps the possible center-center prediction; for
  angular error it is assigned 180 degrees, matching the existing MATLAB
  evaluation rule.
