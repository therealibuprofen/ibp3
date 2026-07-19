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

For 2-target sessions, the default decoder is `cpca_lda`; `--cpca-m` controls
the final CPCA subspace dimension and defaults to MATLAB `trainCPCA.m`'s `m=1`.

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

For stratified k-fold runs, sessions with a combined-direction class count
below 2 are written as `status=skipped` with
`skip_reason=insufficient_class_count`. If the smallest class has 2-9 trials,
the runner automatically uses that count as `actual_n_splits`; otherwise it
uses the requested 10-fold split.

To diagnose the failed sessions without training the decoder:

```bash
MPLCONFIGDIR=/tmp/ppc_matplotlib_cache python run_batch_within_session_decoding.py \
  --data-root /data2/yuq1ngr/dataset/data2 \
  --doppler-dir /data2/yuq1ngr/dataset/data2/doppler \
  --project-record /data2/yuq1ngr/dataset/data2/ProjectRecord_paper.json \
  --output-dir /data2/yuq1ngr/ibp3/output/python_within_session_batch_failed_diagnostics \
  --rerun-failed-from-summary /data2/yuq1ngr/ibp3/output/python_within_session_batch/batch_summary.csv \
  --cv-scheme kfold \
  --diagnostic-only
```

## Cross-Session 2-Target Linear Baseline

The cross-session runner evaluates the predefined 2-target groups without
mixing monkeys or effectors. It writes pairwise train-session to test-session
matrices plus leave-one-session-out pooled-train results. For diagonal
pairwise cells it reads existing within-session 10-fold benchmark outputs.

```bash
cd /Users/ibuprofen/Desktop/griggs3/PPC_direction_decoding_python
MPLCONFIGDIR=/tmp/ppc_matplotlib_cache .venv/bin/python run_cross_session_decoding.py \
  --data-root ../dataset/data1 \
  --doppler-dir ../dataset/data1/doppler \
  --project-record ../dataset/data1/project_record.json \
  --within-session-results-dir /data2/yuq1ngr/ibp3/output/benchmark/data1 \
  --output-dir output/crosss_session/2target \
  --models pca_lda cpca_lda
```

Outputs:

- `cross_session_pairwise_results.csv/json`
- `cross_session_loso_results.csv/json`
- `accuracy_matrix_<group_name>.csv`
- `balanced_accuracy_matrix_<group_name>.csv`
- `angular_error_matrix_<group_name>.csv`
- `group_summary.csv/json`
- `exclusion_log.csv`

Rows that cannot be evaluated are retained with a `skip_reason` such as
`skip_missing_session_file`, `skip_shape_mismatch`, or
`skip_label_mapping_mismatch`.

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
