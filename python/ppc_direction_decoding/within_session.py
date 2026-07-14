"""Within-session single-trial intended-direction decoding.

This module ports the paper repository's MATLAB single-session
``PCA+LDA`` multicoder analysis to Python while keeping the original data
conventions:

* task-aligned Power Doppler data are shaped ``[y, x, time, trial]``;
* direction labels are derived from ``behavior.targetPos`` geometry;
* horizontal and vertical axes are decoded separately as
  negative/center/positive and then recombined into a 3 x 3 multicoder
  output space;
* z-scoring and PCA are fit inside each training fold only.

The task-aligned ``doppler_S*_R*+normcorre.mat`` files distributed with the
paper are already motion corrected by MATLAB NoRMCorre. If an uncorrected
session is passed and rigid correction is requested, this module raises a
clear error unless a Python rigid-registration backend is installed.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class WithinSessionConfig:
    """Configuration for one within-session decoding run."""

    mode: str = "fixed_memory_3frames"
    decoder_type: str | None = None
    frame_rate_hz: float | None = None
    cv_scheme: str = "kfold"
    n_splits: int = 10
    random_seed: int = 12345
    variance_to_keep: float = 0.95
    cpca_m: int = 1
    chance_accuracy: float = 1.0 / 8.0
    detrend_window: int = 50
    spatial_filter_radius: int = 2
    apply_motion_correction: bool = True
    assume_normcorre_if_present: bool = True
    save_motion_corrected: bool = False
    save_detrended: bool = False
    save_spatial_filtered: bool = False
    n_permutations: int = 100_000
    alpha: float = 0.05
    output_dir: str = "output/decoding/python_within_session"
    max_timepoints: int | None = None
    min_trials_per_timepoint: int = 8
    n_jobs: int = 1
    center_tolerance: float = 1e-6
    diagnostic_only: bool = False


VALID_DECODING_MODES = {"fixed_memory_3frames", "dynamic_time_window"}
VALID_DECODER_TYPES = {"multicoder_pca_lda", "pca_lda", "cpca_lda"}


@dataclass
class TaskConfig:
    """Task settings inferred from project/session metadata."""

    task_type: str
    decoder_type: str
    n_targets: int
    chance_accuracy: float
    frame_rate_hz: float
    frame_rate_source: str
    recording_system: str | None = None
    project_record_path: str | None = None


@dataclass
class AlignedSession:
    """Task-aligned fUSI images and behavior-derived labels.

    images:
        Power Doppler data with shape ``[y, x, n_timepoints, n_trials]``.
    target_pos:
        Behavioral target positions with shape ``[n_trials, 2]`` in the
        native task coordinate system.
    time_from_trial_start_s / time_from_cue_s:
        Per-fUSI-frame time axes in seconds. The cue reference follows the
        existing MATLAB code's ``getEpochs``/``fixTime`` convention.
    """

    images: np.ndarray
    behavior: list[dict[str, Any]]
    target_pos: np.ndarray
    valid_trial_mask: np.ndarray
    time_from_trial_start_s: np.ndarray
    time_from_cue_s: np.ndarray
    cue_index: int
    frame_rate_hz: float
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _require(module_name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise ImportError(
            f"Missing Python dependency '{module_name}'. Install '{hint}' to run "
            "within-session decoding on MATLAB v7.3/HDF5 data."
        ) from exc


def _matlab_char_to_str(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.ravel().astype(str)).strip()
    if np.issubdtype(arr.dtype, np.integer):
        return "".join(chr(int(x)) for x in arr.ravel() if int(x) != 0).strip()
    return str(value)


def _load_hdf5_mat_value(handle: Any, obj: Any, refs: dict[int, Any]) -> Any:
    """Recursively load the subset of MATLAB v7.3 objects used here.

    MATLAB stores structs/cells as HDF5 object references. The code below is
    intentionally conservative: it decodes numeric arrays, char arrays,
    structs, and cell/ref arrays; unsupported objects remain visible in the
    returned dict instead of being silently dropped.
    """

    h5py = _require("h5py")

    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        matlab_class = obj.attrs.get("MATLAB_class", b"")
        if isinstance(matlab_class, bytes):
            matlab_class = matlab_class.decode("utf-8", errors="ignore")

        if matlab_class == "char":
            return _matlab_char_to_str(data)

        if data.dtype == h5py.ref_dtype:
            out = np.empty(data.shape, dtype=object)
            for idx, ref in np.ndenumerate(data):
                if not ref:
                    out[idx] = None
                else:
                    key = handle[ref].name
                    if key not in refs:
                        refs[key] = _load_hdf5_mat_value(handle, handle[ref], refs)
                    out[idx] = refs[key]
            return out

        # MATLAB v7.3 arrays are stored with reversed dimension order.
        arr = np.array(data)
        if arr.ndim > 1:
            arr = arr.transpose()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return arr

    if isinstance(obj, h5py.Group):
        out: dict[str, Any] = {}
        for key in obj.keys():
            out[key] = _load_hdf5_mat_value(handle, obj[key], refs)
        return out

    return obj


def _mat_struct_array_to_list(value: Any) -> list[dict[str, Any]]:
    """Convert a MATLAB struct/cell representation to list-of-dicts."""

    if isinstance(value, list):
        return [v if isinstance(v, dict) else {"value": v} for v in value]
    if isinstance(value, np.ndarray) and value.dtype == object:
        flat = value.ravel(order="F")
        return [v if isinstance(v, dict) else {"value": v} for v in flat]
    if isinstance(value, dict):
        lengths = []
        for v in value.values():
            if isinstance(v, np.ndarray) and v.dtype == object:
                lengths.append(v.size)
            elif isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] > 1:
                lengths.append(v.shape[0])
        n = max(lengths) if lengths else 1
        rows: list[dict[str, Any]] = []
        for i in range(n):
            row = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray) and v.dtype == object and v.size == n:
                    row[k] = v.ravel(order="F")[i]
                elif isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
                    row[k] = v[i]
                else:
                    row[k] = v
            rows.append(row)
        return rows
    raise TypeError(f"Cannot convert behavior object of type {type(value)} to records")


def load_mat73_session(mat_path: str | Path) -> dict[str, Any]:
    """Load a paper MATLAB v7.3 session file.

    Parameters
    ----------
    mat_path:
        Path to ``doppler_S*_R*+normcorre.mat`` or an equivalent HDF5 MAT.

    Returns
    -------
    dict
        Keys include any variables found in the MAT file. For decoding, the
        loader expects at least ``iDop`` and ``behavior``.
    """

    h5py = _require("h5py")
    mat_path = Path(mat_path)
    try:
        with h5py.File(mat_path, "r") as handle:
            refs: dict[int, Any] = {}
            out = {
                key: _load_hdf5_mat_value(handle, handle[key], refs)
                for key in handle.keys()
                if key != "#refs#"
            }
    except OSError as exc:
        message = str(exc)
        if "truncated file" in message.lower():
            actual_size = mat_path.stat().st_size if mat_path.exists() else 0
            raise OSError(
                f"Cannot open '{mat_path}': the MATLAB v7.3/HDF5 file appears truncated "
                f"or still copying. Current size is {actual_size} bytes. Re-copy or "
                "re-download the file, wait for the transfer to finish, then rerun decoding."
            ) from exc
        raise
    out["_source_path"] = str(mat_path)
    return out


def _get_scalar(mapping: dict[str, Any], names: Iterable[str], default: float | None) -> float | None:
    for name in names:
        if name in mapping:
            value = mapping[name]
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)[0]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _get_first_present(mapping: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def _get_int(mapping: dict[str, Any], names: Iterable[str]) -> int | None:
    value = _get_first_present(mapping, names)
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else None
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def infer_task_config(
    metadata: dict[str, Any],
    *,
    frame_rate_hz: float | None = None,
    decoder_type: str | None = None,
) -> TaskConfig:
    """Infer task type, decoder, and frame rate from session metadata.

    The function intentionally refuses to invent a generic 1 Hz fallback.
    Frame rate must come from an explicit override/file field or from a
    known ``RecordingSystem`` default.
    """

    n_targets = _get_int(metadata, ("nTargets", "n_targets", "NTargets"))
    if n_targets == 8:
        task_type = "8target"
        default_decoder = "multicoder_pca_lda"
        chance = 1.0 / 8.0
    elif n_targets == 2:
        task_type = "2target"
        default_decoder = "cpca_lda"
        chance = 1.0 / 2.0
    else:
        raise ValueError(
            "Cannot infer task type: metadata must identify nTargets as 8 or 2. "
            f"Found nTargets={n_targets!r}; metadata keys={sorted(metadata.keys())}."
        )

    selected_decoder = decoder_type or str(metadata.get("decoder_type") or default_decoder)
    selected_decoder = selected_decoder.lower()
    if selected_decoder not in VALID_DECODER_TYPES:
        raise ValueError(
            f"Unsupported decoder_type '{selected_decoder}'. Choose one of {sorted(VALID_DECODER_TYPES)}."
        )
    if task_type == "8target" and selected_decoder != "multicoder_pca_lda":
        raise ValueError("8-target decoding currently requires decoder_type='multicoder_pca_lda'.")
    if task_type == "2target" and selected_decoder not in {"pca_lda", "cpca_lda"}:
        raise ValueError("2-target decoding requires decoder_type='cpca_lda' or 'pca_lda'.")

    recording_system = _get_first_present(metadata, ("RecordingSystem", "recording_system"))
    if isinstance(recording_system, np.ndarray):
        recording_system = str(recording_system.reshape(-1)[0])
    if recording_system is not None:
        recording_system = str(recording_system)

    source = "explicit"
    rate = frame_rate_hz
    if rate is None:
        rate = _get_scalar(
            metadata,
            (
                "frame_rate_hz",
                "frameRateHz",
                "framerate",
                "frameRate",
                "FrameRate",
                "fs",
            ),
            default=None,
        )
        source = "file_or_metadata"
    if not rate or rate <= 0:
        recording_to_rate = {
            "prototypeRT": 2.0,
            "offlineVantage": 1.0,
        }
        rate = recording_to_rate.get(recording_system or "")
        source = "ProjectRecord.RecordingSystem"
    if not rate or rate <= 0:
        raise ValueError(
            "Cannot determine frame_rate_hz from file metadata, user config, or "
            "RecordingSystem. Pass --frame-rate-hz explicitly for this dataset."
        )

    return TaskConfig(
        task_type=task_type,
        decoder_type=selected_decoder,
        n_targets=int(n_targets),
        chance_accuracy=chance,
        frame_rate_hz=float(rate),
        frame_rate_source=source,
        recording_system=recording_system,
        project_record_path=metadata.get("project_record_path"),
    )


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1, order="F")[0]
    return value


def _parse_clock_seconds(value: str) -> float:
    match = re.match(r"^\s*(\d+):(\d+):(\d+)(?:[,.](\d+))?\s*$", value)
    if not match:
        return float(value)
    hours, minutes, seconds, frac = match.groups()
    frac_s = float(f"0.{frac}") if frac else 0.0
    return int(hours) * 3600.0 + int(minutes) * 60.0 + int(seconds) + frac_s


def _parse_time(records: list[dict[str, Any]], field: str) -> np.ndarray:
    values = []
    for row in records:
        value = _unwrap_singleton(row.get(field, np.nan))
        if isinstance(value, dict):
            for candidate in ("time", "t", "value"):
                if candidate in value:
                    value = value[candidate]
                    break
        if isinstance(value, str):
            try:
                value = _parse_clock_seconds(value)
            except ValueError:
                value = np.nan
        if isinstance(value, np.ndarray):
            value = _unwrap_singleton(value)
            value = value.reshape(-1)[0] if value.size else np.nan
        values.append(float(value) if np.isfinite(value) else np.nan)
    return np.asarray(values, dtype=float)


def _extract_success(records: list[dict[str, Any]]) -> np.ndarray:
    success = []
    for row in records:
        value = _unwrap_singleton(row.get("success", True))
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else False
        success.append(bool(value))
    return np.asarray(success, dtype=bool)


def _extract_target_pos(records: list[dict[str, Any]]) -> np.ndarray:
    target_pos = []
    missing = []
    for i, row in enumerate(records):
        value = _unwrap_singleton(row.get("targetPos", None))
        if value is None:
            missing.append(i)
            target_pos.append([np.nan, np.nan])
            continue
        value = _unwrap_singleton(value)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 2:
            missing.append(i)
            target_pos.append([np.nan, np.nan])
        else:
            target_pos.append([arr[0], arr[1]])
    if missing:
        LOGGER.warning("Missing targetPos for %d trials", len(missing))
    return np.asarray(target_pos, dtype=float)


def _infer_session_run(path_or_name: str) -> tuple[int, int] | None:
    match = re.search(r"S(\d+)_R(\d+)", Path(path_or_name).name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _project_record_candidates(source_path: str) -> list[Path]:
    source = Path(source_path)
    candidates = []
    if source_path:
        for parent in [source.parent, source.parent.parent, source.parent.parent.parent]:
            candidates.extend(
                [
                    parent / "ProjectRecord_paper.json",
                    parent / "ProjectRecord.json",
                    parent / "project_record.json",
                    parent / "projectrecord.json",
                    parent / "projectrecord",
                ]
            )
        candidates.append(
            source.parent.parent / "PPC_directional_tuning" / "Project Records" / "ProjectRecord_paper.json"
        )
    candidates.extend(
        [
            Path("data") / "ProjectRecord_paper.json",
            Path("dataset") / "data1" / "project_record.json",
            Path("dataset") / "data2" / "ProjectRecord_paper.json",
            Path("PPC_directional_tuning") / "Project Records" / "ProjectRecord_paper.json",
        ]
    )
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _project_record_metadata_from_path(source_path: str) -> tuple[dict[str, Any] | None, str | None]:
    session_run = _infer_session_run(source_path)
    if session_run is None:
        return None, None
    session_id, run_id = session_run
    for project_record in _project_record_candidates(source_path):
        if not project_record.exists():
            continue
        with project_record.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if int(row.get("Session", -1)) == session_id and int(row.get("Run", -1)) == run_id:
                out = dict(row)
                out["project_record_path"] = str(project_record)
                return out, str(project_record)
    return None, None


def _median_finite(values: np.ndarray, name: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot derive task epoch timing: behavior field '{name}' is missing/empty.")
    return float(np.median(finite))


def _finite_unique_target_count(target_pos: np.ndarray, decimals: int = 6) -> int | None:
    target_pos = np.asarray(target_pos, dtype=float)
    finite = np.all(np.isfinite(target_pos), axis=1)
    if not np.any(finite):
        return None
    unique = np.unique(np.round(target_pos[finite], decimals=decimals), axis=0)
    return int(unique.shape[0])


def _get_file_frame_rate(session: dict[str, Any]) -> tuple[float | None, str | None]:
    core_params = session.get("coreParams", {})
    if isinstance(core_params, dict):
        rate = _get_scalar(core_params, ("framerate", "frameRate", "FrameRate", "fs"), default=None)
        if rate and rate > 0:
            return float(rate), "coreParams"
    uf = session.get("UF", {})
    if isinstance(uf, dict):
        rate = _get_scalar(uf, ("FrameRate", "dopFrameRate", "frameRate", "framerate"), default=None)
        if rate and rate > 0:
            return float(rate), "UF"
    return None, None


def _build_trial_aligned_from_continuous(
    session: dict[str, Any],
    behavior: list[dict[str, Any]],
    *,
    frame_rate_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert continuous ``dop[y, x, time]`` data into trial-aligned 4D images."""

    if "dop" not in session or "timestamps" not in session:
        raise KeyError("Continuous RT/BMI sessions require 'dop' and 'timestamps' fields.")

    dop = np.asarray(session["dop"], dtype=np.float32)
    timestamps = np.asarray(session["timestamps"], dtype=float).reshape(-1)
    if dop.ndim != 3:
        raise ValueError(f"Expected continuous dop shape [y, x, time], found {dop.shape}")
    if dop.shape[2] != timestamps.size:
        if dop.shape[0] == timestamps.size:
            dop = np.transpose(dop, (1, 2, 0))
        else:
            raise ValueError(
                f"Cannot match continuous dop shape {dop.shape} to timestamps length {timestamps.size}."
            )

    trial_start = _parse_time(behavior, "trialstart")
    cue = _parse_time(behavior, "cue")
    memory = _parse_time(behavior, "memory")
    target_acquire = _parse_time(behavior, "target_acquire")
    iti = _parse_time(behavior, "iti")

    cue_reference = memory if np.isfinite(memory).any() else cue
    end_rel = iti - trial_start
    fallback_end_rel = target_acquire - trial_start
    finite_end = end_rel[np.isfinite(end_rel) & (end_rel > 0)]
    if finite_end.size == 0:
        finite_end = fallback_end_rel[np.isfinite(fallback_end_rel) & (fallback_end_rel > 0)]
    if finite_end.size == 0:
        raise ValueError("Cannot trial-align continuous dop: no finite trial end timing fields.")

    # Use a robust duration so occasional long/failed trials do not force a
    # very large dense 4D array. The fixed-memory decoder only needs frames up
    # to target acquisition; dynamic mode can still evaluate the common window.
    duration_s = float(np.nanpercentile(finite_end, 90))
    required_rel = target_acquire - trial_start
    finite_required = required_rel[np.isfinite(required_rel) & (required_rel > 0)]
    if finite_required.size:
        duration_s = max(duration_s, float(np.nanpercentile(finite_required, 90)))
    n_time = max(3, int(math.ceil(duration_s * frame_rate_hz)) + 1)

    y, x, _ = dop.shape
    images = np.empty((y, x, n_time, len(behavior)), dtype=np.float32)
    sample_offsets = np.arange(n_time, dtype=float) / frame_rate_hz
    tolerance_s = 0.75 / frame_rate_hz
    missing = 0
    for trial_id, start_s in enumerate(trial_start):
        if not np.isfinite(start_s):
            images[:, :, :, trial_id] = np.nan
            missing += n_time
            continue
        sample_times = start_s + sample_offsets
        right = np.searchsorted(timestamps, sample_times, side="left")
        right = np.clip(right, 0, timestamps.size - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(timestamps[left] - sample_times) <= np.abs(timestamps[right] - sample_times)
        nearest = np.where(choose_left, left, right)
        too_far = np.abs(timestamps[nearest] - sample_times) > tolerance_s
        images[:, :, :, trial_id] = dop[:, :, nearest]
        if np.any(too_far):
            images[:, :, too_far, trial_id] = np.nan
            missing += int(np.sum(too_far))

    finite_cue_rel = cue_reference - trial_start
    finite_cue_rel = finite_cue_rel[np.isfinite(finite_cue_rel) & (finite_cue_rel > 0)]
    cue_index = int(round(_median_finite(finite_cue_rel, "cue/memory - trialstart") * frame_rate_hz))
    cue_index = max(0, min(cue_index, n_time - 1))
    return images, {
        "continuous_source": True,
        "continuous_trial_alignment": "nearest_timestamp_from_trialstart",
        "continuous_missing_frame_count": int(missing),
        "continuous_duration_s": duration_s,
        "continuous_n_timepoints": int(n_time),
        "continuous_cue_index": int(cue_index),
    }


def align_fusi_and_behavior(
    session: dict[str, Any],
    session_id: str | None = None,
    *,
    decoder_type: str | None = None,
    frame_rate_hz: float | None = None,
) -> AlignedSession:
    """Align task-aligned fUSI frames with behavior and construct trial labels.

    The distributed ``doppler_S*_R*+normcorre.mat`` files are already
    task-aligned by the MATLAB loading pipeline. This function verifies shape
    agreement, applies the existing successful-trial rule, extracts
    ``targetPos``, and reconstructs both requested time axes.

    Returns an ``AlignedSession`` with image shape ``[y, x, time, trial]``.
    """

    if "behavior" not in session:
        raise KeyError(f"Session is missing 'behavior'. Found fields: {sorted(session.keys())}")

    behavior = _mat_struct_array_to_list(session["behavior"])
    success = _extract_success(behavior)
    target_pos = _extract_target_pos(behavior)
    valid = success & np.all(np.isfinite(target_pos), axis=1)

    source_path = str(session.get("_source_path", ""))
    project_record, project_record_path = _project_record_metadata_from_path(source_path)
    metadata_for_inference: dict[str, Any] = dict(project_record or {})
    metadata_for_inference["source_path"] = source_path
    if project_record_path:
        metadata_for_inference["project_record_path"] = project_record_path
    if _get_int(metadata_for_inference, ("Session",)) is None:
        sr = _infer_session_run(source_path)
        if sr:
            metadata_for_inference["Session"], metadata_for_inference["Run"] = sr
    if _get_int(metadata_for_inference, ("nTargets",)) is None:
        inferred_n_targets = _finite_unique_target_count(target_pos)
        if inferred_n_targets is not None:
            metadata_for_inference["nTargets"] = inferred_n_targets

    file_rate, file_rate_source = _get_file_frame_rate(session)
    explicit_rate = frame_rate_hz if frame_rate_hz is not None else file_rate
    task_config = infer_task_config(
        metadata_for_inference,
        frame_rate_hz=explicit_rate,
        decoder_type=decoder_type,
    )
    if frame_rate_hz is not None:
        task_config.frame_rate_source = "user"
    elif file_rate is not None:
        task_config.frame_rate_source = file_rate_source or "file"
    frame_rate = task_config.frame_rate_hz

    continuous_log: dict[str, Any] = {}
    if "iDop" in session:
        images = np.asarray(session["iDop"], dtype=np.float32)
        if images.ndim != 4:
            raise ValueError(f"Expected iDop shape [y, x, time, trial], found {images.shape}")
        if len(behavior) != images.shape[3]:
            raise ValueError(
                "Behavior/image trial mismatch. "
                f"behavior has {len(behavior)} rows; iDop has {images.shape[3]} trials. "
                "Use the existing MATLAB loader/metadata to provide task-aligned successful trials."
            )
    elif "dop" in session:
        images, continuous_log = _build_trial_aligned_from_continuous(
            session, behavior, frame_rate_hz=frame_rate
        )
    else:
        raise KeyError(f"Session is missing 'iDop' or continuous 'dop'. Found fields: {sorted(session.keys())}")

    fixation_hold = _parse_time(behavior, "fixationhold")
    cue = _parse_time(behavior, "cue")
    memory = _parse_time(behavior, "memory")
    target_acquire = _parse_time(behavior, "target_acquire")

    if np.isfinite(memory).any():
        cue_reference = memory
        cue_name = "memory"
    else:
        cue_reference = cue
        cue_name = "cue"

    fixation_rel = fixation_hold - cue_reference
    fix_start_s = _median_finite(fixation_rel[valid], "fixationhold")
    if continuous_log.get("continuous_source"):
        cue_index = int(continuous_log["continuous_cue_index"])
    else:
        cue_index = int(math.ceil(abs(fix_start_s) * frame_rate))
    cue_index = max(0, min(cue_index, images.shape[2] - 1))

    frame_index = np.arange(images.shape[2], dtype=float)
    time_from_trial_start_s = frame_index / frame_rate
    time_from_cue_s = (frame_index - cue_index) / frame_rate

    metadata = {
        "cue_reference": cue_name,
        "frame_rate_hz": float(frame_rate),
        "frame_rate_source": task_config.frame_rate_source,
        "recording_system": task_config.recording_system,
        "project_record_path": task_config.project_record_path,
        "task_type": task_config.task_type,
        "decoder_type": task_config.decoder_type,
        "n_targets": task_config.n_targets,
        "task_config": asdict(task_config),
        "project_record": project_record or {},
        "fix_start_s_relative_to_cue": fix_start_s,
        "go_cue_s_relative_to_cue": float(np.nanmedian(target_acquire - cue_reference)),
        "source_path": source_path,
        **continuous_log,
    }
    sid = session_id or _infer_session_id(session.get("_source_path", "session"))
    return AlignedSession(
        images=images,
        behavior=behavior,
        target_pos=target_pos,
        valid_trial_mask=valid,
        time_from_trial_start_s=time_from_trial_start_s,
        time_from_cue_s=time_from_cue_s,
        cue_index=cue_index,
        frame_rate_hz=frame_rate,
        session_id=sid,
        metadata=metadata,
    )


def _infer_session_id(path_or_name: str) -> str:
    name = Path(path_or_name).name
    match = re.search(r"S(\d+)_R(\d+)", name)
    if match:
        return f"S{match.group(1)}_R{match.group(2)}"
    return Path(name).stem or "session"


def _causal_moving_mean(data: np.ndarray, window: int) -> np.ndarray:
    flat = data.reshape((-1, data.shape[2] * data.shape[3]), order="F")
    finite = np.isfinite(flat)
    values = np.where(finite, flat, 0.0)
    csum = np.cumsum(values, axis=1, dtype=np.float64)
    ccount = np.cumsum(finite, axis=1, dtype=np.float64)
    out = np.empty_like(flat, dtype=np.float32)
    for t in range(flat.shape[1]):
        start = max(0, t - window)
        total = csum[:, t] - (csum[:, start - 1] if start > 0 else 0.0)
        count = ccount[:, t] - (ccount[:, start - 1] if start > 0 else 0.0)
        out[:, t] = np.divide(total, count, out=np.zeros(total.shape, dtype=np.float64), where=count > 0)
    return out.reshape(data.shape, order="F")


def _pillbox_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (x * x + y * y) <= radius * radius
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def preprocess_power_doppler_session(
    images: np.ndarray,
    config: WithinSessionConfig,
    *,
    source_path: str = "",
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess a session's Power Doppler images.

    Parameters
    ----------
    images:
        ``[y, x, time, trial]`` Power Doppler array.
    config:
        Preprocessing settings. Motion correction is accepted as already
        complete when the source file name contains ``+normcorre``.

    Returns
    -------
    preprocessed, log:
        Same shape as input. Detrending uses the causal sliding window from
        MATLAB ``detrend_sliding_window``. Spatial filtering uses a pillbox
        filter with radius ``config.spatial_filter_radius``.
    """

    log: dict[str, Any] = {"input_shape": list(images.shape)}
    out = np.asarray(images, dtype=np.float32, order="F").copy()
    output_dir = Path(output_dir or config.output_dir)

    normcorre_present = "+normcorre" in str(source_path).lower()
    rt_continuous_present = "rt_fus_data" in str(source_path).lower()
    if config.apply_motion_correction:
        if normcorre_present and config.assume_normcorre_if_present:
            log["motion_correction"] = "already_normcorre_from_source_file"
        elif rt_continuous_present:
            log["motion_correction"] = "rt_continuous_source_assumed_preprocessed"
        else:
            # The paper repository's implemented motion-correction backend is
            # MATLAB NoRMCorre rigid registration. Python re-running of
            # NoRMCorre is not available in this repository, so refusing here
            # is safer than substituting a different algorithm silently.
            raise NotImplementedError(
                "Rigid NoRMCorre motion correction is not available in this Python port. "
                "Use a task-aligned '+normcorre.mat' file produced by the existing MATLAB "
                "pipeline, or add a verified Python rigid-registration backend."
            )
    else:
        log["motion_correction"] = "disabled_by_config"

    if config.save_motion_corrected:
        _save_intermediate(output_dir, "motion_corrected.npy", out)

    if config.detrend_window and config.detrend_window > 0:
        moving = _causal_moving_mean(out, config.detrend_window)
        mean_per_voxel = np.nanmean(out.reshape((-1, out.shape[2] * out.shape[3]), order="F"), axis=1)
        mean_per_voxel = np.nan_to_num(mean_per_voxel, nan=0.0, posinf=0.0, neginf=0.0)
        mean_per_voxel = mean_per_voxel.reshape((out.shape[0], out.shape[1], 1, 1), order="F")
        out = out - moving + mean_per_voxel
        log["detrend_window"] = config.detrend_window
        if config.save_detrended:
            _save_intermediate(output_dir, "detrended.npy", out)

    if config.spatial_filter_radius and config.spatial_filter_radius > 0:
        scipy_ndimage = _require("scipy.ndimage", "scipy")
        kernel = _pillbox_kernel(config.spatial_filter_radius)
        for trial in range(out.shape[3]):
            for t in range(out.shape[2]):
                out[:, :, t, trial] = scipy_ndimage.convolve(
                    out[:, :, t, trial], kernel, mode="constant", cval=0.0
                )
        log["spatial_filter"] = {"type": "pillbox", "radius_voxels": config.spatial_filter_radius}
        if config.save_spatial_filtered:
            _save_intermediate(output_dir, "spatial_filtered.npy", out)

    log["output_shape"] = list(out.shape)
    return out, log


def _save_intermediate(output_dir: Path, filename: str, array: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / filename, array)


def make_multicoder_labels(target_pos: np.ndarray, center_tolerance: float = 1e-6) -> dict[str, Any]:
    """Create horizontal/vertical multicoder labels from target positions.

    Horizontal and vertical labels match the MATLAB implementation:
    ``1 = negative`` (left/down), ``2 = center``, ``3 = positive``
    (right/up). The combined label is ``h + 3 * (v - 1)``, so center-center
    is label 5 and is retained as a possible prediction even though the
    actual eight-target task has no center target.
    """

    target_pos = np.asarray(target_pos, dtype=float)
    labels = np.full((target_pos.shape[0], 2), np.nan, dtype=float)
    for dim in range(2):
        values = target_pos[:, dim]
        labels[values < -center_tolerance, dim] = 1
        labels[np.abs(values) <= center_tolerance, dim] = 2
        labels[values > center_tolerance, dim] = 3
    combined = labels[:, 0] + 3 * (labels[:, 1] - 1)

    angles_deg = np.degrees(np.arctan2(target_pos[:, 1], target_pos[:, 0]))
    angles_deg = np.mod(angles_deg, 360.0)
    combined_to_angle: dict[int, float] = {}
    for combined_label in sorted(set(combined[np.isfinite(combined)].astype(int))):
        idx = np.where(combined.astype(int) == combined_label)[0]
        angle_values = angles_deg[idx]
        if angle_values.size:
            # Use circular mean in case repeated target positions differ by
            # tiny floating-point jitter.
            radians = np.deg2rad(angle_values)
            mean_angle = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
            combined_to_angle[int(combined_label)] = float(mean_angle % 360.0)

    label_names = {
        1: "left-down",
        2: "center-down",
        3: "right-down",
        4: "left-center",
        5: "center-center",
        6: "right-center",
        7: "left-up",
        8: "center-up",
        9: "right-up",
    }
    axis_int = np.full(labels.shape, -1, dtype=int)
    axis_int[np.isfinite(labels)] = labels[np.isfinite(labels)].astype(int)
    combined_int = np.full(combined.shape, -1, dtype=int)
    combined_int[np.isfinite(combined)] = combined[np.isfinite(combined)].astype(int)

    return {
        "axis_labels": axis_int,
        "combined_labels": combined_int,
        "target_angles_deg": angles_deg,
        "combined_to_angle_deg": combined_to_angle,
        "combined_label_names": label_names,
        "center_tolerance": float(center_tolerance),
    }


def make_binary_labels(target_pos: np.ndarray, decimals: int = 6) -> dict[str, Any]:
    """Encode exactly two unique target positions as labels 0 and 1."""

    target_pos = np.asarray(target_pos, dtype=float)
    finite = np.all(np.isfinite(target_pos), axis=1)
    rounded = np.round(target_pos[finite], decimals=decimals)
    unique = np.unique(rounded, axis=0) if rounded.size else np.empty((0, 2), dtype=float)
    if unique.shape[0] != 2:
        actual = unique.astype(float).tolist()
        raise ValueError(
            "2-target binary decoding requires exactly two unique finite target positions; "
            f"found {unique.shape[0]}: {actual}"
        )

    labels = np.full(target_pos.shape[0], -1, dtype=int)
    mapping: dict[int, list[float]] = {}
    pos_to_label = {tuple(row.tolist()): i for i, row in enumerate(unique)}
    for i, pos in enumerate(np.round(target_pos, decimals=decimals)):
        if np.all(np.isfinite(pos)):
            labels[i] = pos_to_label[tuple(pos.tolist())]
    for pos, label in pos_to_label.items():
        mapping[int(label)] = [float(pos[0]), float(pos[1])]

    angles_deg = np.degrees(np.arctan2(target_pos[:, 1], target_pos[:, 0]))
    angles_deg = np.mod(angles_deg, 360.0)
    label_to_angle: dict[int, float] = {}
    for label in sorted(mapping):
        idx = labels == label
        radians = np.deg2rad(angles_deg[idx])
        mean_angle = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
        label_to_angle[int(label)] = float(mean_angle % 360.0)

    message = f"Binary target position to label mapping: {json.dumps(mapping, sort_keys=True)}"
    print(message)
    LOGGER.info(message)
    return {
        "binary_labels": labels,
        "target_angles_deg": angles_deg,
        "label_to_target_pos": mapping,
        "combined_to_angle_deg": label_to_angle,
        "combined_label_names": {label: f"target_{label}" for label in mapping},
        "round_decimals": int(decimals),
    }


def _distribution_dict(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    values = values[values >= 0]
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _target_pos_distribution(target_pos: np.ndarray, decimals: int = 6) -> list[dict[str, Any]]:
    target_pos = np.asarray(target_pos, dtype=float)
    finite = np.all(np.isfinite(target_pos), axis=1)
    rounded = np.round(target_pos[finite], decimals=decimals)
    if rounded.size == 0:
        return []
    unique, counts = np.unique(rounded, axis=0, return_counts=True)
    return [
        {"target_pos": [float(row[0]), float(row[1])], "count": int(count)}
        for row, count in zip(unique, counts)
    ]


def _build_trial_label_diagnostics(
    *,
    aligned: AlignedSession,
    labels: dict[str, Any],
    valid_mask: np.ndarray,
    trial_indices: np.ndarray,
    requested_n_splits: int,
    cv_scheme: str,
    center_tolerance: float,
) -> dict[str, Any]:
    target_pos = aligned.target_pos[trial_indices]
    axis_labels = labels["axis_labels"][trial_indices]
    combined = labels["combined_labels"][trial_indices]
    combined_distribution = _distribution_dict(combined)
    singleton_labels = sorted(
        int(label) for label, count in combined_distribution.items() if int(count) == 1
    )
    positive_counts = [int(v) for v in combined_distribution.values() if int(v) > 0]
    min_class_count = min(positive_counts) if positive_counts else 0
    if cv_scheme.lower() in {"kfold", "10fold", "stratifiedkfold"}:
        actual_n_splits = min(requested_n_splits, min_class_count) if min_class_count >= 2 else 0
    elif cv_scheme.lower() in {"loo", "leaveoneout", "leave-one-out"}:
        actual_n_splits = int(trial_indices.size)
    else:
        actual_n_splits = 0

    near_zero_x = np.isfinite(target_pos[:, 0]) & (np.abs(target_pos[:, 0]) < center_tolerance) & (target_pos[:, 0] != 0)
    near_zero_y = np.isfinite(target_pos[:, 1]) & (np.abs(target_pos[:, 1]) < center_tolerance) & (target_pos[:, 1] != 0)

    return {
        "n_trials_total": int(aligned.images.shape[3]),
        "n_valid_trials_after_success_and_targetPos": int(valid_mask.sum()),
        "n_trials_entering_CV": int(trial_indices.size),
        "target_pos_distribution_round6": _target_pos_distribution(target_pos, decimals=6),
        "target_pos_near_zero_nonzero_x": bool(np.any(near_zero_x)),
        "target_pos_near_zero_nonzero_y": bool(np.any(near_zero_y)),
        "target_pos_near_zero_nonzero_x_count": int(np.sum(near_zero_x)),
        "target_pos_near_zero_nonzero_y_count": int(np.sum(near_zero_y)),
        "horizontal_label_distribution": _distribution_dict(axis_labels[:, 0]),
        "vertical_label_distribution": _distribution_dict(axis_labels[:, 1]),
        "combined_label_distribution": combined_distribution,
        "combined_labels_with_count_1": singleton_labels,
        "has_real_center_center_label_5": bool(combined_distribution.get("5", 0) > 0),
        "min_class_count": int(min_class_count),
        "requested_n_splits": int(requested_n_splits),
        "actual_n_splits": int(actual_n_splits),
        "center_tolerance": float(center_tolerance),
    }


def _build_binary_label_diagnostics(
    *,
    aligned: AlignedSession,
    labels: dict[str, Any],
    valid_mask: np.ndarray,
    trial_indices: np.ndarray,
    requested_n_splits: int,
    cv_scheme: str,
) -> dict[str, Any]:
    target_pos = aligned.target_pos[trial_indices]
    binary = labels["binary_labels"][trial_indices]
    distribution = _distribution_dict(binary)
    singleton_labels = sorted(int(label) for label, count in distribution.items() if int(count) == 1)
    positive_counts = [int(v) for v in distribution.values() if int(v) > 0]
    min_class_count = min(positive_counts) if positive_counts else 0
    if cv_scheme.lower() in {"kfold", "10fold", "stratifiedkfold"}:
        actual_n_splits = min(requested_n_splits, min_class_count) if min_class_count >= 2 else 0
    elif cv_scheme.lower() in {"loo", "leaveoneout", "leave-one-out"}:
        actual_n_splits = int(trial_indices.size)
    else:
        actual_n_splits = 0
    return {
        "n_trials_total": int(aligned.images.shape[3]),
        "n_valid_trials_after_success_and_targetPos": int(valid_mask.sum()),
        "n_trials_entering_CV": int(trial_indices.size),
        "target_pos_distribution_round6": _target_pos_distribution(target_pos, decimals=6),
        "binary_label_distribution": distribution,
        "combined_label_distribution": distribution,
        "combined_labels_with_count_1": singleton_labels,
        "min_class_count": int(min_class_count),
        "requested_n_splits": int(requested_n_splits),
        "actual_n_splits": int(actual_n_splits),
        "label_to_target_pos": labels["label_to_target_pos"],
    }


def build_dynamic_window_features(
    images: np.ndarray,
    eval_index: int,
    cue_index: int,
    valid_trial_mask: np.ndarray,
    voxel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one timepoint's dynamic spatiotemporal feature matrix.

    Before cue onset, the window is trial start through ``eval_index``.
    From cue onset onward, the window is cue onset through ``eval_index``.
    Frames are concatenated in temporal order, with a fixed voxel order
    within each frame. Missing/non-finite samples are handled at the
    voxel/feature level rather than by dropping whole trials, matching the
    MATLAB offline BCI workflow.
    """

    if eval_index < cue_index:
        frame_indices = np.arange(0, eval_index + 1)
    else:
        frame_indices = np.arange(cue_index, eval_index + 1)

    y, x, _, n_trials = images.shape
    trial_ok = np.asarray(valid_trial_mask, dtype=bool).copy()
    if trial_ok.size != n_trials:
        raise ValueError(f"valid_trial_mask has {trial_ok.size} entries; images have {n_trials} trials.")
    if voxel_mask is None:
        frame_finite = np.isfinite(images[:, :, frame_indices, :][:, :, :, trial_ok]).any(axis=(2, 3))
        voxel_mask = frame_finite
    voxel_mask = np.asarray(voxel_mask, dtype=bool)
    n_voxels = int(voxel_mask.sum())
    if n_voxels == 0:
        raise ValueError("No valid voxels remain after applying finite/background mask.")

    trial_indices = np.where(trial_ok)[0]
    features = np.empty((trial_indices.size, n_voxels * frame_indices.size), dtype=np.float32)
    for row, trial in enumerate(trial_indices):
        chunks = []
        for frame in frame_indices:
            chunks.append(images[:, :, frame, trial][voxel_mask].reshape(-1, order="F"))
        features[row, :] = np.concatenate(chunks)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    info = {
        "eval_index": int(eval_index),
        "frame_indices": frame_indices.astype(int).tolist(),
        "n_window_frames": int(frame_indices.size),
        "n_voxels": n_voxels,
        "feature_dim": int(features.shape[1]),
        "n_trials": int(trial_indices.size),
    }
    return features, trial_indices, info


def _memory_end_frame_index(aligned: AlignedSession) -> int:
    go_cue_s = aligned.metadata.get("go_cue_s_relative_to_cue")
    if go_cue_s is None or not np.isfinite(go_cue_s):
        raise ValueError(
            "Cannot build fixed_memory_3frames features because memory end/go cue timing "
            "could not be derived from behavior.target_acquire."
        )
    memory_end_index = aligned.cue_index + int(math.ceil(float(go_cue_s) * aligned.frame_rate_hz))
    return max(0, min(memory_end_index, aligned.images.shape[2]))


def build_fixed_memory_3frames_features(
    images: np.ndarray,
    aligned: AlignedSession,
    valid_trial_mask: np.ndarray,
    voxel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one fixed feature matrix from the 3 frames before memory end."""

    memory_end_index = _memory_end_frame_index(aligned)
    start = memory_end_index - 3
    if start < 0:
        raise ValueError(
            "Cannot build fixed_memory_3frames features: fewer than 3 frames exist before memory end."
        )
    frame_indices = np.arange(start, memory_end_index, dtype=int)

    _, _, _, n_trials = images.shape
    trial_ok = np.asarray(valid_trial_mask, dtype=bool).copy()
    if trial_ok.size != n_trials:
        raise ValueError(f"valid_trial_mask has {trial_ok.size} entries; images have {n_trials} trials.")
    if voxel_mask is None:
        voxel_mask = np.isfinite(images[:, :, frame_indices, :][:, :, :, trial_ok]).any(axis=(2, 3))
    voxel_mask = np.asarray(voxel_mask, dtype=bool)
    n_voxels = int(voxel_mask.sum())
    if n_voxels == 0:
        raise ValueError("No valid voxels remain after applying finite/background mask.")

    trial_indices = np.where(trial_ok)[0]
    features = np.empty((trial_indices.size, n_voxels * frame_indices.size), dtype=np.float32)
    for row, trial in enumerate(trial_indices):
        chunks = [
            images[:, :, frame, trial][voxel_mask].reshape(-1, order="F")
            for frame in frame_indices
        ]
        features[row, :] = np.concatenate(chunks)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    eval_index = int(frame_indices[-1])
    info = {
        "eval_index": eval_index,
        "memory_end_index": int(memory_end_index),
        "frame_indices": frame_indices.astype(int).tolist(),
        "n_window_frames": int(frame_indices.size),
        "n_voxels": n_voxels,
        "feature_dim": int(features.shape[1]),
        "n_trials": int(trial_indices.size),
    }
    return features, trial_indices, info


def _make_cv_splits(y: np.ndarray, config: WithinSessionConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    sklearn_model_selection = _require("sklearn.model_selection", "scikit-learn")
    n = len(y)
    if config.cv_scheme.lower() in {"loo", "leaveoneout", "leave-one-out"}:
        cv = sklearn_model_selection.LeaveOneOut()
        return list(cv.split(np.arange(n)))

    if config.cv_scheme.lower() in {"kfold", "10fold", "stratifiedkfold"}:
        counts = np.bincount(y.astype(int))
        positive_counts = counts[counts > 0]
        if positive_counts.size == 0:
            raise ValueError("No class labels available for CV.")
        k = min(config.n_splits, int(positive_counts.min()))
        if k < 2:
            raise ValueError(
                f"Cannot run stratified {config.n_splits}-fold CV; smallest class has {positive_counts.min()} sample."
            )
        cv = sklearn_model_selection.StratifiedKFold(
            n_splits=k, shuffle=True, random_state=config.random_seed
        )
        return list(cv.split(np.zeros(n), y))

    raise ValueError(f"Unsupported cv_scheme '{config.cv_scheme}'")


def fit_fold_scaler_pca_lda(
    x_train: np.ndarray,
    y_axis_train: np.ndarray,
    x_test: np.ndarray,
    variance_to_keep: float,
) -> tuple[np.ndarray, int, int]:
    """Fit train-fold z-score, PCA, and LDA, then predict one axis.

    The scaler and PCA are fit on training samples only. This is essential
    because fitting either step before cross-validation would leak test-fold
    distribution information into the dimensionality reduction and inflate
    decoding accuracy.
    """

    sklearn_decomposition = _require("sklearn.decomposition", "scikit-learn")
    sklearn_discriminant = _require("sklearn.discriminant_analysis", "scikit-learn")

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=0)
    tiny = sigma < 1e-12
    zero_std_count = int(tiny.sum())
    sigma[tiny] = 1.0
    z_train = (x_train - mu) / sigma
    z_test = (x_test - mu) / sigma
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # MATLAB ``pca`` chooses enough components for 95% explained variance.
    # scikit-learn's full SVD PCA with n_components in (0,1) is the closest
    # direct equivalent for this use case.
    pca = sklearn_decomposition.PCA(n_components=variance_to_keep, svd_solver="full")
    train_scores = pca.fit_transform(z_train)
    test_scores = pca.transform(z_test)

    lda = sklearn_discriminant.LinearDiscriminantAnalysis(solver="svd")
    lda.fit(train_scores, y_axis_train)
    pred = lda.predict(test_scores).astype(int)
    return pred, int(pca.n_components_), zero_std_count


def _zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=0)
    tiny = sigma < 1e-12
    zero_std_count = int(tiny.sum())
    sigma[tiny] = 1.0
    z_train = (x_train - mu) / sigma
    z_test = (x_test - mu) / sigma
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return z_train, z_test, zero_std_count


def _matlab_principal_components(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA coefficients/eigenvalues following ``dataproc_func_princomp``.

    The MATLAB CPCA code uses the sample-space eigenproblem when features
    outnumber observations, then maps those vectors back into feature space.
    This mirrors that route so class-wise PCA keeps the same small-sample
    behavior as ``PPC_directional_tuning/cbmspccode/dataproc_func_princomp.m``.
    """

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data matrix, found shape {data.shape}.")
    n_obs, n_features = data.shape
    if n_obs < 2 or n_features == 0:
        return np.empty((n_features, 0), dtype=float), np.empty(0, dtype=float)

    demeaned = data - data.mean(axis=0, keepdims=True)
    if n_features >= n_obs:
        gram = demeaned @ demeaned.T
        eval_raw, sample_vecs = np.linalg.eigh((gram + gram.T) / 2.0)
        order = np.argsort(eval_raw)[::-1][: n_obs - 1]
        eval_raw = eval_raw[order]
        sample_vecs = sample_vecs[:, order]
        tol = max(gram.shape) * np.finfo(float).eps * max(float(np.max(np.abs(eval_raw))), 1.0)
        keep = eval_raw > tol
        eval_raw = eval_raw[keep]
        sample_vecs = sample_vecs[:, keep]
        if eval_raw.size == 0:
            return np.empty((n_features, 0), dtype=float), np.empty(0, dtype=float)
        coeff = demeaned.T @ sample_vecs @ np.diag(eval_raw ** -0.5)
        latent = eval_raw / (n_obs - 1)
    else:
        scatter = demeaned.T @ demeaned
        eval_raw, coeff = np.linalg.eigh((scatter + scatter.T) / 2.0)
        order = np.argsort(eval_raw)[::-1]
        eval_raw = eval_raw[order]
        coeff = coeff[:, order]
        latent = eval_raw / (n_obs - 1)
        tol = max(scatter.shape) * np.finfo(float).eps * max(float(np.max(np.abs(eval_raw))), 1.0)
        keep = eval_raw > tol
        coeff = coeff[:, keep]
        latent = latent[keep]

    return np.asarray(coeff, dtype=float), np.asarray(latent, dtype=float)


def _select_cpca_pca_components(
    coeff: np.ndarray,
    latent: np.ndarray,
    eval_keep: tuple[str, float | None] = ("mean", None),
) -> np.ndarray:
    """Apply the MATLAB CPCA ``EvalKeep`` rule to class-wise PCA output."""

    if latent.size == 0 or coeff.shape[1] == 0:
        return coeff[:, :0]

    method, value = eval_keep
    if method == "median":
        keep = latent > np.median(latent)
        return coeff[:, keep]
    if method == "spectrum":
        if latent.size == 1:
            return coeff[:, :1]
        return coeff[:, : int(np.argmax(np.abs(np.diff(latent)))) + 1]
    if method == "energy":
        if value is None:
            raise ValueError("EvalKeep 'energy' requires a fraction value.")
        cutoff = float(value) * float(np.sum(latent))
        n_keep = int(np.searchsorted(np.cumsum(latent), cutoff, side="right") + 1)
        return coeff[:, : min(n_keep, coeff.shape[1])]

    return coeff[:, latent > np.mean(latent)]


def _orth_columns(matrix: np.ndarray) -> np.ndarray:
    """MATLAB ``orth`` equivalent for column spaces."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tol = max(matrix.shape) * np.finfo(float).eps * max(float(singular[0]) if singular.size else 0.0, 1.0)
    return u[:, singular > tol]


def _cov_matrix(data: np.ndarray, n_features: int | None = None) -> np.ndarray:
    """Sample covariance with MATLAB-like zero covariance for singletons."""

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if n_features is None:
        n_features = data.shape[1]
    if data.shape[0] < 2:
        return np.zeros((n_features, n_features), dtype=float)
    cov = np.cov(data, rowvar=False, bias=False)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov


def _regularize_covariance(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.size == 0:
        return cov
    scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    ridge = max(scale, 1.0) * 1e-9
    return cov + np.eye(cov.shape[0]) * ridge


def _linear_disc_analysis_matlab(train_data: np.ndarray, train_labels: np.ndarray, m: int) -> np.ndarray:
    """Feature extraction matrix from ``linear_disc_analysis.m``."""

    train_data = np.asarray(train_data, dtype=float)
    train_labels = np.asarray(train_labels).reshape(-1)
    n_obs, n_features = train_data.shape
    classes = np.unique(train_labels)
    n_classes = classes.size
    if n_obs == 0 or n_features == 0 or n_classes < 2:
        return np.empty((0, n_features), dtype=float)

    mean = np.zeros(n_features, dtype=float)
    sigma_w = np.zeros((n_features, n_features), dtype=float)
    sigma_b_second_moment = np.zeros((n_features, n_features), dtype=float)
    for cls in classes:
        class_data = train_data[train_labels == cls]
        prob = class_data.shape[0] / n_obs
        mean_i = class_data.mean(axis=0)
        sigma_i = _cov_matrix(class_data, n_features)
        mean += prob * mean_i
        sigma_w += prob * sigma_i
        sigma_b_second_moment += prob * np.outer(mean_i, mean_i)

    sigma_b = sigma_b_second_moment - np.outer(mean, mean)
    sigma = sigma_w + sigma_b
    m_new = min(int(np.linalg.matrix_rank(sigma_b)), int(n_classes - 1), int(m))
    if m_new <= 0:
        return np.empty((0, n_features), dtype=float)

    mat = np.linalg.pinv(sigma) @ sigma_b
    eigvals, eigvecs = np.linalg.eig(mat)
    eigvals = np.real_if_close(eigvals, tol=1000).real
    eigvecs = np.real_if_close(eigvecs, tol=1000).real
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:m_new]].T


def _fit_matlab_cpca_subspaces(z_train: np.ndarray, y_train: np.ndarray, *, m: int = 1) -> list[np.ndarray]:
    """Fit class-wise CPCA subspaces using the MATLAB paper code's recipe."""

    y_train = np.asarray(y_train).reshape(-1)
    classes = np.unique(y_train)
    if classes.size < 2:
        raise ValueError("CPCA+LDA requires at least two classes in the training fold.")

    n_obs, n_features = z_train.shape
    class_counts = np.array([np.sum(y_train == cls) for cls in classes], dtype=float)
    sample_means: list[np.ndarray] = []
    class_pca_bases: list[np.ndarray] = []
    for cls in classes:
        class_data = z_train[y_train == cls]
        sample_means.append(class_data.mean(axis=0))
        coeff, latent = _matlab_principal_components(class_data)
        class_pca_bases.append(_select_cpca_pca_components(coeff, latent, ("mean", None)))

    overall_mean = z_train.mean(axis=0)
    data_b = np.vstack(
        [
            math.sqrt(float(class_counts[i]) / float(n_obs)) * (sample_means[i] - overall_mean)
            for i in range(classes.size)
        ]
    )
    between_basis, _ = _matlab_principal_components(data_b)

    subspaces: list[np.ndarray] = []
    for class_basis in class_pca_bases:
        temp_basis = _orth_columns(np.column_stack([class_basis, between_basis]))
        train_projected = z_train @ temp_basis
        dfe = _linear_disc_analysis_matlab(train_projected, y_train, m)
        subspaces.append(temp_basis @ dfe.T)

    empty = [subspace.shape[1] == 0 for subspace in subspaces]
    if any(empty):
        nonempty = next((subspace for subspace in subspaces if subspace.shape[1] > 0), None)
        if nonempty is None:
            coeff, _ = _matlab_principal_components(z_train)
            nonempty = coeff[:, :1] if coeff.shape[1] else np.ones((n_features, 1), dtype=float)
        subspaces = [nonempty.copy() if is_empty else subspace for subspace, is_empty in zip(subspaces, empty)]

    return subspaces


def _choose_cpca_subspaces(
    z_test: np.ndarray,
    z_train: np.ndarray,
    y_train: np.ndarray,
    subspaces: list[np.ndarray],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replicate ``choose_subspace(..., 'unbiased')`` for each test sample."""

    classes = np.unique(y_train)
    priors = np.full(classes.size, 1.0 / classes.size, dtype=float)
    selected = np.empty(z_test.shape[0], dtype=int)

    for row_idx, test_row in enumerate(z_test):
        max_posteriors = np.empty(len(subspaces), dtype=float)
        for subspace_idx, subspace in enumerate(subspaces):
            test_feature = test_row @ subspace
            train_features = z_train @ subspace
            scores = np.empty(classes.size, dtype=float)
            for class_idx, cls in enumerate(classes):
                class_features = train_features[y_train == cls]
                mu = class_features.mean(axis=0)
                cov = _regularize_covariance(_cov_matrix(class_features, subspace.shape[1]))
                delta = np.atleast_1d(test_feature - mu)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0 or not np.isfinite(logdet):
                    cov = _regularize_covariance(cov)
                    sign, logdet = np.linalg.slogdet(cov)
                quad = float(delta @ np.linalg.pinv(cov) @ delta.T)
                scores[class_idx] = -0.5 * quad - 0.5 * float(logdet) + math.log(float(priors[class_idx]))

            divisor = float(np.max(scores))
            scaled = scores / divisor if np.isfinite(divisor) and abs(divisor) > 1e-12 else scores
            scaled = np.clip(scaled, -700.0, 700.0)
            posterior = np.exp(scaled)
            posterior = posterior / posterior.sum()
            max_posteriors[subspace_idx] = float(np.max(posterior))

        best = np.flatnonzero(np.isclose(max_posteriors, np.max(max_posteriors)))
        selected[row_idx] = int(rng.choice(best))

    return selected


def _predict_matlab_cpca_lda(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    *,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Fit MATLAB-style CPCA and classify through selected subspaces."""

    sklearn_discriminant = _require("sklearn.discriminant_analysis", "scikit-learn")

    y_train = np.asarray(y_train).reshape(-1)
    subspaces = _fit_matlab_cpca_subspaces(z_train, y_train, m=m)
    selected = _choose_cpca_subspaces(z_test, z_train, y_train, subspaces, rng=rng)
    pred = np.empty(z_test.shape[0], dtype=int)

    for subspace_idx in np.unique(selected):
        mask = selected == subspace_idx
        train_features = z_train @ subspaces[int(subspace_idx)]
        test_features = z_test[mask] @ subspaces[int(subspace_idx)]
        lda = sklearn_discriminant.LinearDiscriminantAnalysis(solver="svd")
        lda.fit(train_features, y_train)
        pred[mask] = lda.predict(test_features).astype(int)

    return pred, subspaces, selected


def fit_fold_scaler_projection_lda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    variance_to_keep: float,
    decoder_type: str,
    cpca_m: int = 1,
    random_seed: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Fit train-fold z-score, PCA/cPCA, and binary LDA."""

    sklearn_decomposition = _require("sklearn.decomposition", "scikit-learn")
    sklearn_discriminant = _require("sklearn.discriminant_analysis", "scikit-learn")

    z_train, z_test, zero_std_count = _zscore_train_test(x_train, x_test)
    if decoder_type == "pca_lda":
        projector = sklearn_decomposition.PCA(n_components=variance_to_keep, svd_solver="full")
        train_scores = projector.fit_transform(z_train)
        test_scores = projector.transform(z_test)
        n_components = int(projector.n_components_)
    elif decoder_type == "cpca_lda":
        rng = np.random.default_rng(random_seed)
        pred, subspaces, _selected = _predict_matlab_cpca_lda(
            z_train,
            y_train,
            z_test,
            m=cpca_m,
            rng=rng,
        )
        n_components = int(max(subspace.shape[1] for subspace in subspaces))
        return pred, n_components, zero_std_count
    else:
        raise ValueError("Binary timepoint decoding supports decoder_type='pca_lda' or 'cpca_lda'.")

    lda = sklearn_discriminant.LinearDiscriminantAnalysis(solver="svd")
    lda.fit(train_scores, y_train)
    pred = lda.predict(test_scores).astype(int)
    return pred, n_components, zero_std_count


def decode_timepoint(
    x: np.ndarray,
    labels_axis: np.ndarray,
    labels_combined: np.ndarray,
    config: WithinSessionConfig,
) -> dict[str, Any]:
    y = labels_combined.astype(int)
    splits = _make_cv_splits(y, config)
    predictions = np.full_like(y, fill_value=-1)
    fold_results = []
    pca_components = []
    zero_std_counts = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        x_train = x[train_idx]
        x_test = x[test_idx]
        h_pred, h_pca, h_zero = fit_fold_scaler_pca_lda(
            x_train, labels_axis[train_idx, 0], x_test, config.variance_to_keep
        )
        v_pred, v_pca, v_zero = fit_fold_scaler_pca_lda(
            x_train, labels_axis[train_idx, 1], x_test, config.variance_to_keep
        )
        combined_pred = (h_pred + 3 * (v_pred - 1)).astype(int)
        predictions[test_idx] = combined_pred

        correct = combined_pred == y[test_idx]
        fold_results.append(
            {
                "fold": fold_id,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "percent_correct": float(correct.mean() * 100.0),
                "test_indices_local": test_idx.astype(int).tolist(),
                "predicted_combined": combined_pred.astype(int).tolist(),
                "actual_combined": y[test_idx].astype(int).tolist(),
                "pca_components_horizontal": h_pca,
                "pca_components_vertical": v_pca,
                "zero_std_features_horizontal": h_zero,
                "zero_std_features_vertical": v_zero,
            }
        )
        pca_components.extend([h_pca, v_pca])
        zero_std_counts.extend([h_zero, v_zero])

    if np.any(predictions < 0):
        raise RuntimeError("Some samples did not receive a CV prediction.")

    n_correct = int(np.sum(predictions == y))
    percent_correct = float(n_correct / y.size * 100.0)
    return {
        "predictions_combined": predictions.astype(int),
        "actual_combined": y,
        "fold_results": fold_results,
        "actual_n_splits": int(len(splits)),
        "percent_correct": percent_correct,
        "n_correct": n_correct,
        "n_counted": int(y.size),
        "pca_components": np.asarray(pca_components, dtype=int),
        "zero_std_feature_counts": np.asarray(zero_std_counts, dtype=int),
    }


def decode_timepoint_binary(
    x: np.ndarray,
    labels_binary: np.ndarray,
    config: WithinSessionConfig,
    *,
    decoder_type: str,
) -> dict[str, Any]:
    y = labels_binary.astype(int)
    splits = _make_cv_splits(y, config)
    predictions = np.full_like(y, fill_value=-1)
    fold_results = []
    components = []
    zero_std_counts = []

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        pred, n_components, zero_std = fit_fold_scaler_projection_lda(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            variance_to_keep=config.variance_to_keep,
            decoder_type=decoder_type,
            cpca_m=config.cpca_m,
            random_seed=config.random_seed + fold_id,
        )
        predictions[test_idx] = pred
        correct = pred == y[test_idx]
        fold_results.append(
            {
                "fold": fold_id,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "percent_correct": float(correct.mean() * 100.0),
                "test_indices_local": test_idx.astype(int).tolist(),
                "predicted_binary": pred.astype(int).tolist(),
                "actual_binary": y[test_idx].astype(int).tolist(),
                "pca_components": n_components,
                "zero_std_features": zero_std,
                "decoder_type": decoder_type,
            }
        )
        components.append(n_components)
        zero_std_counts.append(zero_std)

    if np.any(predictions < 0):
        raise RuntimeError("Some samples did not receive a CV prediction.")

    n_correct = int(np.sum(predictions == y))
    percent_correct = float(n_correct / y.size * 100.0)
    return {
        "predictions_binary": predictions.astype(int),
        "actual_binary": y,
        "predictions_combined": predictions.astype(int),
        "actual_combined": y,
        "fold_results": fold_results,
        "actual_n_splits": int(len(splits)),
        "percent_correct": percent_correct,
        "n_correct": n_correct,
        "n_counted": int(y.size),
        "pca_components": np.asarray(components, dtype=int),
        "zero_std_feature_counts": np.asarray(zero_std_counts, dtype=int),
    }


def _angle_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def compute_angular_error(
    predicted_combined: np.ndarray,
    actual_combined: np.ndarray,
    combined_to_angle_deg: dict[int, float],
) -> tuple[float, np.ndarray]:
    """Compute absolute circular angular error in degrees.

    The original MATLAB ``calculate_angular_error.m`` keeps the possible
    center-center multicoder output and assigns it the worst possible error
    (180 degrees) instead of forcing it onto one of the eight targets. This
    follows that repository rule and records it explicitly.
    """

    predicted_combined = np.asarray(predicted_combined, dtype=int)
    actual_combined = np.asarray(actual_combined, dtype=int)
    errors = np.empty(predicted_combined.size, dtype=float)
    for i, (pred, actual) in enumerate(zip(predicted_combined, actual_combined)):
        if pred == 5:
            errors[i] = 180.0
            continue
        if pred not in combined_to_angle_deg:
            errors[i] = 180.0
            continue
        if actual not in combined_to_angle_deg:
            raise ValueError(f"Actual combined label {actual} has no target angle mapping.")
        errors[i] = _angle_distance_deg(
            np.asarray(combined_to_angle_deg[pred]), np.asarray(combined_to_angle_deg[actual])
        )
    return float(errors.mean()), errors


def _binomial_greater_pvalue(n_correct: int, n_counted: int, chance: float) -> float:
    scipy_stats = _require("scipy.stats", "scipy")
    return float(scipy_stats.binomtest(n_correct, n_counted, chance, alternative="greater").pvalue)


def permutation_test_angular_error(
    actual_combined: np.ndarray,
    observed_mean_error_deg: float,
    combined_to_angle_deg: dict[int, float],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """One-sided random-direction angular-error permutation test.

    Each replicate samples ``X`` predictions from the eight real target
    directions with uniform probability, compares them to the session's true
    labels, and stores the replicate mean angular error. The p-value is the
    proportion of null means smaller than the observed model error, testing
    whether model error is significantly lower than random guessing.
    """

    rng = np.random.default_rng(random_seed)
    actual_combined = np.asarray(actual_combined, dtype=int)
    possible = np.array(sorted(k for k in combined_to_angle_deg.keys() if k != 5), dtype=int)
    if possible.size not in {2, 8}:
        LOGGER.warning("Expected 2 or 8 real target directions, found %d: %s", possible.size, possible.tolist())

    actual_angles = np.array([combined_to_angle_deg[int(k)] for k in actual_combined], dtype=float)
    null_means = np.empty(n_permutations, dtype=np.float32)
    chunk = 10_000
    done = 0
    while done < n_permutations:
        size = min(chunk, n_permutations - done)
        random_pred = rng.choice(possible, size=(size, actual_combined.size), replace=True)
        pred_angles = np.vectorize(combined_to_angle_deg.__getitem__)(random_pred)
        errors = _angle_distance_deg(pred_angles, actual_angles[None, :])
        null_means[done : done + size] = errors.mean(axis=1)
        done += size

    p_value = float(np.mean(null_means < observed_mean_error_deg))
    return {
        "p_value": p_value,
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std(ddof=1 if n_permutations > 1 else 0)),
        "null_quantiles": {
            "0.001": float(np.quantile(null_means, 0.001)),
            "0.01": float(np.quantile(null_means, 0.01)),
            "0.05": float(np.quantile(null_means, 0.05)),
            "0.5": float(np.quantile(null_means, 0.5)),
            "0.95": float(np.quantile(null_means, 0.95)),
        },
    }


def apply_bonferroni_correction(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    p_values = np.asarray(p_values, dtype=float)
    corrected = np.minimum(p_values * p_values.size, 1.0)
    return corrected, corrected < alpha


def _confusion_matrix(actual: np.ndarray, predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    lookup = {int(label): i for i, label in enumerate(labels)}
    for a, p in zip(actual, predicted):
        matrix[lookup[int(a)], lookup[int(p)]] += 1
    return matrix


def _accuracy_from_confusion(confusion: np.ndarray) -> float:
    total = int(confusion.sum())
    if total == 0:
        return float("nan")
    return float(np.trace(confusion) / total)


def _balanced_accuracy_from_confusion(confusion: np.ndarray) -> float:
    row_sum = confusion.sum(axis=1)
    present = row_sum > 0
    if not np.any(present):
        return float("nan")
    recall = np.divide(
        np.diag(confusion).astype(float),
        row_sum,
        out=np.zeros(row_sum.shape, dtype=float),
        where=row_sum > 0,
    )
    return float(recall[present].mean())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def decode_within_session(
    mat_path: str | Path,
    config: WithinSessionConfig | None = None,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run within-session PCA+LDA multicoder decoding."""

    config = config or WithinSessionConfig()
    if config.mode not in VALID_DECODING_MODES:
        raise ValueError(f"Unsupported mode '{config.mode}'. Choose one of {sorted(VALID_DECODING_MODES)}.")
    rng = np.random.default_rng(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session = load_mat73_session(mat_path)
    aligned = align_fusi_and_behavior(
        session,
        session_id=session_id,
        decoder_type=config.decoder_type,
        frame_rate_hz=config.frame_rate_hz,
    )
    task_type = str(aligned.metadata["task_type"])
    decoder_type = str(aligned.metadata["decoder_type"])
    chance_accuracy = float(aligned.metadata["task_config"]["chance_accuracy"])
    images, preprocess_log = preprocess_power_doppler_session(
        aligned.images, config, source_path=str(mat_path), output_dir=output_dir
    )

    if task_type == "8target":
        labels = make_multicoder_labels(aligned.target_pos, center_tolerance=config.center_tolerance)
    elif task_type == "2target":
        labels = make_binary_labels(aligned.target_pos)
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")
    valid_mask = aligned.valid_trial_mask.copy()
    axis_labels_all = labels.get("axis_labels")
    if task_type == "8target":
        combined_all = labels["combined_labels"]
    else:
        combined_all = labels["binary_labels"]

    n_time = images.shape[2]
    if config.max_timepoints:
        n_time = min(n_time, config.max_timepoints)

    timepoint_results = []
    diagnostic_results = []
    all_acc_p = []
    all_ang_p = []
    if config.mode == "fixed_memory_3frames":
        decode_windows = [("fixed_memory_3frames", None)]
    else:
        # Dynamic mode intentionally reproduces the older behavior: one model
        # is trained/evaluated for each trial timepoint.
        voxel_mask = np.isfinite(images).all(axis=(2, 3))
        decode_windows = [("dynamic_time_window", i) for i in range(n_time)]

    for window_mode, eval_index in decode_windows:
        if window_mode == "fixed_memory_3frames":
            x, trial_indices, winfo = build_fixed_memory_3frames_features(
                images, aligned, valid_mask, voxel_mask=None
            )
            result_index = int(winfo["eval_index"])
        else:
            assert eval_index is not None
            x, trial_indices, winfo = build_dynamic_window_features(
                images, eval_index, aligned.cue_index, valid_mask, voxel_mask=voxel_mask
            )
            result_index = int(eval_index)

        if trial_indices.size < config.min_trials_per_timepoint:
            LOGGER.warning("Skipping timepoint %d: only %d trials", result_index, trial_indices.size)
            continue

        combined = combined_all[trial_indices]
        if task_type == "8target":
            assert axis_labels_all is not None
            axis_labels = axis_labels_all[trial_indices]
            diagnostics = _build_trial_label_diagnostics(
                aligned=aligned,
                labels=labels,
                valid_mask=valid_mask,
                trial_indices=trial_indices,
                requested_n_splits=config.n_splits,
                cv_scheme=config.cv_scheme,
                center_tolerance=config.center_tolerance,
            )
        else:
            axis_labels = None
            diagnostics = _build_binary_label_diagnostics(
                aligned=aligned,
                labels=labels,
                valid_mask=valid_mask,
                trial_indices=trial_indices,
                requested_n_splits=config.n_splits,
                cv_scheme=config.cv_scheme,
            )
        diagnostics.update(
            {
                "mode": window_mode,
                "task_type": task_type,
                "decoder_type": decoder_type,
                "eval_index": int(result_index),
                "frame_indices": winfo["frame_indices"],
                "n_window_frames": winfo["n_window_frames"],
                "n_voxels": winfo["n_voxels"],
                "feature_dim": winfo["feature_dim"],
            }
        )
        diagnostic_results.append(diagnostics)
        LOGGER.info(
            "label diagnostics t=%d task=%s decoder=%s trials total=%d valid=%d cv=%d "
            "min_class=%d actual_splits=%d labels=%s singletons=%s",
            result_index,
            task_type,
            decoder_type,
            diagnostics["n_trials_total"],
            diagnostics["n_valid_trials_after_success_and_targetPos"],
            diagnostics["n_trials_entering_CV"],
            diagnostics["min_class_count"],
            diagnostics["actual_n_splits"],
            json.dumps(diagnostics["combined_label_distribution"], sort_keys=True),
            diagnostics["combined_labels_with_count_1"],
        )

        if config.diagnostic_only:
            continue
        if len(np.unique(combined)) < 2:
            LOGGER.warning("Skipping timepoint %d: fewer than 2 classes", result_index)
            continue
        if diagnostics["min_class_count"] < 2:
            LOGGER.warning(
                "Skipping timepoint %d: insufficient class count; combined labels with count 1: %s",
                result_index,
                diagnostics["combined_labels_with_count_1"],
            )
            continue

        if task_type == "8target":
            assert axis_labels is not None
            decoded = decode_timepoint(x, axis_labels, combined, config)
        else:
            decoded = decode_timepoint_binary(x, combined, config, decoder_type=decoder_type)
        mean_ang, per_trial_ang = compute_angular_error(
            decoded["predictions_combined"], decoded["actual_combined"], labels["combined_to_angle_deg"]
        )
        for fold in decoded["fold_results"]:
            predicted_key = "predicted_combined" if "predicted_combined" in fold else "predicted_binary"
            actual_key = "actual_combined" if "actual_combined" in fold else "actual_binary"
            fold_mean_ang, _ = compute_angular_error(
                np.asarray(fold[predicted_key]),
                np.asarray(fold[actual_key]),
                labels["combined_to_angle_deg"],
            )
            fold["mean_angular_error_deg"] = fold_mean_ang

        acc_p = _binomial_greater_pvalue(
            decoded["n_correct"], decoded["n_counted"], chance_accuracy
        )
        perm = permutation_test_angular_error(
            decoded["actual_combined"],
            mean_ang,
            labels["combined_to_angle_deg"],
            n_permutations=config.n_permutations,
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

        confusion_labels = np.arange(1, 10) if task_type == "8target" else np.arange(0, 2)
        conf = _confusion_matrix(decoded["actual_combined"], decoded["predictions_combined"], confusion_labels)
        accuracy = _accuracy_from_confusion(conf)
        balanced_accuracy = _balanced_accuracy_from_confusion(conf)
        row_sum = conf.sum(axis=1, keepdims=True)
        conf_pct = np.divide(conf * 100.0, row_sum, out=np.zeros_like(conf, dtype=float), where=row_sum > 0)

        result = {
            **winfo,
            "mode": config.mode,
            "task_type": task_type,
            "decoder_type": decoder_type,
            "time_from_trial_start_s": float(aligned.time_from_trial_start_s[result_index]),
            "time_from_cue_s": float(aligned.time_from_cue_s[result_index]),
            "task_state": _task_state(
                float(aligned.time_from_cue_s[result_index]),
                aligned.metadata.get("go_cue_s_relative_to_cue"),
            ),
            "cv_scheme": config.cv_scheme,
            "requested_n_splits": int(config.n_splits),
            "actual_n_splits": int(decoded["actual_n_splits"]),
            "label_diagnostics": diagnostics,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "percent_correct": decoded["percent_correct"],
            "mean_angular_error_deg": mean_ang,
            "angular_error_per_trial_deg": per_trial_ang,
            "fold_results": decoded["fold_results"],
            "pca_components": decoded["pca_components"],
            "pca_component_min": int(decoded["pca_components"].min()),
            "pca_component_max": int(decoded["pca_components"].max()),
            "zero_std_feature_counts": decoded["zero_std_feature_counts"],
            "accuracy_p_uncorrected": acc_p,
            "angular_error_p_uncorrected": perm["p_value"],
            "chance_accuracy": chance_accuracy,
            "permutation_null_angular_error": perm,
            "actual_combined": decoded["actual_combined"],
            "predicted_combined": decoded["predictions_combined"],
            "global_trial_indices": trial_indices,
            "confusion_matrix_labels": confusion_labels,
            "confusion_matrix_counts": conf,
            "confusion_matrix_row_percent": conf_pct,
        }
        if task_type == "8target":
            result["confusion_matrix_counts_1_to_9"] = conf
            result["confusion_matrix_row_percent_1_to_9"] = conf_pct
        else:
            result["actual_binary"] = decoded["actual_binary"]
            result["predicted_binary"] = decoded["predictions_binary"]
            result["confusion_matrix_counts_0_to_1"] = conf
            result["confusion_matrix_row_percent_0_to_1"] = conf_pct
        timepoint_results.append(result)
        all_acc_p.append(acc_p)
        all_ang_p.append(perm["p_value"])
        LOGGER.info(
            "t=%d n=%d acc=%.2f%% ang=%.2f deg PCA=%d-%d",
            result_index,
            trial_indices.size,
            decoded["percent_correct"],
            mean_ang,
            result["pca_component_min"],
            result["pca_component_max"],
        )

    if not timepoint_results:
        final_diag = diagnostic_results[-1] if diagnostic_results else {}
        skip_reason = "diagnostic_only" if config.diagnostic_only else "insufficient_class_count"
        if final_diag and final_diag.get("min_class_count", 0) >= 2 and not config.diagnostic_only:
            skip_reason = "no_decodable_timepoints"
        summary = {
            "status": "diagnostic_only" if config.diagnostic_only else "skipped",
            "skip_reason": skip_reason,
            "session_id": aligned.session_id,
            "source_path": str(mat_path),
            "task_type": task_type,
            "decoder_type": decoder_type,
            "n_targets": int(aligned.metadata["n_targets"]),
            "n_trials_total": int(images.shape[3]),
            "n_valid_trials": int(valid_mask.sum()),
            "n_valid_trials_after_success_and_targetPos": int(valid_mask.sum()),
            "n_trials_entering_CV": final_diag.get("n_trials_entering_CV", ""),
            "mode": config.mode,
            "cv_scheme": config.cv_scheme,
            "requested_n_splits": int(config.n_splits),
            "actual_n_splits": final_diag.get("actual_n_splits", ""),
            "min_class_count": final_diag.get("min_class_count", ""),
            "center_tolerance": float(config.center_tolerance),
            "frame_rate_hz": float(aligned.frame_rate_hz),
            "frame_rate_source": aligned.metadata.get("frame_rate_source"),
            "recording_system": aligned.metadata.get("recording_system"),
            "chance_accuracy": chance_accuracy,
            "class_distribution_combined": final_diag.get("combined_label_distribution", {}),
            "combined_label_distribution": final_diag.get("combined_label_distribution", {}),
            "binary_label_distribution": final_diag.get("binary_label_distribution", {}),
            "horizontal_label_distribution": final_diag.get("horizontal_label_distribution", {}),
            "vertical_label_distribution": final_diag.get("vertical_label_distribution", {}),
            "combined_labels_with_count_1": final_diag.get("combined_labels_with_count_1", []),
            "has_real_center_center_label_5": final_diag.get("has_real_center_center_label_5", False),
            "target_pos_near_zero_nonzero_x": final_diag.get("target_pos_near_zero_nonzero_x", False),
            "target_pos_near_zero_nonzero_y": final_diag.get("target_pos_near_zero_nonzero_y", False),
            "target_pos_near_zero_nonzero_x_count": final_diag.get("target_pos_near_zero_nonzero_x_count", 0),
            "target_pos_near_zero_nonzero_y_count": final_diag.get("target_pos_near_zero_nonzero_y_count", 0),
            "target_pos_distribution_round6": final_diag.get("target_pos_distribution_round6", []),
        }
        result_dict = {
            "summary": summary,
            "config": asdict(config),
            "preprocess_log": preprocess_log,
            "alignment_metadata": aligned.metadata,
            "direction_labels": {
                "combined_to_angle_deg": labels["combined_to_angle_deg"],
                "combined_label_names": labels["combined_label_names"],
                "label_to_target_pos": labels.get("label_to_target_pos"),
                "center_tolerance": float(config.center_tolerance),
                "center_center_rule": (
                    "Multicoder center-center predictions are retained and assigned "
                    "180 deg angular error, matching the existing MATLAB evaluation."
                    if task_type == "8target"
                    else "Binary labels are evaluated directly; angular error uses the two target angles."
                ),
            },
            "diagnostics": diagnostic_results,
            "timepoints": [],
        }
        save_results(result_dict, output_dir, aligned.session_id)
        print_summary(summary)
        return result_dict

    acc_p_corr, acc_sig = apply_bonferroni_correction(np.asarray(all_acc_p), config.alpha)
    ang_p_corr, ang_sig = apply_bonferroni_correction(np.asarray(all_ang_p), config.alpha)
    for i, result in enumerate(timepoint_results):
        result["accuracy_p_bonferroni"] = float(acc_p_corr[i])
        result["accuracy_significant"] = bool(acc_sig[i])
        result["angular_error_p_bonferroni"] = float(ang_p_corr[i])
        result["angular_error_significant"] = bool(ang_sig[i])

    class_distribution = {
        str(k): int(v)
        for k, v in zip(*np.unique(combined_all[valid_mask], return_counts=True))
    }
    final = timepoint_results[-1]
    pca_all = np.concatenate([r["pca_components"] for r in timepoint_results])
    summary = {
        "session_id": aligned.session_id,
        "source_path": str(mat_path),
        "task_type": task_type,
        "decoder_type": decoder_type,
        "n_targets": int(aligned.metadata["n_targets"]),
        "n_trials_total": int(images.shape[3]),
        "n_valid_trials": int(valid_mask.sum()),
        "n_valid_trials_after_success_and_targetPos": int(valid_mask.sum()),
        "n_trials_entering_CV": int(final["n_trials"]),
        "mode": config.mode,
        "cv_scheme": config.cv_scheme,
        "requested_n_splits": int(config.n_splits),
        "actual_n_splits": int(final["actual_n_splits"]),
        "min_class_count": int(final["label_diagnostics"]["min_class_count"]),
        "center_tolerance": float(config.center_tolerance),
        "frame_rate_hz": float(aligned.frame_rate_hz),
        "frame_rate_source": aligned.metadata.get("frame_rate_source"),
        "recording_system": aligned.metadata.get("recording_system"),
        "chance_accuracy": chance_accuracy,
        "final_timepoint_index": int(final["eval_index"]),
        "accuracy": float(final["accuracy"]),
        "balanced_accuracy": float(final["balanced_accuracy"]),
        "final_accuracy_percent": float(final["percent_correct"]),
        "final_mean_angular_error_deg": float(final["mean_angular_error_deg"]),
        "earliest_accuracy_significant_time_s": _earliest_sig_time(timepoint_results, "accuracy_significant"),
        "earliest_angular_error_significant_time_s": _earliest_sig_time(
            timepoint_results, "angular_error_significant"
        ),
        "pca_component_range": [int(pca_all.min()), int(pca_all.max())],
        "class_distribution_combined": class_distribution,
        "combined_label_distribution": final["label_diagnostics"]["combined_label_distribution"],
        "binary_label_distribution": final["label_diagnostics"].get("binary_label_distribution", {}),
        "horizontal_label_distribution": final["label_diagnostics"].get("horizontal_label_distribution", {}),
        "vertical_label_distribution": final["label_diagnostics"].get("vertical_label_distribution", {}),
        "combined_labels_with_count_1": final["label_diagnostics"]["combined_labels_with_count_1"],
        "has_real_center_center_label_5": final["label_diagnostics"].get("has_real_center_center_label_5", False),
        "target_pos_near_zero_nonzero_x": final["label_diagnostics"].get("target_pos_near_zero_nonzero_x", False),
        "target_pos_near_zero_nonzero_y": final["label_diagnostics"].get("target_pos_near_zero_nonzero_y", False),
        "target_pos_near_zero_nonzero_x_count": final["label_diagnostics"].get(
            "target_pos_near_zero_nonzero_x_count", 0
        ),
        "target_pos_near_zero_nonzero_y_count": final["label_diagnostics"].get(
            "target_pos_near_zero_nonzero_y_count", 0
        ),
        "target_pos_distribution_round6": final["label_diagnostics"]["target_pos_distribution_round6"],
    }
    result_dict = {
        "summary": summary,
        "config": asdict(config),
        "preprocess_log": preprocess_log,
        "alignment_metadata": aligned.metadata,
        "direction_labels": {
            "combined_to_angle_deg": labels["combined_to_angle_deg"],
            "combined_label_names": labels["combined_label_names"],
            "label_to_target_pos": labels.get("label_to_target_pos"),
            "center_tolerance": float(config.center_tolerance),
            "center_center_rule": (
                "Multicoder center-center predictions are retained and assigned "
                "180 deg angular error, matching the existing MATLAB evaluation."
                if task_type == "8target"
                else "Binary labels are evaluated directly; angular error uses the two target angles."
            ),
        },
        "diagnostics": diagnostic_results,
        "timepoints": timepoint_results,
    }

    save_results(result_dict, output_dir, aligned.session_id)
    plot_within_session_results(result_dict, output_dir)
    print_summary(summary)
    return result_dict


def _earliest_sig_time(timepoints: list[dict[str, Any]], key: str) -> float | None:
    for result in timepoints:
        if result.get(key):
            return float(result["time_from_cue_s"])
    return None


def _task_state(time_from_cue_s: float, go_cue_s: float | None) -> str:
    if time_from_cue_s < 0:
        return "fixation/cue-before-reference"
    if go_cue_s is not None and np.isfinite(go_cue_s) and time_from_cue_s >= float(go_cue_s):
        return "movement"
    return "memory"


def save_results(result: dict[str, Any], output_dir: Path, session_id: str) -> None:
    json_path = output_dir / f"{session_id}_within_session_decoding.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result), f, indent=2)

    if not result["timepoints"]:
        summary_path = output_dir / "summary.csv"
        summary_fields = [
            "session_id",
            "source_path",
            "status",
            "skip_reason",
            "task_type",
            "decoder_type",
            "n_targets",
            "mode",
            "cv_scheme",
            "requested_n_splits",
            "actual_n_splits",
            "min_class_count",
            "center_tolerance",
            "n_trials_total",
            "n_valid_trials_after_success_and_targetPos",
            "n_trials_entering_CV",
            "combined_label_distribution",
            "binary_label_distribution",
            "horizontal_label_distribution",
            "vertical_label_distribution",
            "combined_labels_with_count_1",
            "has_real_center_center_label_5",
            "target_pos_near_zero_nonzero_x",
            "target_pos_near_zero_nonzero_y",
            "target_pos_near_zero_nonzero_x_count",
            "target_pos_near_zero_nonzero_y_count",
            "target_pos_distribution_round6",
        ]
        summary = result["summary"]
        summary_row = {
            field: json.dumps(_to_jsonable(summary[field]))
            if isinstance(summary.get(field), (dict, list))
            else summary.get(field, "")
            for field in summary_fields
        }
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerow(summary_row)
        LOGGER.info("Saved %s and %s", json_path, summary_path)
        return

    npz_path = output_dir / f"{session_id}_within_session_decoding_arrays.npz"
    np.savez_compressed(
        npz_path,
        time_from_cue_s=np.asarray([r["time_from_cue_s"] for r in result["timepoints"]]),
        time_from_trial_start_s=np.asarray(
            [r["time_from_trial_start_s"] for r in result["timepoints"]]
        ),
        percent_correct=np.asarray([r["percent_correct"] for r in result["timepoints"]]),
        accuracy=np.asarray([r["accuracy"] for r in result["timepoints"]]),
        balanced_accuracy=np.asarray([r["balanced_accuracy"] for r in result["timepoints"]]),
        mean_angular_error_deg=np.asarray(
            [r["mean_angular_error_deg"] for r in result["timepoints"]]
        ),
        accuracy_p_bonferroni=np.asarray(
            [r["accuracy_p_bonferroni"] for r in result["timepoints"]]
        ),
        angular_error_p_bonferroni=np.asarray(
            [r["angular_error_p_bonferroni"] for r in result["timepoints"]]
        ),
        n_trials=np.asarray([r["n_trials"] for r in result["timepoints"]]),
        feature_dim=np.asarray([r["feature_dim"] for r in result["timepoints"]]),
    )

    final = result["timepoints"][-1]
    summary_path = output_dir / "summary.csv"
    summary_fields = [
        "session_id",
        "source_path",
        "task_type",
        "decoder_type",
        "n_targets",
        "mode",
        "cv_scheme",
        "n_splits",
        "frame_rate_hz",
        "frame_rate_source",
        "recording_system",
        "n_valid_trials",
        "n_counted",
        "accuracy",
        "balanced_accuracy",
        "percent_correct",
        "mean_angular_error_deg",
        "time_from_cue_s",
        "eval_index",
        "frame_indices",
        "confusion_matrix_labels",
        "confusion_matrix_counts",
        "confusion_matrix_row_percent",
    ]
    summary_row = {
        "session_id": result["summary"]["session_id"],
        "source_path": result["summary"]["source_path"],
        "task_type": result["summary"]["task_type"],
        "decoder_type": result["summary"]["decoder_type"],
        "n_targets": result["summary"]["n_targets"],
        "mode": result["summary"]["mode"],
        "cv_scheme": result["summary"]["cv_scheme"],
        "n_splits": result["config"]["n_splits"],
        "frame_rate_hz": result["summary"]["frame_rate_hz"],
        "frame_rate_source": result["summary"]["frame_rate_source"],
        "recording_system": result["summary"]["recording_system"],
        "n_valid_trials": result["summary"]["n_valid_trials"],
        "n_counted": final["n_trials"],
        "accuracy": final["accuracy"],
        "balanced_accuracy": final["balanced_accuracy"],
        "percent_correct": final["percent_correct"],
        "mean_angular_error_deg": final["mean_angular_error_deg"],
        "time_from_cue_s": final["time_from_cue_s"],
        "eval_index": final["eval_index"],
        "frame_indices": json.dumps(_to_jsonable(final["frame_indices"])),
        "confusion_matrix_labels": json.dumps(_to_jsonable(final["confusion_matrix_labels"])),
        "confusion_matrix_counts": json.dumps(_to_jsonable(final["confusion_matrix_counts"])),
        "confusion_matrix_row_percent": json.dumps(_to_jsonable(final["confusion_matrix_row_percent"])),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)

    LOGGER.info("Saved %s, %s, and %s", json_path, npz_path, summary_path)


def _plot_direction_confusion_matrix(
    plt: Any,
    confusion_row_percent: np.ndarray,
    output_path: Path,
    *,
    title: str = "Confusion matrix",
) -> None:
    """Save a direction-labeled confusion matrix image."""

    direction_order = [7, 8, 9, 6, 3, 2, 1, 4, 5]
    direction_symbols = ["↖", "↑", "↗", "→", "↘", "↓", "↙", "←", "•"]
    idx = [label - 1 for label in direction_order]
    cm = np.asarray(confusion_row_percent, dtype=float)[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(5.2, 5.8), constrained_layout=True)
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=100, interpolation="nearest")

    ax.set_xticks(range(len(direction_symbols)), labels=direction_symbols)
    ax.set_yticks(range(len(direction_symbols)), labels=direction_symbols)
    ax.tick_params(axis="both", length=0, labelsize=20)
    ax.set_xlabel("Predicted class", fontsize=16)
    ax.set_ylabel("True class", fontsize=16)
    ax.set_title(title, fontsize=16, pad=14)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.08, pad=0.14)
    cbar.set_label("Percent correct (%)", fontsize=14)
    cbar.set_ticks([0, 50, 100])
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_binary_confusion_matrix(
    plt: Any,
    confusion_row_percent: np.ndarray,
    output_path: Path,
    *,
    title: str = "Binary confusion matrix",
) -> None:
    cm = np.asarray(confusion_row_percent, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Predicted class", fontsize=13)
    ax.set_ylabel("True class", fontsize=13)
    ax.set_title(title, fontsize=14, pad=12)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]:.1f}", ha="center", va="center", color="white", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1, pad=0.14)
    cbar.set_label("Percent (%)", fontsize=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_within_session_results(result: dict[str, Any], output_dir: str | Path) -> None:
    """Generate performance, confusion, and diagnostic plots."""

    if not result["timepoints"]:
        return

    plt = _require("matplotlib.pyplot", "matplotlib")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timepoints = result["timepoints"]
    t = np.asarray([r["time_from_cue_s"] for r in timepoints])
    acc = np.asarray([r["percent_correct"] for r in timepoints])
    ang = np.asarray([r["mean_angular_error_deg"] for r in timepoints])
    acc_sem = np.asarray([
        np.std([f["percent_correct"] for f in r["fold_results"]], ddof=1)
        / math.sqrt(len(r["fold_results"]))
        if len(r["fold_results"]) > 1
        else 0.0
        for r in timepoints
    ])
    ang_sem = np.asarray([
        np.std([f["mean_angular_error_deg"] for f in r["fold_results"]], ddof=1)
        / math.sqrt(len(r["fold_results"]))
        if len(r["fold_results"]) > 1
        else 0.0
        for r in timepoints
    ])
    acc_sig = np.asarray([r["accuracy_significant"] for r in timepoints], dtype=bool)
    ang_sig = np.asarray([r["angular_error_significant"] for r in timepoints], dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, acc, color="#1f77b4", marker="o", ms=3)
    axes[0].fill_between(t, acc - acc_sem, acc + acc_sem, color="#1f77b4", alpha=0.18, linewidth=0)
    axes[0].axhline(result["summary"].get("chance_accuracy", result["config"]["chance_accuracy"]) * 100, color="0.35", ls="--", lw=1)
    axes[0].scatter(t[acc_sig], acc[acc_sig], color="#d62728", zorder=3, s=24)
    axes[0].axvline(0, color="0.2", lw=1)
    axes[0].set_ylabel("Percent correct")
    axes[0].set_title(
        f"{result['summary']['session_id']} {result['config']['mode']} {result['config']['cv_scheme']} decoding"
    )

    axes[1].plot(t, ang, color="#2ca02c", marker="o", ms=3)
    axes[1].fill_between(t, ang - ang_sem, ang + ang_sem, color="#2ca02c", alpha=0.18, linewidth=0)
    axes[1].scatter(t[ang_sig], ang[ang_sig], color="#d62728", zorder=3, s=24)
    axes[1].axvline(0, color="0.2", lw=1)
    axes[1].set_ylabel("Mean angular error (deg)")
    axes[1].set_xlabel("Time from cue onset (s)")
    fig.savefig(output_dir / f"{result['summary']['session_id']}_performance.png", dpi=200)
    plt.close(fig)

    final = timepoints[-1]
    if result["summary"].get("task_type") == "2target":
        cm = np.asarray(final["confusion_matrix_row_percent"], dtype=float)
        _plot_binary_confusion_matrix(
            plt,
            cm,
            output_dir / f"{result['summary']['session_id']}_confusion_final.png",
        )
    else:
        cm = np.asarray(final["confusion_matrix_row_percent_1_to_9"], dtype=float)
        _plot_direction_confusion_matrix(
            plt,
            cm,
            output_dir / f"{result['summary']['session_id']}_confusion_final.png",
        )

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, [r["n_trials"] for r in timepoints], color="#4c78a8")
    axes[0].set_ylabel("Trials")
    axes[1].plot(t, [r["feature_dim"] for r in timepoints], color="#f58518")
    axes[1].set_ylabel("Feature dim")
    axes[2].plot(t, [r["pca_component_min"] for r in timepoints], color="#54a24b", label="min")
    axes[2].plot(t, [r["pca_component_max"] for r in timepoints], color="#e45756", label="max")
    axes[2].legend()
    axes[2].set_ylabel("PCA comps")
    axes[2].set_xlabel("Time from cue onset (s)")
    fig.savefig(output_dir / f"{result['summary']['session_id']}_diagnostics.png", dpi=200)
    plt.close(fig)


def print_summary(summary: dict[str, Any]) -> None:
    print("\nWithin-session decoding summary")
    print(f"  session id: {summary['session_id']}")
    print(f"  status: {summary.get('status', 'ok')}")
    if summary.get("skip_reason"):
        print(f"  skip reason: {summary['skip_reason']}")
    print(f"  valid trials: {summary['n_valid_trials']}")
    if "n_trials_entering_CV" in summary:
        print(f"  trials entering CV: {summary['n_trials_entering_CV']}")
    if "task_type" in summary:
        print(f"  task/decoder: {summary['task_type']} / {summary['decoder_type']}")
    print(f"  mode: {summary['mode']}")
    print(f"  CV scheme: {summary['cv_scheme']}")
    if "actual_n_splits" in summary:
        print(f"  requested/actual splits: {summary.get('requested_n_splits')} / {summary['actual_n_splits']}")
    if "min_class_count" in summary:
        print(f"  min class count: {summary['min_class_count']}")
    if "combined_label_distribution" in summary:
        print(f"  combined labels: {summary['combined_label_distribution']}")
    print(f"  frame rate: {summary['frame_rate_hz']} Hz ({summary['frame_rate_source']})")
    if summary.get("status") in {"skipped", "diagnostic_only"}:
        return
    print(f"  accuracy: {summary['accuracy']:.4f}")
    print(f"  balanced accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"  final accuracy: {summary['final_accuracy_percent']:.2f}%")
    print(f"  final angular error: {summary['final_mean_angular_error_deg']:.2f} deg")
    print(f"  earliest significant accuracy time: {summary['earliest_accuracy_significant_time_s']}")
    print(
        "  earliest significant angular-error time: "
        f"{summary['earliest_angular_error_significant_time_s']}"
    )
    print(f"  PCA component range: {summary['pca_component_range']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", help="Path to doppler_S*_R*+normcorre.mat")
    parser.add_argument("--output-dir", default=WithinSessionConfig.output_dir)
    parser.add_argument("--decoder-type", choices=sorted(VALID_DECODER_TYPES), default=None)
    parser.add_argument("--frame-rate-hz", type=float, default=None)
    parser.add_argument(
        "--mode",
        default=WithinSessionConfig.mode,
        choices=sorted(VALID_DECODING_MODES),
        help=(
            "fixed_memory_3frames builds one feature matrix from the 3 frames before memory end. "
            "dynamic_time_window reproduces the older per-timepoint analysis."
        ),
    )
    parser.add_argument("--cv-scheme", default=WithinSessionConfig.cv_scheme, choices=["loo", "kfold"])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--cpca-m",
        type=int,
        default=WithinSessionConfig.cpca_m,
        help="Final CPCA subspace dimension, matching trainCPCA.m's m parameter.",
    )
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--center-tolerance", type=float, default=WithinSessionConfig.center_tolerance)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--no-motion-correction", action="store_true")
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--no-spatial-filter", action="store_true")
    args = parser.parse_args(argv)

    config = WithinSessionConfig(
        mode=args.mode,
        decoder_type=args.decoder_type,
        frame_rate_hz=args.frame_rate_hz,
        cv_scheme=args.cv_scheme,
        n_splits=args.n_splits,
        random_seed=args.seed,
        cpca_m=args.cpca_m,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        max_timepoints=args.max_timepoints,
        center_tolerance=args.center_tolerance,
        diagnostic_only=args.diagnostic_only,
        apply_motion_correction=not args.no_motion_correction,
        detrend_window=0 if args.no_detrend else WithinSessionConfig.detrend_window,
        spatial_filter_radius=0 if args.no_spatial_filter else WithinSessionConfig.spatial_filter_radius,
    )
    decode_within_session(args.mat_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
