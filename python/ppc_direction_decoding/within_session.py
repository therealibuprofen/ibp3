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
    cv_scheme: str = "kfold"
    n_splits: int = 10
    random_seed: int = 12345
    variance_to_keep: float = 0.95
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


VALID_DECODING_MODES = {"fixed_memory_3frames", "dynamic_time_window"}


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
        candidates.extend(
            [
                source.parent / "ProjectRecord_paper.json",
                source.parent.parent / "ProjectRecord_paper.json",
                source.parent.parent / "PPC_directional_tuning" / "Project Records" / "ProjectRecord_paper.json",
            ]
        )
    candidates.extend(
        [
            Path("data") / "ProjectRecord_paper.json",
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


def _frame_rate_from_project_record(source_path: str) -> tuple[float | None, str | None, str | None]:
    session_run = _infer_session_run(source_path)
    if session_run is None:
        return None, None, None
    session_id, run_id = session_run
    recording_to_rate = {
        "prototypeRT": 2.0,
        "offlineVantage": 1.0,
    }
    for project_record in _project_record_candidates(source_path):
        if not project_record.exists():
            continue
        with project_record.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if int(row.get("Session", -1)) == session_id and int(row.get("Run", -1)) == run_id:
                recording_system = row.get("RecordingSystem")
                frame_rate = recording_to_rate.get(recording_system)
                return frame_rate, recording_system, str(project_record)
    return None, None, None


def _median_finite(values: np.ndarray, name: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot derive task epoch timing: behavior field '{name}' is missing/empty.")
    return float(np.median(finite))


def align_fusi_and_behavior(session: dict[str, Any], session_id: str | None = None) -> AlignedSession:
    """Align task-aligned fUSI frames with behavior and construct trial labels.

    The distributed ``doppler_S*_R*+normcorre.mat`` files are already
    task-aligned by the MATLAB loading pipeline. This function verifies shape
    agreement, applies the existing successful-trial rule, extracts
    ``targetPos``, and reconstructs both requested time axes.

    Returns an ``AlignedSession`` with image shape ``[y, x, time, trial]``.
    """

    if "iDop" not in session:
        raise KeyError(f"Session is missing 'iDop'. Found fields: {sorted(session.keys())}")
    if "behavior" not in session:
        raise KeyError(f"Session is missing 'behavior'. Found fields: {sorted(session.keys())}")

    images = np.asarray(session["iDop"], dtype=np.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected iDop shape [y, x, time, trial], found {images.shape}")

    behavior = _mat_struct_array_to_list(session["behavior"])
    if len(behavior) != images.shape[3]:
        raise ValueError(
            "Behavior/image trial mismatch. "
            f"behavior has {len(behavior)} rows; iDop has {images.shape[3]} trials. "
            "Use the existing MATLAB loader/metadata to provide task-aligned successful trials."
        )

    core_params = session.get("coreParams", {})
    if not isinstance(core_params, dict):
        core_params = {}
    frame_rate_source = "coreParams"
    recording_system = None
    project_record_path = None
    frame_rate = _get_scalar(core_params, ("framerate", "frameRate", "fs"), default=None)
    if not frame_rate or frame_rate <= 0:
        inferred_rate, recording_system, project_record_path = _frame_rate_from_project_record(
            str(session.get("_source_path", ""))
        )
        if inferred_rate:
            frame_rate = inferred_rate
            frame_rate_source = "ProjectRecord.RecordingSystem"
            LOGGER.info(
                "Missing coreParams.framerate; inferred %.1f Hz from %s in %s",
                frame_rate,
                recording_system,
                project_record_path,
            )
        else:
            LOGGER.warning(
                "Missing coreParams.framerate and no ProjectRecord match; using 1 Hz fallback."
            )
            frame_rate = 1.0
            frame_rate_source = "fallback_1hz"

    success = _extract_success(behavior)
    target_pos = _extract_target_pos(behavior)
    valid = success & np.all(np.isfinite(target_pos), axis=1)

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
    cue_index = int(math.ceil(abs(fix_start_s) * frame_rate))
    cue_index = max(0, min(cue_index, images.shape[2] - 1))

    frame_index = np.arange(images.shape[2], dtype=float)
    time_from_trial_start_s = frame_index / frame_rate
    time_from_cue_s = (frame_index - cue_index) / frame_rate

    metadata = {
        "cue_reference": cue_name,
        "frame_rate_hz": float(frame_rate),
        "frame_rate_source": frame_rate_source,
        "recording_system": recording_system,
        "project_record_path": project_record_path,
        "fix_start_s_relative_to_cue": fix_start_s,
        "go_cue_s_relative_to_cue": float(np.nanmedian(target_acquire - cue_reference)),
        "source_path": session.get("_source_path", ""),
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
    csum = np.cumsum(flat, axis=1, dtype=np.float64)
    out = np.empty_like(flat, dtype=np.float32)
    for t in range(flat.shape[1]):
        start = max(0, t - window)
        total = csum[:, t] - (csum[:, start - 1] if start > 0 else 0.0)
        out[:, t] = total / (t - start + 1)
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
    if config.apply_motion_correction:
        if normcorre_present and config.assume_normcorre_if_present:
            log["motion_correction"] = "already_normcorre_from_source_file"
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
        mean_per_voxel = out.reshape((-1, out.shape[2] * out.shape[3]), order="F").mean(axis=1)
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


def make_multicoder_labels(target_pos: np.ndarray) -> dict[str, Any]:
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
        labels[target_pos[:, dim] < 0, dim] = 1
        labels[target_pos[:, dim] == 0, dim] = 2
        labels[target_pos[:, dim] > 0, dim] = 3
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
    within each frame. Missing/non-finite samples are excluded from this
    timepoint rather than imputed with future data.
    """

    if eval_index < cue_index:
        frame_indices = np.arange(0, eval_index + 1)
    else:
        frame_indices = np.arange(cue_index, eval_index + 1)

    y, x, _, n_trials = images.shape
    if voxel_mask is None:
        frame_finite = np.isfinite(images[:, :, frame_indices, :]).all(axis=(2, 3))
        voxel_mask = frame_finite
    voxel_mask = np.asarray(voxel_mask, dtype=bool)
    n_voxels = int(voxel_mask.sum())
    if n_voxels == 0:
        raise ValueError("No valid voxels remain after applying finite/background mask.")

    trial_ok = np.asarray(valid_trial_mask, dtype=bool).copy()
    trial_ok &= np.isfinite(images[:, :, frame_indices, :]).all(axis=(0, 1, 2))
    trial_indices = np.where(trial_ok)[0]
    features = np.empty((trial_indices.size, n_voxels * frame_indices.size), dtype=np.float32)
    for row, trial in enumerate(trial_indices):
        chunks = []
        for frame in frame_indices:
            chunks.append(images[:, :, frame, trial][voxel_mask].reshape(-1, order="F"))
        features[row, :] = np.concatenate(chunks)

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
    if voxel_mask is None:
        voxel_mask = np.isfinite(images[:, :, frame_indices, :]).all(axis=(2, 3))
    voxel_mask = np.asarray(voxel_mask, dtype=bool)
    n_voxels = int(voxel_mask.sum())
    if n_voxels == 0:
        raise ValueError("No valid voxels remain after applying finite/background mask.")

    trial_ok = np.asarray(valid_trial_mask, dtype=bool).copy()
    if trial_ok.size != n_trials:
        raise ValueError(f"valid_trial_mask has {trial_ok.size} entries; images have {n_trials} trials.")
    trial_ok &= np.isfinite(images[:, :, frame_indices, :]).all(axis=(0, 1, 2))
    trial_indices = np.where(trial_ok)[0]
    features = np.empty((trial_indices.size, n_voxels * frame_indices.size), dtype=np.float32)
    for row, trial in enumerate(trial_indices):
        chunks = [
            images[:, :, frame, trial][voxel_mask].reshape(-1, order="F")
            for frame in frame_indices
        ]
        features[row, :] = np.concatenate(chunks)

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
        "percent_correct": percent_correct,
        "n_correct": n_correct,
        "n_counted": int(y.size),
        "pca_components": np.asarray(pca_components, dtype=int),
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
    if possible.size != 8:
        LOGGER.warning("Expected 8 real target directions, found %d: %s", possible.size, possible.tolist())

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
    aligned = align_fusi_and_behavior(session, session_id=session_id)
    images, preprocess_log = preprocess_power_doppler_session(
        aligned.images, config, source_path=str(mat_path), output_dir=output_dir
    )

    labels = make_multicoder_labels(aligned.target_pos)
    valid_mask = aligned.valid_trial_mask.copy()
    axis_labels_all = labels["axis_labels"]
    combined_all = labels["combined_labels"]

    n_time = images.shape[2]
    if config.max_timepoints:
        n_time = min(n_time, config.max_timepoints)

    timepoint_results = []
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

        axis_labels = axis_labels_all[trial_indices]
        combined = combined_all[trial_indices]
        if len(np.unique(combined)) < 2:
            LOGGER.warning("Skipping timepoint %d: fewer than 2 classes", result_index)
            continue

        decoded = decode_timepoint(x, axis_labels, combined, config)
        mean_ang, per_trial_ang = compute_angular_error(
            decoded["predictions_combined"], decoded["actual_combined"], labels["combined_to_angle_deg"]
        )
        for fold in decoded["fold_results"]:
            fold_mean_ang, _ = compute_angular_error(
                np.asarray(fold["predicted_combined"]),
                np.asarray(fold["actual_combined"]),
                labels["combined_to_angle_deg"],
            )
            fold["mean_angular_error_deg"] = fold_mean_ang

        acc_p = _binomial_greater_pvalue(
            decoded["n_correct"], decoded["n_counted"], config.chance_accuracy
        )
        perm = permutation_test_angular_error(
            decoded["actual_combined"],
            mean_ang,
            labels["combined_to_angle_deg"],
            n_permutations=config.n_permutations,
            random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

        conf = _confusion_matrix(
            decoded["actual_combined"], decoded["predictions_combined"], np.arange(1, 10)
        )
        accuracy = _accuracy_from_confusion(conf)
        balanced_accuracy = _balanced_accuracy_from_confusion(conf)
        row_sum = conf.sum(axis=1, keepdims=True)
        conf_pct = np.divide(conf * 100.0, row_sum, out=np.zeros_like(conf, dtype=float), where=row_sum > 0)

        result = {
            **winfo,
            "mode": config.mode,
            "time_from_trial_start_s": float(aligned.time_from_trial_start_s[result_index]),
            "time_from_cue_s": float(aligned.time_from_cue_s[result_index]),
            "task_state": _task_state(
                float(aligned.time_from_cue_s[result_index]),
                aligned.metadata.get("go_cue_s_relative_to_cue"),
            ),
            "cv_scheme": config.cv_scheme,
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
            "chance_accuracy": config.chance_accuracy,
            "permutation_null_angular_error": perm,
            "actual_combined": decoded["actual_combined"],
            "predicted_combined": decoded["predictions_combined"],
            "global_trial_indices": trial_indices,
            "confusion_matrix_counts_1_to_9": conf,
            "confusion_matrix_row_percent_1_to_9": conf_pct,
        }
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
        raise RuntimeError("No decodable timepoints were produced. Check trial labels and data completeness.")

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
        "n_trials_total": int(images.shape[3]),
        "n_valid_trials": int(valid_mask.sum()),
        "mode": config.mode,
        "cv_scheme": config.cv_scheme,
        "frame_rate_hz": float(aligned.frame_rate_hz),
        "frame_rate_source": aligned.metadata.get("frame_rate_source"),
        "recording_system": aligned.metadata.get("recording_system"),
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
    }
    result_dict = {
        "summary": summary,
        "config": asdict(config),
        "preprocess_log": preprocess_log,
        "alignment_metadata": aligned.metadata,
        "direction_labels": {
            "combined_to_angle_deg": labels["combined_to_angle_deg"],
            "combined_label_names": labels["combined_label_names"],
            "center_center_rule": (
                "Multicoder center-center predictions are retained and assigned "
                "180 deg angular error, matching the existing MATLAB evaluation."
            ),
        },
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
        "confusion_matrix_counts_1_to_9",
        "confusion_matrix_row_percent_1_to_9",
    ]
    summary_row = {
        "session_id": result["summary"]["session_id"],
        "source_path": result["summary"]["source_path"],
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
        "confusion_matrix_counts_1_to_9": json.dumps(
            _to_jsonable(final["confusion_matrix_counts_1_to_9"])
        ),
        "confusion_matrix_row_percent_1_to_9": json.dumps(
            _to_jsonable(final["confusion_matrix_row_percent_1_to_9"])
        ),
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


def plot_within_session_results(result: dict[str, Any], output_dir: str | Path) -> None:
    """Generate performance, confusion, and diagnostic plots."""

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
    axes[0].axhline(result["config"]["chance_accuracy"] * 100, color="0.35", ls="--", lw=1)
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
    print(f"  valid trials: {summary['n_valid_trials']}")
    print(f"  mode: {summary['mode']}")
    print(f"  CV scheme: {summary['cv_scheme']}")
    print(f"  frame rate: {summary['frame_rate_hz']} Hz ({summary['frame_rate_source']})")
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
    parser.add_argument("--n-permutations", type=int, default=100_000)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--no-motion-correction", action="store_true")
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--no-spatial-filter", action="store_true")
    args = parser.parse_args(argv)

    config = WithinSessionConfig(
        mode=args.mode,
        cv_scheme=args.cv_scheme,
        n_splits=args.n_splits,
        random_seed=args.seed,
        n_permutations=args.n_permutations,
        output_dir=args.output_dir,
        max_timepoints=args.max_timepoints,
        apply_motion_correction=not args.no_motion_correction,
        detrend_window=0 if args.no_detrend else WithinSessionConfig.detrend_window,
        spatial_filter_radius=0 if args.no_spatial_filter else WithinSessionConfig.spatial_filter_radius,
    )
    decode_within_session(args.mat_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
