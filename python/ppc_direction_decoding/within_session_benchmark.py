"""Unified within-session fUS decoding benchmark.

This module intentionally reuses :mod:`within_session` for MATLAB/HDF5
loading, fUS/behavior alignment, preprocessing, target labels, and feature
window construction.  The benchmark layer only adds shared cross-validation,
additional decoders, common metrics, and result serialization.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import importlib.util
import json
import logging
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# Required by CUDA/cuBLAS when PyTorch deterministic algorithms are enabled.
# This must be set before CUDA is initialized, so do it at module import time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LOGGER = logging.getLogger(__name__)
ALL_MODELS = ("pca_lda", "cpca_lda", "cnn", "cnn_lstm")
LINEAR_MODELS = {"pca_lda", "cpca_lda"}
DEEP_MODELS = {"cnn", "cnn_lstm"}


@dataclass
class BenchmarkConfig:
    """Configuration for the unified benchmark."""

    models: tuple[str, ...] = ALL_MODELS
    mode: str = "fixed_memory_3frames"
    n_splits: int = 5
    repeats: int = 3
    random_seed: int = 12345
    variance_to_keep: float = 0.95
    cpca_m: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    validation_fraction: float = 0.2
    epoch_log_interval: int = 10
    cnn_spatial_pool: int = 4
    deep_hidden_dim: int = 64
    deep_dropout: float = 0.2
    use_class_weights: bool = True
    voxel_mask_percentile: float = 20.0
    voxel_mask_min_fraction: float = 0.05
    deep_train_mask_percentile: float = 20.0
    deep_train_mask_min_fraction: float = 0.05
    deep_foreground_mode: str = "crop"
    foreground_epsilon: float = 1e-8
    foreground_margin: int = 2
    normalize_foreground_only: bool = True
    device: str = "auto"
    num_workers: int = 0
    frame_rate_hz: float | None = None
    center_tolerance: float = 1e-6
    output_dir: str = "output/decoding/within_session_benchmark"
    max_timepoints: int | None = None
    min_trials_per_timepoint: int = 8
    apply_motion_correction: bool = True
    detrend_window: int = 50
    spatial_filter_radius: int = 2
    direct_8class: bool = False
    merge_existing: bool = True
    deterministic_torch: bool = True


@dataclass
class PreparedWindow:
    """One decoded time window shared by every benchmark model."""

    name: str
    eval_index: int
    x_flat: np.ndarray
    x_frames: np.ndarray
    voxel_mask: np.ndarray
    trial_indices_global: np.ndarray
    window_info: dict[str, Any]
    labels_combined: np.ndarray
    labels_axis: np.ndarray | None


def _default_core_path() -> Path:
    return Path(__file__).resolve().with_name("within_session.py")


def load_core_module(core_script: str | Path | None = None) -> Any:
    """Load the existing ``within_session.py`` module from a path or package."""

    if core_script is None:
        return importlib.import_module("ppc_direction_decoding.within_session")

    raw = Path(core_script)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, _default_core_path().parent / raw.name])
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("ppc_within_session_core", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load core script from {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Core script not found: {core_script}")


def _require(module_name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise ImportError(f"Missing dependency '{module_name}'. Install '{hint}'.") from exc


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _select_device(requested: str) -> str:
    if requested == "gpu":
        requested = "cuda"
    if requested != "auto":
        return requested
    if not _torch_available():
        return "cpu"
    torch = _require("torch")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if not _torch_available():
        return
    torch = _require("torch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def _distribution_dict(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    values = values[values >= 0]
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def make_shared_splits(y: np.ndarray, n_splits: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Create one shared stratified split list for every model."""

    sklearn_model_selection = _require("sklearn.model_selection", "scikit-learn")
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y[y >= 0])
    positive = counts[counts > 0]
    if positive.size < 2:
        raise ValueError("Need at least two classes for stratified CV.")
    actual = min(int(n_splits), int(positive.min()))
    if actual < 2:
        raise ValueError(
            f"Cannot run stratified CV: requested {n_splits}, smallest class has {int(positive.min())} sample."
        )
    cv = sklearn_model_selection.StratifiedKFold(n_splits=actual, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(y.size), y)), actual


def _angle_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _angles_for_labels(labels: np.ndarray, label_to_angle: dict[int, float]) -> np.ndarray:
    out = np.full(np.asarray(labels).shape, np.nan, dtype=float)
    for i, label in np.ndenumerate(np.asarray(labels, dtype=int)):
        if int(label) in label_to_angle:
            out[i] = float(label_to_angle[int(label)])
    return out


def _precision_recall_f1(confusion: np.ndarray, labels: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    f1_values = []
    for i, label in enumerate(labels):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - confusion[i, i])
        fn = float(confusion[i, :].sum() - confusion[i, i])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        support = int(confusion[i, :].sum())
        rows.append(
            {
                "label": int(label),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
        if support > 0:
            f1_values.append(f1)
    macro_f1 = float(np.mean(f1_values)) if f1_values else float("nan")
    return rows, macro_f1


def compute_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    combined_to_angle_deg: dict[int, float],
    pred_horizontal: np.ndarray | None = None,
    pred_vertical: np.ndarray | None = None,
    true_horizontal: np.ndarray | None = None,
    true_vertical: np.ndarray | None = None,
    proba: np.ndarray | None = None,
    chance_accuracy: float,
) -> dict[str, Any]:
    sklearn_metrics = _require("sklearn.metrics", "scikit-learn")
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.asarray(labels, dtype=int)
    confusion = sklearn_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(sklearn_metrics.accuracy_score(y_true, y_pred))
    row_sum = confusion.sum(axis=1)
    present = row_sum > 0
    recall = np.divide(
        np.diag(confusion).astype(float),
        row_sum,
        out=np.zeros(row_sum.shape, dtype=float),
        where=row_sum > 0,
    )
    balanced = float(recall[present].mean()) if np.any(present) else float("nan")
    per_class, macro_f1 = _precision_recall_f1(confusion, labels)

    true_angles = _angles_for_labels(y_true, combined_to_angle_deg)
    pred_angles = _angles_for_labels(y_pred, combined_to_angle_deg)
    valid_angle = np.isfinite(true_angles) & np.isfinite(pred_angles)
    if task_type == "8target":
        valid_angle &= y_pred != 5
    angular_errors = np.full(y_true.shape, np.nan, dtype=float)
    angular_errors[valid_angle] = _angle_distance_deg(pred_angles[valid_angle], true_angles[valid_angle])
    mean_ang = float(np.nanmean(angular_errors)) if np.any(valid_angle) else float("nan")
    median_ang = float(np.nanmedian(angular_errors)) if np.any(valid_angle) else float("nan")

    center_center_rate = float(np.mean(y_pred == 5)) if task_type == "8target" else 0.0
    h_acc = None
    v_acc = None
    if task_type == "8target" and pred_horizontal is not None and pred_vertical is not None:
        assert true_horizontal is not None and true_vertical is not None
        h_acc = float(np.mean(np.asarray(pred_horizontal) == np.asarray(true_horizontal)))
        v_acc = float(np.mean(np.asarray(pred_vertical) == np.asarray(true_vertical)))

    top2 = None
    if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        top = np.argsort(proba, axis=1)[:, -2:]
        top_labels = labels[top]
        top2 = float(np.mean(np.any(top_labels == y_true[:, None], axis=1)))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "mean_angular_error_deg": mean_ang,
        "median_angular_error_deg": median_ang,
        "angular_error_per_trial_deg": angular_errors,
        "valid_angular_error_count": int(np.sum(valid_angle)),
        "center_center_prediction_rate": center_center_rate,
        "horizontal_accuracy": h_acc,
        "vertical_accuracy": v_acc,
        "top2_accuracy": top2,
        "chance_accuracy": float(chance_accuracy),
    }


def _make_core_config(core: Any, config: BenchmarkConfig, output_dir: Path) -> Any:
    return core.WithinSessionConfig(
        mode=config.mode,
        decoder_type=None,
        frame_rate_hz=config.frame_rate_hz,
        cv_scheme="kfold",
        n_splits=config.n_splits,
        random_seed=config.random_seed,
        variance_to_keep=config.variance_to_keep,
        cpca_m=config.cpca_m,
        output_dir=str(output_dir),
        max_timepoints=config.max_timepoints,
        min_trials_per_timepoint=config.min_trials_per_timepoint,
        center_tolerance=config.center_tolerance,
        apply_motion_correction=config.apply_motion_correction,
        detrend_window=config.detrend_window,
        spatial_filter_radius=config.spatial_filter_radius,
        n_permutations=1,
    )


def _window_frame_tensor(
    images: np.ndarray,
    trial_indices: np.ndarray,
    frame_indices: list[int],
    voxel_mask: np.ndarray,
) -> np.ndarray:
    frames = []
    for trial in trial_indices:
        chunks = [images[:, :, int(frame), int(trial)].astype(np.float32, copy=False) for frame in frame_indices]
        trial_frames = np.stack(chunks, axis=0)
        trial_frames[:, ~voxel_mask] = 0.0
        frames.append(trial_frames)
    arr = np.stack(frames, axis=0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _power_doppler_voxel_mask(
    values: np.ndarray,
    *,
    percentile: float,
    min_fraction: float,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an anatomical/angiogram-like mask from mean Power Doppler strength."""

    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim < 3:
        raise ValueError(f"Cannot build voxel mask from array with shape {arr.shape}")
    height, width = arr.shape[:2]
    samples = arr.reshape(height, width, -1)
    mean_power = np.mean(np.abs(samples), axis=2)
    stability = mean_power / (np.std(samples, axis=2) + 1e-6)
    if candidate_mask is None:
        candidate = np.ones((height, width), dtype=bool)
    else:
        candidate = np.asarray(candidate_mask, dtype=bool)
        if candidate.shape != (height, width):
            raise ValueError(f"candidate_mask shape {candidate.shape} does not match frame shape {(height, width)}")
    positive = (mean_power > 0) & candidate
    n_pixels = int(height * width)
    n_candidate = int(candidate.sum())
    min_pixels = max(1, int(math.ceil(float(min_fraction) * max(1, n_candidate))))

    info: dict[str, Any] = {
        "method": "mean_power_doppler_percentile",
        "percentile": float(percentile),
        "min_fraction": float(min_fraction),
        "n_pixels": n_pixels,
        "n_candidate_pixels": n_candidate,
        "n_positive_power_pixels": int(positive.sum()),
        "mean_power_min": float(mean_power[positive].min()) if np.any(positive) else 0.0,
        "mean_power_max": float(mean_power[candidate].max()) if np.any(candidate) else 0.0,
        "stability_mean": float(stability[positive].mean()) if np.any(positive) else 0.0,
    }

    if not np.any(positive):
        mask = candidate.copy()
        info.update({"threshold": 0.0, "status": "empty_power_fallback_candidate_mask"})
    else:
        candidate_values = mean_power[positive]
        threshold = float(np.percentile(candidate_values, float(percentile)))
        mask = (mean_power >= threshold) & candidate
        if int(mask.sum()) < min_pixels:
            flat = np.where(candidate, mean_power, -np.inf).reshape(-1)
            keep = min(min_pixels, int(np.count_nonzero(np.isfinite(flat) & (flat > 0))))
            if keep > 0:
                selected = np.argpartition(flat, -keep)[-keep:]
                mask = np.zeros_like(flat, dtype=bool)
                mask[selected] = True
                mask = mask.reshape(height, width)
                info["status"] = "top_power_min_fraction"
            else:
                mask = candidate.copy()
                info["status"] = "too_few_positive_fallback_candidate_mask"
        else:
            info["status"] = "percentile_threshold"
        info["threshold"] = threshold

    info["n_mask_pixels"] = int(mask.sum())
    info["mask_fraction"] = float(mask.mean()) if mask.size else 0.0
    return mask.astype(bool, copy=False), info


def _window_power_mask(
    images: np.ndarray,
    trial_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    percentile: float,
    min_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    stack = images[:, :, np.asarray(frame_indices, dtype=int), :][:, :, :, np.asarray(trial_indices, dtype=int)]
    return _power_doppler_voxel_mask(stack, percentile=percentile, min_fraction=min_fraction)


def _train_fold_power_mask(
    x_train_frames: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    percentile: float,
    min_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    stack = np.transpose(np.asarray(x_train_frames, dtype=np.float32), (2, 3, 1, 0))
    mask, info = _power_doppler_voxel_mask(
        stack,
        percentile=percentile,
        min_fraction=min_fraction,
        candidate_mask=candidate_mask,
    )
    info["source"] = "inner_train_mean_power_doppler"
    return mask, info


def prepare_benchmark_windows(
    *,
    core: Any,
    mat_path: str | Path,
    config: BenchmarkConfig,
    output_dir: Path,
    session_id: str | None = None,
) -> tuple[Any, dict[str, Any], list[PreparedWindow], dict[str, Any], dict[str, Any], float]:
    """Run the reused loading/alignment/preprocessing/label/window pipeline."""

    core_config = _make_core_config(core, config, output_dir)
    session = core.load_mat73_session(mat_path)
    aligned = core.align_fusi_and_behavior(
        session,
        session_id=session_id,
        decoder_type=None,
        frame_rate_hz=config.frame_rate_hz,
    )
    task_type = str(aligned.metadata["task_type"])
    images, preprocess_log = core.preprocess_power_doppler_session(
        aligned.images,
        core_config,
        source_path=str(mat_path),
        output_dir=output_dir,
    )

    if task_type == "8target":
        label_info = core.make_multicoder_labels(aligned.target_pos, center_tolerance=config.center_tolerance)
        combined_all = label_info["combined_labels"]
        axis_all = label_info["axis_labels"]
    elif task_type == "2target":
        label_info = core.make_binary_labels(aligned.target_pos, center_tolerance=config.center_tolerance)
        combined_all = label_info["binary_labels"]
        axis_all = None
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'")

    valid_mask = aligned.valid_trial_mask.copy()
    n_time = images.shape[2]
    if config.max_timepoints is not None:
        n_time = min(n_time, int(config.max_timepoints))

    decode_windows: list[tuple[str, int | None]] = []
    if config.mode == "fixed_memory_3frames":
        decode_windows.append(("fixed_memory_3frames", None))
    elif config.mode == "dynamic_time_window":
        decode_windows.extend(("dynamic_time_window", i) for i in range(n_time))
    else:
        raise ValueError(f"Unsupported mode '{config.mode}'")

    prepared: list[PreparedWindow] = []
    for window_mode, eval_index in decode_windows:
        if window_mode == "fixed_memory_3frames":
            memory_end_index = core._memory_end_frame_index(aligned)
            start = memory_end_index - 3
            if start < 0:
                raise ValueError("Cannot build fixed_memory_3frames features: fewer than 3 frames exist.")
            frame_indices = np.arange(start, memory_end_index, dtype=int)
            trial_ok = np.asarray(valid_mask, dtype=bool)
            candidate_trials = np.flatnonzero(trial_ok)
            voxel_mask, voxel_mask_info = _window_power_mask(
                images,
                candidate_trials,
                frame_indices,
                percentile=config.voxel_mask_percentile,
                min_fraction=config.voxel_mask_min_fraction,
            )
            x_flat, trial_indices, winfo = core.build_fixed_memory_3frames_features(
                images, aligned, valid_mask, voxel_mask=voxel_mask
            )
        else:
            assert eval_index is not None
            if eval_index < aligned.cue_index:
                frame_indices = np.arange(0, eval_index + 1, dtype=int)
            else:
                frame_indices = np.arange(aligned.cue_index, eval_index + 1, dtype=int)
            candidate_trials = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
            voxel_mask, voxel_mask_info = _window_power_mask(
                images,
                candidate_trials,
                frame_indices,
                percentile=config.voxel_mask_percentile,
                min_fraction=config.voxel_mask_min_fraction,
            )
            x_flat, trial_indices, winfo = core.build_dynamic_window_features(
                images,
                int(eval_index),
                aligned.cue_index,
                valid_mask,
                voxel_mask=voxel_mask,
            )

        if trial_indices.size < config.min_trials_per_timepoint:
            LOGGER.warning("Skipping window %s: only %d trials", window_mode, trial_indices.size)
            continue
        winfo["voxel_mask_info"] = voxel_mask_info
        x_frames = _window_frame_tensor(images, trial_indices, winfo["frame_indices"], voxel_mask)
        labels_combined = combined_all[trial_indices].astype(int)
        labels_axis = axis_all[trial_indices].astype(int) if axis_all is not None else None
        prepared.append(
            PreparedWindow(
                name=window_mode,
                eval_index=int(winfo["eval_index"]),
                x_flat=x_flat,
                x_frames=x_frames,
                voxel_mask=np.asarray(voxel_mask, dtype=bool),
                trial_indices_global=trial_indices.astype(int),
                window_info=winfo,
                labels_combined=labels_combined,
                labels_axis=labels_axis,
            )
        )

    if not prepared:
        raise ValueError("No decodable benchmark windows were produced.")
    chance = float(aligned.metadata["task_config"]["chance_accuracy"])
    return aligned, preprocess_log, prepared, label_info, asdict(core_config), chance


def _linear_predict_fold(
    *,
    core: Any,
    model_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    labels_axis_train: np.ndarray | None,
    task_type: str,
    config: BenchmarkConfig,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if task_type == "8target":
        if labels_axis_train is None:
            raise ValueError("8-target linear decoding requires horizontal/vertical axis labels.")
        h_cpca_m = max(int(config.cpca_m), int(np.unique(labels_axis_train[:, 0]).size) - 1)
        v_cpca_m = max(int(config.cpca_m), int(np.unique(labels_axis_train[:, 1]).size) - 1)
        if model_name == "pca_lda":
            h_pred, h_comp, h_zero = core.fit_fold_scaler_pca_lda(
                x_train, labels_axis_train[:, 0], x_test, config.variance_to_keep
            )
            v_pred, v_comp, v_zero = core.fit_fold_scaler_pca_lda(
                x_train, labels_axis_train[:, 1], x_test, config.variance_to_keep
            )
        elif model_name == "cpca_lda":
            h_pred, h_comp, h_zero = core.fit_fold_scaler_projection_lda(
                x_train,
                labels_axis_train[:, 0],
                x_test,
                variance_to_keep=config.variance_to_keep,
                decoder_type="cpca_lda",
                cpca_m=h_cpca_m,
                random_seed=seed,
            )
            v_pred, v_comp, v_zero = core.fit_fold_scaler_projection_lda(
                x_train,
                labels_axis_train[:, 1],
                x_test,
                variance_to_keep=config.variance_to_keep,
                decoder_type="cpca_lda",
                cpca_m=v_cpca_m,
                random_seed=seed + 1,
            )
        else:
            raise ValueError(model_name)
        pred = (h_pred + 3 * (v_pred - 1)).astype(int)
        model_info = {
            "components_horizontal": int(h_comp),
            "components_vertical": int(v_comp),
            "zero_std_features_horizontal": int(h_zero),
            "zero_std_features_vertical": int(v_zero),
            "requested_cpca_m": int(config.cpca_m),
            "effective_cpca_m_horizontal": int(h_cpca_m) if model_name == "cpca_lda" else None,
            "effective_cpca_m_vertical": int(v_cpca_m) if model_name == "cpca_lda" else None,
        }
        return {
            "pred_combined": pred,
            "pred_horizontal": h_pred.astype(int),
            "pred_vertical": v_pred.astype(int),
            "proba_combined": None,
            "train_log": {"training_time": float(time.perf_counter() - t0), "best_epoch": None},
            "model_info": model_info,
        }

    pred, n_components, zero_std = core.fit_fold_scaler_projection_lda(
        x_train,
        y_train,
        x_test,
        variance_to_keep=config.variance_to_keep,
        decoder_type=model_name,
        cpca_m=config.cpca_m,
        random_seed=seed,
    )
    return {
        "pred_combined": pred.astype(int),
        "pred_horizontal": None,
        "pred_vertical": None,
        "proba_combined": None,
        "train_log": {"training_time": float(time.perf_counter() - t0), "best_epoch": None},
        "model_info": {"components": int(n_components), "zero_std_features": int(zero_std)},
    }


class _FrameDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        axis: np.ndarray | None,
        indices: np.ndarray,
        model_name: str,
        task_type: str,
        normalization: dict[str, Any],
        spatial_roi: dict[str, Any] | None = None,
    ) -> None:
        torch = _require("torch")
        arr = np.asarray(x[indices], dtype=np.float32)
        arr = _apply_spatial_roi(arr, spatial_roi)
        valid_mask = np.asarray(normalization["valid_mask"], dtype=bool)
        invalid = ~valid_mask.reshape(1, 1, *valid_mask.shape)
        mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, *normalization["mean"].shape)
        std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, *normalization["std"].shape)
        arr = (arr - mean) / std
        arr = np.where(invalid, 0.0, arr)
        if model_name == "cnn":
            pass
        else:
            arr = arr[:, :, None, :, :]
        self.x = torch.from_numpy(arr.astype(np.float32, copy=False))
        self.y = torch.from_numpy(np.asarray(y[indices], dtype=np.int64))
        self.axis = None if axis is None else torch.from_numpy(np.asarray(axis[indices], dtype=np.int64))
        self.model_name = model_name
        self.task_type = task_type

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if self.task_type == "8target":
            assert self.axis is not None
            return self.x[idx], self.axis[idx, 0] - 1, self.axis[idx, 1] - 1
        return self.x[idx], self.y[idx]


def _normalization_from_train(
    x_train_frames: np.ndarray,
    model_name: str,
    *,
    valid_mask: np.ndarray | None = None,
    foreground_epsilon: float = 0.0,
    foreground_only: bool = False,
) -> dict[str, Any]:
    x_train_frames = np.asarray(x_train_frames, dtype=np.float32)
    _, n_frames, height, width = x_train_frames.shape
    if valid_mask is None:
        valid = np.ones((height, width), dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != (height, width):
            raise ValueError(f"valid_mask shape {valid.shape} does not match frame shape {(height, width)}")
    mean = np.zeros((n_frames, height, width), dtype=np.float32)
    std = np.ones((n_frames, height, width), dtype=np.float32)
    if np.any(valid):
        mean[:, valid] = x_train_frames[:, :, valid].mean(axis=0)
        voxel_std = x_train_frames[:, :, valid].std(axis=0)
        zero_std = voxel_std < 1e-6
        voxel_std = np.where(zero_std, 1.0, voxel_std)
        std[:, valid] = voxel_std.astype(np.float32)
    else:
        zero_std = np.asarray([], dtype=bool)
    return {
        "method": "train_fold_per_frame_per_voxel_zscore",
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "valid_mask": valid,
        "valid_voxel_count": int(valid.sum()),
        "feature_count": int(valid.sum()) * int(n_frames),
        "zero_std_features": int(np.sum(zero_std)),
    }


def _spatial_roi_from_voxel_mask(
    voxel_mask: np.ndarray,
    *,
    mode: str,
    margin: int,
) -> dict[str, Any]:
    """Create a deep-model ROI from the same voxel mask used for flat features."""

    full_mask = np.asarray(voxel_mask, dtype=bool)
    height, width = full_mask.shape
    mode = str(mode).lower()
    roi = {
        "mode": mode,
        "source": "window_voxel_mask",
        "original_shape": [int(height), int(width)],
        "bbox": [0, int(height), 0, int(width)],
        "roi_shape": [int(height), int(width)],
        "active_pixel_fraction": float(full_mask.mean()) if full_mask.size else 0.0,
        "valid_voxel_count": int(full_mask.sum()),
        "status": "disabled" if mode == "none" else "full_image",
        "_mask": full_mask,
    }
    if mode == "none":
        return roi
    if mode != "crop":
        raise ValueError(f"Unsupported deep_foreground_mode '{mode}'. Use 'crop' or 'none'.")

    if not np.any(full_mask):
        roi["status"] = "empty_voxel_mask_fallback_full_image"
        return roi

    ys, xs = np.where(full_mask)
    pad = max(0, int(margin))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(int(height), int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(int(width), int(xs.max()) + pad + 1)
    if y1 <= y0 or x1 <= x0:
        roi["status"] = "invalid_bbox_fallback_full_image"
        return roi
    roi["bbox"] = [int(y0), int(y1), int(x0), int(x1)]
    roi["roi_shape"] = [int(y1 - y0), int(x1 - x0)]
    roi["_mask"] = full_mask[y0:y1, x0:x1]
    roi["status"] = "cropped" if (y0, y1, x0, x1) != (0, height, 0, width) else "full_image_no_crop"
    return roi


def _apply_spatial_roi(x_frames: np.ndarray, spatial_roi: dict[str, Any] | None) -> np.ndarray:
    if not spatial_roi or spatial_roi.get("mode") != "crop":
        return x_frames
    y0, y1, x0, x1 = [int(v) for v in spatial_roi["bbox"]]
    return x_frames[:, :, y0:y1, x0:x1]


def _spatial_roi_summary(spatial_roi: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spatial_roi.items() if not key.startswith("_")}


def _normalization_summary(normalization: dict[str, Any]) -> dict[str, Any]:
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    valid = np.asarray(normalization["valid_mask"], dtype=bool)
    finite_mean = mean[:, valid] if np.any(valid) else np.asarray([], dtype=np.float32)
    finite_std = std[:, valid] if np.any(valid) else np.asarray([], dtype=np.float32)
    return {
        "method": normalization.get("method"),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "valid_voxel_count": int(normalization.get("valid_voxel_count", 0)),
        "feature_count": int(normalization.get("feature_count", 0)),
        "zero_std_features": int(normalization.get("zero_std_features", 0)),
        "mean_of_train_means": float(finite_mean.mean()) if finite_mean.size else float("nan"),
        "mean_of_train_stds": float(finite_std.mean()) if finite_std.size else float("nan"),
    }


def _make_validation_split(
    train_idx: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    sklearn_model_selection = _require("sklearn.model_selection", "scikit-learn")
    if validation_fraction <= 0 or train_idx.size < 6:
        return train_idx, np.empty(0, dtype=int), "no_validation_small_train"
    y_train = np.asarray(y[train_idx], dtype=int)
    counts = np.bincount(y_train[y_train >= 0])
    positive = counts[counts > 0]
    n_classes = int(positive.size)
    n_val = int(round(train_idx.size * validation_fraction))
    n_val = max(n_classes, n_val)
    if positive.size < 2 or positive.min() < 2 or n_val >= train_idx.size:
        return train_idx, np.empty(0, dtype=int), "no_validation_insufficient_stratified_counts"
    splitter = sklearn_model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=n_val, random_state=seed
    )
    inner_train_local, val_local = next(splitter.split(np.zeros(train_idx.size), y_train))
    return train_idx[inner_train_local], train_idx[val_local], "stratified_train_validation"


def _build_model(
    model_name: str,
    task_type: str,
    input_frames: int,
    height: int,
    width: int,
    config: BenchmarkConfig,
) -> Any:
    torch = _require("torch")
    nn = torch.nn

    class SpatialBinMeanPool2d(nn.Module):
        """Deterministic coarse spatial pooling without CUDA adaptive-pool backward."""

        def __init__(self, output_size: int) -> None:
            super().__init__()
            self.output_size = max(1, int(output_size))

        def forward(self, x: Any) -> Any:
            _, _, height, width = x.shape
            bins = []
            for y_bin in range(self.output_size):
                y0 = (y_bin * height) // self.output_size
                y1 = ((y_bin + 1) * height + self.output_size - 1) // self.output_size
                row = []
                for x_bin in range(self.output_size):
                    x0 = (x_bin * width) // self.output_size
                    x1 = ((x_bin + 1) * width + self.output_size - 1) // self.output_size
                    row.append(x[:, :, y0:y1, x0:x1].mean(dim=(-2, -1)))
                bins.append(torch.stack(row, dim=-1))
            return torch.stack(bins, dim=-2)

    class ConvEncoder(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            pool = max(1, int(config.cnn_spatial_pool))
            self.output_dim = 32 * pool * pool
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.GroupNorm(4, 16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
                SpatialBinMeanPool2d(pool),
                nn.Flatten(),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if task_type == "8target":
                self.encoder_h = ConvEncoder(input_frames)
                self.encoder_v = ConvEncoder(input_frames)
                self.project_h = nn.Sequential(
                    nn.Linear(self.encoder_h.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(config.deep_dropout)),
                )
                self.project_v = nn.Sequential(
                    nn.Linear(self.encoder_v.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(config.deep_dropout)),
                )
                self.head_h = nn.Linear(int(config.deep_hidden_dim), 3)
                self.head_v = nn.Linear(int(config.deep_hidden_dim), 3)
            else:
                self.encoder = ConvEncoder(input_frames)
                self.project = nn.Sequential(
                    nn.Linear(self.encoder.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(config.deep_dropout)),
                )
                self.head = nn.Linear(int(config.deep_hidden_dim), 2)

        def forward(self, x: Any) -> Any:
            if task_type == "8target":
                z_h = self.project_h(self.encoder_h(x))
                z_v = self.project_v(self.encoder_v(x))
                return self.head_h(z_h), self.head_v(z_v)
            z = self.project(self.encoder(x))
            return self.head(z)

    class SmallCNNLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(float(config.deep_dropout))
            if task_type == "8target":
                self.frame_encoder_h = ConvEncoder(1)
                self.frame_encoder_v = ConvEncoder(1)
                self.frame_project_h = nn.Sequential(
                    nn.Linear(self.frame_encoder_h.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                )
                self.frame_project_v = nn.Sequential(
                    nn.Linear(self.frame_encoder_v.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                )
                self.lstm_h = nn.LSTM(
                    input_size=int(config.deep_hidden_dim),
                    hidden_size=int(config.deep_hidden_dim),
                    num_layers=1,
                    batch_first=True,
                )
                self.lstm_v = nn.LSTM(
                    input_size=int(config.deep_hidden_dim),
                    hidden_size=int(config.deep_hidden_dim),
                    num_layers=1,
                    batch_first=True,
                )
                self.head_h = nn.Linear(int(config.deep_hidden_dim), 3)
                self.head_v = nn.Linear(int(config.deep_hidden_dim), 3)
            else:
                self.frame_encoder = ConvEncoder(1)
                self.frame_project = nn.Sequential(
                    nn.Linear(self.frame_encoder.output_dim, int(config.deep_hidden_dim)),
                    nn.ReLU(inplace=True),
                )
                self.lstm = nn.LSTM(
                    input_size=int(config.deep_hidden_dim),
                    hidden_size=int(config.deep_hidden_dim),
                    num_layers=1,
                    batch_first=True,
                )
                self.head = nn.Linear(int(config.deep_hidden_dim), 2)

        def forward(self, x: Any) -> Any:
            b, t, c, h, w = x.shape
            if task_type == "8target":
                flat = x.reshape(b * t, c, h, w)
                z_h = self.frame_project_h(self.frame_encoder_h(flat)).reshape(b, t, -1)
                z_v = self.frame_project_v(self.frame_encoder_v(flat)).reshape(b, t, -1)
                out_h, _ = self.lstm_h(z_h)
                out_v, _ = self.lstm_v(z_v)
                last_h = self.dropout(out_h[:, -1, :])
                last_v = self.dropout(out_v[:, -1, :])
                return self.head_h(last_h), self.head_v(last_v)
            z = self.frame_encoder(x.reshape(b * t, c, h, w))
            z = self.frame_project(z).reshape(b, t, -1)
            out, _ = self.lstm(z)
            last = self.dropout(out[:, -1, :])
            return self.head(last)

    if model_name == "cnn":
        return SmallCNN()
    if model_name == "cnn_lstm":
        return SmallCNNLSTM()
    raise ValueError(model_name)


def _class_weights_from_labels(labels: np.ndarray, n_classes: int, device: str) -> Any:
    torch = _require("torch")
    labels = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(labels[(labels >= 0) & (labels < n_classes)], minlength=n_classes).astype(float)
    weights = np.ones(n_classes, dtype=np.float32)
    present = counts > 0
    if np.any(present):
        weights[present] = float(counts[present].sum()) / (float(np.sum(present)) * counts[present])
        weights = weights / float(weights[present].mean())
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _make_deep_class_weights(
    *,
    y: np.ndarray,
    labels_axis: np.ndarray | None,
    train_idx: np.ndarray,
    task_type: str,
    device: str,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {}
    if task_type == "8target":
        if labels_axis is None:
            raise ValueError("8-target class weights require axis labels.")
        return {
            "horizontal": _class_weights_from_labels(labels_axis[train_idx, 0] - 1, 3, device),
            "vertical": _class_weights_from_labels(labels_axis[train_idx, 1] - 1, 3, device),
        }
    return {"combined": _class_weights_from_labels(y[train_idx], 2, device)}


def _deep_loss(outputs: Any, batch: tuple[Any, ...], task_type: str, class_weights: dict[str, Any] | None = None) -> Any:
    torch = _require("torch")
    class_weights = class_weights or {}
    if task_type == "8target":
        logits_h, logits_v = outputs
        _, y_h, y_v = batch
        return torch.nn.functional.cross_entropy(
            logits_h,
            y_h,
            weight=class_weights.get("horizontal"),
        ) + torch.nn.functional.cross_entropy(
            logits_v,
            y_v,
            weight=class_weights.get("vertical"),
        )
    _, y = batch
    return torch.nn.functional.cross_entropy(outputs, y, weight=class_weights.get("combined"))


def _balanced_accuracy_for_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    recalls = []
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    for label in np.asarray(labels, dtype=int):
        mask = y_true == int(label)
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == int(label))))
    return float(np.mean(recalls)) if recalls else float("nan")


def _evaluate_deep(
    model: Any,
    loader: Any,
    task_type: str,
    device: str,
    class_weights: dict[str, Any] | None = None,
) -> dict[str, float]:
    torch = _require("torch")
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    true_all: list[int] = []
    pred_all: list[int] = []
    true_h_all: list[int] = []
    pred_h_all: list[int] = []
    true_v_all: list[int] = []
    pred_v_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            x = batch[0]
            outputs = model(x)
            loss = _deep_loss(outputs, batch, task_type, class_weights)
            n = int(x.shape[0])
            total_loss += float(loss.item()) * n
            total += n
            if task_type == "8target":
                logits_h, logits_v = outputs
                pred_h = torch.argmax(logits_h, dim=1) + 1
                pred_v = torch.argmax(logits_v, dim=1) + 1
                y_h = batch[1] + 1
                y_v = batch[2] + 1
                pred = pred_h + 3 * (pred_v - 1)
                true = y_h + 3 * (y_v - 1)
                true_h_all.extend(y_h.cpu().numpy().astype(int).tolist())
                pred_h_all.extend(pred_h.cpu().numpy().astype(int).tolist())
                true_v_all.extend(y_v.cpu().numpy().astype(int).tolist())
                pred_v_all.extend(pred_v.cpu().numpy().astype(int).tolist())
            else:
                pred = torch.argmax(outputs, dim=1)
                true = batch[1]
            correct += int((pred == true).sum().item())
            true_all.extend(true.cpu().numpy().astype(int).tolist())
            pred_all.extend(pred.cpu().numpy().astype(int).tolist())
    if task_type == "8target":
        h_bal = _balanced_accuracy_for_labels(np.asarray(true_h_all), np.asarray(pred_h_all), np.arange(1, 4))
        v_bal = _balanced_accuracy_for_labels(np.asarray(true_v_all), np.asarray(pred_v_all), np.arange(1, 4))
        axis_bal = float(np.nanmean([h_bal, v_bal]))
        labels = np.arange(1, 10)
    else:
        h_bal = float("nan")
        v_bal = float("nan")
        axis_bal = float("nan")
        labels = np.arange(0, 2)
    combined_bal = _balanced_accuracy_for_labels(np.asarray(true_all), np.asarray(pred_all), labels)
    return {
        "validation_loss": total_loss / total if total else float("nan"),
        "validation_accuracy": correct / total if total else float("nan"),
        "validation_balanced_accuracy": combined_bal,
        "validation_horizontal_balanced_accuracy": h_bal,
        "validation_vertical_balanced_accuracy": v_bal,
        "validation_axis_balanced_accuracy": axis_bal,
    }


def _predict_deep(model: Any, loader: Any, task_type: str, device: str) -> dict[str, Any]:
    torch = _require("torch")
    model.eval()
    pred_combined = []
    pred_h_all = []
    pred_v_all = []
    proba = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            outputs = model(batch[0])
            if task_type == "8target":
                logits_h, logits_v = outputs
                ph = torch.softmax(logits_h, dim=1)
                pv = torch.softmax(logits_v, dim=1)
                pred_h = torch.argmax(ph, dim=1) + 1
                pred_v = torch.argmax(pv, dim=1) + 1
                combined = pred_h + 3 * (pred_v - 1)
                joint_cols = []
                for combined_label in range(1, 10):
                    h_idx = (combined_label - 1) % 3
                    v_idx = (combined_label - 1) // 3
                    joint_cols.append(ph[:, h_idx] * pv[:, v_idx])
                joint = torch.stack(joint_cols, dim=1)
                pred_h_all.extend(pred_h.cpu().numpy().astype(int).tolist())
                pred_v_all.extend(pred_v.cpu().numpy().astype(int).tolist())
                pred_combined.extend(combined.cpu().numpy().astype(int).tolist())
                proba.append(joint.cpu().numpy())
            else:
                p = torch.softmax(outputs, dim=1)
                pred = torch.argmax(p, dim=1)
                pred_combined.extend(pred.cpu().numpy().astype(int).tolist())
                proba.append(p.cpu().numpy())
    return {
        "pred_combined": np.asarray(pred_combined, dtype=int),
        "pred_horizontal": np.asarray(pred_h_all, dtype=int) if task_type == "8target" else None,
        "pred_vertical": np.asarray(pred_v_all, dtype=int) if task_type == "8target" else None,
        "proba_combined": np.concatenate(proba, axis=0) if proba else None,
    }


def _deep_train_predict_fold(
    *,
    model_name: str,
    x_frames: np.ndarray,
    voxel_mask: np.ndarray,
    y: np.ndarray,
    labels_axis: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task_type: str,
    config: BenchmarkConfig,
    seed: int,
) -> dict[str, Any]:
    torch = _require("torch")
    data = _require("torch.utils.data")
    set_global_seed(seed, deterministic_torch=config.deterministic_torch)
    device = _select_device(config.device)
    inner_train_idx, val_idx, val_strategy = _make_validation_split(
        train_idx, y, config.validation_fraction, seed
    )
    train_voxel_mask, train_voxel_mask_info = _train_fold_power_mask(
        x_frames[inner_train_idx],
        np.asarray(voxel_mask, dtype=bool),
        percentile=config.deep_train_mask_percentile,
        min_fraction=config.deep_train_mask_min_fraction,
    )
    spatial_roi = _spatial_roi_from_voxel_mask(
        train_voxel_mask,
        mode=config.deep_foreground_mode,
        margin=config.foreground_margin,
    )
    spatial_roi["train_voxel_mask_info"] = train_voxel_mask_info
    normalization = _normalization_from_train(
        _apply_spatial_roi(x_frames[inner_train_idx], spatial_roi),
        model_name,
        valid_mask=np.asarray(spatial_roi["_mask"], dtype=bool),
        foreground_epsilon=config.foreground_epsilon,
        foreground_only=bool(config.normalize_foreground_only),
    )
    train_ds = _FrameDataset(
        x_frames,
        y,
        labels_axis,
        inner_train_idx,
        model_name,
        task_type,
        normalization,
        spatial_roi,
    )
    val_ds = (
        _FrameDataset(x_frames, y, labels_axis, val_idx, model_name, task_type, normalization, spatial_roi)
        if val_idx.size
        else None
    )
    test_ds = _FrameDataset(x_frames, y, labels_axis, test_idx, model_name, task_type, normalization, spatial_roi)
    batch_size = max(1, min(int(config.batch_size), len(train_ds)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = (
        data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)
        if val_ds is not None
        else None
    )
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)

    _, t_frames, h, w = _apply_spatial_roi(x_frames[:1], spatial_roi).shape
    model = _build_model(model_name, task_type, t_frames, h, w, config).to(device)
    class_weights = _make_deep_class_weights(
        y=y,
        labels_axis=labels_axis,
        train_idx=inner_train_idx,
        task_type=task_type,
        device=device,
        enabled=bool(config.use_class_weights),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_score: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[0])
            loss = _deep_loss(outputs, batch, task_type, class_weights)
            loss.backward()
            optimizer.step()
            n = int(batch[0].shape[0])
            total_loss += float(loss.item()) * n
            total += n
        train_loss = total_loss / total if total else float("nan")
        if val_loader is not None:
            val_metrics = _evaluate_deep(model, val_loader, task_type, device, class_weights)
            if task_type == "8target":
                monitor = val_metrics["validation_axis_balanced_accuracy"]
                monitor_name = "validation_axis_balanced_accuracy"
                monitor_mode = "max"
            else:
                monitor = val_metrics["validation_balanced_accuracy"]
                monitor_name = "validation_balanced_accuracy"
                monitor_mode = "max"
        else:
            val_metrics = {
                "validation_loss": float("nan"),
                "validation_accuracy": float("nan"),
                "validation_balanced_accuracy": float("nan"),
                "validation_horizontal_balanced_accuracy": float("nan"),
                "validation_vertical_balanced_accuracy": float("nan"),
                "validation_axis_balanced_accuracy": float("nan"),
            }
            monitor = train_loss
            monitor_name = "train_loss"
            monitor_mode = "min"
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "monitor_name": monitor_name,
                "monitor_value": float(monitor),
                **{key: float(value) for key, value in val_metrics.items()},
            }
        )
        if config.epoch_log_interval and (
            epoch == 1 or epoch % int(config.epoch_log_interval) == 0 or epoch == int(config.max_epochs)
        ):
            LOGGER.info(
                "%s epoch %d/%d train_loss=%.4f val_loss=%s val_acc=%s val_axis_bal_acc=%s monitor=%s:%s",
                model_name,
                epoch,
                int(config.max_epochs),
                train_loss,
                f"{val_metrics['validation_loss']:.4f}" if np.isfinite(val_metrics["validation_loss"]) else "nan",
                f"{val_metrics['validation_accuracy']:.4f}" if np.isfinite(val_metrics["validation_accuracy"]) else "nan",
                f"{val_metrics['validation_axis_balanced_accuracy']:.4f}"
                if np.isfinite(val_metrics["validation_axis_balanced_accuracy"])
                else "nan",
                monitor_name,
                f"{monitor:.4f}" if np.isfinite(monitor) else "nan",
            )
        if not np.isfinite(monitor):
            improved = False
        elif best_score is None:
            improved = True
        elif monitor_mode == "max":
            improved = monitor > best_score + 1e-8
        else:
            improved = monitor < best_score - 1e-8
        if improved:
            best_score = monitor
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if val_loader is not None and epochs_without_improvement >= int(config.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = _predict_deep(model, test_loader, task_type, device)
    pred["train_log"] = {
        "training_time": float(time.perf_counter() - t0),
        "best_epoch": int(best_epoch),
        "best_monitor_name": history[best_epoch - 1]["monitor_name"] if best_epoch > 0 else None,
        "best_monitor_value": float(best_score) if best_score is not None else None,
        "epochs_run": len(history),
        "validation_strategy": val_strategy,
        "validation_indices_local": val_idx.astype(int).tolist(),
        "inner_train_indices_local": inner_train_idx.astype(int).tolist(),
        "history": history,
    }
    pred["model_info"] = {
        "normalization": _normalization_summary(normalization),
        "spatial_roi": _spatial_roi_summary(spatial_roi),
        "device": device,
        "batch_size": int(batch_size),
        "validation_strategy": val_strategy,
        "cnn_spatial_pool": int(config.cnn_spatial_pool),
        "deep_hidden_dim": int(config.deep_hidden_dim),
        "deep_dropout": float(config.deep_dropout),
        "branching": "independent_horizontal_vertical" if task_type == "8target" else "single",
        "use_class_weights": bool(config.use_class_weights),
        "class_weights": {
            key: value.detach().cpu().numpy().astype(float).tolist()
            for key, value in class_weights.items()
        },
    }
    return pred


def _model_repeats(model_name: str, config: BenchmarkConfig) -> range:
    return range(int(config.repeats)) if model_name in DEEP_MODELS else range(1)


def run_window_benchmark(
    *,
    core: Any,
    window: PreparedWindow,
    task_type: str,
    label_info: dict[str, Any],
    chance_accuracy: float,
    config: BenchmarkConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every configured model on one prepared window."""

    y = window.labels_combined.astype(int)
    labels = np.arange(1, 10, dtype=int) if task_type == "8target" else np.arange(0, 2, dtype=int)
    splits, actual_n_splits = make_shared_splits(y, config.n_splits, config.random_seed)
    split_records = [
        {
            "fold": int(i),
            "train_indices_local": train.astype(int).tolist(),
            "test_indices_local": test.astype(int).tolist(),
            "train_trial_indices_global": window.trial_indices_global[train].astype(int).tolist(),
            "test_trial_indices_global": window.trial_indices_global[test].astype(int).tolist(),
        }
        for i, (train, test) in enumerate(splits)
    ]
    fold_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    true_h = window.labels_axis[:, 0] if window.labels_axis is not None else None
    true_v = window.labels_axis[:, 1] if window.labels_axis is not None else None

    for model_name in config.models:
        for repeat in _model_repeats(model_name, config):
            repeat_seed = int(config.random_seed + 1009 * repeat + 7919 * (ALL_MODELS.index(model_name) + 1))
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                fold_seed = int(repeat_seed + fold_id)
                status = "ok"
                error_message = ""
                pred_result: dict[str, Any] | None = None
                LOGGER.info(
                    "Running window=%s model=%s repeat=%d/%d fold=%d/%d n_train=%d n_test=%d seed=%d",
                    window.name,
                    model_name,
                    int(repeat) + 1,
                    int(config.repeats) if model_name in DEEP_MODELS else 1,
                    fold_id + 1,
                    len(splits),
                    int(train_idx.size),
                    int(test_idx.size),
                    fold_seed,
                )
                try:
                    if model_name in LINEAR_MODELS:
                        pred_result = _linear_predict_fold(
                            core=core,
                            model_name=model_name,
                            x_train=window.x_flat[train_idx],
                            x_test=window.x_flat[test_idx],
                            y_train=y[train_idx],
                            labels_axis_train=window.labels_axis[train_idx] if window.labels_axis is not None else None,
                            task_type=task_type,
                            config=config,
                            seed=fold_seed,
                        )
                    elif model_name in DEEP_MODELS:
                        pred_result = _deep_train_predict_fold(
                            model_name=model_name,
                            x_frames=window.x_frames,
                            voxel_mask=window.voxel_mask,
                            y=y,
                            labels_axis=window.labels_axis,
                            train_idx=train_idx,
                            test_idx=test_idx,
                            task_type=task_type,
                            config=config,
                            seed=fold_seed,
                        )
                    else:
                        raise ValueError(f"Unsupported model '{model_name}'")
                    metrics = compute_metrics(
                        y_true=y[test_idx],
                        y_pred=pred_result["pred_combined"],
                        labels=labels,
                        task_type=task_type,
                        combined_to_angle_deg=label_info["combined_to_angle_deg"],
                        pred_horizontal=pred_result.get("pred_horizontal"),
                        pred_vertical=pred_result.get("pred_vertical"),
                        true_horizontal=true_h[test_idx] if true_h is not None else None,
                        true_vertical=true_v[test_idx] if true_v is not None else None,
                        proba=pred_result.get("proba_combined"),
                        chance_accuracy=chance_accuracy,
                    )
                except Exception as exc:
                    status = "error"
                    error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    metrics = {
                        "accuracy": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "macro_f1": float("nan"),
                        "mean_angular_error_deg": float("nan"),
                        "median_angular_error_deg": float("nan"),
                        "center_center_prediction_rate": float("nan"),
                        "horizontal_accuracy": None,
                        "vertical_accuracy": None,
                        "top2_accuracy": None,
                        "valid_angular_error_count": 0,
                        "confusion_matrix": np.zeros((labels.size, labels.size), dtype=int),
                        "per_class": [],
                    }
                    pred_result = {
                        "pred_combined": np.full(test_idx.shape, -1, dtype=int),
                        "pred_horizontal": None,
                        "pred_vertical": None,
                        "proba_combined": None,
                        "train_log": {"training_time": 0.0, "best_epoch": None},
                        "model_info": {},
                    }
                    LOGGER.exception("Model %s repeat %s fold %s failed", model_name, repeat, fold_id)

                train_log = pred_result.get("train_log", {}) if pred_result else {}
                row = {
                    "session_id": "",
                    "task_type": task_type,
                    "model": model_name,
                    "repeat": int(repeat),
                    "seed": int(fold_seed),
                    "fold": int(fold_id),
                    "window": window.name,
                    "eval_index": int(window.eval_index),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "horizontal_accuracy": metrics["horizontal_accuracy"],
                    "vertical_accuracy": metrics["vertical_accuracy"],
                    "mean_angular_error_deg": metrics["mean_angular_error_deg"],
                    "median_angular_error_deg": metrics["median_angular_error_deg"],
                    "center_center_prediction_rate": metrics["center_center_prediction_rate"],
                    "top2_accuracy": metrics["top2_accuracy"],
                    "valid_angular_error_count": metrics["valid_angular_error_count"],
                    "best_epoch": train_log.get("best_epoch"),
                    "best_monitor_name": train_log.get("best_monitor_name"),
                    "best_monitor_value": train_log.get("best_monitor_value"),
                    "training_time": train_log.get("training_time", 0.0),
                    "status": status,
                    "error_message": error_message,
                    "class_distribution_train": _distribution_dict(y[train_idx]),
                    "class_distribution_test": _distribution_dict(y[test_idx]),
                    "predicted_distribution": _distribution_dict(pred_result["pred_combined"]),
                    "predicted_horizontal_distribution": _distribution_dict(pred_result["pred_horizontal"])
                    if pred_result.get("pred_horizontal") is not None
                    else {},
                    "predicted_vertical_distribution": _distribution_dict(pred_result["pred_vertical"])
                    if pred_result.get("pred_vertical") is not None
                    else {},
                    "train_indices_local": train_idx.astype(int).tolist(),
                    "test_indices_local": test_idx.astype(int).tolist(),
                    "train_trial_indices_global": window.trial_indices_global[train_idx].astype(int).tolist(),
                    "test_trial_indices_global": window.trial_indices_global[test_idx].astype(int).tolist(),
                }
                fold_rows.append(row)
                detailed.append(
                    {
                        **row,
                        "actual_combined": y[test_idx].astype(int),
                        "predicted_combined": pred_result["pred_combined"],
                        "actual_horizontal": true_h[test_idx].astype(int) if true_h is not None else None,
                        "actual_vertical": true_v[test_idx].astype(int) if true_v is not None else None,
                        "predicted_horizontal": pred_result.get("pred_horizontal"),
                        "predicted_vertical": pred_result.get("pred_vertical"),
                        "confusion_matrix_labels": labels,
                        "confusion_matrix": metrics["confusion_matrix"],
                        "per_class": metrics["per_class"],
                        "model_info": pred_result.get("model_info", {}),
                        "train_log": train_log,
                    }
                )

    for row in fold_rows:
        row["actual_n_splits"] = int(actual_n_splits)
    detailed.append({"split_records": split_records, "actual_n_splits": int(actual_n_splits)})
    return fold_rows, detailed


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([row.get(key, np.nan) for row in rows if row.get("status") == "ok"], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1 if values.size > 1 else 0))


def build_summary_rows(
    *,
    session_id: str,
    task_type: str,
    fold_rows: list[dict[str, Any]],
    chance_accuracy: float,
) -> list[dict[str, Any]]:
    summary = []
    models = sorted({row["model"] for row in fold_rows})
    for model in models:
        rows = [row for row in fold_rows if row["model"] == model]
        acc_mean, acc_std = _mean_std(rows, "accuracy")
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        n_test_total = sum(int(row.get("n_test", 0)) for row in ok_rows)
        accuracy_pooled = (
            float(sum(float(row.get("accuracy", 0.0)) * int(row.get("n_test", 0)) for row in ok_rows) / n_test_total)
            if n_test_total > 0
            else float("nan")
        )
        bal_mean, bal_std = _mean_std(rows, "balanced_accuracy")
        f1_mean, f1_std = _mean_std(rows, "macro_f1")
        ang_mean, _ = _mean_std(rows, "mean_angular_error_deg")
        h_mean, _ = _mean_std(rows, "horizontal_accuracy")
        v_mean, _ = _mean_std(rows, "vertical_accuracy")
        summary.append(
            {
                "session_id": session_id,
                "task_type": task_type,
                "model": model,
                "n_repeats": len({int(row["repeat"]) for row in rows}),
                "n_folds": len({int(row["fold"]) for row in rows}),
                "n_success": sum(row["status"] == "ok" for row in rows),
                "n_failed": sum(row["status"] != "ok" for row in rows),
                "accuracy_pooled": accuracy_pooled,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "balanced_accuracy_mean": bal_mean,
                "balanced_accuracy_std": bal_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "mean_angular_error_deg": ang_mean,
                "horizontal_accuracy_mean": h_mean,
                "vertical_accuracy_mean": v_mean,
                "chance_accuracy": float(chance_accuracy),
            }
        )
    return summary


def _jsonable(value: Any) -> Any:
    if _torch_available():
        torch = importlib.import_module("torch")
        if isinstance(value, torch.Tensor):
            return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _read_input_list(path: str | Path) -> list[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line)
    return rows


def _expand_mat_paths(
    *,
    mat_paths: list[str] | None,
    mat_globs: list[str] | None,
    input_list: str | None,
) -> list[Path]:
    inputs: list[str] = []
    inputs.extend(mat_paths or [])
    if input_list:
        inputs.extend(_read_input_list(input_list))
    for pattern in mat_globs or []:
        matches = sorted(glob.glob(pattern))
        if not matches:
            LOGGER.warning("No MAT files matched glob pattern: %s", pattern)
        inputs.extend(matches)

    expanded: list[Path] = []
    seen: set[str] = set()
    for item in inputs:
        path = Path(item).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        expanded.append(path)
    return expanded


def run_batch_benchmark(
    mat_paths: list[Path],
    config: BenchmarkConfig,
    *,
    core_script: str | Path | None,
    stop_on_error: bool = False,
) -> list[dict[str, Any]]:
    batch_rows: list[dict[str, Any]] = []
    base_output = Path(config.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    for idx, mat_path in enumerate(mat_paths, start=1):
        LOGGER.info("Batch %d/%d: %s", idx, len(mat_paths), mat_path)
        t0 = time.perf_counter()
        row: dict[str, Any] = {
            "batch_index": idx - 1,
            "mat_path": str(mat_path),
            "models": ",".join(config.models),
            "status": "ok",
            "error_message": "",
            "runtime_seconds": 0.0,
            "session_id": "",
            "task_type": "",
            "output_json": "",
            "output_summary_csv": "",
            "output_folds_csv": "",
        }
        try:
            result = run_benchmark(mat_path, config, core_script=core_script)
            session_id = str(result.get("session_id", ""))
            row["session_id"] = session_id
            row["task_type"] = str(result.get("task_type", ""))
            if session_id:
                session_dir = base_output / session_id
                row["output_json"] = str(session_dir / f"{session_id}_benchmark.json")
                row["output_summary_csv"] = str(session_dir / f"{session_id}_benchmark_summary.csv")
                row["output_folds_csv"] = str(session_dir / f"{session_id}_benchmark_folds.csv")
        except Exception as exc:
            row["status"] = "error"
            row["error_message"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            LOGGER.exception("Batch item failed: %s", mat_path)
            if stop_on_error:
                row["runtime_seconds"] = float(time.perf_counter() - t0)
                batch_rows.append(row)
                break
        row["runtime_seconds"] = float(time.perf_counter() - t0)
        batch_rows.append(row)

    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    batch_csv = base_output / f"batch_benchmark_{timestamp}.csv"
    batch_json = base_output / f"batch_benchmark_{timestamp}.json"
    fields = [
        "batch_index",
        "mat_path",
        "session_id",
        "task_type",
        "models",
        "status",
        "error_message",
        "runtime_seconds",
        "output_json",
        "output_summary_csv",
        "output_folds_csv",
    ]
    _write_csv(batch_csv, batch_rows, fields)
    with batch_json.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(
                {
                    "created_at_utc": started.isoformat(),
                    "config": asdict(config),
                    "n_inputs": len(mat_paths),
                    "n_success": sum(row["status"] == "ok" for row in batch_rows),
                    "n_failed": sum(row["status"] != "ok" for row in batch_rows),
                    "rows": batch_rows,
                }
            ),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    LOGGER.info("Saved batch report %s and %s", batch_json, batch_csv)
    return batch_rows


def _merge_existing_result_by_model(
    *,
    result: dict[str, Any],
    json_path: Path,
    current_models: tuple[str, ...],
    session_id: str,
    task_type: str,
    chance_accuracy: float,
) -> dict[str, Any]:
    """Preserve previous benchmark rows for models not run this time.

    This lets users run expensive models separately into the same output
    directory without losing earlier model results. Rows for models requested
    in the current invocation are replaced, so rerunning one model updates it
    cleanly instead of duplicating stale folds.
    """

    if not json_path.exists():
        result["merged_existing_result"] = False
        return result

    try:
        existing = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not merge existing benchmark JSON %s: %s", json_path, exc)
        result["merged_existing_result"] = False
        result["merge_warning"] = f"Could not read existing result: {exc}"
        return result

    current = set(current_models)
    old_folds = existing.get("folds", [])
    old_details = existing.get("details", [])
    preserved_folds = [
        row for row in old_folds
        if isinstance(row, dict) and row.get("model") not in current
    ]
    preserved_details = [
        row for row in old_details
        if isinstance(row, dict) and row.get("model") and row.get("model") not in current
    ]

    if preserved_folds:
        LOGGER.info(
            "Merging existing benchmark result %s; preserving models: %s",
            json_path,
            sorted({str(row.get("model")) for row in preserved_folds}),
        )

    result["folds"] = preserved_folds + result.get("folds", [])
    result["details"] = preserved_details + result.get("details", [])
    result["summary"] = build_summary_rows(
        session_id=session_id,
        task_type=task_type,
        fold_rows=result["folds"],
        chance_accuracy=chance_accuracy,
    )
    history = existing.get("run_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "merged_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_models_replaced": sorted(current),
            "preserved_models": sorted({str(row.get("model")) for row in preserved_folds}),
        }
    )
    result["run_history"] = history
    result["merged_existing_result"] = bool(preserved_folds or preserved_details)
    return result


def environment_info(device: str) -> dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": None,
        "torch": None,
        "cuda_available": False,
        "mps_available": False,
        "device": device,
        "device_name": device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        sklearn = _require("sklearn")
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    if _torch_available():
        torch = _require("torch")
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if device == "cuda" and torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
        elif device == "mps":
            info["device_name"] = "Apple MPS"
    return info


def run_benchmark(
    mat_path: str | Path,
    config: BenchmarkConfig | None = None,
    *,
    core_script: str | Path | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "all" in config.models:
        config.models = ALL_MODELS
    unknown = sorted(set(config.models) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}. Choose from {ALL_MODELS} or 'all'.")
    if config.direct_8class:
        raise NotImplementedError("Direct 8-class CNN mode is intentionally not the default benchmark path.")
    if config.device == "gpu":
        config.device = "cuda"
    if any(model in DEEP_MODELS for model in config.models) and not _torch_available():
        raise ImportError(
            "CNN/CNN+LSTM models require PyTorch in the same Python environment that runs "
            f"this script. Current executable: {sys.executable}. Install with "
            "'python -m pip install torch' using that executable, or omit cnn/cnn_lstm."
        )
    if any(model in DEEP_MODELS for model in config.models) and config.device == "cuda":
        torch = _require("torch")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You requested --device cuda, but PyTorch cannot initialize CUDA in this environment. "
                "This is usually a driver/PyTorch CUDA build mismatch. Run `nvidia-smi` and "
                "`python -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"`, "
                "then either update the NVIDIA driver, install a PyTorch build compatible with that "
                "driver, or rerun with --device cpu."
            )

    core = load_core_module(core_script)
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file does not exist: {mat_path}")
    base_output = Path(config.output_dir)
    provisional_session_id = session_id or getattr(core, "_infer_session_id")(str(mat_path))
    output_dir = base_output / provisional_session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_seed, deterministic_torch=config.deterministic_torch)

    aligned, preprocess_log, windows, label_info, core_config, chance = prepare_benchmark_windows(
        core=core,
        mat_path=mat_path,
        config=config,
        output_dir=output_dir,
        session_id=session_id,
    )
    task_type = str(aligned.metadata["task_type"])
    session_id = str(aligned.session_id)
    if output_dir.name != session_id:
        output_dir = base_output / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for window in windows:
        fold_rows, detailed = run_window_benchmark(
            core=core,
            window=window,
            task_type=task_type,
            label_info=label_info,
            chance_accuracy=chance,
            config=config,
        )
        for row in fold_rows:
            row["session_id"] = session_id
        for row in detailed:
            if isinstance(row, dict):
                row["session_id"] = session_id
                row["window"] = getattr(window, "name", row.get("window", ""))
                row["eval_index"] = getattr(window, "eval_index", row.get("eval_index", ""))
        all_fold_rows.extend(fold_rows)
        details.extend(detailed)

    summary_rows = build_summary_rows(
        session_id=session_id,
        task_type=task_type,
        fold_rows=all_fold_rows,
        chance_accuracy=chance,
    )
    selected_device = _select_device(config.device)
    result = {
        "summary": summary_rows,
        "folds": all_fold_rows,
        "details": details,
        "config": asdict(config),
        "core_config": core_config,
        "environment": environment_info(selected_device),
        "input": {
            "mat_path": str(mat_path),
            "mat_size_bytes": int(mat_path.stat().st_size),
            "core_script": str(core_script or _default_core_path()),
        },
        "alignment_metadata": aligned.metadata,
        "preprocess_log": preprocess_log,
        "direction_labels": {
            "combined_to_angle_deg": label_info["combined_to_angle_deg"],
            "combined_label_names": label_info["combined_label_names"],
            "label_to_target_pos": label_info.get("label_to_target_pos"),
            "center_tolerance": float(config.center_tolerance),
            "center_center_rule": (
                "Benchmark metrics record center-center prediction rate; angular error is NaN "
                "when a predicted label has no real target angle."
            ),
        },
        "windows": [
            {
                **w.window_info,
                "window": w.name,
                "eval_index": int(w.eval_index),
                "trial_indices_global": w.trial_indices_global.astype(int),
                "x_flat_shape": list(w.x_flat.shape),
                "x_cnn_shape": list(w.x_frames.shape),
                "voxel_mask_shape": list(w.voxel_mask.shape),
                "voxel_mask_count": int(np.asarray(w.voxel_mask, dtype=bool).sum()),
                "combined_label_distribution": _distribution_dict(w.labels_combined),
                "horizontal_label_distribution": _distribution_dict(w.labels_axis[:, 0])
                if w.labels_axis is not None
                else {},
                "vertical_label_distribution": _distribution_dict(w.labels_axis[:, 1])
                if w.labels_axis is not None
                else {},
            }
            for w in windows
        ],
    }

    json_path = output_dir / f"{session_id}_benchmark.json"
    summary_path = output_dir / f"{session_id}_benchmark_summary.csv"
    folds_path = output_dir / f"{session_id}_benchmark_folds.csv"
    if config.merge_existing:
        result = _merge_existing_result_by_model(
            result=result,
            json_path=json_path,
            current_models=tuple(config.models),
            session_id=session_id,
            task_type=task_type,
            chance_accuracy=chance,
        )
        all_fold_rows = result["folds"]
        summary_rows = result["summary"]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, indent=2, ensure_ascii=False, allow_nan=False)

    fold_fields = [
        "session_id",
        "task_type",
        "model",
        "repeat",
        "seed",
        "fold",
        "window",
        "eval_index",
        "n_train",
        "n_test",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "horizontal_accuracy",
        "vertical_accuracy",
        "mean_angular_error_deg",
        "median_angular_error_deg",
        "center_center_prediction_rate",
        "top2_accuracy",
        "valid_angular_error_count",
        "best_epoch",
        "best_monitor_name",
        "best_monitor_value",
        "training_time",
        "status",
        "error_message",
        "class_distribution_train",
        "class_distribution_test",
        "predicted_distribution",
        "predicted_horizontal_distribution",
        "predicted_vertical_distribution",
        "train_trial_indices_global",
        "test_trial_indices_global",
    ]
    summary_fields = [
        "session_id",
        "task_type",
        "model",
        "n_repeats",
        "n_folds",
        "n_success",
        "n_failed",
        "accuracy_pooled",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "mean_angular_error_deg",
        "horizontal_accuracy_mean",
        "vertical_accuracy_mean",
        "chance_accuracy",
    ]
    _write_csv(folds_path, all_fold_rows, fold_fields)
    _write_csv(summary_path, summary_rows, summary_fields)
    LOGGER.info("Saved %s, %s, and %s", json_path, summary_path, folds_path)
    return result


def _synthetic_multicoder_labels(n_repeats: int = 3) -> tuple[np.ndarray, dict[str, Any]]:
    positions = np.array(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
        ],
        dtype=float,
    )
    target_pos = np.tile(positions, (n_repeats, 1))
    core = load_core_module(None)
    return target_pos, core.make_multicoder_labels(target_pos)


def run_synthetic_tests() -> None:
    """Small offline tests for models, metrics, splits, and leakage guards."""

    core = load_core_module(None)
    config = BenchmarkConfig(
        models=ALL_MODELS,
        n_splits=2,
        repeats=1,
        random_seed=7,
        batch_size=4,
        max_epochs=1,
        patience=1,
        validation_fraction=0.25,
        device="cpu",
        detrend_window=0,
        spatial_filter_radius=0,
    )
    set_global_seed(config.random_seed)

    # Binary data.
    n = 24
    y = np.array([0, 1] * (n // 2), dtype=int)
    x_frames = np.random.normal(size=(n, 3, 8, 8)).astype(np.float32)
    x_frames[y == 1, :, 3:5, 3:5] += 1.5
    x_flat = np.concatenate([x_frames[:, i].reshape(n, -1, order="F") for i in range(3)], axis=1)
    splits, _ = make_shared_splits(y, 2, 7)
    train_idx, test_idx = splits[0]
    for model in ALL_MODELS:
        if model in LINEAR_MODELS:
            out = _linear_predict_fold(
                core=core,
                model_name=model,
                x_train=x_flat[train_idx],
                x_test=x_flat[test_idx],
                y_train=y[train_idx],
                labels_axis_train=None,
                task_type="2target",
                config=config,
                seed=11,
            )
        else:
            out = _deep_train_predict_fold(
                model_name=model,
                x_frames=x_frames,
                voxel_mask=np.ones(x_frames.shape[-2:], dtype=bool),
                y=y,
                labels_axis=None,
                train_idx=train_idx,
                test_idx=test_idx,
                task_type="2target",
                config=config,
                seed=11,
            )
        assert out["pred_combined"].shape == y[test_idx].shape
    metrics = compute_metrics(
        y_true=y[test_idx],
        y_pred=y[test_idx],
        labels=np.array([0, 1]),
        task_type="2target",
        combined_to_angle_deg={0: 180.0, 1: 0.0},
        chance_accuracy=0.5,
    )
    assert math.isclose(metrics["accuracy"], 1.0)

    # Eight-target multicoder data.
    target_pos, labels = _synthetic_multicoder_labels(n_repeats=3)
    y8 = labels["combined_labels"]
    axis8 = labels["axis_labels"]
    assert set(np.unique(axis8[:, 0])) == {1, 2, 3}
    assert set(np.unique(axis8[:, 1])) == {1, 2, 3}
    x8_frames = np.random.normal(size=(y8.size, 3, 8, 8)).astype(np.float32)
    for i, cls in enumerate(y8):
        x8_frames[i, :, int(cls) % 8, int(cls * 2) % 8] += 1.0
    x8_flat = np.concatenate([x8_frames[:, i].reshape(y8.size, -1, order="F") for i in range(3)], axis=1)
    splits8, _ = make_shared_splits(y8, 2, 7)
    train8, test8 = splits8[0]
    for model in ALL_MODELS:
        if model in LINEAR_MODELS:
            out = _linear_predict_fold(
                core=core,
                model_name=model,
                x_train=x8_flat[train8],
                x_test=x8_flat[test8],
                y_train=y8[train8],
                labels_axis_train=axis8[train8],
                task_type="8target",
                config=config,
                seed=13,
            )
        else:
            out = _deep_train_predict_fold(
                model_name=model,
                x_frames=x8_frames,
                voxel_mask=np.ones(x8_frames.shape[-2:], dtype=bool),
                y=y8,
                labels_axis=axis8,
                train_idx=train8,
                test_idx=test8,
                task_type="8target",
                config=config,
                seed=13,
            )
        assert out["pred_combined"].shape == y8[test8].shape
        if model in DEEP_MODELS:
            assert out["proba_combined"].shape[1] == 9
    center_metrics = compute_metrics(
        y_true=np.array([1, 9]),
        y_pred=np.array([5, 9]),
        labels=np.arange(1, 10),
        task_type="8target",
        combined_to_angle_deg=labels["combined_to_angle_deg"],
        pred_horizontal=np.array([2, 3]),
        pred_vertical=np.array([2, 3]),
        true_horizontal=np.array([1, 3]),
        true_vertical=np.array([1, 3]),
        chance_accuracy=1 / 8,
    )
    assert math.isclose(center_metrics["center_center_prediction_rate"], 0.5)
    assert center_metrics["valid_angular_error_count"] == 1
    circ = _angle_distance_deg(np.asarray([0.0]), np.asarray([315.0]))[0]
    assert math.isclose(float(circ), 45.0)

    # Leakage/split tests: outer splits are identical objects consumed by all models;
    # validation is strictly a subset of the training fold and excludes test.
    train_inner, val_idx, _ = _make_validation_split(train_idx, y, 0.25, 19)
    assert set(train_inner).issubset(set(train_idx))
    assert set(val_idx).issubset(set(train_idx))
    assert set(val_idx).isdisjoint(set(test_idx))
    splits_again, _ = make_shared_splits(y, 2, 7)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(splits, splits_again))

    # Deep model spatial preprocessing: ROI comes from the feature voxel mask,
    # and normalization is per frame and per voxel over train trials.
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:7, 4:8] = True
    masked_frames = np.zeros((6, 3, 10, 12), dtype=np.float32)
    masked_frames[:, :, mask] = np.arange(6 * 3 * int(mask.sum()), dtype=np.float32).reshape(6, 3, int(mask.sum()))
    roi = _spatial_roi_from_voxel_mask(mask, mode="crop", margin=1)
    cropped = _apply_spatial_roi(masked_frames, roi)
    norm = _normalization_from_train(cropped, "cnn", valid_mask=roi["_mask"])
    masked_ds = _FrameDataset(masked_frames, np.array([0, 1, 0, 1, 0, 1]), None, np.arange(6), "cnn", "2target", norm, roi)
    assert roi["bbox"] == [2, 8, 3, 9]
    assert tuple(norm["mean"].shape) == (3, 6, 6)
    assert norm["feature_count"] == int(mask.sum()) * 3
    assert float(masked_ds.x[:, :, 0, 0].abs().sum()) == 0.0

    finite_values = np.ones((10, 12, 3, 4), dtype=np.float32) * 0.01
    finite_values[3:7, 4:8, :, :] = 10.0
    power_mask, power_info = _power_doppler_voxel_mask(finite_values, percentile=90.0, min_fraction=0.01)
    assert power_info["method"] == "mean_power_doppler_percentile"
    assert int(power_mask.sum()) < finite_values.shape[0] * finite_values.shape[1]
    assert power_mask[4, 5]
    assert not power_mask[0, 0]

    h_bal = _balanced_accuracy_for_labels(np.array([1, 1, 2, 2, 3, 3]), np.array([1, 2, 2, 2, 1, 3]), np.arange(1, 4))
    v_bal = _balanced_accuracy_for_labels(np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 1, 2, 3, 1]), np.arange(1, 4))
    assert math.isclose(float(np.nanmean([h_bal, v_bal])), (h_bal + v_bal) / 2.0)

    expanded = _expand_mat_paths(
        mat_paths=["/tmp/session_a.mat", "/tmp/session_a.mat"],
        mat_globs=[],
        input_list=None,
    )
    assert [str(path) for path in expanded] == ["/tmp/session_a.mat"]

    print("Synthetic benchmark tests passed.")


def _parse_models(values: list[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    for value in values or []:
        expanded.extend(item.strip() for item in str(value).split(",") if item.strip())
    if not expanded or expanded == ["all"] or "all" in expanded:
        return ALL_MODELS
    return tuple(expanded)


def _normalize_cli_argv(argv: list[str] | None) -> list[str] | None:
    r"""Make pasted commands robust to chat apps mangling shell continuations.

    A common failure mode is copying a multi-line command through WeChat or a
    notebook and pasting it as ``\  --flag``. In POSIX shells, ``\ `` becomes a
    literal single-space argument, so argparse assigns that whitespace token to
    the positional ``mat_path`` and reports the real MAT path as unrecognized.
    Stripping and dropping pure-whitespace arguments keeps normal shell usage
    unchanged while making these pasted commands parse as intended.
    """

    if argv is None:
        argv = sys.argv[1:]
    out: list[str] = []
    for arg in argv:
        cleaned = str(arg).strip()
        if not cleaned or cleaned == "\\":
            continue
        if " " in cleaned and (" --" in cleaned or cleaned.startswith("--")):
            out.extend(part for part in cleaned.split() if part and part != "\\")
        else:
            out.append(cleaned)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fair within-session fUS decoding benchmark sharing trials, windows, labels, and CV splits."
    )
    parser.add_argument(
        "mat_paths",
        nargs="*",
        help="One or more MAT files, e.g. doppler_S*_R*+normcorre.mat or rt_fUS_data_S*_R*.mat.",
    )
    parser.add_argument(
        "--mat-glob",
        nargs="+",
        default=None,
        help="Glob pattern(s) for batch input, for example '../dataset/data2/doppler/*.mat'. Quote patterns in the shell.",
    )
    parser.add_argument(
        "--input-list",
        default=None,
        help="Text file containing one MAT path per line. Blank lines and lines starting with # are ignored.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="In batch mode, stop after the first failed session instead of continuing.",
    )
    parser.add_argument(
        "--core-script",
        default=str(_default_core_path()),
        help="Path to the existing within_session.py core script to reuse for loading/preprocessing/labels.",
    )
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), help="Models to run, or 'all'.")
    parser.add_argument(
        "--mode",
        choices=["fixed_memory_3frames", "dynamic_time_window"],
        default=BenchmarkConfig.mode,
        help="Feature window mode reused from within_session.py.",
    )
    parser.add_argument("--n-splits", type=int, default=BenchmarkConfig.n_splits, help="Requested outer StratifiedKFold splits.")
    parser.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats, help="Random initializations for CNN models.")
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.random_seed, help="Base random seed for splits and repeats.")
    parser.add_argument("--device", default=BenchmarkConfig.device, help="Deep-learning device: auto, cpu, cuda, or mps.")
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size, help="Mini-batch size for CNN models.")
    parser.add_argument("--learning-rate", type=float, default=BenchmarkConfig.learning_rate, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=BenchmarkConfig.weight_decay, help="AdamW weight decay.")
    parser.add_argument("--max-epochs", type=int, default=BenchmarkConfig.max_epochs, help="Maximum CNN training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=BenchmarkConfig.patience,
        help=(
            "Early stopping patience. 8-target deep models monitor validation_axis_balanced_accuracy; "
            "2-target deep models monitor validation_balanced_accuracy."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=BenchmarkConfig.validation_fraction, help="Fraction of outer-training trials held out for validation.")
    parser.add_argument("--epoch-log-interval", type=int, default=BenchmarkConfig.epoch_log_interval, help="Log deep-model training progress every N epochs; set 0 to disable.")
    parser.add_argument(
        "--cnn-spatial-pool",
        type=int,
        default=BenchmarkConfig.cnn_spatial_pool,
        help=(
            "CNN adaptive spatial pooling size. 4 preserves coarse fUS spatial layout; "
            "1 is global pooling and is more likely to collapse on small 8-target data."
        ),
    )
    parser.add_argument(
        "--deep-hidden-dim",
        type=int,
        default=BenchmarkConfig.deep_hidden_dim,
        help="Hidden feature dimension for CNN and CNN+LSTM heads.",
    )
    parser.add_argument(
        "--deep-dropout",
        type=float,
        default=BenchmarkConfig.deep_dropout,
        help="Dropout probability before the deep-model classifier heads.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable train-fold class weights for CNN/CNN+LSTM losses.",
    )
    parser.add_argument(
        "--voxel-mask-percentile",
        type=float,
        default=BenchmarkConfig.voxel_mask_percentile,
        help=(
            "Window-level mask threshold: keep pixels at or above this percentile of "
            "mean Power Doppler among positive candidate pixels."
        ),
    )
    parser.add_argument(
        "--voxel-mask-min-fraction",
        type=float,
        default=BenchmarkConfig.voxel_mask_min_fraction,
        help="Minimum fraction of candidate pixels retained in the window-level Power Doppler mask.",
    )
    parser.add_argument(
        "--deep-train-mask-percentile",
        type=float,
        default=BenchmarkConfig.deep_train_mask_percentile,
        help=(
            "Deep-model fold mask threshold computed from inner-train mean Power Doppler "
            "inside the window-level mask."
        ),
    )
    parser.add_argument(
        "--deep-train-mask-min-fraction",
        type=float,
        default=BenchmarkConfig.deep_train_mask_min_fraction,
        help="Minimum fraction of window-mask pixels retained in the deep train-fold mask.",
    )
    parser.add_argument(
        "--deep-foreground-mode",
        choices=["crop", "none"],
        default=BenchmarkConfig.deep_foreground_mode,
        help=(
            "Foreground handling for CNN/CNN+LSTM. 'crop' computes an ROI from the "
            "inner-train Power Doppler mask; 'none' keeps the full image."
        ),
    )
    parser.add_argument(
        "--foreground-epsilon",
        type=float,
        default=BenchmarkConfig.foreground_epsilon,
        help=(
            "Legacy compatibility option. Deep ROI cropping now uses the window voxel_mask "
            "from feature construction rather than image-value thresholds."
        ),
    )
    parser.add_argument(
        "--foreground-margin",
        type=int,
        default=BenchmarkConfig.foreground_margin,
        help="Pixel margin added around the train-fold foreground bounding box.",
    )
    parser.add_argument(
        "--normalize-all-pixels",
        action="store_true",
        help=(
            "Legacy compatibility option. CNN/CNN+LSTM normalization is now train-fold "
            "per-frame/per-voxel z-scoring over voxel_mask features."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=BenchmarkConfig.num_workers, help="PyTorch DataLoader worker count.")
    parser.add_argument("--frame-rate-hz", type=float, default=None, help="Optional frame-rate override passed to within_session.py.")
    parser.add_argument("--variance-to-keep", type=float, default=BenchmarkConfig.variance_to_keep, help="PCA explained variance fraction.")
    parser.add_argument("--cpca-m", type=int, default=BenchmarkConfig.cpca_m, help="Final cPCA/LDA subspace dimension.")
    parser.add_argument("--center-tolerance", type=float, default=BenchmarkConfig.center_tolerance, help="Tolerance for multicoder center axis labels.")
    parser.add_argument("--max-timepoints", type=int, default=None, help="Limit dynamic_time_window to the first N timepoints.")
    parser.add_argument("--min-trials-per-timepoint", type=int, default=BenchmarkConfig.min_trials_per_timepoint, help="Minimum valid trials required per window.")
    parser.add_argument("--output-dir", default=BenchmarkConfig.output_dir, help="Base directory for benchmark outputs.")
    parser.add_argument("--no-motion-correction", action="store_true", help="Disable motion-correction check in preprocessing.")
    parser.add_argument("--no-detrend", action="store_true", help="Disable causal detrending in preprocessing.")
    parser.add_argument("--no-spatial-filter", action="store_true", help="Disable pillbox spatial filtering in preprocessing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing benchmark JSON/CSV instead of merging by model.")
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "Disable torch deterministic algorithms. This can avoid CUDA/cuBLAS deterministic "
            "errors on older server stacks, at the cost of less reproducible CNN training."
        ),
    )
    parser.add_argument("--self-test", action="store_true", help="Run synthetic benchmark tests instead of a real MAT file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    argv = _normalize_cli_argv(argv)
    args = parser.parse_args(argv)
    if args.self_test:
        run_synthetic_tests()
        return 0
    mat_paths = _expand_mat_paths(
        mat_paths=args.mat_paths,
        mat_globs=args.mat_glob,
        input_list=args.input_list,
    )
    if not mat_paths:
        parser.error("At least one MAT path, --mat-glob pattern, or --input-list is required unless --self-test is used.")
    config = BenchmarkConfig(
        models=_parse_models(args.models),
        mode=args.mode,
        n_splits=args.n_splits,
        repeats=args.repeats,
        random_seed=args.seed,
        variance_to_keep=args.variance_to_keep,
        cpca_m=args.cpca_m,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        epoch_log_interval=args.epoch_log_interval,
        cnn_spatial_pool=args.cnn_spatial_pool,
        deep_hidden_dim=args.deep_hidden_dim,
        deep_dropout=args.deep_dropout,
        use_class_weights=not args.no_class_weights,
        voxel_mask_percentile=args.voxel_mask_percentile,
        voxel_mask_min_fraction=args.voxel_mask_min_fraction,
        deep_train_mask_percentile=args.deep_train_mask_percentile,
        deep_train_mask_min_fraction=args.deep_train_mask_min_fraction,
        deep_foreground_mode=args.deep_foreground_mode,
        foreground_epsilon=args.foreground_epsilon,
        foreground_margin=args.foreground_margin,
        normalize_foreground_only=not args.normalize_all_pixels,
        device=args.device,
        num_workers=args.num_workers,
        frame_rate_hz=args.frame_rate_hz,
        center_tolerance=args.center_tolerance,
        output_dir=args.output_dir,
        max_timepoints=args.max_timepoints,
        min_trials_per_timepoint=args.min_trials_per_timepoint,
        apply_motion_correction=not args.no_motion_correction,
        detrend_window=0 if args.no_detrend else BenchmarkConfig.detrend_window,
        spatial_filter_radius=0 if args.no_spatial_filter else BenchmarkConfig.spatial_filter_radius,
        merge_existing=not args.overwrite,
        deterministic_torch=not args.no_deterministic,
    )
    if len(mat_paths) == 1:
        run_benchmark(mat_paths[0], config, core_script=args.core_script)
    else:
        run_batch_benchmark(mat_paths, config, core_script=args.core_script, stop_on_error=args.stop_on_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
