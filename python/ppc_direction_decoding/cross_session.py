"""Cross-session fixed-memory direction decoding baselines.

This module evaluates 2-target linear decoders across sessions without
cross-validation leakage: the scaler, PCA/cPCA projection, and LDA are fit
only on the training session or pooled LOSO training sessions. Test sessions
are only transformed and predicted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import within_session as core


LOGGER = logging.getLogger(__name__)
LINEAR_MODELS = ("pca_lda", "cpca_lda")


@dataclass(frozen=True)
class CrossSessionGroup:
    group_name: str
    monkey: str
    task: str
    sessions: tuple[str, ...]


@dataclass
class CrossSessionConfig:
    """Configuration for cross-session fixed-memory 2-target decoding."""

    data_root: str = ""
    doppler_dir: str = ""
    project_record: str = ""
    output_dir: str = ""
    within_session_results_dir: str = "/data2/yuq1ngr/ibp3/output/benchmark/data1"
    models: tuple[str, ...] = LINEAR_MODELS
    mode: str = "fixed_memory_3frames"
    frame_rate_hz: float | None = None
    random_seed: int = 12345
    variance_to_keep: float = 0.95
    cpca_m: int = 1
    detrend_window: int = 50
    spatial_filter_radius: int = 2
    apply_motion_correction: bool = True
    min_train_class_count: int = 2
    min_test_class_count: int = 1
    min_trials_per_session: int = 2
    center_tolerance: float = 1e-6
    max_timepoints: int | None = None


@dataclass
class PreparedSession:
    session_id: str
    mat_path: Path
    group_name: str
    expected_monkey: str
    expected_task: str
    record: dict[str, Any]
    aligned: core.AlignedSession
    images: np.ndarray
    labels: np.ndarray
    label_info: dict[str, Any]
    valid_trial_mask: np.ndarray
    frame_indices: np.ndarray
    default_voxel_mask: np.ndarray
    preprocess_log: dict[str, Any]


@dataclass
class BinaryDecoder:
    model: str
    y_train: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    zero_std_features: int
    n_components: int
    projector: Any | None = None
    lda: Any | None = None
    cpca_subspaces: list[np.ndarray] = field(default_factory=list)
    cpca_ldas: list[Any] = field(default_factory=list)
    z_train: np.ndarray | None = None
    random_seed: int = 0


DEFAULT_GROUPS: tuple[CrossSessionGroup, ...] = (
    CrossSessionGroup(
        group_name="P_saccade",
        monkey="P",
        task="saccade",
        sessions=("S1_R1", "S2_R1", "S3_R1", "S4_R1", "S5_R1"),
    ),
    CrossSessionGroup(
        group_name="L_saccade",
        monkey="L",
        task="saccade",
        sessions=("S6_R1", "S8_R1", "S10_R1", "S11_R1", "S12_R1"),
    ),
    CrossSessionGroup(
        group_name="P_reach",
        monkey="P",
        task="reach",
        sessions=("S18_R1", "S20_R1", "S22_R1", "S24_R1"),
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    local = _repo_root() / "dataset" / "data1"
    return local if local.exists() else Path("/data2/yuq1ngr/dataset/data1")


def default_output_dir() -> Path:
    return _python_project_root() / "output" / "crosss_session" / "2target"


def _with_defaults(config: CrossSessionConfig) -> CrossSessionConfig:
    if not config.data_root:
        config.data_root = str(default_data_root())
    data_root = Path(config.data_root)
    if not config.doppler_dir:
        config.doppler_dir = str(data_root / "doppler")
    if not config.project_record:
        candidates = [
            data_root / "project_record.json",
            data_root / "ProjectRecord_paper.json",
            data_root / "ProjectRecord.json",
            data_root / "projectrecord.json",
            data_root / "projectrecord",
        ]
        config.project_record = str(next((p for p in candidates if p.exists()), candidates[0]))
    if not config.output_dir:
        config.output_dir = str(default_output_dir())
    if not Path(config.within_session_results_dir).exists():
        local = _python_project_root() / "output" / "benchmark" / "data1"
        if local.exists():
            config.within_session_results_dir = str(local)
    return config


def _session_token(record: dict[str, Any]) -> str:
    return f"S{int(record['Session'])}_R{int(record['Run'])}"


def _parse_session_token(session_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"S(\d+)_R(\d+)", str(session_id))
    if not match:
        raise ValueError(f"Expected session token like S1_R1, got {session_id!r}")
    return int(match.group(1)), int(match.group(2))


def _load_project_record(path: str | Path) -> dict[str, dict[str, Any]]:
    record_path = Path(path)
    if not record_path.exists():
        LOGGER.warning("Project record does not exist: %s", record_path)
        return {}
    with record_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Project record must contain a JSON list: {record_path}")
    return {_session_token(row): dict(row) for row in rows}


def _expected_mat_path(session_id: str, doppler_dir: str | Path) -> Path:
    doppler = Path(doppler_dir)
    candidates = [
        doppler / f"doppler_{session_id}+normcorre.mat",
        doppler / f"doppler_{session_id}.mat",
        doppler / f"rt_fUS_data_{session_id}.mat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_safe_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _safe_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    return value


def _csv_value(value: Any) -> Any:
    value = _safe_jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _distribution_dict(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    values = values[values >= 0]
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(int), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _min_class_count(values: np.ndarray) -> int:
    dist = _distribution_dict(values)
    return min(dist.values()) if dist else 0


def _row_percent(confusion: np.ndarray) -> np.ndarray:
    row_sum = confusion.sum(axis=1, keepdims=True)
    return np.divide(
        confusion * 100.0,
        row_sum,
        out=np.zeros(confusion.shape, dtype=float),
        where=row_sum > 0,
    )


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    lookup = {int(label): idx for idx, label in enumerate(labels)}
    for actual, pred in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        if int(actual) in lookup and int(pred) in lookup:
            matrix[lookup[int(actual)], lookup[int(pred)]] += 1
    return matrix


def _accuracy(confusion: np.ndarray) -> float:
    total = int(confusion.sum())
    return float(np.trace(confusion) / total) if total else float("nan")


def _balanced_accuracy(confusion: np.ndarray) -> float:
    row_sum = confusion.sum(axis=1)
    present = row_sum > 0
    if not np.any(present):
        return float("nan")
    recalls = np.divide(
        np.diag(confusion).astype(float),
        row_sum,
        out=np.zeros(row_sum.shape, dtype=float),
        where=row_sum > 0,
    )
    return float(recalls[present].mean())


def _fixed_memory_frame_indices(aligned: core.AlignedSession) -> np.ndarray:
    memory_end_index = core._memory_end_frame_index(aligned)
    start = memory_end_index - 3
    if start < 0:
        raise ValueError("Cannot build fixed_memory_3frames features: fewer than 3 frames before memory end.")
    return np.arange(start, memory_end_index, dtype=int)


def _finite_fixed_memory_voxel_mask(
    images: np.ndarray,
    frame_indices: np.ndarray,
    valid_trial_mask: np.ndarray,
) -> np.ndarray:
    valid_trials = np.asarray(valid_trial_mask, dtype=bool)
    if valid_trials.size != images.shape[3]:
        raise ValueError("valid_trial_mask length does not match image trials.")
    return np.isfinite(images[:, :, frame_indices, :][:, :, :, valid_trials]).any(axis=(2, 3))


def _build_features_from_prepared(
    prepared: PreparedSession,
    voxel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, trial_indices, info = core.build_fixed_memory_3frames_features(
        prepared.images,
        prepared.aligned,
        prepared.valid_trial_mask,
        voxel_mask=voxel_mask,
    )
    labels = prepared.labels[trial_indices].astype(int)
    ok = labels >= 0
    if not np.all(ok):
        x = x[ok]
        trial_indices = trial_indices[ok]
        labels = labels[ok]
        info = dict(info)
        info["n_trials"] = int(trial_indices.size)
    info["labels"] = labels
    return x, trial_indices, info


def load_and_build_fixed_features_for_session(
    mat_path: str | Path,
    config: CrossSessionConfig,
    *,
    session_id: str,
    group_name: str = "",
    expected_monkey: str = "",
    expected_task: str = "",
    record: dict[str, Any] | None = None,
    voxel_mask: np.ndarray | None = None,
) -> tuple[PreparedSession, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load, preprocess, label, and build fixed-memory features for one session."""

    core_config = core.WithinSessionConfig(
        mode=config.mode,
        frame_rate_hz=config.frame_rate_hz,
        random_seed=config.random_seed,
        variance_to_keep=config.variance_to_keep,
        cpca_m=config.cpca_m,
        detrend_window=config.detrend_window,
        spatial_filter_radius=config.spatial_filter_radius,
        apply_motion_correction=config.apply_motion_correction,
        min_trials_per_timepoint=config.min_trials_per_session,
        center_tolerance=config.center_tolerance,
        max_timepoints=config.max_timepoints,
        n_permutations=1,
    )
    mat_path = Path(mat_path)
    session = core.load_mat73_session(mat_path)
    aligned = core.align_fusi_and_behavior(
        session,
        session_id=session_id,
        decoder_type=None,
        frame_rate_hz=config.frame_rate_hz,
    )
    images, preprocess_log = core.preprocess_power_doppler_session(
        aligned.images,
        core_config,
        source_path=str(mat_path),
        output_dir=Path(config.output_dir) / "_preprocess_cache" / session_id,
    )
    if str(aligned.metadata["task_type"]) != "2target":
        raise ValueError("skip_not_2target")
    label_info = core.make_binary_labels(aligned.target_pos, center_tolerance=config.center_tolerance)
    labels = label_info["binary_labels"].astype(int)
    valid_mask = aligned.valid_trial_mask.copy() & (labels >= 0)
    frame_indices = _fixed_memory_frame_indices(aligned)
    default_mask = _finite_fixed_memory_voxel_mask(images, frame_indices, valid_mask)
    prepared = PreparedSession(
        session_id=session_id,
        mat_path=mat_path,
        group_name=group_name,
        expected_monkey=expected_monkey,
        expected_task=expected_task,
        record=dict(record or {}),
        aligned=aligned,
        images=images,
        labels=labels,
        label_info=label_info,
        valid_trial_mask=valid_mask,
        frame_indices=frame_indices,
        default_voxel_mask=default_mask,
        preprocess_log=preprocess_log,
    )
    x, _trial_indices, info = _build_features_from_prepared(prepared, voxel_mask=voxel_mask)
    return prepared, x, info["labels"], info


def _load_prepared_session(
    session_id: str,
    group: CrossSessionGroup,
    config: CrossSessionConfig,
    project_records: dict[str, dict[str, Any]],
) -> PreparedSession:
    mat_path = _expected_mat_path(session_id, config.doppler_dir)
    if not mat_path.exists():
        raise FileNotFoundError("skip_missing_session_file")
    prepared, _x, _y, _info = load_and_build_fixed_features_for_session(
        mat_path,
        config,
        session_id=session_id,
        group_name=group.group_name,
        expected_monkey=group.monkey,
        expected_task=group.task,
        record=project_records.get(session_id, {}),
    )
    return prepared


def _record_value(prepared: PreparedSession, key: str) -> str:
    value = prepared.record.get(key)
    if value is None:
        value = prepared.aligned.metadata.get("project_record", {}).get(key)
    return "" if value is None else str(value)


def _validate_session_metadata(prepared: PreparedSession) -> str | None:
    monkey = _record_value(prepared, "Monkey")
    task = _record_value(prepared, "Task")
    effector = _record_value(prepared, "Effector") or _record_value(prepared, "effector")
    n_targets = prepared.record.get("nTargets") or prepared.aligned.metadata.get("n_targets")
    if prepared.expected_monkey and monkey and monkey != prepared.expected_monkey:
        return "skip_monkey_mismatch"
    if prepared.expected_task and task and task != prepared.expected_task:
        return "skip_task_mismatch"
    if prepared.expected_task and effector and effector != prepared.expected_task:
        return "skip_effector_mismatch"
    if int(n_targets or -1) != 2 or str(prepared.aligned.metadata.get("task_type")) != "2target":
        return "skip_not_2target"
    return None


def _angle_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float((a - b + 180.0) % 360.0 - 180.0)) <= tol


def _label_mapping_matches(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ref_angles = {int(k): float(v) for k, v in reference["combined_to_angle_deg"].items()}
    cand_angles = {int(k): float(v) for k, v in candidate["combined_to_angle_deg"].items()}
    if set(ref_angles) != set(cand_angles):
        return False
    return all(_angle_close(ref_angles[label], cand_angles[label]) for label in ref_angles)


def _fit_zscore(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mu = np.asarray(x_train, dtype=np.float64).mean(axis=0)
    sigma = np.asarray(x_train, dtype=np.float64).std(axis=0, ddof=0)
    tiny = sigma < 1e-12
    zero_std_count = int(tiny.sum())
    sigma[tiny] = 1.0
    z_train = (x_train - mu) / sigma
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return z_train, mu, sigma, zero_std_count


def fit_binary_decoder_on_session_or_pooled_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model: str,
    config: CrossSessionConfig,
    random_seed: int,
) -> BinaryDecoder:
    """Fit train-only z-score, PCA/cPCA projection, and LDA."""

    sklearn_decomposition = core._require("sklearn.decomposition", "scikit-learn")
    sklearn_discriminant = core._require("sklearn.discriminant_analysis", "scikit-learn")

    y_train = np.asarray(y_train, dtype=int)
    z_train, mu, sigma, zero_std = _fit_zscore(np.asarray(x_train, dtype=np.float32))
    if model == "pca_lda":
        projector = sklearn_decomposition.PCA(n_components=config.variance_to_keep, svd_solver="full")
        train_scores = projector.fit_transform(z_train)
        lda = sklearn_discriminant.LinearDiscriminantAnalysis(solver="svd")
        lda.fit(train_scores, y_train)
        return BinaryDecoder(
            model=model,
            y_train=y_train,
            mu=mu,
            sigma=sigma,
            zero_std_features=zero_std,
            n_components=int(projector.n_components_),
            projector=projector,
            lda=lda,
            random_seed=random_seed,
        )

    if model == "cpca_lda":
        subspaces = core._fit_matlab_cpca_subspaces(z_train, y_train, m=config.cpca_m)
        ldas = []
        for subspace in subspaces:
            lda = sklearn_discriminant.LinearDiscriminantAnalysis(solver="svd")
            lda.fit(z_train @ subspace, y_train)
            ldas.append(lda)
        return BinaryDecoder(
            model=model,
            y_train=y_train,
            mu=mu,
            sigma=sigma,
            zero_std_features=zero_std,
            n_components=int(max(subspace.shape[1] for subspace in subspaces)),
            cpca_subspaces=subspaces,
            cpca_ldas=ldas,
            z_train=z_train,
            random_seed=random_seed,
        )

    raise ValueError(f"Unsupported linear model {model!r}")


def predict_binary_decoder_on_test_session(decoder: BinaryDecoder, x_test: np.ndarray) -> np.ndarray:
    """Transform test features using train-fit parameters, then predict."""

    z_test = (np.asarray(x_test, dtype=np.float32) - decoder.mu) / decoder.sigma
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if decoder.model == "pca_lda":
        assert decoder.projector is not None and decoder.lda is not None
        return decoder.lda.predict(decoder.projector.transform(z_test)).astype(int)

    if decoder.model == "cpca_lda":
        if decoder.z_train is None:
            raise RuntimeError("CPCA decoder is missing train scores.")
        rng = np.random.default_rng(decoder.random_seed)
        selected = core._choose_cpca_subspaces(
            z_test,
            decoder.z_train,
            decoder.y_train,
            decoder.cpca_subspaces,
            rng=rng,
        )
        pred = np.empty(z_test.shape[0], dtype=int)
        for subspace_idx in np.unique(selected):
            idx = int(subspace_idx)
            mask = selected == idx
            pred[mask] = decoder.cpca_ldas[idx].predict(z_test[mask] @ decoder.cpca_subspaces[idx]).astype(int)
        return pred

    raise ValueError(f"Unsupported decoder model {decoder.model!r}")


def _metrics_row(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_info: dict[str, Any],
) -> dict[str, Any]:
    labels = np.arange(0, 2, dtype=int)
    confusion = _confusion_matrix(y_true, y_pred, labels)
    row_percent = _row_percent(confusion)
    mean_ang, _per_trial = core.compute_angular_error(y_pred, y_true, label_info["combined_to_angle_deg"])
    return {
        "accuracy": _accuracy(confusion),
        "balanced_accuracy": _balanced_accuracy(confusion),
        "confusion_matrix_counts": confusion,
        "confusion_matrix_row_percent": row_percent,
        "mean_angular_error_deg": float(mean_ang),
    }


def _base_pair_row(
    *,
    group: CrossSessionGroup,
    model: str,
    train_session: str,
    test_session: str,
    evaluation_type: str,
    train_prepared: PreparedSession | None = None,
    test_prepared: PreparedSession | None = None,
) -> dict[str, Any]:
    return {
        "group_name": group.group_name,
        "model": model,
        "train_session": train_session,
        "test_session": test_session,
        "train_monkey": _record_value(train_prepared, "Monkey") if train_prepared else group.monkey,
        "test_monkey": _record_value(test_prepared, "Monkey") if test_prepared else group.monkey,
        "task": group.task,
        "effector": group.task,
        "evaluation_type": evaluation_type,
        "n_train_trials": "",
        "n_test_trials": "",
        "train_class_distribution": {},
        "test_class_distribution": {},
        "min_train_class_count": "",
        "min_test_class_count": "",
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "confusion_matrix_counts": [],
        "confusion_matrix_row_percent": [],
        "mean_angular_error_deg": float("nan"),
        "status": "skipped",
        "skip_reason": "",
    }


def _skip_pair_row(
    *,
    group: CrossSessionGroup,
    model: str,
    train_session: str,
    test_session: str,
    evaluation_type: str,
    skip_reason: str,
    train_prepared: PreparedSession | None = None,
    test_prepared: PreparedSession | None = None,
) -> dict[str, Any]:
    row = _base_pair_row(
        group=group,
        model=model,
        train_session=train_session,
        test_session=test_session,
        evaluation_type=evaluation_type,
        train_prepared=train_prepared,
        test_prepared=test_prepared,
    )
    row["skip_reason"] = skip_reason
    return row


def _within_result_candidates(session_id: str, within_root: str | Path) -> list[Path]:
    root = Path(within_root)
    if not root.exists():
        return []
    direct = [
        root / session_id / f"{session_id}_benchmark.json",
        root / session_id / f"{session_id}_within_session_decoding.json",
    ]
    found = [path for path in direct if path.exists()]
    found.extend(sorted(root.glob(f"*/{session_id}_benchmark.json")))
    found.extend(sorted(root.glob(f"*/{session_id}_within_session_decoding.json")))
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _aggregate_benchmark_confusion(result: dict[str, Any], model: str) -> tuple[np.ndarray, int]:
    confusion = np.zeros((2, 2), dtype=int)
    n_test = 0
    for detail in result.get("details", []):
        if not isinstance(detail, dict) or detail.get("model") != model or detail.get("status") != "ok":
            continue
        matrix = detail.get("confusion_matrix")
        if matrix is None:
            continue
        arr = np.asarray(matrix, dtype=int)
        if arr.shape == (2, 2):
            confusion += arr
            n_test += int(arr.sum())
    return confusion, n_test


def _load_within_session_diagonal_result(
    *,
    group: CrossSessionGroup,
    session_id: str,
    model: str,
    config: CrossSessionConfig,
    prepared: PreparedSession | None,
) -> dict[str, Any]:
    row = _base_pair_row(
        group=group,
        model=model,
        train_session=session_id,
        test_session=session_id,
        evaluation_type="within_session_10fold",
        train_prepared=prepared,
        test_prepared=prepared,
    )
    for candidate in _within_result_candidates(session_id, config.within_session_results_dir):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "folds" in payload and "summary" in payload:
            summary_rows = [
                item for item in payload.get("summary", [])
                if isinstance(item, dict) and item.get("model") == model
            ]
            if not summary_rows:
                continue
            summary = summary_rows[0]
            confusion, n_test = _aggregate_benchmark_confusion(payload, model)
            row.update(
                {
                    "n_train_trials": n_test if n_test else "",
                    "n_test_trials": n_test if n_test else "",
                    "accuracy": summary.get("accuracy_pooled", summary.get("accuracy_mean", float("nan"))),
                    "balanced_accuracy": summary.get("balanced_accuracy_mean", float("nan")),
                    "mean_angular_error_deg": summary.get("mean_angular_error_deg", float("nan")),
                    "confusion_matrix_counts": confusion,
                    "confusion_matrix_row_percent": _row_percent(confusion),
                    "status": "ok",
                    "skip_reason": "",
                    "within_session_result_path": str(candidate),
                }
            )
            if prepared is not None:
                _, _trial_indices, info = _build_features_from_prepared(prepared, prepared.default_voxel_mask)
                y = info["labels"]
                row["train_class_distribution"] = _distribution_dict(y)
                row["test_class_distribution"] = _distribution_dict(y)
                row["min_train_class_count"] = _min_class_count(y)
                row["min_test_class_count"] = _min_class_count(y)
            return row

        summary = payload.get("summary", {})
        if summary.get("decoder_type") and summary.get("decoder_type") != model:
            continue
        final = payload.get("timepoints", [{}])[-1] if payload.get("timepoints") else {}
        confusion = np.asarray(final.get("confusion_matrix_counts", [[0, 0], [0, 0]]), dtype=int)
        row.update(
            {
                "n_train_trials": final.get("n_trials", summary.get("n_trials_entering_CV", "")),
                "n_test_trials": final.get("n_trials", summary.get("n_trials_entering_CV", "")),
                "train_class_distribution": summary.get("combined_label_distribution", {}),
                "test_class_distribution": summary.get("combined_label_distribution", {}),
                "min_train_class_count": summary.get("min_class_count", ""),
                "min_test_class_count": summary.get("min_class_count", ""),
                "accuracy": summary.get("accuracy", final.get("accuracy", float("nan"))),
                "balanced_accuracy": summary.get("balanced_accuracy", final.get("balanced_accuracy", float("nan"))),
                "confusion_matrix_counts": confusion,
                "confusion_matrix_row_percent": final.get("confusion_matrix_row_percent", _row_percent(confusion)),
                "mean_angular_error_deg": summary.get(
                    "final_mean_angular_error_deg",
                    final.get("mean_angular_error_deg", float("nan")),
                ),
                "status": "ok",
                "skip_reason": "",
                "within_session_result_path": str(candidate),
            }
        )
        return row
    row["skip_reason"] = "skip_missing_session_file"
    row["within_session_results_dir"] = str(config.within_session_results_dir)
    return row


def _train_mask_from_sessions(sessions: list[PreparedSession]) -> np.ndarray:
    if not sessions:
        raise ValueError("No sessions supplied for train mask.")
    shape = sessions[0].images.shape[:2]
    mask = np.zeros(shape, dtype=bool)
    for prepared in sessions:
        if prepared.images.shape[:2] != shape:
            raise ValueError("skip_shape_mismatch")
        if prepared.frame_indices.size != 3:
            raise ValueError("skip_feature_build_failed")
        mask |= _finite_fixed_memory_voxel_mask(
            prepared.images,
            prepared.frame_indices,
            prepared.valid_trial_mask,
        )
    if not np.any(mask):
        raise ValueError("skip_feature_build_failed")
    return mask


def run_cross_session_pair(
    train_session: str,
    test_session: str,
    config: CrossSessionConfig,
    *,
    group: CrossSessionGroup,
    model: str,
    prepared_sessions: dict[str, PreparedSession],
) -> dict[str, Any]:
    """Train on one session and evaluate on another session."""

    if train_session == test_session:
        return _load_within_session_diagonal_result(
            group=group,
            session_id=test_session,
            model=model,
            config=config,
            prepared=prepared_sessions.get(test_session),
        )
    train = prepared_sessions.get(train_session)
    test = prepared_sessions.get(test_session)
    if train is None or test is None:
        return _skip_pair_row(
            group=group,
            model=model,
            train_session=train_session,
            test_session=test_session,
            evaluation_type="cross_session_train_test",
            skip_reason="skip_missing_session_file",
            train_prepared=train,
            test_prepared=test,
        )
    row = _base_pair_row(
        group=group,
        model=model,
        train_session=train_session,
        test_session=test_session,
        evaluation_type="cross_session_train_test",
        train_prepared=train,
        test_prepared=test,
    )
    for prepared in (train, test):
        reason = _validate_session_metadata(prepared)
        if reason:
            row["skip_reason"] = reason
            return row
    if not _label_mapping_matches(train.label_info, test.label_info):
        row["skip_reason"] = "skip_label_mapping_mismatch"
        return row
    if train.images.shape[:2] != test.images.shape[:2]:
        row["skip_reason"] = "skip_shape_mismatch"
        return row
    try:
        train_mask = _train_mask_from_sessions([train])
        x_train, _train_trials, train_info = _build_features_from_prepared(train, train_mask)
        x_test, _test_trials, test_info = _build_features_from_prepared(test, train_mask)
    except ValueError as exc:
        reason = str(exc) if str(exc).startswith("skip_") else "skip_feature_build_failed"
        row["skip_reason"] = reason
        return row
    except Exception:
        row["skip_reason"] = "skip_feature_build_failed"
        row["error_message"] = traceback.format_exc(limit=2)
        return row

    y_train = train_info["labels"]
    y_test = test_info["labels"]
    row.update(
        {
            "n_train_trials": int(y_train.size),
            "n_test_trials": int(y_test.size),
            "train_class_distribution": _distribution_dict(y_train),
            "test_class_distribution": _distribution_dict(y_test),
            "min_train_class_count": _min_class_count(y_train),
            "min_test_class_count": _min_class_count(y_test),
        }
    )
    if row["min_train_class_count"] < config.min_train_class_count:
        row["skip_reason"] = "skip_insufficient_train_class_count"
        return row
    if row["min_test_class_count"] < config.min_test_class_count:
        row["skip_reason"] = "skip_insufficient_test_class_count"
        return row
    if x_train.shape[1] != x_test.shape[1]:
        row["skip_reason"] = "skip_shape_mismatch"
        return row
    try:
        seed = int(config.random_seed + 1009 * LINEAR_MODELS.index(model) + 17 * _parse_session_token(test_session)[0])
        decoder = fit_binary_decoder_on_session_or_pooled_train(
            x_train,
            y_train,
            model=model,
            config=config,
            random_seed=seed,
        )
        y_pred = predict_binary_decoder_on_test_session(decoder, x_test)
        row.update(_metrics_row(y_true=y_test, y_pred=y_pred, label_info=train.label_info))
        row.update(
            {
                "status": "ok",
                "skip_reason": "",
                "n_components": int(decoder.n_components),
                "zero_std_features": int(decoder.zero_std_features),
                "feature_dim": int(x_train.shape[1]),
            }
        )
    except Exception:
        row["skip_reason"] = "skip_decode_failed"
        row["error_message"] = traceback.format_exc(limit=3)
    return row


def run_pairwise_cross_session_group(
    group: CrossSessionGroup,
    config: CrossSessionConfig,
    *,
    prepared_sessions: dict[str, PreparedSession],
) -> list[dict[str, Any]]:
    rows = []
    for model in config.models:
        for train_session in group.sessions:
            for test_session in group.sessions:
                rows.append(
                    run_cross_session_pair(
                        train_session,
                        test_session,
                        config,
                        group=group,
                        model=model,
                        prepared_sessions=prepared_sessions,
                    )
                )
    return rows


def _base_loso_row(
    *,
    group: CrossSessionGroup,
    model: str,
    train_sessions: list[str],
    test_session: str,
) -> dict[str, Any]:
    return {
        "group_name": group.group_name,
        "model": model,
        "test_session": test_session,
        "train_sessions": train_sessions,
        "n_train_sessions": int(len(train_sessions)),
        "task": group.task,
        "effector": group.task,
        "n_train_trials": "",
        "n_test_trials": "",
        "train_class_distribution": {},
        "test_class_distribution": {},
        "min_train_class_count": "",
        "min_test_class_count": "",
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "confusion_matrix_counts": [],
        "confusion_matrix_row_percent": [],
        "mean_angular_error_deg": float("nan"),
        "status": "skipped",
        "skip_reason": "",
    }


def run_loso_cross_session_group(
    group: CrossSessionGroup,
    config: CrossSessionConfig,
    *,
    prepared_sessions: dict[str, PreparedSession],
) -> list[dict[str, Any]]:
    """Run leave-one-session-out pooled-train cross-session decoding."""

    rows = []
    for model in config.models:
        for test_session in group.sessions:
            train_ids = [sid for sid in group.sessions if sid != test_session]
            row = _base_loso_row(group=group, model=model, train_sessions=train_ids, test_session=test_session)
            train_prepared = [prepared_sessions[sid] for sid in train_ids if sid in prepared_sessions]
            test = prepared_sessions.get(test_session)
            if test is None or len(train_prepared) != len(train_ids):
                row["skip_reason"] = "skip_missing_session_file"
                rows.append(row)
                continue
            all_prepared = train_prepared + [test]
            reason = next((r for r in (_validate_session_metadata(p) for p in all_prepared) if r), None)
            if reason:
                row["skip_reason"] = reason
                rows.append(row)
                continue
            reference = train_prepared[0].label_info
            if any(not _label_mapping_matches(reference, p.label_info) for p in all_prepared[1:]):
                row["skip_reason"] = "skip_label_mapping_mismatch"
                rows.append(row)
                continue
            if any(p.images.shape[:2] != train_prepared[0].images.shape[:2] for p in all_prepared):
                row["skip_reason"] = "skip_shape_mismatch"
                rows.append(row)
                continue
            try:
                train_mask = _train_mask_from_sessions(train_prepared)
                train_chunks = []
                train_labels = []
                for prepared in train_prepared:
                    x_part, _trial_indices, info = _build_features_from_prepared(prepared, train_mask)
                    train_chunks.append(x_part)
                    train_labels.append(info["labels"])
                x_train = np.vstack(train_chunks)
                y_train = np.concatenate(train_labels).astype(int)
                x_test, _test_trials, test_info = _build_features_from_prepared(test, train_mask)
                y_test = test_info["labels"]
            except ValueError as exc:
                row["skip_reason"] = str(exc) if str(exc).startswith("skip_") else "skip_feature_build_failed"
                rows.append(row)
                continue
            except Exception:
                row["skip_reason"] = "skip_feature_build_failed"
                row["error_message"] = traceback.format_exc(limit=2)
                rows.append(row)
                continue

            row.update(
                {
                    "n_train_trials": int(y_train.size),
                    "n_test_trials": int(y_test.size),
                    "train_class_distribution": _distribution_dict(y_train),
                    "test_class_distribution": _distribution_dict(y_test),
                    "min_train_class_count": _min_class_count(y_train),
                    "min_test_class_count": _min_class_count(y_test),
                }
            )
            if row["min_train_class_count"] < config.min_train_class_count:
                row["skip_reason"] = "skip_insufficient_train_class_count"
                rows.append(row)
                continue
            if row["min_test_class_count"] < config.min_test_class_count:
                row["skip_reason"] = "skip_insufficient_test_class_count"
                rows.append(row)
                continue
            if any(x.shape[1] != x_train.shape[1] for x in [x_test]):
                row["skip_reason"] = "skip_shape_mismatch"
                rows.append(row)
                continue
            try:
                seed = int(config.random_seed + 7919 * LINEAR_MODELS.index(model) + _parse_session_token(test_session)[0])
                decoder = fit_binary_decoder_on_session_or_pooled_train(
                    x_train,
                    y_train,
                    model=model,
                    config=config,
                    random_seed=seed,
                )
                y_pred = predict_binary_decoder_on_test_session(decoder, x_test)
                row.update(_metrics_row(y_true=y_test, y_pred=y_pred, label_info=reference))
                row.update(
                    {
                        "status": "ok",
                        "skip_reason": "",
                        "n_components": int(decoder.n_components),
                        "zero_std_features": int(decoder.zero_std_features),
                        "feature_dim": int(x_train.shape[1]),
                    }
                )
            except Exception:
                row["skip_reason"] = "skip_decode_failed"
                row["error_message"] = traceback.format_exc(limit=3)
            rows.append(row)
    return rows


def _finite_metric_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            value = float(row.get(metric, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _pairwise_macro_by_test(
    rows: list[dict[str, Any]],
    *,
    group: CrossSessionGroup,
    model: str,
    metric: str,
) -> float:
    per_test = []
    for test_session in group.sessions:
        test_rows = [
            row for row in rows
            if row.get("group_name") == group.group_name
            and row.get("model") == model
            and row.get("evaluation_type") == "cross_session_train_test"
            and row.get("test_session") == test_session
            and row.get("train_session") != test_session
        ]
        values = _finite_metric_values(test_rows, metric)
        if values.size:
            per_test.append(float(values.mean()))
    return _mean_or_nan(np.asarray(per_test, dtype=float))


def summarize_cross_session_results(
    pairwise_rows: list[dict[str, Any]],
    loso_rows: list[dict[str, Any]],
    groups: tuple[CrossSessionGroup, ...],
    config: CrossSessionConfig,
) -> list[dict[str, Any]]:
    summary_rows = []
    for group in groups:
        for model in config.models:
            base = {
                "group_name": group.group_name,
                "model": model,
                "n_sessions": int(len(group.sessions)),
                "n_cross_session_pairs": int(
                    sum(
                        row.get("group_name") == group.group_name
                        and row.get("model") == model
                        and row.get("evaluation_type") == "cross_session_train_test"
                        and row.get("status") == "ok"
                        for row in pairwise_rows
                    )
                ),
            }
            diag_rows = [
                row for row in pairwise_rows
                if row.get("group_name") == group.group_name
                and row.get("model") == model
                and row.get("evaluation_type") == "within_session_10fold"
            ]
            group_loso = [
                row for row in loso_rows
                if row.get("group_name") == group.group_name and row.get("model") == model
            ]
            for metric in ("accuracy", "balanced_accuracy", "mean_angular_error_deg"):
                base[f"mean_within_session_{metric}"] = _mean_or_nan(_finite_metric_values(diag_rows, metric))
                base[f"mean_pairwise_cross_session_{metric}_macro_by_test"] = _pairwise_macro_by_test(
                    pairwise_rows,
                    group=group,
                    model=model,
                    metric=metric,
                )
                base[f"mean_loso_{metric}"] = _mean_or_nan(_finite_metric_values(group_loso, metric))
            summary_rows.append(base)
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)


def _write_metric_matrices(
    output_dir: Path,
    pairwise_rows: list[dict[str, Any]],
    groups: tuple[CrossSessionGroup, ...],
    config: CrossSessionConfig,
) -> None:
    for group in groups:
        for metric, filename in (
            ("accuracy", f"accuracy_matrix_{group.group_name}.csv"),
            ("balanced_accuracy", f"balanced_accuracy_matrix_{group.group_name}.csv"),
            ("mean_angular_error_deg", f"angular_error_matrix_{group.group_name}.csv"),
        ):
            rows = []
            for model in config.models:
                for train_session in group.sessions:
                    row = {"model": model, "train_session": train_session}
                    for test_session in group.sessions:
                        match = next(
                            (
                                item for item in pairwise_rows
                                if item.get("group_name") == group.group_name
                                and item.get("model") == model
                                and item.get("train_session") == train_session
                                and item.get("test_session") == test_session
                            ),
                            None,
                        )
                        row[test_session] = match.get(metric, "") if match and match.get("status") == "ok" else ""
                    rows.append(row)
            _write_csv(output_dir / filename, rows, ["model", "train_session", *group.sessions])


def _exclusion_rows(pairwise_rows: list[dict[str, Any]], loso_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in pairwise_rows:
        if row.get("status") == "ok":
            continue
        rows.append(
            {
                "experiment": "pairwise",
                "group_name": row.get("group_name", ""),
                "model": row.get("model", ""),
                "train_session": row.get("train_session", ""),
                "test_session": row.get("test_session", ""),
                "train_sessions": "",
                "evaluation_type": row.get("evaluation_type", ""),
                "status": row.get("status", ""),
                "skip_reason": row.get("skip_reason", ""),
                "error_message": row.get("error_message", ""),
            }
        )
    for row in loso_rows:
        if row.get("status") == "ok":
            continue
        rows.append(
            {
                "experiment": "loso",
                "group_name": row.get("group_name", ""),
                "model": row.get("model", ""),
                "train_session": "",
                "test_session": row.get("test_session", ""),
                "train_sessions": row.get("train_sessions", ""),
                "evaluation_type": "loso_cross_session",
                "status": row.get("status", ""),
                "skip_reason": row.get("skip_reason", ""),
                "error_message": row.get("error_message", ""),
            }
        )
    return rows


def _load_all_group_sessions(
    groups: tuple[CrossSessionGroup, ...],
    config: CrossSessionConfig,
    project_records: dict[str, dict[str, Any]],
) -> dict[str, PreparedSession]:
    prepared: dict[str, PreparedSession] = {}
    for group in groups:
        for session_id in group.sessions:
            if session_id in prepared:
                continue
            try:
                LOGGER.info("Loading %s (%s)", session_id, group.group_name)
                prepared[session_id] = _load_prepared_session(session_id, group, config, project_records)
            except FileNotFoundError:
                LOGGER.warning("Missing session file for %s", session_id)
            except ValueError as exc:
                if str(exc).startswith("skip_"):
                    LOGGER.warning("Skipping session preload %s: %s", session_id, exc)
                else:
                    LOGGER.exception("Could not prepare session %s", session_id)
            except Exception:
                LOGGER.exception("Could not prepare session %s", session_id)
    return prepared


def run_cross_session_decoding(
    config: CrossSessionConfig | None = None,
    *,
    groups: tuple[CrossSessionGroup, ...] = DEFAULT_GROUPS,
) -> dict[str, Any]:
    """Run pairwise and LOSO cross-session experiments and save outputs."""

    config = _with_defaults(config or CrossSessionConfig())
    if config.mode != "fixed_memory_3frames":
        raise ValueError("Cross-session baseline currently supports only fixed_memory_3frames.")
    unknown = sorted(set(config.models) - set(LINEAR_MODELS))
    if unknown:
        raise ValueError(f"Cross-session currently supports only {LINEAR_MODELS}; unknown={unknown}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_records = _load_project_record(config.project_record)
    prepared = _load_all_group_sessions(groups, config, project_records)

    pairwise_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for group in groups:
        LOGGER.info("Running pairwise group %s", group.group_name)
        pairwise_rows.extend(
            run_pairwise_cross_session_group(group, config, prepared_sessions=prepared)
        )
        LOGGER.info("Running LOSO group %s", group.group_name)
        loso_rows.extend(run_loso_cross_session_group(group, config, prepared_sessions=prepared))

    summary_rows = summarize_cross_session_results(pairwise_rows, loso_rows, groups, config)
    exclusion_rows = _exclusion_rows(pairwise_rows, loso_rows)

    pairwise_fields = [
        "group_name",
        "model",
        "train_session",
        "test_session",
        "train_monkey",
        "test_monkey",
        "task",
        "effector",
        "evaluation_type",
        "n_train_trials",
        "n_test_trials",
        "train_class_distribution",
        "test_class_distribution",
        "min_train_class_count",
        "min_test_class_count",
        "accuracy",
        "balanced_accuracy",
        "confusion_matrix_counts",
        "confusion_matrix_row_percent",
        "mean_angular_error_deg",
        "status",
        "skip_reason",
        "feature_dim",
        "n_components",
        "zero_std_features",
        "within_session_result_path",
        "error_message",
    ]
    loso_fields = [
        "group_name",
        "model",
        "test_session",
        "train_sessions",
        "n_train_sessions",
        "task",
        "effector",
        "n_train_trials",
        "n_test_trials",
        "train_class_distribution",
        "test_class_distribution",
        "min_train_class_count",
        "min_test_class_count",
        "accuracy",
        "balanced_accuracy",
        "confusion_matrix_counts",
        "confusion_matrix_row_percent",
        "mean_angular_error_deg",
        "status",
        "skip_reason",
        "feature_dim",
        "n_components",
        "zero_std_features",
        "error_message",
    ]
    summary_fields = list(summary_rows[0].keys()) if summary_rows else ["group_name", "model"]
    exclusion_fields = [
        "experiment",
        "group_name",
        "model",
        "train_session",
        "test_session",
        "train_sessions",
        "evaluation_type",
        "status",
        "skip_reason",
        "error_message",
    ]

    _write_csv(output_dir / "cross_session_pairwise_results.csv", pairwise_rows, pairwise_fields)
    _write_json(
        output_dir / "cross_session_pairwise_results.json",
        {"config": asdict(config), "groups": [asdict(g) for g in groups], "rows": pairwise_rows},
    )
    _write_csv(output_dir / "cross_session_loso_results.csv", loso_rows, loso_fields)
    _write_json(
        output_dir / "cross_session_loso_results.json",
        {"config": asdict(config), "groups": [asdict(g) for g in groups], "rows": loso_rows},
    )
    _write_metric_matrices(output_dir, pairwise_rows, groups, config)
    _write_csv(output_dir / "group_summary.csv", summary_rows, summary_fields)
    _write_json(
        output_dir / "group_summary.json",
        {"config": asdict(config), "groups": [asdict(g) for g in groups], "rows": summary_rows},
    )
    _write_csv(output_dir / "exclusion_log.csv", exclusion_rows, exclusion_fields)

    return {
        "config": asdict(config),
        "groups": [asdict(g) for g in groups],
        "pairwise": pairwise_rows,
        "loso": loso_rows,
        "summary": summary_rows,
        "exclusions": exclusion_rows,
        "output_dir": str(output_dir),
    }


def _parse_models(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return LINEAR_MODELS
    out = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                out.append(item)
    if "all" in out:
        return LINEAR_MODELS
    return tuple(out)


def _parse_groups(values: list[str] | None) -> tuple[CrossSessionGroup, ...]:
    if not values:
        return DEFAULT_GROUPS
    requested = {item.strip() for value in values for item in value.split(",") if item.strip()}
    selected = tuple(group for group in DEFAULT_GROUPS if group.group_name in requested)
    missing = sorted(requested - {group.group_name for group in selected})
    if missing:
        raise ValueError(f"Unknown group(s): {missing}. Choose from {[g.group_name for g in DEFAULT_GROUPS]}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="", help="Directory containing project record and doppler/.")
    parser.add_argument("--doppler-dir", default="", help="Directory containing session MAT files.")
    parser.add_argument("--project-record", default="", help="ProjectRecord JSON path.")
    parser.add_argument("--output-dir", default="", help="Output directory for cross-session results.")
    parser.add_argument(
        "--within-session-results-dir",
        default=CrossSessionConfig.within_session_results_dir,
        help="Directory containing existing within-session 10-fold benchmark outputs for diagonal cells.",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Linear models: pca_lda cpca_lda or all.")
    parser.add_argument("--groups", nargs="*", default=None, help="Group names to run.")
    parser.add_argument("--frame-rate-hz", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=CrossSessionConfig.random_seed)
    parser.add_argument("--variance-to-keep", type=float, default=CrossSessionConfig.variance_to_keep)
    parser.add_argument("--cpca-m", type=int, default=CrossSessionConfig.cpca_m)
    parser.add_argument("--min-train-class-count", type=int, default=CrossSessionConfig.min_train_class_count)
    parser.add_argument("--min-test-class-count", type=int, default=CrossSessionConfig.min_test_class_count)
    parser.add_argument("--detrend-window", type=int, default=CrossSessionConfig.detrend_window)
    parser.add_argument("--spatial-filter-radius", type=int, default=CrossSessionConfig.spatial_filter_radius)
    parser.add_argument("--no-motion-correction", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ppc_matplotlib_cache")

    config = CrossSessionConfig(
        data_root=args.data_root,
        doppler_dir=args.doppler_dir,
        project_record=args.project_record,
        output_dir=args.output_dir,
        within_session_results_dir=args.within_session_results_dir,
        models=_parse_models(args.models),
        frame_rate_hz=args.frame_rate_hz,
        random_seed=args.random_seed,
        variance_to_keep=args.variance_to_keep,
        cpca_m=args.cpca_m,
        min_train_class_count=args.min_train_class_count,
        min_test_class_count=args.min_test_class_count,
        detrend_window=args.detrend_window,
        spatial_filter_radius=args.spatial_filter_radius,
        apply_motion_correction=not args.no_motion_correction,
    )
    result = run_cross_session_decoding(config, groups=_parse_groups(args.groups))
    n_pair_ok = sum(row.get("status") == "ok" for row in result["pairwise"])
    n_loso_ok = sum(row.get("status") == "ok" for row in result["loso"])
    print("Cross-session decoding complete")
    print(f"  output_dir: {result['output_dir']}")
    print(f"  pairwise ok/skipped: {n_pair_ok}/{len(result['pairwise']) - n_pair_ok}")
    print(f"  loso ok/skipped: {n_loso_ok}/{len(result['loso']) - n_loso_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
