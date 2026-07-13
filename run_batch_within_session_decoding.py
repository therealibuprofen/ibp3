#!/usr/bin/env python3
"""Run within-session decoding for all sessions listed in a project record."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ppc_matplotlib_cache")

from ppc_direction_decoding.within_session import (  # noqa: E402
    VALID_DECODING_MODES,
    WithinSessionConfig,
    decode_within_session,
)


DEFAULT_DATA_ROOT = Path("/data2/yuq1ngr/dataset/data2")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output" / "python_within_session_batch"


def _find_project_record(data_root: Path) -> Path:
    candidates = [
        data_root / "projectrecord",
        data_root / "ProjectRecord_paper.json",
        data_root / "ProjectRecord.json",
        data_root / "projectrecord.json",
        ROOT.parent / "data" / "ProjectRecord_paper.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find project record. Checked: {names}")


def _load_project_record(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Project record must contain a JSON list, got {type(records).__name__}.")
    return records


def _session_token(record: dict[str, Any]) -> str:
    session = int(record["Session"])
    run = int(record["Run"])
    return f"S{session}_R{run}"


def _expected_mat_path(record: dict[str, Any], doppler_dir: Path) -> Path:
    return doppler_dir / f"doppler_{_session_token(record)}+normcorre.mat"


def _record_is_decodable(record: dict[str, Any]) -> bool:
    return int(record.get("nTrials") or 0) > 0 and int(record.get("nTargets") or 0) == 8


def _parse_session_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("S") and "_R" in item:
                out.add(item)
            elif item.startswith("S"):
                out.add(item)
            elif "_R" in item:
                out.add(f"S{item}")
            elif ":" in item:
                session, run = item.split(":", 1)
                out.add(f"S{int(session)}_R{int(run)}")
            else:
                out.add(f"S{int(item)}")
    return out


def _matches_session_filter(token: str, filters: set[str] | None) -> bool:
    if filters is None:
        return True
    session_part = token.split("_R", 1)[0]
    return token in filters or session_part in filters


def _load_failed_session_filters(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Failed-session summary CSV does not exist: {summary_path}")

    failed: set[str] = set()
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            error = (row.get("error") or "").strip()
            if status != "error" and not error:
                continue

            token = (row.get("session_id") or "").strip()
            if not token:
                session = (row.get("session") or "").strip()
                run = (row.get("run") or "").strip()
                if session and run:
                    token = f"S{int(float(session))}_R{int(float(run))}"
            if token:
                failed.add(token)

    if not failed:
        raise ValueError(f"No failed sessions found in {summary_path}")
    return failed


def _summary_row(
    *,
    status: str,
    record: dict[str, Any],
    mat_path: Path,
    output_dir: Path,
    summary: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    token = _session_token(record)
    row = {
        "status": status,
        "session_id": token,
        "session": record.get("Session"),
        "run": record.get("Run"),
        "monkey": record.get("Monkey"),
        "date": record.get("Date"),
        "project": record.get("Project"),
        "recording_system": record.get("RecordingSystem"),
        "slot": record.get("Slot"),
        "ap_plane": record.get("ap_plane"),
        "record_n_trials": record.get("nTrials"),
        "record_n_targets": record.get("nTargets"),
        "mat_path": str(mat_path),
        "output_dir": str(output_dir),
        "n_valid_trials": "",
        "n_trials_total": "",
        "n_valid_trials_after_success_and_targetPos": "",
        "n_trials_entering_CV": "",
        "skip_reason": "",
        "min_class_count": "",
        "requested_n_splits": "",
        "actual_n_splits": "",
        "center_tolerance": "",
        "combined_label_distribution": "",
        "horizontal_label_distribution": "",
        "vertical_label_distribution": "",
        "combined_labels_with_count_1": "",
        "has_real_center_center_label_5": "",
        "target_pos_near_zero_nonzero_x": "",
        "target_pos_near_zero_nonzero_y": "",
        "target_pos_near_zero_nonzero_x_count": "",
        "target_pos_near_zero_nonzero_y_count": "",
        "target_pos_distribution_round6": "",
        "accuracy": "",
        "balanced_accuracy": "",
        "final_accuracy_percent": "",
        "final_mean_angular_error_deg": "",
        "earliest_accuracy_significant_time_s": "",
        "earliest_angular_error_significant_time_s": "",
        "pca_component_range": "",
        "error": error,
    }
    if summary:
        row.update(
            {
                "n_valid_trials": summary.get("n_valid_trials", ""),
                "n_trials_total": summary.get("n_trials_total", ""),
                "n_valid_trials_after_success_and_targetPos": summary.get(
                    "n_valid_trials_after_success_and_targetPos", ""
                ),
                "n_trials_entering_CV": summary.get("n_trials_entering_CV", ""),
                "skip_reason": summary.get("skip_reason", ""),
                "min_class_count": summary.get("min_class_count", ""),
                "requested_n_splits": summary.get("requested_n_splits", ""),
                "actual_n_splits": summary.get("actual_n_splits", ""),
                "center_tolerance": summary.get("center_tolerance", ""),
                "combined_label_distribution": json.dumps(
                    summary.get("combined_label_distribution", summary.get("class_distribution_combined", ""))
                ),
                "horizontal_label_distribution": json.dumps(summary.get("horizontal_label_distribution", "")),
                "vertical_label_distribution": json.dumps(summary.get("vertical_label_distribution", "")),
                "combined_labels_with_count_1": json.dumps(summary.get("combined_labels_with_count_1", "")),
                "has_real_center_center_label_5": summary.get("has_real_center_center_label_5", ""),
                "target_pos_near_zero_nonzero_x": summary.get("target_pos_near_zero_nonzero_x", ""),
                "target_pos_near_zero_nonzero_y": summary.get("target_pos_near_zero_nonzero_y", ""),
                "target_pos_near_zero_nonzero_x_count": summary.get(
                    "target_pos_near_zero_nonzero_x_count", ""
                ),
                "target_pos_near_zero_nonzero_y_count": summary.get(
                    "target_pos_near_zero_nonzero_y_count", ""
                ),
                "target_pos_distribution_round6": json.dumps(summary.get("target_pos_distribution_round6", "")),
                "accuracy": summary.get("accuracy", ""),
                "balanced_accuracy": summary.get("balanced_accuracy", ""),
                "final_accuracy_percent": summary.get("final_accuracy_percent", ""),
                "final_mean_angular_error_deg": summary.get("final_mean_angular_error_deg", ""),
                "earliest_accuracy_significant_time_s": summary.get(
                    "earliest_accuracy_significant_time_s", ""
                ),
                "earliest_angular_error_significant_time_s": summary.get(
                    "earliest_angular_error_significant_time_s", ""
                ),
                "pca_component_range": json.dumps(summary.get("pca_component_range", "")),
            }
        )
    return row


def _write_batch_outputs(rows: list[dict[str, Any]], config: WithinSessionConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "batch_summary.csv"
    json_path = output_dir / "batch_summary.json"

    fieldnames = list(rows[0].keys()) if rows else ["status", "session_id", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": asdict(config),
        "rows": rows,
        "counts": {
            "ok": sum(row["status"] == "ok" for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "skipped": sum(row["status"] == "skipped" for row in rows),
            "error": sum(row["status"] == "error" for row in rows),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nBatch summary written to:\n  {csv_path}\n  {json_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--doppler-dir", type=Path, default=None)
    parser.add_argument("--project-record", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Optional filters, e.g. S27_R99 S76_R1, 27:99, or 27.",
    )
    parser.add_argument(
        "--rerun-failed-from-summary",
        type=Path,
        default=None,
        help=(
            "Read a previous batch_summary.csv and only run rows with status=error "
            "or a non-empty error column. Can be combined with --sessions to further narrow the set."
        ),
    )
    parser.add_argument("--include-nondecodable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
    parser.add_argument("--center-tolerance", type=float, default=WithinSessionConfig.center_tolerance)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--no-motion-correction", action="store_true")
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--no-spatial-filter", action="store_true")
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser().resolve()
    doppler_dir = (args.doppler_dir or data_root / "doppler").expanduser().resolve()
    project_record_path = (args.project_record or _find_project_record(data_root)).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    session_filters = _parse_session_filters(args.sessions)
    failed_session_filters = (
        _load_failed_session_filters(args.rerun_failed_from_summary.expanduser().resolve())
        if args.rerun_failed_from_summary
        else None
    )

    records = _load_project_record(project_record_path)
    config = WithinSessionConfig(
        mode=args.mode,
        cv_scheme=args.cv_scheme,
        n_splits=args.n_splits,
        random_seed=args.seed,
        n_permutations=args.n_permutations,
        output_dir=str(output_dir),
        max_timepoints=args.max_timepoints,
        center_tolerance=args.center_tolerance,
        diagnostic_only=args.diagnostic_only,
        apply_motion_correction=not args.no_motion_correction,
        detrend_window=0 if args.no_detrend else WithinSessionConfig.detrend_window,
        spatial_filter_radius=0 if args.no_spatial_filter else WithinSessionConfig.spatial_filter_radius,
    )

    selected_records = []
    for record in records:
        token = _session_token(record)
        if not _matches_session_filter(token, session_filters):
            continue
        if not _matches_session_filter(token, failed_session_filters):
            continue
        if not args.include_nondecodable and not _record_is_decodable(record):
            continue
        selected_records.append(record)

    print(f"Project record: {project_record_path}")
    print(f"Doppler dir:    {doppler_dir}")
    print(f"Output dir:     {output_dir}")
    if failed_session_filters is not None:
        print(f"Failed summary: {args.rerun_failed_from_summary.expanduser().resolve()}")
        print(f"Failed rows:    {len(failed_session_filters)} session ids")
    print(f"Sessions:       {len(selected_records)} selected from {len(records)} records")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=1):
        token = _session_token(record)
        mat_path = _expected_mat_path(record, doppler_dir)
        session_output_dir = output_dir / token
        print(f"\n[{index}/{len(selected_records)}] {token}")
        print(f"  input:  {mat_path}")
        print(f"  output: {session_output_dir}")

        if not mat_path.exists():
            print("  status: missing input file")
            rows.append(_summary_row(status="missing", record=record, mat_path=mat_path, output_dir=session_output_dir))
            _write_batch_outputs(rows, config, output_dir)
            continue

        if args.dry_run:
            print("  status: dry-run")
            rows.append(_summary_row(status="skipped", record=record, mat_path=mat_path, output_dir=session_output_dir))
            _write_batch_outputs(rows, config, output_dir)
            continue

        session_config = WithinSessionConfig(**{**asdict(config), "output_dir": str(session_output_dir)})
        try:
            result = decode_within_session(mat_path, session_config, session_id=token)
            result_status = result["summary"].get("status", "ok")
            if result_status == "diagnostic_only":
                result_status = "skipped"
            rows.append(
                _summary_row(
                    status=result_status,
                    record=record,
                    mat_path=mat_path,
                    output_dir=session_output_dir,
                    summary=result["summary"],
                )
            )
            del result
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - batch mode should continue and report failures.
            message = f"{type(exc).__name__}: {exc}"
            print(f"  status: error: {message}", file=sys.stderr)
            traceback.print_exc()
            rows.append(
                _summary_row(
                    status="error",
                    record=record,
                    mat_path=mat_path,
                    output_dir=session_output_dir,
                    error=message,
                )
            )
            if args.stop_on_error:
                _write_batch_outputs(rows, config, output_dir)
                return 1

        _write_batch_outputs(rows, config, output_dir)

    _write_batch_outputs(rows, config, output_dir)

    ok = sum(row["status"] == "ok" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    print(f"\nDone. ok={ok}, missing={missing}, skipped={skipped}, error={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
