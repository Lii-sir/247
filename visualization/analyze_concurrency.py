"""Summarize wall-clock task overlap from existing Bridge timing logs."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


STAGES = {
    "cpu_prepare_phase_ms": "CPU prepare",
    "silver_prepare_phase_ms": "Silver boundary",
    "gpu_phase_ms": "GPU worker task",
    "yolo_predict_wall_ms": "YOLO call",
    "efficientad_total_ms": "Chip inspection",
    "geometry_phase_ms": "Joint input geometry",
    "cpu_postprocess_phase_ms": "CPU postprocess",
    "joint_phase_ms": "Joint 3D",
}


def read_intervals(path: Path):
    pending = {}
    intervals = []
    warnings = []
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            _, marker, payload = line.partition("[TIMING] ")
            if not marker:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                warnings.append(f"line {number}: malformed timing JSON")
                continue
            if not isinstance(record, dict):
                continue
            stage, event = record.get("stage"), record.get("event")
            if stage not in STAGES or event not in {"begin", "end"}:
                continue
            clock = record.get("clock_ms")
            if not isinstance(clock, (int, float)) or not math.isfinite(clock):
                warnings.append(f"line {number}: invalid clock")
                continue
            key = tuple(record.get(name) for name in (
                "pid", "board_id", "request_id", "piece_id", "thread", "phase", "stage"
            ))
            if event == "begin":
                if key in pending:
                    warnings.append(f"line {number}: duplicate begin for {stage}")
                pending[key] = record
                continue
            start = pending.pop(key, None)
            if start is None:
                warnings.append(f"line {number}: unmatched end for {stage}")
            elif record.get("error") is not None:
                warnings.append(f"line {number}: failed {stage}: {record['error']}")
            elif clock <= start["clock_ms"]:
                warnings.append(f"line {number}: nonpositive interval for {stage}")
            else:
                intervals.append({**start, "start_ms": start["clock_ms"], "end_ms": clock})
    warnings.extend(f"unfinished {row['stage']} board={row.get('board_id')} "
                    f"piece={row.get('piece_id')}" for row in pending.values())
    return intervals, warnings


def overlap_stats(intervals):
    """End events precede starts at the same timestamp: touching is not overlap."""
    if not intervals:
        raise ValueError("no complete intervals")
    events = sorted(event for row in intervals for event in (
        (row["start_ms"], 1), (row["end_ms"], -1)
    ))
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    span = max(row["end_ms"] for row in intervals) - min(row["start_ms"] for row in intervals)
    total = sum(row["end_ms"] - row["start_ms"] for row in intervals)
    return {"count": len(intervals), "peak": peak, "span_s": span / 1000,
            "avg_active": total / span}


def print_run_summary(folder: Path):
    startup = folder / "startup.json"
    if startup.is_file():
        data = json.loads(startup.read_text(encoding="utf-8-sig"))
        settings = data.get("algorithm_response", {}).get("threading")
        print("Effective threading:", json.dumps(settings, ensure_ascii=False))
    summary = folder / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8-sig"))
        print("Run success:", data.get("success"))
        print("Batch times (include input preparation; exclude initialization/report rendering):")
        for batch in data.get("batches", []):
            wall = batch.get("batch_wall_ms")
            seconds = f"{wall / 1000:.3f}s" if isinstance(wall, (int, float)) else "unknown"
            print(f"  {batch.get('board_id')}: {seconds}, request_ok={batch.get('request_ok')}")
    results = sorted(folder.glob("batch-*/result.json"))
    if not results and (folder / "result.json").is_file():
        results = [folder / "result.json"]
    for path in results:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        requests = [row for row in data.get("timeline", []) if row.get("cmd") == "detect_2d_piece"]
        if requests:
            span = max(row["received_ms"] for row in requests) - min(row["sent_ms"] for row in requests)
            ok = sum(bool(row.get("ok")) for row in requests)
            print(f"2D request window {data.get('board_id')}: {span / 1000:.3f}s, "
                  f"successful={ok}/{len(requests)} (includes queueing)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run directory or Python log (also accepts [PYTHON] prefixed logs)")
    args = parser.parse_args(argv)
    path = args.path.resolve()
    folder = path if path.is_dir() else path.parent
    log = path / "python.log" if path.is_dir() else path
    try:
        print_run_summary(folder)
        intervals, warnings = read_intervals(log)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot analyze: {exc}\n")
    groups = defaultdict(list)
    for row in intervals:
        groups[(row.get("pid"), row.get("board_id"))].append(row)
    for (pid, board), rows in groups.items():
        print(f"\nPID={pid} Board={board}")
        print(f"{'Stage':<20} {'Tasks':>5} {'Peak':>5} {'Avg active':>11} {'Window(s)':>10}")
        for stage, label in STAGES.items():
            selected = [row for row in rows if row["stage"] == stage]
            if selected:
                stats = overlap_stats(selected)
                print(f"{label:<20} {stats['count']:>5} {stats['peak']:>5} "
                      f"{stats['avg_active']:>11.2f} {stats['span_s']:>10.3f}")
    print("\nPeak > 1 proves overlapping software task/call intervals, NOT simultaneous CUDA kernels.")
    print("Avg active = sum of interval durations / stage window; it is NOT a speedup ratio.")
    print("Nested stages overlap each other; do not add their times. Missing stages are unknown, not serial.")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}): failed/incomplete intervals were excluded.")
        for warning in warnings[:20]:
            print(" ", warning)
    if not intervals:
        print("No complete timing intervals found. Enable BRIDGE_TIMING_LOG=1 before starting the service.")
    return 0 if intervals and not warnings else 1


if __name__ == "__main__":
    raise SystemExit(main())
