import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_concurrency import overlap_stats, read_intervals


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_intervals(self):
        stats = overlap_stats([{"start_ms": 0, "end_ms": 1000},
                               {"start_ms": 250, "end_ms": 750}])
        self.assertEqual(stats["peak"], 2)
        self.assertEqual(stats["avg_active"], 1.5)

    def test_touching_intervals_are_serial(self):
        stats = overlap_stats([{"start_ms": 0, "end_ms": 10},
                               {"start_ms": 10, "end_ms": 20}])
        self.assertEqual(stats["peak"], 1)

    def test_read_interleaved_requests_and_reject_failed_or_unfinished(self):
        def row(request, event, clock, **fields):
            return {"pid": 1, "board_id": "A", "request_id": request,
                    "piece_id": 1, "thread": "gpu", "phase": "gpu",
                    "stage": "gpu_phase_ms", "event": event, "clock_ms": clock, **fields}

        records = [row("a", "begin", 1), row("b", "begin", 2),
                   row("a", "end", 3), row("b", "end", 4),
                   row("c", "begin", 5), row("c", "end", 6, error="RuntimeError"),
                   row("d", "begin", 7)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "python.log"
            path.write_text("unrelated output\n" + "\n".join(
                "[PYTHON] [TIMING] " + json.dumps(r) for r in records), encoding="utf-8")
            intervals, warnings = read_intervals(path)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(overlap_stats(intervals)["peak"], 2)
        self.assertEqual(len(warnings), 2)

    def test_pid_board_and_thread_separate_matching(self):
        records = []
        for event, clock in [("begin", 1), ("end", 3)]:
            for pid, board, thread in [(1, "A", "g0"), (2, "A", "g0"),
                                       (1, "B", "g0"), (1, "A", "g1")]:
                records.append(dict(pid=pid, board_id=board, thread=thread,
                                    event=event, clock_ms=clock, stage="gpu_phase_ms"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "python.log"
            path.write_text("\n".join("[TIMING] " + json.dumps(r) for r in records), encoding="utf-8")
            intervals, warnings = read_intervals(path)
        self.assertEqual(len(intervals), 4)
        self.assertFalse(warnings)


if __name__ == "__main__":
    unittest.main()
