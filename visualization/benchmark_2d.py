"""Benchmark real 2D-only streaming with raw BGR named shared memory on Windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
import json
import mmap
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

import cv2
import numpy as np
from detect_and_view import ROOT, ProtocolClient, algorithm_request
from generate_report import image_files


def map_bgr(stack, raw, width, height, piece_id):
    name = "benchmark_2d_" + uuid.uuid4().hex
    mapping = mmap.mmap(-1, len(raw), tagname=name, access=mmap.ACCESS_WRITE)
    stack.callback(mapping.close)
    mapping.write(raw)
    return dict(piece_id=piece_id, shm_name=name, image_size=len(raw),
                image_width=width, image_height=height, image_channels=3, image_format="bgr8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-2d", type=Path, default=ROOT / "demodatasets/2D")
    parser.add_argument("--server", type=Path, default=ROOT / "python/transport/bridge_server.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=10, help="1 submits serially; 10 submits all pieces concurrently")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("Windows named shared memory is required")
    if args.repeat < 1 or args.max_inflight < 1 or args.timeout <= 0:
        parser.error("repeat, max-inflight and timeout must be positive")
    paths = image_files(args.images_2d.resolve())
    if len(paths) != 10:
        parser.error(f"expected ten images, found {len(paths)}")
    output = args.output_dir or ROOT / "visualization/output" / (
        "2d-only-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(mode="2d_only", format="bgr8", max_inflight=args.max_inflight,
                   server=str(args.server.resolve()), batches=[], success=False)
    client = None
    try:
        # Decode only once, outside inference timing, simulating already captured frames.
        frames = []
        for path in paths:
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read {path}")
            height, width = image.shape[:2]
            frames.append((image.tobytes(), width, height))
        client = ProtocolClient(args.python.resolve(), args.server.resolve(), output / "python.log")
        if not client.request(dict(cmd="init"), args.timeout).get("ok"):
            raise RuntimeError("protocol initialization failed")
        init = client.request(algorithm_request(None, None), args.timeout)
        summary["initialization"] = init
        if not init.get("ok"):
            raise RuntimeError(init.get("error") or "algorithm initialization failed")
        print("Effective threading:", json.dumps(init.get("threading")), flush=True)
        for batch in range(1, args.repeat + 1):
            board = "2d-benchmark-" + uuid.uuid4().hex
            started = time.perf_counter()
            timeline_index = len(client.timeline)
            responses = []
            try:
                with ExitStack() as stack:
                    descriptors = [map_bgr(stack, raw, width, height, index)
                                   for index, (raw, width, height) in enumerate(frames, 1)]
                    input_ms = (time.perf_counter() - started) * 1000
                    request_started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
                        jobs = [pool.submit(client.request, dict(
                            cmd="detect_2d_piece", stage="2d", board_id=board, **descriptor), args.timeout)
                            for descriptor in descriptors]
                        for job in as_completed(jobs):
                            responses.append(job.result())
                    request_ms = (time.perf_counter() - request_started) * 1000
            finally:
                # No depth scan will follow: do not let board caches accumulate.
                client.request(dict(cmd="release_board", board_id=board), args.timeout)
            wall_ms = (time.perf_counter() - started) * 1000
            responses.sort(key=lambda item: item.get("piece_id", 0))
            item = dict(batch=batch, board_id=board, first_batch=batch == 1,
                        request_ok=len(responses) == 10 and all(r.get("ok") for r in responses),
                        input_prepare_ms=input_ms, request_window_ms=request_ms,
                        batch_wall_ms=wall_ms, images_per_second=10000 / request_ms)
            folder = output / f"batch-{batch:02d}"
            folder.mkdir()
            # Preserve full real results for business/geometry comparison, not just timing.
            (folder / "result.json").write_text(json.dumps(dict(
                mode="2d_only", board_id=board, responses_2d=responses, benchmark=item,
                timeline=client.timeline[timeline_index:]), ensure_ascii=False, indent=2), encoding="utf-8")
            summary["batches"].append(item)
            print(f"Batch {batch}: 10x2D request window={request_ms / 1000:.3f}s; "
                  f"input={input_ms / 1000:.3f}s; success={item['request_ok']}", flush=True)
            if not item["request_ok"]:
                summary["errors"] = [r.get("error") or r.get("message") for r in responses if not r.get("ok")]
                break
        hot = [b["request_window_ms"] for b in summary["batches"] if not b["first_batch"] and b["request_ok"]]
        summary["hot_median_ms"] = statistics.median(hot) if hot else None
        summary["success"] = len(summary["batches"]) == args.repeat and all(
            b["request_ok"] for b in summary["batches"])
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        summary["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        if client is not None:
            try:
                client.close()
            except (OSError, RuntimeError, TimeoutError) as exc:
                summary["cleanup_error"] = str(exc)
                summary["success"] = False
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Summary:", output / "summary.json", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
