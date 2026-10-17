# Streaming thread optimization handoff

- Completed: bounded independent postprocessing pool and startup native thread defaults. Default is 9 preparation/joint + 3 postprocess + 1 GPU, preserving the 13-worker total.
- Files: `python/transport/bridge_server.py`, `python/algorithm/board_pipeline.py`, `tests/test_concurrent_pipeline.py`, `visualization/README.md`.
- Evidence: 63 tests passed; real init/exit defaults checked; nine real batches completed. Raw comparison: `visualization/output/thread-comparison-20261015.json`. Baseline first/reused: 78.101/42.427 s; selected first/reused/reused: 39.505/37.439/37.036 s.
- Remaining/blockers: none for the requested optimization. Silver registration fallback varies before and after; other compared decision fields and defect areas match. Benchmark samples are limited and no peak-memory measurement was made. Revisit worker counts when hardware/load changes.

## CPU postprocess queue optimization

- Completed: equivalent local postprocess reductions and default 5-way postprocess pool. Total budget is 7 preparation/joint + 5 postprocess + 1 GPU.
- Evidence: 67 tests passed; 50 differential continuity fixtures passed; selected 2D outputs unchanged. Hot batches improved from 36.152s baseline to 30.759s and 28.877s; queue average from 6.698s to 1.826s/1.892s.
- Files: `python/algorithm/2D/silver_inspection.py`, `python/algorithm/2D/defect_geometry.py`, `python/algorithm/detect_board.py`, `python/transport/bridge_server.py`, `tests/test_postprocess_equivalence.py`.
- Remaining: GPU inference architecture unchanged; retest worker count on other hardware.

## Thread counts moved to stage config

- `config/2D/config.json`: `common_2d.parameters.threading.postprocess_workers=5`, `gpu_workers=1`.
- `config/3D/config.json`: `common_3d.parameters.threading.worker_budget=13`, `cpu_workers=7`.
- Bridge reads the merged stage config during algorithm initialization and reconfigures idle pools; `BRIDGE_MAX_WORKERS`, `BRIDGE_CPU_WORKERS`, and `BRIDGE_POSTPROCESS_WORKERS` remain temporary overrides.
- 70 tests passed; config JSON and syntax checks passed.

## GPU workers configurable; worker budget has no hard-coded ceiling

- Current stage settings: 2D `gpu_workers=2`, `postprocess_workers=5`; 3D `worker_budget=14`, `cpu_workers=7`.
- The unified board detector loads one independent YOLO and EfficientAD resource pair per configured GPU worker. The configured YOLO device selects each CUDA stream even before lazy predictor setup moves the model off CPU. Standalone legacy 2D/3D APIs continue to reject `gpu_workers > 1`.
- `worker_budget` is read from merged stage configuration and has no code-level maximum; configured worker-pool totals must still fit the budget. Environment variables are fallback values when stage keys are absent.
- Verification: `.venv\\Scripts\\python.exe -B -m unittest discover -s tests -p 'test_*.py' -v` passed 75 tests. No real GPU inference, sample batch, or hardware benchmark was run.
- Remaining: on-hardware verification of replica initialization, peak VRAM, output equivalence, and throughput is still required before relying on increased GPU concurrency.

# Python staged optimization checkpoint (2026-10-15)

- Completed with evidence: early joint-input publication and final diagnostic join; silver boundary preparation overlaps GPU branch; local height reference morphology and selected-point validity; per-piece chip-region reuse.
- Files: `python/algorithm/board_pipeline.py`, `python/algorithm/detect_board.py`, `python/algorithm/2D/silver_inspection.py`, `python/algorithm/height_measurement.py`, `python/transport/bridge_server.py`, tests and visualization concurrency analyzer.
- Validation: 91 Python tests, 26 visualization tests; final real 4-batch run successful with 10 pieces and valid registration per batch; final batch times 25.402, 27.577, 31.378, 26.334 seconds.
- Remaining: controlled comparison for `early_joint_workers`/pool sizes, output differential beyond current structural checks, and later affine/height algorithm reductions. User-owned config/DLL/process changes preserved.

# Python 2D optimization checkpoint (2026-10-16)

- Completed: precomputed gray image reuse across silver SIFT and later registration; local silver continuity rings with one sector aggregation; local contour extraction and one-time ROI instance index; raw BGR 2D benchmark utility and differential reference tests.
- Backup: Git stash `before-2d-optimization-20261016` preserves the pre-optimization worktree snapshot.
- Evidence: 95 Python tests and 26 visualization tests pass; baseline 2D-only hot median 13.546 s versus optimized 12.767 s for the same ten BGR shared-memory inputs; 2D result statuses in batch 2 matched.
- Remaining: no SIFT resolution/feature reduction was applied; future optimization requires precision differential and hardware-specific tuning. User changes remain in the worktree.

# Split silver/joint pools complete (2026-10-16)

- Completed: silver boundary preparation uses configurable silver_pool; joint registration/mapping/height uses configurable joint_pool. Input prep/geometry/depth remain cpu_pool. GPU and complete 2D diagnostics keep their existing pools.
- Config: common_2d.parameters.threading.silver_workers=5; common_3d.parameters.threading.joint_workers=5. Current distinct pools are CPU5/GPU3/silver5/postprocess5/joint5 = 23 slots, budget50. JSON wins over BRIDGE_SILVER_WORKERS/BRIDGE_JOINT_WORKERS; missing both key and env keeps aliases. early_joint_workers defaults to and cannot exceed joint_workers. Restart/init or idle reinit applies; no hot reload.
- Evidence: 119 Python tests including 24 split-pool cases; 26 visualization tests; AST/JSON and git diff --check pass. Real 2D raw BGR: 30/30 requests, windows 17.301/12.940/11.931s. Real 2D+3D: 3 batches with 10 pieces and valid registration each, input-inclusive times 26.731/23.206/22.909s. Logs show silver peak5, joint peak5, GPU software task peak3. Second-batch key 2D decisions/statuses match prior optimized output.
- Files: python/algorithm/board_pipeline.py; python/transport/bridge_server.py; config/2D/config.json; config/3D/config.json; tests/test_split_stage_pools.py; visualization/README.md. Evidence at visualization/output/split-pools-2d-20261016-085649 and visualization/output/split-pools-joint-20261016-085811 (verification.json).
- Remaining/blockers: none for the two requested pool separations. No controlled same-turn baseline, peak RAM/VRAM, or CUDA kernel concurrency proof; do not claim fixed speedup or target-machine optimality. Existing user changes and before-2d-optimization-20261016 stash preserved; DLL/C# untouched this task.

## ROI four-intersection wire bending (2026-10-17)
- Completed: shared quadrilateral construction, per-instance ROI denominator, diagnostics and consistent ROI assignment in both 2D entry points and joint height fallback; DTO source and parameter docs updated.
- Evidence: 134 algorithm + 26 report tests passed; Bridge Debug/Release builds passed; real 10-piece sample completed, 72 2D ratios and 52 joint cached measurements verified. Artifacts: visualization/output/wire-bending-validation-20261017/.
- Remaining: no implementation work; sample-label threshold calibration advised. Existing 0.75 threshold and user config/DLL edits preserved.
- Limits: five silver SIFT allocation failures triggered configured fallback during the real batch. src/ remains ignored by existing repository rules; DTO edits are local and compiled under src/Bridge/bin, root DLL preserved.
