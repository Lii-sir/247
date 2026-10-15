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
