# Python priority optimization, 2026-10-15

Scope: Python service only. Preserve input/output contracts, detection thresholds,
calibration integrity, shared-memory ownership, board/piece association and final
OK/NG semantics. Existing user changes in configs, DLL and process notes stay intact.

Evidence: current flow publishes joint inputs only after finish_2d_piece completes;
the heavy silver boundary registration does not require segmentation results.
Baseline: visualization/output/python-priority-baseline-20261015 (three real batches).

Stages and acceptance:
1. Publish a separate joint-input snapshot before heavy 2D diagnostics. Start 3D
   only when depth exists; join complete diagnostics before any final board response.
   Event-gated regression tests must demonstrate early execution and final NG merge.
   Cover 2D-first, 3D-first, late 3D, failure, retries, release and independent boards.
2. Move independent silver boundary preparation before/alongside inference using
   bounded existing pools and callbacks, without worker-blocking joins. Preserve
   boundary failure/fallback and source-buffer lifetime. Measure before retaining.
3. Reduce equivalent repeated height computation (validity scans/local rings),
   with differential tests on numeric values, coordinate order and edge behavior.
4. Compare real outputs and hot batch timing at every meaningful checkpoint.
   Retain proven improvements; document stochastic silver registration differences.

Deferred: reduced SIFT accuracy, changed registration optimizer/thresholds, extra
GPU workers, nested per-object executors, new depth wire protocol and DLL changes.
These require independent evidence or protocol decisions, rather than blind rollout.

Dependency map: bridge_server dispatch -> BoardPipeline -> DetectBoard and silver
inspection/height measurement -> Python protocol/client report. No new dependencies.
Risks: mutable instances (separate dicts/read-only mask inputs); early NG/return
(join both outputs); callbacks after release (state identity guard); failed early
jobs (terminal board failure once claimed, remove unclaimed early cache for retry);
CPU competition (fixed pool counts, benchmark same inputs/config); stale source
images (retain ownership until all branches finish).

Validation: CPU event-gated tests, existing scheduling/height/silver tests, full
Python and visualization regression suites, real service shared-memory batches,
business-output differential, log interval analysis, git diff check.
Fallback: remove only task-owned changes or disable a staged experiment if it
regresses; preserve user changes. No production deployment/config mutation.

Initial regression evidence: test_early_joint_pipeline.py fails three overlap
cases on the original scheduler, while two release/retry cases pass.
Stage 1: event-gated early-3D and final-diagnostics joins pass. Baseline batch
times: 46.080/33.029/26.650 s. Unrestricted early-3D experiment:
35.066/29.805/27.502 s. Hot run did not improve; do not claim a speedup.

Stage 2: image-only silver boundary work now branches alongside GPU work;
geometry publishes joint inputs independently. All joins are callbacks (no
executor worker waits for a sibling). Source ownership is released once, after
all branches settle, including failures. Early 3D is bounded to one task per
incomplete board by default; optional common_3d.parameters.threading.
early_joint_workers must be <= cpu_workers. Full CPU concurrency resumes after
all 2D diagnostics complete. 87 Python tests pass; real stage-2 batches running.

Stage 2 experiment: 31.183/32.077/31.617 s with early_joint_workers=1.
Restricting early joint work extended the 3D tail, so the default was reverted
to the configured cpu_workers (optional early_joint_workers remains tunable).
Do not use the experiment as a claimed improvement.

Stage 3: local platform morphology now operates inside the target bounding box
plus the complete halo; fitting keeps original coordinates and sample order.
Height validity filtering visits selected mask points instead of full frames.
Filled chip external regions/areas are cached inside each piece's private aligned
instance dicts (no cross-piece cache). Silver total work includes the independent
boundary branch, so the timing rename cannot manufacture a speedup.
Validation: 91 Python tests pass, including differential height cases, boundary
preparation/fallback equivalence, resource-lifetime/failure gates and diagnostic NG
joins. Final four-batch real benchmark in progress; business diff still required.

Existing test issue fixed: checked-in-config assertions pinned old thread
counts (14/7/5/2), inconsistent with user settings (50/5/5/3). Test now checks
that effective settings match file values; production configs were not changed.
