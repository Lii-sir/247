"""Standalone synthetic process/affinity probe; no models, safe spawn main."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinity", action="store_true")
    args = parser.parse_args()
    import os
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    from algorithm.registration import RegistrationConfig
    from algorithm.registration_process import RegistrationProcesses, SharedBoardInputs
    from algorithm.registration_worker import register_arrays
    from algorithm.cpu_partition import current_cpus
    from multiprocessing.shared_memory import SharedMemory
    cv2.setNumThreads(1)
    before = current_cpus() if os.name == "nt" else None
    service = RegistrationProcesses(2, dict(enabled=args.affinity, physical_cores=2))
    inputs = SharedBoardInputs()
    rng = np.random.default_rng(251)
    mask = (rng.random((96, 128)) > .6).astype(np.uint8)*255
    depth = np.tile((mask > 0).astype(np.uint16)*33000, (2, 1))
    entry = dict(gray_image=mask, registration_mask=mask)
    config = RegistrationConfig.from_dict(dict(scale=1, morphology_kernel_size=1,
        roi_3d=[2, 2, 30, 40], search_2d_scaled=[0, 0, 128, 96]))
    affine = dict(score_mode="full", cache_enabled=True, preview_scale=1,
                  morphology_kernel_size=1, rois=[[0, 0, 64, 96], [64, 0, 128, 96]],
                  maxiter=1, maxfev=20, translation_delta=[2, 2])
    direct = register_arrays(mask, mask, depth[:96], config, affine, {})
    rows, timings = [], []
    try:
        with ThreadPoolExecutor(2) as preparation:
            for repeat in range(2):
                futures = []
                for pid in (1, 2):
                    durations = {}
                    timings.append(durations)
                    futures.append(service.submit(preparation, inputs, pid, entry, depth,
                        dict(y1=(pid-1)*96, y2=pid*96), {}, config, affine, durations,
                        dict(board_id="A", piece_id=pid, request_id="probe"), lambda: True))
                rows += [future.result(timeout=30) for future in futures]
        for row in rows:
            worker = row.pop("registration_worker")
            assert not worker["torch_loaded"]
            for key in ("matrix_2d_to_3d", "affine_score_after", "affine_zone_losses",
                        "affine_function_evaluations", "affine_warp_evaluations", "affine_valid"):
                assert row[key] == direct[key], (key, row[key], direct[key])
        names = [block[0].name for block in inputs._blocks.values()]
        assert len(names) == 5, names  # One depth plus two gray/mask pairs, reused in round 2.
        during = current_cpus() if os.name == "nt" else None
        info = service.info
        if args.affinity:
            assert during == info["affinity"]["main_cpus"]
            assert set(during).isdisjoint(info["affinity"]["registration_cpus"])
            for worker in info["workers"]:
                assert worker["cpu_ids"] == worker["assigned_cpu_ids"]
        inputs.close()
        for name in names:
            try:
                leftover = SharedMemory(name=name)
            except FileNotFoundError:
                pass
            else:
                leftover.close()
                raise AssertionError("shared memory still exists after last lease: "+name)
    finally:
        inputs.close()
        service.close()
    after = current_cpus() if os.name == "nt" else None
    assert before == after, (before, after)
    print(json.dumps(dict(ok=True, rounds=2, pieces=4, buffers=5, execution=info,
                          affinity_before=before, affinity_during=during, affinity_after=after,
                          timings=timings, torch_loaded="torch" in sys.modules)))


if __name__ == "__main__":
    main()
