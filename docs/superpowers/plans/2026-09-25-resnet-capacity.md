# ResNet capacity implementation plan

Goal: implement the approved whole-student widening, channel-scaled feature AE,
and resnet50_layer3 support, then verify and document the result in this workspace.

Design: frozen native pretrained teacher C; randomly initialized student with
2x stem/stage widths and a signed 3x3 output head producing 2C. Feature AE uses
small=max(32, min(256, C//2)//2), large=min(256, max(64, C//2)), and produces C
channels directly at teacher spatial size. PDN stays on the original architecture.
Version 2 is the new ResNet default; legacy checkpoints rebuild version 1.

Tech: existing PyTorch 2.7.1, torchvision 0.22.1, local EfficientAD, unittest.
Spec: user-approved design in the current task (2026-09-25).

## Work items

- [x] Add failing regression tests in tests/test_resnet_capacity.py:
  ResNet50 acceptance; intermediate widened feature shapes; exact AE target shape;
  signed outputs; finite backward into both student halves and AE; frozen teacher;
  legacy and new CLI/Lightning checkpoint loading.
- [x] Implement teacher/whole-student factories in self_efficientad/backbones.py.
  Use torchvision residual blocks with standard strides and stage depths.
- [x] Parameterize Encoder widths; add self_efficientad/resnet_autoencoder.py.
  Preserve encoder compression, decoder convolution/interpolation/dropout and no skips.
- [x] Wire architecture version into torch_model.py, lightning_model.py and CLI
  construction/read/save. Preserve old state keys and old prediction behavior.
- [x] Extend tests/smoke_pipeline.py and CLI choices for ResNet50. Run fresh full
  unittest discovery, actual CLI train/calibrate/evaluate/predict/resume, plus
  two-rank DDP checks if supported locally.
- [x] Review changes, fix failures, run git diff --check; write
  docs/resnet_capacity_upgrade.md with exact structures, commands, compatibility,
  measured parameter counts, validation results and hardware limitations.

No dependency changes, no data/mask changes, no defect-mask loss in this task.
Do not overwrite the user's current circle_config.json, pyproject.toml or uv.lock.
