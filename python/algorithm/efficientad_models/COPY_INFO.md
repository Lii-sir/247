# Anomalib EfficientAD 2.2.0

This folder contains the inference-only EfficientAD network used by Bridge.
It was copied from the user's `ccd_efficientad/models` training project on
2026-10-10; that project in turn tracks `anomalib==2.2.0`.

Copied source files:

- `torch_model.py`: PyTorch network, losses, and anomaly-map computation.
- `backbones.py`: local backbone definitions used by compatible checkpoints.
- `resnet_autoencoder.py`: feature autoencoder used by ResNet checkpoints.
- `__init__.py`: inference-only package export.

`torch_model.py` replaces anomalib's result container with a small local
`NamedTuple`.  No anomalib or Lightning installation is required to run the
Bridge inference path.  The checkpoint scoring, calibration and image
preprocessing logic lives in `algorithm/2D/chip_anomaly.py`.

