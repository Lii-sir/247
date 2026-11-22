"""Inference-only EfficientAD network copied from the CCD training project."""

from .torch_model import EfficientAdModel, EfficientAdModelSize

__all__ = ["EfficientAdModel", "EfficientAdModelSize"]
