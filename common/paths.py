"""Project defaults resolved from source location, not the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "datasets"

