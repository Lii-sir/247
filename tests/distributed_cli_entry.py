"""Test-only launcher: exercise two ranks on a CPU or one physical test GPU."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import efficientad_ccd as cli


if __name__ == "__main__":
    if sys.argv[1] == "train":
        device = os.environ.get("CCD_DDP_TEST_DEVICE", "cpu")
        with patch.object(cli, "choose_devices", return_value=[device, device]):
            cli.main()
    else:
        cli.main()
