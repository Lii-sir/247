"""项目统一启动入口；具体业务代码位于 ccd_efficientad/ 和 tools/。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ccd_efficientad.__main__ import entrypoint


if __name__ == "__main__":
    entrypoint()

