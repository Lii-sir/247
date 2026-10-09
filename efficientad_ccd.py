"""兼容旧命令/导入；新代码位于 ccd_efficientad.cli，推荐 python run.py。"""

import sys

if __name__ == "__main__":
    from ccd_efficientad.__main__ import entrypoint
    entrypoint()
else:
    from ccd_efficientad import cli
    sys.modules[__name__] = cli
