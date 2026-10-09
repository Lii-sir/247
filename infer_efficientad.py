"""兼容旧命令/导入；新代码位于 ccd_efficientad.inference。"""

import sys

if __name__ == "__main__":
    from ccd_efficientad.__main__ import entrypoint
    sys.argv.insert(1, "infer")
    entrypoint()
else:
    from ccd_efficientad import inference
    sys.modules[__name__] = inference
