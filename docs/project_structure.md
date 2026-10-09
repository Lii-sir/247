# 项目目录结构

整个研发项目已按职责分层，模型、训练、推理和工具不再平铺于根目录。根目录只保留统一入口、环境配置和两个旧命令兼容入口。

```text
247/
├─ run.py                         # 统一启动入口
├─ README.md                      # 项目总说明
├─ pyproject.toml / uv.lock        # 锁定依赖
├─ requirements.txt               # 已有依赖清单
├─ .python-version                # Python 3.12
├─ ccd_efficientad/                # 唯一维护的核心代码
│  ├─ __main__.py                 # 子命令分发
│  ├─ paths.py                    # 集中定义项目路径
│  ├─ cli.py                      # 数据检查、训练、校准、评估
│  ├─ inference.py                # 单图/目录推理及 API
│  ├─ data.py                     # 数据检查、快照和划分
│  ├─ distributed.py              # 优化循环与 DDP
│  ├─ mask.py                     # ignore mask、圆检测、池化
│  ├─ localization.py             # 响应、连通域、定位框
│  ├─ report.py                   # 指标、报告、热图
│  └─ models/                     # Teacher / Student / AE
├─ tools/
│  ├─ data/                       # VisA 下载、划分与异常合成
│  ├─ masks/                      # GUI、圆检测、mask 生成
│  ├─ diagnostics/                # 分支异常图导出
│  └─ build_inference_delivery.py  # 同步推理交付快照
├─ configs/
│  ├─ circle_config.json          # 原配置内容不变，只移动位置
│  └─ masks/                      # 建议存放 ignore mask
├─ tests/                         # 回归测试、DDP 与合成链路
├─ docs/                          # 使用与网络说明
├─ references/                    # 参考检出，保留各自 Git 元数据
│  ├─ efficientad_official/       # 原 .efficientad_check
│  └─ efficientad_alternative/    # 原 .efficientad_rximg_check
├─ assets/                        # 资源缓存，原路径不变
├─ outputs/                       # 权重、报告、结果，原路径不变
│  └─ migration_backup/           # 迁移前交付源码与旧 process 缓存
├─ delivery/efficientad/          # 对外推理包
├─ efficientad_ccd.py             # 旧 CLI/导入兼容薄入口
└─ infer_efficientad.py           # 旧推理/导入兼容薄入口
```

`.git/`、`.venv/`、`.uv-cache/` 和 `__pycache__/` 是版本控制、环境或缓存，不是业务源码，不要随意移动或交付。

## 统一入口

在项目根目录使用：

```powershell
uv run python run.py --help
uv run python run.py inspect --category CCD1
uv run python run.py train --category CCD1 --circle-config "configs/circle_config.json"
uv run python run.py evaluate --checkpoint "D:\models\CCD1\model.pt"
uv run python run.py infer --checkpoint "D:\models\CCD1\model.pt" --image "D:\images\sample.bmp"
uv run python run.py mask-gui
uv run python run.py generate-mask --help
uv run python run.py detect-circle --help
uv run python run.py branch-maps --help
uv run python run.py download-visa --help
uv run python run.py make-dataset --help
```

也支持 `python -m ccd_efficientad <命令>`。工具支持 `python -m tools.data.make_anomaly_dataset` 等模块调用；核心包内文件不要直接作为脚本执行。

## 迁移对照

| 原位置 | 新位置 |
|---|---|
| efficientad_ccd.py 业务代码 | ccd_efficientad/cli.py |
| infer_efficientad.py 业务代码 | ccd_efficientad/inference.py |
| ccd_data.py / ccd_distributed.py | ccd_efficientad/data.py / distributed.py |
| ccd_localization.py / ccd_report.py | ccd_efficientad/localization.py / report.py |
| circle_mask.py | ccd_efficientad/mask.py |
| self_efficientad/ | ccd_efficientad/models/ |
| circle_config.json | configs/circle_config.json |
| 三个圆检测脚本 | tools/masks/ |
| VisA / 合成数据脚本 | tools/data/ |
| export_branch_maps.py | tools/diagnostics/export_branch_maps.py |

旧的 `python efficientad_ccd.py train ...`、`python infer_efficientad.py ...` 仍可用。新 API 使用正式包路径：

```python
from ccd_efficientad.inference import EfficientAdPredictor
from ccd_efficientad.models import EfficientAd
from ccd_efficientad import cli
```

网络结构、权重 key、阈值及预处理不因目录调整而改变。历史日志/checkpoint 的原路径不批量改写；旧 mask 路径失效时用 --mask / --circle-config 指定等效本机文件。

## 交付快照

运行 `python tools/build_inference_delivery.py`，从核心包同步到 `delivery/efficientad/src/ccd_efficientad/`，不要手工改交付源码。构建不覆盖权重、mask、验收图片或结果。

旧平铺快照归档在 `outputs/migration_backup/delivery_src_flat_20261009/`，不再作为入口依赖。对外交付仍需主动选择生产权重和匹配 mask。
