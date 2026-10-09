# EfficientAD 推理交付包

**只做推理，不包含训练数据。** 原项目源码在 `src/`，统一启动入口是 `run_inference.py`。

## 目录怎么用

```text
efficientad/
├─ README.md                # 先看这个文件
├─ run_inference.py         # 唯一推荐入口
├─ pyproject.toml           # 环境配置（构建时从原项目复制）
├─ uv.lock                  # 锁定环境（构建时从原项目复制）
├─ .python-version          # Python 3.12
├─ src/                     # 运行源码快照，不建议直接修改
│  └─ ccd_efficientad/      # 与研发项目一致的包结构
│     ├─ inference.py       # 推理 API
│     ├─ cli.py             # 恢复权重与 score
│     ├─ data.py
│     ├─ localization.py
│     ├─ report.py
│     ├─ mask.py
│     ├─ paths.py
│     └─ models/            # 网络结构及上游来源说明
├─ weights/                 # 放已校准 model.pt，例如 CCD1/model.pt
├─ masks/                   # 放与训练一致的 ignore mask
├─ configs/                 # 可选：相机类别 → mask 配置
├─ examples/                # 放允许对外交付的验收图片
├─ results/                 # 推理结果自动生成，不是输入资料
├─ docs/                    # 文件清单、交付检查和参数说明
└─ source_manifest.json     # 源文件清单与 SHA256，标识源码快照
```

## 1. 安装环境

在此文件所在目录执行：

```powershell
uv sync --locked
uv run python .\run_inference.py --help
```

已有对应环境时，也可以直接 `python run_inference.py ...`。环境按原项目锁定，仍依赖 anomalib / Lightning，不是纯 torch 单文件程序。

## 2. 放好权重和 mask

本次整理**没有自动选择和复制已有训练权重**，也没有用全黑图冒充训练 mask。
请将准备交付的已校准模型放入 `weights/CCD1/model.pt`，将对应训练 mask 放入 `masks/CCD1.png`。

权重须为原项目 CLI 保存的、带 `calibration` 的 model.pt；未校准 last.pt、单独 Teacher 权重和 Lightning .ckpt 不能替代。

## 3. 单张图片

```powershell
uv run python .\run_inference.py `
  --checkpoint ".\weights\CCD1\model.pt" `
  --image ".\examples\sample.bmp" `
  --mask ".\masks\CCD1.png" `
  --device auto
```

如果原训练未使用 mask，应省略 `--mask`。如使用 mask，迁移时显式指定本机路径，但有效检测区必须保持一致。mask 的 **0/黑色参与检测，非零/白色忽略**。

## 4. 文件夹批量推理

```powershell
uv run python .\run_inference.py `
  --checkpoint ".\weights\CCD1\model.pt" `
  --image-dir ".\examples" `
  --mask ".\masks\CCD1.png" `
  --device auto
```

结果默认写入本交付包的 `results/inference_<随机标识>/`，包含 JSON、CSV、异常数组和可视化。
`--output-dir` 可指定其他位置。`--no-heatmaps --no-maps` 只保存 JSON/CSV。

单图和目录推理均可添加 `--threshold 0.5`，仅覆盖本次阈值，不修改权重、不重新校准；省略时沿用 checkpoint。
阈值须为有限数值（允许负数、0 或大于 1，拒绝 NaN/Inf）。`score > threshold` 为 NG，否则 OK。
判定、定位、热图和 JSON/CSV 统一使用指定阈值，分数公式不变；手动调整后不保证原校准的目标召回率。

也可指定 `--score-mode top`、`--score-mode "pool+top"` 或 `--score-mode multiscale_pool`；默认 `checkpoint` 保留原公式。
改变公式或池化参数时必须同时给出 `--threshold`，例如：

```powershell
uv run python .\run_inference.py --checkpoint ".\weights\CCD1\model.pt" `
  --image ".\examples\sample.bmp" --mask ".\masks\CCD1.png" `
  --score-mode "pool+top" --score-pool-kernel 21 --score-topk-ratio 0.001 --threshold 0.5
```

池化核和 Top-K 比例仅可与显式 `pool+top` 一起使用；正奇数核、比例范围 `(0, 1]`。
多尺度必须使用 checkpoint 已保存的归一化基准，缺失时须在原项目重新校准，不能用待测图片估计。
score、定位及所有结果统一使用本次计算方式；JSON 的 `score_method` 记录实际参数，不修改原权重。

## 5. 交付和验收

- 交付整个本目录，不必交付原项目、训练数据、缓存或虚拟环境。
- 必须补齐生产权重及对应 mask；空目录不代表已经包含可用模型。
- 多个 CCD 分别提供权重和 mask，不会自动根据图片所在目录切换模型。
- 先看 `docs/HANDOVER.md` 完成交付检查。
- 源码更新后，在原项目运行 `python tools/build_inference_delivery.py` 同步快照；不要手工改 `src/` 后再运行构建，否则更改会被替换。
- 模型推理不下载预训练权重，但首次依赖安装需要联网；离线交付需另行准备匹配平台的安装包。

