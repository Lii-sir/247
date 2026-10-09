# 交付检查清单

## 交付人

- [ ] `src/ccd_efficientad/` 和其中的 `models/` 已完整生成。
- [ ] 提供已校准权重，明确模型适用的 CCD、工件和输入条件。
- [ ] 提供与训练/阈值校准相同的 mask（原训练无 mask 时不必提供）。
- [ ] mask 路径使用交付目录内路径，不要求接收人保留你的 Linux/Windows 路径。
- [ ] 提供可公开的正常/异常验收图片，以及本次权重的预期分数/OK/NG；不要伪造检测精度。
- [ ] 接收平台支持 Python 3.12；按 pyproject.toml / uv.lock 安装环境。
- [ ] 在新目录或接收方环境中实际运行一次，而不是只验证原项目。
- [ ] 对比 score、threshold、score_mode、mask 与验收记录。
- [ ] 发包前移除不适合对外公开的示例、运行结果和敏感信息。

## 文件依赖

| 文件/目录 | 为什么要保留 |
|---|---|
| `src/ccd_efficientad/inference.py` | 单图/目录推理、输出和 API |
| `src/ccd_efficientad/cli.py` | 模型恢复、预处理及整图 score 公式 |
| `src/ccd_efficientad/data.py` | 被模型恢复入口顶层导入，即使不训练也需要 |
| `src/ccd_efficientad/mask.py` | mask 和 mask-aware pooling |
| `src/ccd_efficientad/localization.py` | 空间响应、连通域及定位框 |
| `src/ccd_efficientad/report.py` | 默认热图输出 |
| `src/ccd_efficientad/paths.py` | 源码与运行资源的路径定义 |
| `src/ccd_efficientad/models/` | 模型定义、Lightning 包装及来源信息 |

虽复用含训练函数的模块，本包不提供训练/多卡/GUI 的完整依赖集合；只通过根目录的 run_inference.py 进行推理。

## 可选相机配置

把 `configs/circle_config.example.json` 复制为 `configs/circle_config.json`，修改相机名和 mask 路径。
配置中的相对路径以 JSON 所在目录为基准，例如 `../masks/CCD1.png`。

```powershell
uv run python .\run_inference.py `
  --checkpoint ".\weights\CCD1\model.pt" `
  --image-dir ".\examples" `
  --circle-config ".\configs\circle_config.json"
```

`--mask` 与 `--circle-config` 二选一。checkpoint 原 mask 路径不可用时不会静默取消忽略区域；改用全黑 mask 可能需要重新校准。

## 结果解释

- `score > threshold` 为 NG，等于阈值为 OK。
- 可用 `--threshold 0.5` 覆盖本次推理阈值，默认沿用 checkpoint；不修改权重或分数公式，不保证原目标召回率。
- score 不是概率，默认沿用权重定义；可用 `--score-mode` 选择其他公式，但实际改变公式/池化参数时必须指定 `--threshold`，不能套用旧阈值。
- `pool+top` 可调整 `--score-pool-kernel` / `--score-topk-ratio`；多尺度仅复用模型已有的归一化参数。`score_method` 记录本次实际参数。
- `prediction.json` 包含原图坐标框；PNG 与二值异常 mask 是定位启发式，不是像素真值。
- NPY 保留未遮挡的有符号异常图；ignore mask 单独输出。
- 整批完成才有 summary.json。损坏图片或配置错误会报错，前面已经生成的单图文件不代表整批成功。
- checkpoint 通常还包含 manifest/config 中的原路径；即使不交付训练图片，也应在对外发包前检查其中是否有敏感信息。

