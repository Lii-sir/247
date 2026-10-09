# 加载权重进行 EfficientAD 推理

入口命令：`python run.py infer`；核心实现位于 `ccd_efficientad/inference.py`。

这是一个**独立命令行入口**，复用本项目 `ccd_efficientad/cli.py`、`models/`、`mask.py`、`localization.py` 和 `report.py`。它不是复制一个 Python 文件就能脱离项目运行的单文件部署包；目录迁移见 `docs/project_structure.md`。

当前项目采用 **EfficientAD 的 Teacher / Student / AutoEncoder**，包括 PDN 和 ResNet 变体；不是 PatchCore，也不需要加载 memory bank。推理不训练、不访问原训练/验证集、不重新选择阈值，也不下载预训练教师或 ImageNette。

## 1. 使用什么权重

使用本项目训练结束生成的：

```text
D:\python_programs\LXD_project\247\outputs\CCD1\实际运行目录\model.pt
```

或者用 `run.py evaluate --score-mode ...` 重新校准后生成的 `model.pt`。文件名不决定兼容性，**内容格式**必须由本项目 `save_checkpoint()` 保存：

| 字段                   | 用途                                                                     |
| ---------------------- | ------------------------------------------------------------------------ |
| `format_version = 1` | 检查权重包格式                                                           |
| `model_state`        | Teacher、Student、AE 权重，以及教师均值/标准差、异常图分位数             |
| `config`             | backbone、训练输入尺寸、ResNet 版本、输出边界模式、Teacher 激活、mask 等 |
| `calibration`        | 整图阈值、score 定义、多尺度归一化参数、热图显示范围                     |
| `manifest.category`  | 模型所属相机/类别                                                        |

**不支持直接传入：**

- 未校准的 `checkpoints/last.pt`：通常 `calibration` 为 `None`，会明确拒绝。
- 仅预训练 Teacher 的 `.pth`、仅 `state_dict` 的文件、PatchCore 权重。
- anomalib / Lightning 生成的 `.ckpt`：其包装格式、预处理和阈值信息与本项目 CLI 不同；不能仅改扩展名使用。

权重使用 `weights_only=True` 读取并严格加载参数。缺少统计量、统计量退化、NaN/Inf、架构不匹配均应先修正，不能忽略后继续输出 OK/NG。

## 2. 环境准备

PowerShell：

```powershell
cd D:\python_programs\LXD_project\247
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --locked
uv run python .\run.py infer --help
```

环境已安装时可直接使用：

```powershell
.\.venv\Scripts\python.exe .\run.py infer --help
```

依赖版本以项目的 `pyproject.toml` / `uv.lock` 为准，不需要为了推理升级模型库。

## 3. 单张图片推理

将以下两个变量替换成**实际存在**的权重和原始图片路径：

```powershell
$checkpoint = "D:\python_programs\LXD_project\247\outputs\CCD1\实际运行目录\model.pt"
$image = "D:\datasets\待检测\sample.bmp"

uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image $image `
  --device auto `
  --output-dir "D:\python_programs\LXD_project\247\outputs\inference"
```

- `auto`：CUDA 可用时选第一张可见 GPU，否则 CPU。
- `cpu`：明确使用 CPU。
- `cuda` 或 `0`：第一张可见 GPU；`1` 表示第二张。一次只支持一个设备，编号受 `CUDA_VISIBLE_DEVICES` 影响。
- 不需要再指定 `--backbone` / `--image-size`；score 默认从权重恢复，也可按下文显式选择。
- 不要把热图拼图作为输入，应输入拍摄的原始图像。

### 手动指定 threshold

单图和文件夹推理都可添加 `--threshold`，例如：

```powershell
uv run python .\run.py infer --checkpoint $checkpoint --image $image --threshold 0.5
```

- 不指定时，沿用 checkpoint 中保存的阈值；指定后仅覆盖本次推理，不修改原 `model.pt`，也不重新校准。
- 阈值必须是有限数值，拒绝 NaN/Inf；score 不是概率，因此允许负数、0 和大于 1 的阈值。
- OK/NG 判定、定位框、二值异常 mask、热图以及 JSON/CSV 都使用覆盖后的阈值；score 公式与归一化参数不变。
- `score > threshold` 为 NG，否则 OK。增大阈值会减少 NG，降低阈值会增加 NG；手动调整后不再保证原校准时的目标召回率。
- 负阈值的空间定位仍遵循项目既有的下限 0 规则，避免整幅图被选中；整图分类使用指定的原始阈值。

### 指定 score 计算方式

单图和目录推理均支持 `--score-mode`：

| 参数值 | 计算方式 |
|---|---|
| `checkpoint`（默认） | 完整沿用模型保存的 score 公式及参数 |
| `top` | mask 后最大单像素异常值 |
| `pool+top` / `pool_topk` | mask-aware 单尺度平均池化后 Top-K 均值 |
| `multiscale_pool` | 使用模型保存的正常集基准进行多尺度归一化融合 |

```powershell
# 改为单像素最大值，必须为新公式指定阈值。
uv run python .\run.py infer --checkpoint $checkpoint --image $image --score-mode top --threshold 0.5

# 单尺度池化 + Top-K，可选调整正奇数池化核和 (0, 1] 的 Top-K 比例。
uv run python .\run.py infer --checkpoint $checkpoint --image $image `
  --score-mode "pool+top" --score-pool-kernel 21 --score-topk-ratio 0.001 --threshold 0.5
```

- **实际改变 score 公式或池化参数时，必须同时指定 `--threshold`**，否则在生成结果前报错。不自动沿用不同公式的旧阈值，也不使用待测图片重新校准。
- 显式选择与 checkpoint 相同的公式且未改变参数时，可以继续使用保存的阈值。
- `pool+top` 已是模型保存的模式时，保留其池化核和比例；从其他模式切换时按模型 config 取值，缺省为核 21、比例 0.001。可用上述两个参数覆盖；它们仅允许与显式的 `pool+top` 一起使用。
- `multiscale_pool` 必须已有 checkpoint 保存的尺度列表、Top-K 比例及归一化参数；不能从缺少这些参数的 top/pool 模型凭空切换。缺失时需先用 `evaluate --score-mode multiscale_pool` 生成对应的 `model.pt`。推理不更改多尺度尺度列表或比例，以免归一化基准失配。
- score、OK/NG、定位、热图和结果文件统一使用所选公式与实际阈值；模型网络、异常图校准统计和原 `model.pt` 不变。JSON 的 `score_method` 记录实际计算参数。
- 此操作不做效果校准，不保证原目标召回率；需要自动选择匹配阈值时仍使用 `evaluate --score-mode ...`。

## 4. 文件夹批量推理

```powershell
uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image-dir "D:\datasets\待检测" `
  --device cuda `
  --output-dir "D:\python_programs\LXD_project\247\outputs\inference"
```

递归查找 PNG、JPG/JPEG、BMP、TIF/TIFF、WEBP，不要求 `train/good` 或 `test/ng` 目录结构。模型只加载一次，之后**逐张**前向（batch size 1），不自动按不同 CCD 子目录切换权重。不同相机/工件分布应使用各自模型分别运行。

输出目录位于输入目录内部时会被排除，防止重复推理上次生成的 PNG；输出目录不能等于输入目录，也不能是输入目录的祖先。目录符号链接和 Windows junction 不递归扫描。损坏图片或无效 mask 会终止并返回非零退出码，已经写出的单图结果保留；没有 `summary.json` 时不能认定整批完成。

只要 JSON/CSV 结果、不要热图和数组：

```powershell
uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image-dir "D:\datasets\待检测" `
  --device auto `
  --no-heatmaps `
  --no-maps
```

`--no-heatmaps` 关闭可视化；`--no-maps` 关闭 NPY 和二值 mask 的保存。定位信息仍计算并写入 JSON；此选项不是对纯网络延迟的基准测试。

## 5. mask 设置，尤其是跨机器使用

本项目的 mask 是 **ignore mask**，不是缺陷标签：

- **0 / 黑色：保留，参与检测。**
- **非零 / 白色：忽略，不参与 score 和定位。**
- mask 原尺寸不同时先按原图最近邻缩放，再缩放到模型输入大小。
- mask 不会将网络输入图像涂白或置零；只用于异常响应的后处理。

默认复现 checkpoint 中的 `circle_config` / `circle_params`。嵌入的 `circle_params` 可避免再次读取原配置 JSON，但其 `default_mask` PNG 仍须存在。当前流程始终使用配置的默认 mask，不在推理期间重新找圆。

从 Linux 搬到 Windows 时，权重里的 `/media/.../mask.png` 通常不可用。此时显式指定**与训练等效**的本机 mask：

```powershell
uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image $image `
  --mask "D:\datasets\mask\CCD1.png" `
  --device auto
```

也可以指定本机 JSON：

```powershell
uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image-dir "D:\datasets\待检测" `
  --circle-config "D:\datasets\mask\circle_config.json"
```

JSON 按 checkpoint 类别读取，例如：

```json
{
  "CCD1": {
    "default_mask": "CCD1.png"
  }
}
```

这里的相对路径以 JSON 所在目录为基准。`--mask` 和 `--circle-config` 互斥，显式覆盖会给出警告。默认 mask 不可用时**不会静默退化为无 mask**。

使用 **738×1144 全黑 mask**（检测全图；先将实际文件放到 `configs/masks/`）：

```powershell
uv run python .\run.py infer `
  --checkpoint $checkpoint `
  --image $image `
  --mask "D:\python_programs\LXD_project\247\configs\masks\black_mask_738x1144.png"
```

**注意：**如果训练/校准时忽略了某些区域，改成全黑会扩大检测区域，原阈值可能失效。仅当有效区域与原校准一致时才直接复用阈值；真正改变 mask 应重新校准，不要把这一步当作单纯路径迁移。全白或缩放后遮住全部像素的 mask 会报错。

## 6. 输出是什么

每次创建独立的 `inference_<随机标识>` 目录，不覆盖上次 CLI 结果：

```text
outputs/inference/inference_<随机标识>/
├─ summary.json                    # 模型、阈值、mask 设置、图片数量、OK/NG 数量
├─ predictions.csv                 # 每图路径、分数、阈值、结果、框数、单图输出路径
└─ images/
   └─ nested/sample.bmp/           # 保留相对目录和完整文件名（含扩展名）
      ├─ prediction.json           # 完整分数、定位摘要、原图坐标框
      ├─ prediction.png            # 2×3 热图/叠加图/定位图
      ├─ prediction_scales.png     # 仅 multiscale_pool 模型生成
      ├─ anomaly_map.npy           # float32 原始异常图，模型输入尺寸，未按 mask 置零
      ├─ ignore_mask.png           # 实际使用的 ignore mask，模型输入尺寸，0/255
      └─ anomaly_mask.png          # 阈值响应二值图，恢复到 EXIF 校正后的原图尺寸
```

`sample.png` 和 `sample.bmp` 输出到不同目录，不会因为文件 stem 相同而覆盖。单图模式同样保存到 `images/<完整文件名>/`。

`prediction.json` 主要字段：

| 字段                    | 说明                                               |
| ----------------------- | -------------------------------------------------- |
| `prediction`          | `OK` 或 `NG`                                   |
| `score`               | 按本次实际 score 公式计算的最终整图异常分数         |
| `score_max`           | 有效区域异常图的单像素最大值，不一定是最终判定分数 |
| `threshold`           | 本次实际阈值：默认来自 checkpoint，可用 `--threshold` 覆盖，不在本批图片上重新估计 |
| `score_mode`          | `top`、`pool_topk` 或 `multiscale_pool`      |
| `score_method`        | 本次实际 score 定义、池化参数及可用的归一化基准 |
| `localization.boxes`  | 模型异常图坐标空间的定位框，保留原定位字段         |
| `boxes_original`      | 映射到原图的`x0, y0, x1, y1` 半开区间坐标        |
| `image_size_original` | EXIF 校正后的原图宽高                              |
| `anomaly_map_shape`   | `[高度, 宽度]`，等于权重保存的方形输入尺寸       |

判定统一为 **`score > threshold` 则 NG，否则 OK**；等于阈值时是 OK。分数不是缺陷概率，不必限制在 `[0,1]`，多尺度分数可以大于 1。异常图 NPY 保留负值，不截断、不按单图 min-max 归一化；PNG 为展示会裁剪颜色范围。

定位框和 `anomaly_mask.png` 是后处理启发式结果，**不是像素级真值，也不是保证精确的分割**。NG 但没有合格连通域时项目会使用峰值兜底框，因此框可能不完全对应二值 mask。CSV 不推断图片真实标签，也不计算 AUROC/召回率；有标签数据的效果评估请使用原 `evaluate` 入口。

## 7. 预处理和前向过程

1. 读取并 EXIF 校正，转 RGB。
2. 双线性缩放为 checkpoint 的 `image_size × image_size`；不保持长宽比、不裁 ROI、不分块。
3. 转为 float Tensor，范围 `[0,1]`，形状 `[1,3,H,W]`。**不要外部再做 ImageNet Normalize**；网络内部有归一化。
4. 恢复 Teacher、Student、AE 和统计量，调用 `eval()`，在 `torch.inference_mode()` 中前向。
5. 得到校准后的融合异常图，用保存的 mask 和保存/显式选择的 score 公式计算整图分数。
6. 用保存的阈值（或显式指定的 `--threshold`）给出 OK/NG，生成定位信息和可选输出。

ResNet 的架构版本、`valid/native` 输出边界、Teacher 最终激活沿用 checkpoint；缺少相关字段的旧权重使用项目现有兼容逻辑（版本 1、native、relu），不会自动变成新版结构。`resnet50_layer1v2` 的 Student 前向同样复用当前项目实现。

score 不是一律 `prediction.pred_score`：后者只是异常图最大值，不能替代池化/多尺度模式的最终分数。推理入口复用 `prediction_scores()`：

- `top`：mask 后的单像素最大值；缺少 `score_method` 的旧权重也按此处理。
- `pool_topk`：mask-aware 局部平均池化后 Top-K 均值。
- `multiscale_pool`：各尺度 Top-K 均值按保存的正常集基准归一化，再取尺度最大值。

不能换 score 公式后直接沿用旧 threshold。推理中切换公式须显式指定新阈值；需要自动校准匹配阈值或建立多尺度归一化参数时，用 `evaluate --score-mode ...` 生成新的 `model.pt`，再传给本脚本。

## 8. 在自己的 Python 程序中调用

在项目环境和可导入项目模块的路径中运行：

```python
from ccd_efficientad.inference import EfficientAdPredictor

# 启动阶段只构造一次；实际路径请自行替换。
predictor = EfficientAdPredictor(
    checkpoint=r"D:\models\CCD1\model.pt",
    device="auto",
    # threshold=0.5,  # 可选：仅覆盖本实例阈值；默认 None 沿用 checkpoint
    # score_mode="pool+top",  # 可选：切换公式时须同时提供 threshold
    # score_pool_kernel=21,  # 可选：仅支持显式 pool+top
    # score_topk_ratio=0.001,  # 可选：仅支持显式 pool+top
    # mask=r"D:\datasets\mask\CCD1.png",  # 可选：默认沿用训练设置
)

# 无 output_dir：只返回字典，不保存文件。
result = predictor.predict(r"D:\datasets\待检测\sample.bmp")
print(result["prediction"], result["score"], result["threshold"])
print(result["boxes_original"])

# 传 output_dir：保存单图结果。API 写入指定目录，调用方应为每图分配不同目录。
result = predictor.predict(
    r"D:\datasets\待检测\sample.bmp",
    output_dir=r"D:\results\sample.bmp",
    save_heatmaps=False,
    save_maps=True,
)
```

## 9. 验证与常见错误

```powershell
uv run python -m unittest tests.test_infer_efficientad -v
```

测试使用合成图片和随机模型权重，验证 PDN、ResNet（含 layer1v2）、权重读取、旧版兼容、三种 score、mask、原图坐标、单图/目录 CLI、数组和热图输出；**不是训练模型检测效果的证明**。

| 错误/现象                     | 处理                                                           |
| ----------------------------- | -------------------------------------------------------------- |
| 找不到权重                    | 替换示例路径，确认文件存在                                     |
| 尚未校准                      | 使用训练结束或重新校准后的`model.pt`，不是续训 `last.pt`   |
| 格式不匹配、参数键/形状不匹配 | 确认来自本项目；不能用教师单独权重或 Lightning`.ckpt` 代替   |
| 找不到配置/mask               | 保留等效 mask，使用`--mask` 或 `--circle-config` 重定位    |
| mask 遮住全部像素             | 黑色是检测区，白色是忽略区，检查是否反了                       |
| CUDA 不可用                   | 用`--device cpu`，或检查锁定环境中的 GPU 支持                |
| 异常图分位数/教师标准差退化   | 检查训练/校准统计，不要绕过验证或随意填阈值                    |
| 分数与历史报告不同            | 确认权重、原始图片、mask 和 score 模式一致；不要重复 Normalize |
| 全黑 mask 后误报增多          | 检查是否扩大了原先忽略区，必要时重新校准                       |

完整部署需保留上述项目模块和依赖；只有权重中的旧训练图片路径可以不迁移，mask 文件不能遗漏。
