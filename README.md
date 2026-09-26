# CCD 数据集 EfficientAD 实验

这个项目提供数据检查、训练、正常样本阈值校准、独立测试和单图预测脚本。代码注释使用中文，环境由 **uv** 管理。网络结构、教师网络和三项训练损失基于 **anomalib 2.2.0 的 EfficientAD 实现**；训练循环自行管理，便于固定数据快照、按步调整学习率和断点续训。

## 1. 已适配的数据目录

本机实际找到的路径为：

```text
D:\datasets\20260909_ccd1-6_ok+v5ng\
├─ CCD1\
│  ├─ train\good\*.bmp
│  └─ test\
│     ├─ good\*.bmp
│     └─ defect\*.bmp
└─ CCD2\ ...
```

用户最初提供的 `D:\datasets\20260909\_ccd1-6\_ok+v5ng` 当前不存在；脚本默认优先查找上面的实际路径。以后目录变动时，请用 `--data-root` 指定**包含 CCD1 等子目录**的路径。

当前检查到 CCD1 原始目录有 52 张训练良品、10 张测试良品、318 张测试缺陷图，示例 BMP 为 **2000×1500、RGB**。这些是下载过程中的原始计数；实际参与实验的数量以 `inspect` 输出为准，因为损坏文件和尚在写入的文件会被剔除。

`inspect` 会一次扫描并按 SHA256 汇总所有内容完全相同的图片，完整结果写入 `duplicate_report.json`。下载继续进行后计数可能改变。

每个 CCD 独立训练模型，避免不同相机画面分布混在一起。未来 `test` 下可以有多个缺陷目录，除 `good` 外均视为异常。没有像素级 mask，因此只计算图像级指标，热图用于查看异常位置。

## 2. 安装 uv 环境

本机已安装 uv。PowerShell 中执行：

```powershell
cd D:\python_programs\LXD_project\247

# 将 uv 缓存放入项目目录，避免全局 D:\uv_cache 的权限问题。
$env:UV_CACHE_DIR = "$PWD\.uv-cache"

# 使用项目锁定的 Python 3.12 和依赖版本。
uv sync --locked

# 确认当前环境能识别 GPU。
uv run python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未识别 GPU')"
```

项目固定 `anomalib==2.2.0`、`torch==2.7.1`、`torchvision==0.22.1` 和 `lightning==2.5.5`，PyTorch 使用官方 **CUDA 12.6** wheel。这个组合是固定实验基线，不是追踪 anomalib 最新版。本机为 **RTX 3060 12GB**，默认 `--device auto` 会在可用时使用 CUDA。运行预编译 PyTorch wheel 不需要额外安装 CUDA Toolkit；首次同步会下载体积较大的 GPU 依赖。

新开 PowerShell 窗口时可再次设置上面的 `UV_CACHE_DIR`。也可以逐条使用 `uv --cache-dir .uv-cache run ...` / `uv --cache-dir .uv-cache sync --locked`，命令行参数优先于已有全局变量。

## 3. 先检查 CCD1

```powershell
uv run python efficientad_ccd.py inspect --category CCD1

# 显式指定数据根目录的等价写法。
uv run python efficientad_ccd.py inspect --data-root "D:\datasets\20260909_ccd1-6_ok+v5ng" --category CCD1
```

检查会保存 `outputs/inspection/CCD1/<时间>/manifest.json`，其中包含文件清单、原始尺寸、SHA256、跳过原因和最终划分：

- 仅使用 `train/good` 的正常图。固定随机种子 42，默认约 80% 用于训练、20% 留作正常验证。
- `test/good` 和 `test/defect` 留给最终评估，不用于拟合网络、异常图分位数或判定阈值。
- 默认跳过最近 **60 秒**写入的文件、临时文件、不能完整解码的图片，以及扫描期间大小/修改时间变化的文件。
- 每个文件只计算一次 SHA256，再按哈希一次性汇总全部重复组，包括训练集内部、测试集内部、train/test 交叉重复和标签冲突；`inspect` 保存并打印报告，`train` 在保存完整报告后停止，避免数据泄漏。
- 每个运行使用固定快照。训练过程中出现的新图片不会悄悄加入，快照内图片发生变化会报错。

至少需要 4 张不同内容的训练良品，训练与验证各至少 2 张。这只是程序可运行下限；要判断效果，应准备更多有代表性的正常样本。哈希识别的是字节完全相同的文件，连续拍摄的近似图、同一工件的不同编码副本仍需按工件/批次人工检查隔离。

由于数据还在下载，建议先只运行 CCD1。全目录检查可使用 `--category all`；未完成的相机会明确报错，已完成相机的检查结果仍保留。扫描对原始数据只读，首次完整解码和计算哈希会读取数 GB 数据。

## 4. 训练并自动评估

先用 1,000 步确认效果输出和流程：

```powershell
uv run python efficientad_ccd.py train --category CCD1 --max-steps 1000 --image-size 256 --heatmaps 32
```

进行较完整的初步实验：

```powershell
uv run python efficientad_ccd.py train --category CCD1 --max-steps 10000 --image-size 256
```

考虑到原图为 2000×1500，细小缺陷缩到 256×256 后可能不明显，可再运行 512 分辨率对照：

```powershell
uv run python efficientad_ccd.py train --category CCD1 --max-steps 10000 --image-size 512
```

1,000 步仅用于快速试跑，10,000 步也不保证收敛。需要更长训练时可设 `--max-steps 70000`；在相同数据快照、相同训练参数下比较效果更有意义。默认将整张图缩放为方形，保留整张画面但会改变长宽比；本脚本未实现 ROI 或分块。若小缺陷在缩放后消失，后续应根据实际缺陷位置增加 ROI/分块处理。

训练默认使用 **batch size=1**、Adam、初始学习率 `1e-4`、权重衰减 `1e-5`；在总步数 95% 处将学习率乘以 0.1。输入只转 RGB、缩放和映射到 `[0,1]`，不能在外部再做 ImageNet Normalize。默认 backbone 是 `pdn_small`，也支持 `pdn_medium`、`resnet18_layer2`、`resnet18_layer3`、`resnet50_layer1`、`resnet50_layer2` 和 `resnet50_layer3`。PDN 使用对应的预训练教师权重；ResNet 默认使用 torchvision 的 ImageNet 预训练主干（ResNet-18 V1、ResNet-50 V2），第一次运行会下载对应权重。

可以用同一份数据快照比较不同 backbone：

```powershell
uv run python efficientad_ccd.py train `
  --category CCD1 `
  --backbone resnet18_layer2 `
  --max-steps 10000 `
  --image-size 256 `
  --circle-config "D:\python_programs\LXD_project\247\circle_config.json"
```

`--backbone` 与旧版 `--model-size` 互斥，不能同时传入；未指定 `--backbone` 时，`small/medium` 分别映射到 `pdn_small/pdn_medium`。续训默认沿用 checkpoint 的结构，显式指定不同 backbone 会报错；切换结构必须重新训练并重新校准阈值。

ResNet 新训练默认采用架构版本 2：teacher 保留原生预训练特征并冻结；student 的 stem 和所有保留的残差 stage 通道整体扩大 2 倍，末尾使用无激活的 `3×3` 输出卷积；AE 保留卷积编码、插值解码和 Dropout 的形式，中间容量按 teacher 通道数扩展，直接输出 teacher 的空间尺寸。256×256 输入时：

| Backbone | Teacher / student / AE 通道 | 特征图 | AE hidden |
|---|---|---|---:|
| resnet18_layer2 | 128 / 256 / 128 | 32×32 | 64 |
| resnet18_layer3 | 256 / 512 / 256 | 16×16 | 128 |
| resnet50_layer1 | 256 / 512 / 256 | 64×64 | 128 |
| resnet50_layer2 | 512 / 1024 / 512 | 32×32 | 256 |
| resnet50_layer3 | 1024 / 2048 / 1024 | 16×16 | 256 |

ResNet-50 layer1/layer2 分别只保留 3 个、3+4 个 Bottleneck；student 的块内通道和残差支路同步扩宽，不包含后续 stage。AE 按实际特征尺寸解码，无额外 PDN 边界补零。详见 [ResNet 容量升级说明](docs/resnet_capacity_upgrade.md)。

这是实验性 EfficientAD 变体。Student 的 BatchNorm 使用每卡局部统计；layer3 空间分辨率较低，整体扩宽不能保证提高小缺陷召回。ResNet-50 layer3 student 约 7184 万参数；layer1/layer2 参数更少，但特征图更大，仍需实测显存。旧 checkpoint 缺少 `resnet_architecture_version` 时自动按版本 1 原结构加载，可继续旧训练；要使用新结构，应启动新训练并重新校准，不能把旧权重直接装进版本 2。

ResNet 通道统计使用双精度累积；若部分 ReLU 通道在训练集上恒定，则只减均值、使用单位标准差并给出警告，避免除零。全部通道恒定仍会报错。PDN 保留原有统计行为。

当前也支持显式批量训练。例如：

```powershell
uv run python efficientad_ccd.py train --category CCD1 --batch-size 4 --max-images 70000
```

`batch-size` 是**每张卡**的训练 batch，正常样本统计、异常图校准、测试和热图仍使用单张图片。每卡 `batch-size` 大于 1 或使用多卡时，程序启用逐图片 Q99.9 hard loss，并让辅助 loader 使用相同的每卡 batch。单卡 `batch-size=1` 保留原始全局 hard loss 和旧 checkpoint 语义。

`--max-images` 不能被 batch 整除时，实际图片预算会向上取整到完整 batch；例如 `--batch-size 4 --max-images 101` 会执行 26 步、实际预算 104 张。`--device auto` 会优先使用 `cuda:0`，也可以明确指定 `--device cuda` 或 `--device cpu`；`--num-workers` 同时用于训练、评估、统计和 ImageNette 辅助 loader。默认 score 选取方式是多尺度池化 `multiscale_pool`，使用 `1,7,21` 三个窗口和 `0.001` 的 Top-K 比例。训练时 `--score-pool-kernel`（单尺度）与 `--score-pool-kernels`（多尺度）只能二选一。

### 单机多卡训练

```powershell
uv run python efficientad_ccd.py train `
  --category CCD1 `
  --device 0 1 `
  --batch-size 4 `
  --backbone resnet18_layer2 `
  --max-images 70000 `
  --imagenette-dir "assets\visa_aux" `
  --circle-config "circle_config.json"
```

此命令由两张卡共同训练一个模型：每卡 4 张，全局 batch 为 8，执行 8,750 个 optimizer steps。`--max-images` 是所有卡合计的图片预算，向上取整到完整全局 batch；例如两卡、每卡 4 张、预算 101 张，实际执行 13 步、104 张。`--max-steps` 则始终是同步优化步数。学习率保持 `--lr` 指定值，不自动按卡数放大。

`--device 1` 只使用编号 1 的 GPU；`auto`/`cuda` 仍只选第一张可见 GPU，`cpu` 为单进程。编号相对于 `CUDA_VISIBLE_DEVICES` 筛选后的可见设备，重复、越界或混合 CPU/GPU 会报错。独立评估和预测只接受一个设备，例如 `--device 0`。

多卡通过 DDP 一卡一进程训练，训练集和辅助集分别分片，不补重复图片，每轮丢弃不完整的全局 batch；两套数据各自必须至少包含一个全局 batch。`--num-workers` 是每个进程中每个 loader 的 worker 数，Windows 建议先用 0。教师统计在启动前统一计算；训练结束后只在主进程进行完整校准、评估与报告保存。ResNet 学生使用每卡本地 BatchNorm，DDP 在每次前向前同步 rank 0 的缓冲区，最终保存 rank 0 状态；未启用 SyncBatchNorm，因此它与同全局 batch 的单卡训练不保证数值等价。

续训必须保持原来的卡数，每卡 batch、优化器和总步数沿用 checkpoint，允许换用其他 GPU 编号。旧 checkpoint 视为单卡；已训练的多卡 `model.pt` 可在单 GPU 或 CPU 推理。多卡中断时保留最近一次成功写入的 `checkpoints/last.pt`，保存间隔由 `--save-every` 控制。多卡恢复会分别恢复训练集和辅助集的 sampler epoch，并跳过本轮已处理的 batch；因此样本顺序可以接续，但随机增强和 Dropout 的 RNG 状态未保存，仍不保证与不中断训练逐位一致。

优先使用 NCCL（通常是 Linux CUDA），不可用时使用 Gloo。本功能针对 `efficientad_ccd.py` 单机训练；未扩展为多机训练，也不代表独立 Lightning/Engine 入口已完成 DDP 适配。

训练结束后会自动完成：正常验证集异常图校准 → 图像阈值校准 → 测试集推理 → 指标和热图保存。默认每 1,000 步保存 `checkpoints/last.pt`；Ctrl+C 在训练循环中会保存最近完成步数的续训文件。

### 首次训练需要下载的资源

除了 uv 环境，EfficientAD 还需要：

1. 官方预训练教师权重，默认缓存在 `assets/pre_trained/efficientad_pretrained_weights/`。
2. **ImageNette 辅助图像集**，默认自动下载官方完整归档（约 1.5 GB），用于学生网络的正则项，缓存在 `assets/imagenette/`。它不是你的 CCD 测试集。

下载使用 anomalib 的官方地址和校验信息。有现成文件时可以显式指定：

```powershell
uv run python efficientad_ccd.py train --category CCD1 --max-steps 10000 --teacher-weights "D:\models\pretrained_teacher_small.pth" --imagenette-dir "D:\datasets\imagenette2\train"
```

这里的路径是示例，必须改成实际存在的位置。教师权重必须与所选 `--backbone` 匹配；`pdn_small/pdn_medium` 使用官方对应权重，三个 ResNet 选项均可以省略该参数并自动使用 torchvision 的 ImageNet 权重。辅助图片目录需要兼容 `ImageFolder`，例如 `train/n01440764/*.JPEG`。仅有 CCD 图片不足以复现官方带 ImageNette 正则项的配置。**不要预先创建空的 `assets/imagenette` 目录**：官方辅助函数看到目录存在就会尝试读取，空目录应删除或改用新的有效目录后重试。

ResNet 自定义 `--teacher-weights` 应保存本项目适配器的 `model.teacher.state_dict()`（Lightning 包装实例为 `model.model.teacher.state_dict()`）；完整 torchvision ResNet 的原始 state dict 键名与其不同，不能直接传入。已训练的 `model.pt` 包含教师参数，推理和 CLI 续训不需要重新下载教师。

直接使用 `self_efficientad.EfficientAd` 的 Lightning 入口时，学习率也按 optimizer step 调度；`EfficientAd.load_from_checkpoint(...)` 从 checkpoint 恢复教师，即使保存时设置了 `teacher_pretrained=True`，也不再读取外部预训练权重。

### 使用 VisA 作为工业辅助数据集

如果不使用 ImageNette，也可以下载 VisA 的训练正常图片作为辅助数据。项目中的 `download_visa_aux.py` 会读取官方 `VisA_20220922.tar`，只保留 `train` 且标签为 `normal/good` 的图片，并整理为当前 `ImageFolder` 需要的格式：

```text
D:\datasets\visa_aux\
├─ candle\*.JPG
├─ capsules\*.JPG
└─ ...
```

PowerShell 下载并整理命令如下；归档路径放在输出目录之外：

```powershell
uv run python download_visa_aux.py `
  --output "D:\datasets\visa_aux" `
  --archive "D:\datasets\VisA_20220922.tar"
```

脚本默认使用 VisA 官方地址，并在输出目录写入 `manifest.json`。它不会把 VisA 的异常测试图放入辅助集；`--exclude-class candle` 可以排除某个类别，`--max-per-class 500` 可以限制每类图片数量。下载失败时，也可以手动取得 `.tar` 文件后重新执行同一命令，脚本会通过 `--archive` 直接读取本地归档。

整理完成后，把输出目录传给现有的兼容参数 `--imagenette-dir` 即可；参数名沿用旧版，但底层只要求 `ImageFolder` 目录：

```powershell
uv run python efficientad_ccd.py train `
  --data-root "D:\datasets\20260909_ccd1-6_ok+v5ng" `
  --category CCD1 `
  --imagenette-dir "D:\datasets\visa_aux" `
  --circle-config "D:\python_programs\LXD_project\247\circle_config.json"
```

## 5. 读取效果报告

每次运行单独保存，例如：

```text
outputs/CCD1/<运行时间>/
├─ manifest.json          # 当次实际使用的样本及固定划分
├─ config.json            # 超参数和设备配置
├─ loss.csv               # 每一步的总损失和三项损失
├─ checkpoints/last.pt    # 模型、优化器和调度器，可用于续训
├─ model.pt               # 已校准模型；包含教师参数，可独立推理
├─ calibration.json       # 正常验证分数、阈值和异常图分位数
├─ metrics.json           # 图像级指标与混淆矩阵
├─ inference_speed.json   # 模型+score 与验证端到端的耗时、FPS
├─ predictions.csv        # 每张图片的标签、分数、预测和是否正确
├─ score_distribution.png # 正常/异常分数分布
└─ heatmaps/               # 原图、异常热图、叠加图，按测试子目录/预测结果分类
   ├─ good/
   │  ├─ normal/*.png      # test/good 中预测为正常
   │  └─ anomaly/*.png     # test/good 中被误报为异常
   ├─ defect1/
   │  ├─ normal/*.png      # test/defect1 中被漏检
   │  └─ anomaly/*.png     # test/defect1 中预测为异常
   └─ defect2/             # 其他 test 子目录同样保留
```

验证结束后终端会显示平均检测毫秒数和 FPS。`model_and_score` 包含模型前向与 mask 后整图 score，不含图片读取、缩放和热力图保存；`evaluation_total` 还包含测试 DataLoader 读取，但仍不包含报告和热力图保存。同一统计也保存在 `inference_speed.json`，并写入 `metrics.json` 的 `inference_speed` 字段。

整图 `score` 默认使用 `1,7,21` 三个尺度：各尺度先在有效区域进行局部平均池化并取最高 0.1% 位置的均值，再使用正常验证集各自的中位数与 Q99 归一化，最终取三个归一化分数的最大值。`predictions.csv` 同时保存各尺度的原始分数、归一化分数和 `score_max`，便于追踪触发异常的尺度。原始 `test` 会按 `good/缺陷类别` 分层划出 20% 的 `threshold_val`，阈值按每种异常子目录分别约束，选择令所有异常类型达到 `target_recall`（默认 99%）的最高值，剩余图片才作为最终测试集。可以通过 `--threshold-val-ratio`、`--target-recall`、`--score-pool-kernels 1,7,21` 和 `--score-topk-ratio` 调整。

重点看 `roc_auc`、`average_precision`、异常召回率 `recall`、正常误报率 `false_positive_rate` 和漏检率 `false_negative_rate`。当前正常测试图只有 10 张，误报 1 张就会改变误报率 10 个百分点；异常图明显更多，单看 accuracy 容易误判。

尺度归一化只使用训练良品中留出的正常验证集；最终阈值则使用独立的带标签 `threshold_val`。程序先对每种异常子目录求出满足目标召回率的边界，再取其中最低的边界作为统一阈值；`score > threshold` 判为 NG，否则 OK。阈值验证集同时会统计正常误报率。分数不是概率，也不保证位于 `[0,1]`，最终测试集不参与尺度归一化或阈值选择。

热图统一使用正常验证集确定的显示色阶，默认优先保存误判图，再保存接近阈值的图。主可视化采用 2×3：上排为原图、原始异常热力图、原始 Overlay；下排为当前 Score 对应的定位响应、阈值二值图、定位 Overlay 与异常框。`top` 使用原始异常图，`pool+top` 使用与 Score 完全相同的 mask-aware 池化图，`multiscale_pool` 只融合整图尺度分数超过阈值的归一化尺度。多尺度模式还会在 `heatmap_scales/` 保存逐尺度池化图与 Overlay 诊断图。

主结果按测试集原始子目录和整图预测结果保存：`heatmaps/<test子目录>/normal/` 或 `heatmaps/<test子目录>/anomaly/`。例如 `test/defect1` 的图片会进入 `heatmaps/defect1/normal/` 或 `heatmaps/defect1/anomaly/`；逐尺度诊断图保持相同层级放在 `heatmap_scales/`。`--heatmaps 32` 是所有子目录合计最多 32 张主图，`--heatmaps 0` 不保存，`--heatmaps -1` 保存全部。预测为正常的图片默认不画框；预测异常但形态学和面积过滤后没有连通域时，会在最强响应位置生成兜底框；响应完全平坦时改用有效区域中心，避免固定落在左上角。热图和异常框均为模型定位的启发式展示，不是像素标注或经过像素指标验证的分割结果。

当校准边界为 0 时，严格大于判定会令整图分类阈值成为一个极小负数。由于异常定位响应通常大于等于 0，三种 Score 模式的画框阶段都会把空间阈值下限限制为 0，避免整幅图被选中；整图分类仍使用 checkpoint 中的原始阈值。该调整会记录在单图 `prediction.json` 的 `threshold_adjusted_for_localization` 字段中。

画框参数在训练、评估和单图预测命令中复用，并保存到 `config.json` 的 `localization` 字段。常用参数是 `--box-min-area-ratio`（最小连通域面积比例）、`--box-morph-kernel`、`--box-open-iterations`、`--box-close-iterations`、`--box-merge-iou`、`--box-merge-containment`（小尺度框被大尺度框覆盖时的合并比例）、`--box-merge-distance-ratio` 和 `--box-padding-ratio`。命令行未指定时沿用 checkpoint；旧 checkpoint 自动补齐默认值。

## 6. 重跑评估、预测和续训

以下命令中的 `<运行时间>` 必须替换为实际目录名。

```powershell
# 完整复现 checkpoint 保存时的 score 方式和 threshold（默认）。
uv run python efficientad_ccd.py evaluate --checkpoint "outputs\CCD1\<运行时间>\model.pt" --heatmaps -1

# 单像素最大值（Git 中“单像素最大作为 score”的方式）。
uv run python efficientad_ccd.py evaluate --checkpoint "outputs\CCD1\<运行时间>\model.pt" --score-mode top --heatmaps -1

# 单尺度局部平均池化 + Top-K 均值。
uv run python efficientad_ccd.py evaluate --checkpoint "outputs\CCD1\<运行时间>\model.pt" --score-mode "pool+top" --score-pool-kernel 21 --score-topk-ratio 0.001 --heatmaps -1

# 多尺度池化；各尺度用正常验证集归一化后取最大值。
uv run python efficientad_ccd.py evaluate --checkpoint "outputs\CCD1\<运行时间>\model.pt" --score-mode multiscale_pool --score-pool-kernels 1,7,21 --score-topk-ratio 0.001 --heatmaps -1

# 下载完成后先 inspect，生成新快照，再评估其中的测试图片。
uv run python efficientad_ccd.py evaluate --checkpoint "outputs\CCD1\<运行时间>\model.pt" --manifest "outputs\inspection\CCD1\<检查时间>\manifest.json"

# 单张新图推理，输出 prediction.json、2×3 prediction.png；多尺度模型另存 prediction_scales.png。
uv run python efficientad_ccd.py predict --checkpoint "outputs\CCD1\<运行时间>\model.pt" --image "D:\datasets\新图片.bmp"

# 中断后续训：沿用 checkpoint 内的数据快照、分辨率和总训练步数。
uv run python efficientad_ccd.py train --resume "outputs\CCD1\<运行时间>\checkpoints\last.pt"
```

`checkpoint` 是 `--score-mode` 的默认值，严格沿用模型里保存的分数公式和阈值。显式选择 `top`、`pool+top` 或 `multiscale_pool` 时，程序使用同一份 `threshold_val` 为所选公式重新选择匹配阈值，并在本次 `evaluation` 目录保存 `calibration.json`、`config.json` 和新的 `model.pt`；原 checkpoint 不会被覆盖。若快照没有 `threshold_val`，程序会从 `test` 各子目录分层划出一部分，因此用于最终报告的测试图片会相应减少。

续训结果放入新的运行目录，不覆盖原结果。续训恢复模型、优化器和调度器；单卡会重新开始随机数据顺序，多卡会接续两套数据的分片顺序。随机增强和 Dropout 的 RNG 状态未保存，因此不保证与不中断训练逐位相同。若要改变总步数、分辨率、阈值分位数、加入新下载的训练图片或改用另一份辅助数据，请启动一次新的训练；`--resume` 会沿用原配置，只允许改变运行设备（卡数保持一致）、数据读取进程数、保存频率和热图数量。

新快照评估不会因内容与原训练/校准样本重复而拒绝；相机类别仍必须一致。快照内文件需要保持路径、大小和修改时间不变。

等 CCD1～CCD6 都下载完整并检查通过后，可依次训练：

```powershell
uv run python efficientad_ccd.py inspect --category all
uv run python efficientad_ccd.py train --category all --max-steps 10000 --image-size 256
```

`all` 按目录顺序串行运行，每个相机各有自己的权重和报告。训练遇到未就绪类别会停止并报错；下载阶段请选择已经完整的相机。

## 7. 验证脚本与常见参数

本次交付已完成：真实 CCD1 数据扫描，27 项数据/评估/命令行测试，以及合成数据的训练、校准、保存/加载、评估、热图、单图预测和恢复后继续优化的完整链路验证。完整链路使用本机已有 **Python 3.14 / anomalib 2.3.3 / PyTorch 2.11 CPU** 环境作为补充检查，随机教师权重只用于验证程序。

2026-09-11 后续修复已将 anomalib 间接导入所需的 `requests` 加入项目依赖和锁文件。**项目锁定的 Python 3.12 / anomalib 2.2.0 / PyTorch 2.7.1 CUDA 12.6 环境现已安装完成，并识别 RTX 3060。** 已在该环境通过合成数据的 GPU 训练、校准、保存/加载、评估、热图、单图预测及恢复后继续训练的完整链路测试。合成测试使用随机教师，只验证程序运行；本次没有用官方预训练教师完成 CCD1 的正式训练，因此尚无可信的 CCD1 检测分数。

```powershell
uv run python -m unittest discover -s tests -v

# 可选：用合成数据和随机教师测试完整链路，不下载模型，不代表实际检测效果。
uv run python tests/smoke_pipeline.py

# ResNet 完整链路：需要本机已缓存 torchvision ImageNet 预训练教师权重。
uv run python tests/smoke_pipeline.py --backbone resnet18_layer2
uv run python tests/smoke_pipeline.py --backbone resnet50_layer3
uv run python tests/smoke_pipeline.py --backbone resnet50_layer1
uv run python tests/smoke_pipeline.py --backbone resnet50_layer2

# 测试专用：在 CPU 上运行两个真实 DDP 进程，验证完整流程。
uv run python tests/smoke_pipeline.py --ddp-test-device cpu

# 测试专用：同一张 GPU 上运行两个进程，验证 Windows Gloo/CUDA 和 DataLoader。
# 不代表两张物理 GPU 的性能验证；正式训练禁止重复选择同一 GPU。
uv run python tests/smoke_pipeline.py --ddp-test-device cuda:0 --num-workers 1

uv run python efficientad_ccd.py --help
uv run python efficientad_ccd.py train --help
```

- Windows 默认 `--num-workers 0` 便于定位读取问题；稳定后可尝试 `--num-workers 2`。
- `--device cuda` 要求 CUDA 可用，否则直接报错；`--device cpu` 可用于调试，但 GPU 环境依赖仍会安装。
- 默认 `--min-age-seconds 60`；确认所有文件已完整下载后可以设 0。
- 样本不足、图片损坏、重复内容、校准分位数退化或出现 NaN/Inf 会明确报错；重复内容的全部路径保存在运行目录的 `duplicate_report.json`。
- 只有一种测试类别时 AUROC 无定义，报告记录为 `null`；没有测试图时仍会保存训练完成的模型与校准结果。

```
uv run python efficientad_ccd.py train `
  --data-root "D:\datasets\20260909_ccd1-6_ok+v5ng" `
  --category CCD1 `
  --model-size small `
  --image-size 512 `
  --max-steps 20000 `
  --device cuda `
  --num-workers 0 `
  --save-every 1000 `
  --heatmaps 32


  $env:UV_CACHE_DIR=".uv-cache"

# 无需修改 JSON：打开界面，拖框选择 ROI，检测并保存 mask。
uv run python circle_mask_gui.py
# 也可打开整个图片文件夹，选定统一 ROI 后批量检测，再用按钮或左右方向键逐张查看。

uv run python generate_circle_mask.py `
  --input "D:\datasets\20260913_caijian_liugongwei\2lixiaodong\CCD1\good\B20260731_01_CCD1_0003.bmp" `
  --output-dir "circle_mask_generated_CCD1" `
  --circle-config "circle_config.json" `
  --category CCD1

uv run python detect_background_circle.py `
  --input "D:\datasets\20260913_caijian_liugongwei\2lixiaodong\CCD1\good" `
  --output-dir "circle_test_bmp_tight" `
  --roi "0.58,0.33,0.28,0.36" `
  --detection-method hybrid `
  --circle-target outer `
  --group-target best_score `
  --min-radius-ratio 0.15 `
  --max-radius-ratio 0.55 `
  --dark-threshold-offset 0 `
  --morph-kernel 5 `
  --min-axis-ratio 0.65 `
  --min-contour-score 0.40 `
  --param1 100 `
  --param2 24 `
  --mask-radius-scale 0.95 `
  --mask-margin 2



CUDA_VISIBLE_DEVICES=1 python efficientad_ccd.py evaluate \
  --checkpoint "/media/pe/5fe0ba86-cd64-483b-bfc5-dd83088ea652/lxd/outputs/CCD1-v2/<时间目录>/model.pt" \
  --output-dir "outputs" \
  --heatmaps -1 \
  --device cuda \
  --num-workers 0
```

目录输入会递归检测，并在输出目录中保留输入图片的相对子目录，避免不同缺陷目录中的同名图片互相覆盖。输出目录可以放在输入目录下，脚本会排除该目录；但输入目录与输出目录不能完全相同。

`generate_circle_mask.py` 从一张参考图生成原图尺寸的单通道 `default_mask.png`：圆内为 255、圆外为 0。同时输出检测叠加图、白色填充预览和带检测耗时的 `circle_mask.json`。确认结果后，可将该 PNG 配置为对应型号的 `default_mask`。

圆检测界面固定使用 `outer_inner_ring`：先检测外侧大圆，再利用大小圆之间的黑色环带定位内圆。该模式只要求内圆完整位于外圆内部，不要求两者
同心。程序先用最大暗色圆形外轮廓直接拟合外圆，再保留外圆内部与外侧环带相连的
黑色区域，从该黑区的内侧边缘点一次性稳健拟合内圆，不会再对内圆运行另一套 Hough
候选搜索：

```json
{
  "detection_method": "outer_inner_ring",
  "min_radius_ratio": 0.12,
  "max_radius_ratio": 0.48,
  "inner_radius_min_ratio": 0.15,
  "inner_radius_max_ratio": 0.75,
  "black_ring_width_ratio": 0.06,
  "min_black_ring_coverage": 0.45,
  "min_inner_angular_coverage": 0.35
}
```

其中前两个半径比例用于寻找外圆，内圆上下限和黑环宽度均相对于已检测的外圆半径。
局部反光或工件遮挡可以由覆盖率容忍；若黑环确实只露出较少部分，可适当降低
`min_black_ring_coverage`，但不建议一开始低于 `0.30`。

## 参考实现

- [EfficientAD 论文与作者实现](https://github.com/nelson1425/EfficientAD)
- [本项目使用的 anomalib 2.2.0 EfficientAd 模型](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/efficient_ad/lightning_model.py)
- [PyTorch 2.7.1 官方版本配对](https://pytorch.org/get-started/previous-versions/#v271)
- [uv 官方 PyTorch 集成说明](https://docs.astral.sh/uv/guides/integration/pytorch/)
