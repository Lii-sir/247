# ResNet student 与 AE 容量升级说明

日期：2026-09-25
更新：2026-09-26，补充 ResNet-50 layer1/layer2。

## 1. 实现范围

新训练默认使用 ResNet 架构版本 2，完成以下三项：

1. 支持 `resnet50_layer1`、`resnet50_layer2` 和 `resnet50_layer3`，使用原生 ImageNet 预训练 teacher。
2. Student 的 stem 和所有保留的残差 stage 通道整体扩大 2 倍，随机初始化。
3. AE 参考 EfficientAD 原有卷积编码/解码形式，中间通道随 teacher 通道数变化，直接生成与 teacher 对齐的特征图。

PDN small/medium 保持原结构和原 AE。本次未加入缺陷 mask 辅助损失，也未改变数据划分、评分方式或阈值算法。

这是一种自定义 EfficientAD 变体，扩大容量并不等于检测效果必然更好。

## 2. Teacher、student、AE 的接口

下面的空间尺寸以 256×256 RGB 输入为例：

| Backbone | Teacher C | Student 2C | AE C | 特征图 | 步长 |
|---|---:|---:|---:|---|---:|
| resnet18_layer2 | 128 | 256 | 128 | 32×32 | 8 |
| resnet18_layer3 | 256 | 512 | 256 | 16×16 | 16 |
| resnet50_layer1 | 256 | 512 | 256 | 64×64 | 4 |
| resnet50_layer2 | 512 | 1024 | 512 | 32×32 | 8 |
| resnet50_layer3 | 1024 | 2048 | 1024 | 16×16 | 16 |

Teacher 保留 torchvision ResNet 原生 stem 和 stage；无新增随机输出投影。
ResNet-18 使用 IMAGENET1K_V1 权重，ResNet-50 明确选择 IMAGENET1K_V2。
CLI 正常新训练会加载预训练权重；自定义 `--teacher-weights` 则优先加载所给权重。
Teacher 全部参数禁用梯度，模型调用 `train()` 后仍保持 teacher 为 `eval()`，包括 BatchNorm。

输入接口仍然是 RGB、[0,1]；ImageNet 标准化在网络内部完成，不要在外部重复 Normalize。
输出特征的每通道均值/标准差使用正常训练图像计算。ResNet 统计使用 float64 累积，对恒定通道使用单位标准差，防止除零。

低层 `EfficientAdModel(...)` 默认不下载 teacher 权重；直接使用它时必须加载权重，
或传 `teacher_pretrained=True`。CLI 已负责此步骤，Lightning 训练也会准备预训练 teacher。

## 3. Student：整体扩宽

每个阶段的输出通道如下，不只是最后一个 stage 扩宽：

| 阶段 | ResNet-18 teacher | ResNet-18 student | ResNet-50 teacher | ResNet-50 student |
|---|---:|---:|---:|---:|
| stem | 64 | 128 | 64 | 128 |
| layer1 | 64 | 128 | 256 | 512 |
| layer2 | 128 | 256 | 512 | 1024 |
| layer3 | 256 | 512 | 1024 | 2048 |

`resnet18_layer2` 截止 layer2，不构建 layer3。
`resnet50_layer1` 仅保留 layer1；`resnet50_layer2` 保留 layer1/layer2，不构建后续 stage。
ResNet-18 使用 BasicBlock，各保留 stage 为 2 个块；ResNet-50 使用 Bottleneck，各 stage 为 3、4、6 个块。
保持原阶段的步长；Bottleneck 内部通道和残差下采样支路一起扩宽。
没有构建 layer4、分类池化或全连接分类层。

整个 student 从随机权重训练。所有保留 stage 后增加一个 stride=1、padding=1、
输入输出均为 2C 的密集 3×3 卷积，不使用末尾 ReLU 或 BatchNorm：

```text
RGB → 扩宽 stem → 扩宽 stage → 3×3 Conv(2C, 2C)
                                      ├─ [:C]：teacher 对齐输出
                                      └─ [C:]：AE 对齐输出
```

保留线性输出是必要的：经过每通道标准化的 teacher 特征可以为负，
不能直接将末尾 ReLU 后的非负 ResNet 特征当作最终 student 预测。
两个输出切片有各自的卷积滤波器，但共享上游主干。

这与 torchvision 的 `wide_resnet50_2` 不是同一种配置：本项目扩宽 stem、
stage 输出和 Bottleneck 内部通道，不能直接加载那个模型的 student 权重。

## 4. AE：容量和空间尺寸适配

内部宽度使用下面的固定规则：

```python
hidden = min(256, max(64, C // 2))
small = hidden // 2
```

| Backbone | small | hidden | AE 最终输出 |
|---|---:|---:|---:|
| resnet18_layer2 | 32 | 64 | 128 |
| resnet18_layer3 | 64 | 128 | 256 |
| resnet50_layer1 | 64 | 128 | 256 |
| resnet50_layer2 | 128 | 256 | 512 |
| resnet50_layer3 | 128 | 256 | 1024 |

hidden 的上限 256 是本版本明确的容量选择：ResNet-50 layer2/layer3 的 hidden 均为 256，layer3 未继续扩大到 512。
它仍比原来的 64 通道 AE 更宽。这不是官方规定，也不是已验证的最优值；
若后续修改上限，需要相应升级架构版本并重新训练。

编码器保留六层卷积：

```text
3 → small → small → hidden → hidden → hidden → hidden
前五层：4×4，stride=2，padding=1，ReLU
第六层：8×8，stride=1，无 padding，无末尾激活
```

256×256 输入得到 1×1 的编码特征；更大的输入仍按原编码器公式产生更大的瓶颈，
例如 384 输入为 5×5，512 为 9×9，768 为 17×17。没有跳跃连接，也没有改成 U-Net。

解码器仍为八层卷积：六层 hidden→hidden 的 4×4 卷积，
一层 hidden→hidden 的 3×3 卷积，最后一层 hidden→C 的 3×3 线性输出卷积。
前六层使用 ReLU 和 Dropout(p=0.2)，第七层 ReLU，最后一层无激活。

传入真实 teacher 特征尺寸 (Hf,Wf)，前六层的输出空间计划为：

```text
ceil(Hf/4) → ceil(Hf/2) → Hf → Hf → Hf → Hf
```

宽度同理，每个维度至少为 2；4×4、padding=2 的卷积会增加一个像素，
因此每层卷积前先插值到计划尺寸减 1。最后两层 3×3 保持尺寸。
16×16 目标对应 4→8→16→16→16→16，最终直接输出 C×16×16。
ResNet-50 layer1 的 64×64 目标对应 16→32→64→64→64→64；
layer2 的 32×32 目标对应 8→16→32→32→32→32。
因此 layer1 与 ResNet-18 layer3 虽然同为 256 通道，空间解码路径仍按各自 teacher 的实际尺寸区分。
训练和推理都走这一条路径，不先生成 PDN 的 56×56 特征再缩小。

## 5. 损失与多卡语义

通道规则始终是 teacher=C、AE=C、student=2C。

| 损失 | 数据与目标 | 更新参数 |
|---|---|---|
| hard loss | 正常图：student[:C] 匹配标准化 teacher | student |
| auxiliary penalty | 辅助图：student[:C] 的平方均值 | student |
| AE loss | 增强正常图：AE 匹配标准化 teacher | AE |
| student-AE loss | 同一增强正常图：student[C:] 与 AE 差异 | student 和 AE |

总损失仍为上述各项之和。当前 student-AE loss 未对 AE 输出 detach，因此它也会更新 AE；
此次保持这一已有行为。辅助图像不直接训练 AE，但 student 主干共享，
penalty 更新主干后会间接改变另一半输出，不能理解成完全独立的两个网络。

多卡仍是一卡一进程。每卡 batch 含 B 张正常图和 B 张辅助图，
全局正常图 batch 为 B×world_size；`--max-images` 按正常图计数。
多 batch 的 hard loss 按图选择困难元素再平均，DDP 平均各 rank 的梯度，
不再手动给总损失除一次 world_size。Teacher 的统计在训练前准备并传给各进程。

Student 继续使用每卡局部 BatchNorm，而不是 SyncBatchNorm。因而 DDP 的整体训练数值
不保证与同一全局 batch 的单卡训练完全相同；也有增强与 Dropout 的随机性。
Teacher 的 BatchNorm 保持冻结推理状态。

## 6. Checkpoint 兼容

`resnet_architecture_version` 标记网络结构：

- 新训练默认写入 2，恢复时必须构建相同结构。
- 旧 CLI checkpoint 没有此字段时按 1 恢复，保留旧 student、旧 AE 和权重键。
- 旧 Lightning checkpoint 同样在应用 state dict 前恢复旧结构。
- ResNet-50 仅支持版本 2。
- PDN 架构不受该字段影响。

因此旧 ResNet 模型仍可评估和沿旧结构续训；它们不会因为软件升级而自动扩宽。
若要采用新结构，需要启动新训练并重新统计、校准。
已有 checkpoint 的 teacher 权重内嵌其中，恢复不需要重新下载。
新增 layer1/layer2 也使用版本 2；现有 layer3 的 state dict 键和形状保持一致。
layer1、layer2、layer3 的完整模型 checkpoint 不能互换，切换 stage 必须新训练并重新校准。
直接保存裸 state dict 的 API 使用者需要自行保存 backbone 和架构版本。

## 7. 参数量与资源

通过本次构建的模型参数实测（不含均值/方差和分位数标量）：

| Backbone | Teacher 参数 | Student 参数 | AE 参数 |
|---|---:|---:|---:|
| resnet18_layer2 | 683,072 | 3,299,712 | 948,608 |
| resnet18_layer3 | 2,782,784 | 13,463,168 | 3,789,568 |
| resnet50_layer1 | 225,344 | 3,236,480 | 3,789,568 |
| resnet50_layer2 | 1,444,928 | 15,178,880 | 15,148,544 |
| resnet50_layer3 | 8,543,296 | 71,843,968 | 16,328,704 |

ResNet-50 layer3 的 2048→2048 密集 3×3 输出卷积本身就约有 3775 万参数。
这里保留已确认的空间卷积设计；不应将本变体宣传为原版 EfficientAD 的毫秒级轻量网络。
实际显存还包含激活、梯度、Adam 状态和 DDP 通信缓冲，不能仅按权重大小估算 batch。
layer1/layer2 的特征图更大，AE 也在更大网格解码；参数更少不意味着任意输入尺寸和 batch 都能更省显存。

更宽的网络未提高 layer3 的空间分辨率；它仍然是 stride 16。
检测效果尤其是小缺陷召回，需要在真实数据上做相同划分和图片预算的对照。
layer1/layer2 提供更细的特征网格，但 teacher/student 的推理感受野也更小。
AE 在 256×256 输入下仍压缩到 1×1 瓶颈，再重建 64×64 或 32×32 网格；
增加解码分辨率不会取消该瓶颈，也不保证小缺陷或全局结构异常的检测效果更好。

## 8. 使用命令

在项目根目录运行，替换数据与 mask 配置路径。Linux 双卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 python efficientad_ccd.py train \
  --category CCD5-V3 \
  --data-root /path/to/dataset \
  --device 0 1 \
  --batch-size 1 \
  --backbone resnet50_layer3 \
  --image-size 256 \
  --max-images 70000 \
  --circle-config /path/to/circle_config.json \
  --num-workers 4
```

此例每卡 batch=1，全局 batch=2；脚本自行启动 DDP，无需另用 torchrun。
将 backbone 改成 `resnet18_layer3` 即使用新版整体扩宽的 ResNet-18 与 AE。
改为 `resnet50_layer1` 或 `resnet50_layer2` 即选择对应的 ResNet-50 浅层特征、整体扩宽 student 和匹配 AE。
首轮可先用较小图片预算核实显存。

恢复同一架构：

```bash
CUDA_VISIBLE_DEVICES=0,1 python efficientad_ccd.py train \
  --resume /path/to/run/checkpoints/last.pt --device 0 1 --num-workers 4
```

评估已校准模型：

```bash
python efficientad_ccd.py evaluate \
  --checkpoint /path/to/run/model.pt --device 0 --heatmaps -1
```

## 9. 验证记录

### 初始容量升级（2026-09-25）

环境：Windows、Python 3.12、PyTorch 2.7.1+cu126、torchvision 0.22.1，
单张 RTX 3060 12 GiB。测试使用现有虚拟环境，未修改依赖版本。

复现命令（Windows 可将 python 换成 .venv\Scripts\python.exe）：

```bash
python -m unittest discover -s tests -p "test*.py" -q
python tests/smoke_pipeline.py --backbone resnet50_layer3
python tests/smoke_pipeline.py --backbone resnet50_layer3 --ddp-test-device cuda:0 --batch-size 1
python tests/smoke_pipeline.py --backbone resnet18_layer3 --ddp-test-device cuda:0
git diff --check
```

测试范围包括：所有 stage 扩宽、teacher 完整预训练权重加载、student/AE 负数输出、
实际反传和更新、teacher 参数与 BN 统计不变、AE 不同输入尺寸、
新旧 CLI/Lightning checkpoint 恢复，以及 CLI 训练、校准、评估、预测、评分切换和续训。
ResNet smoke 使用本机缓存的真实预训练 teacher 和合成图片，默认每卡 batch=2。

执行结果：

| 检查 | 结果 |
|---|---|
| 完整 unittest discovery（第二轮） | 113 项通过，67.993 秒，退出码 0 |
| ResNet-50 layer3 单卡、batch=2 全流程（第二轮重跑） | 通过，退出码 0 |
| ResNet-50 layer3 双进程、每进程 batch=1 全流程 | 通过，退出码 0 |
| ResNet-18 layer3 双进程、每进程 batch=2 全流程 | 通过，退出码 0 |
| git diff --check | 通过；仅提示 Windows 行尾转换 |

全流程均包含训练、校准、保存/加载、默认与切换 score 评估、热图、单图预测，
以及从第 2 步恢复后实际执行第 3 步优化。
本次排查修正了公共 pair 工厂仍默认生成旧 student 的不一致，以及 Python API
省略版本时 DDP 启动文件将新版模型误判为旧版的问题，并加入对应回归测试。

双进程 GPU 测试使用同一张 GPU 的两个 rank 和 Gloo，不能证明 Linux NCCL 双物理卡性能，
也不能用合成图片指标判断真实缺陷检测效果。

### 第二轮复核与修正

第二轮发现旧 Lightning checkpoint 触发架构重建时，新建 core 默认回到 float32
和训练模式，没有保留调用方原先的 dtype 与 train/eval 状态。已先用回归测试复现
float64 被重置为 float32，再修正为同时继承设备、dtype 和原 core 的模式；teacher 仍始终冻结并保持 eval。

新增三项测试：

1. 旧架构重建后保留 float64 和 eval，实际推理返回 float64 异常图。
2. 分别反传三项损失，验证 student–teacher 只更新 student 第一半输出及共享主干，
   AE 重建只更新 AE，student–AE 更新 student 第二半输出、共享主干和 AE；teacher 无梯度。
3. 使用真实 Lightning Trainer 训练旧版模型一步，保存后移除架构版本字段，
   通过新版默认实例加载并继续第二步。验证优化器参数引用属于重建后的 student/AE，
   Adam 状态步数延续到 2，student 权重实际更新且 teacher 参数和 BN 缓冲区不变。
   此测试使用合成张量并替换外部数据/预训练准备，不包含数据采样顺序的精确恢复验证。

第二轮完整测试集 113 项通过，独立代码复核未发现其他需要修正的损失或恢复问题。
另重跑 ResNet-50 layer3 单卡 batch=2 全流程，训练、校准、评分切换、预测与实际续训均通过，退出码 0。

### ResNet-50 layer1/layer2 扩展（2026-09-26）

新增 `tests/test_resnet50_stages.py`，共 8 项测试，覆盖：

- CLI 选项、原生通道及 stride，拒绝错误 teacher 通道和旧架构版本。
- 所有保留 Bottleneck 的三层卷积、残差下采样支路均扩宽 2 倍，且未构建后续 stage。
- 256×256 与 257×289 输入下 teacher/student/AE 对齐，异常图恢复输入尺寸，输出允许负值。
- 384、512、768 输入下 AE 直接匹配实际 teacher 的大尺寸特征网格。
- 原生预训练权重前缀加载，不额外增加随机 teacher 投影；原 layer3 的键和形状保持兼容。
- batch=2 的真实三损失反传，student 两半输出和 AE 都有有限梯度并实际更新，teacher 参数和 BN 缓冲不变。
- CLI 与 Lightning checkpoint 保存/恢复的输出一致，恢复不下载初始化权重。

复现命令：

```bash
python -m unittest discover -s tests -p "test*.py" -q
python tests/smoke_pipeline.py --backbone resnet50_layer1
python tests/smoke_pipeline.py --backbone resnet50_layer2 --ddp-test-device cuda:0 --batch-size 2
```

| 检查 | 实测结果 |
|---|---|
| 当前工作区完整 unittest discovery | 124 项通过，122.953 秒，退出码 0 |
| layer1 单卡 batch=2 完整 CLI 流程 | 通过，退出码 0 |
| layer2 两个 DDP rank、每 rank batch=2 完整 CLI 流程 | 通过，退出码 0 |
| 独立结构/兼容性代码复核 | 未发现需要修正的问题 |
| git diff --check | 通过，仅 Windows 行尾提示 |

两个 CLI 流程均使用缓存的真实 ResNet-50 V2 teacher 权重和合成图片，
覆盖训练、校准、评分切换、评估、预测及从第 2 步恢复后完成第 3 步优化。
DDP 实测使用同一 RTX 3060 上的两个进程、Gloo 后端；尚未实测 Linux 双物理卡 NCCL。
这些结果验证程序和梯度链路，不能替代真实缺陷数据上的效果评估。
完整测试第一次遇到工作区并行新增分支导出测试的导入失败；其配套模块出现后重跑全部通过，
本次未修改该模块或对应测试。

### 扩展后的再次复核（2026-09-26）

本轮没有发现需要修改的网络或训练逻辑错误，新增一项损失语义测试（专项测试现共 9 项）：
对 layer1/layer2 分别使用 batch=1/global 与 batch=2/per_image，启用非零 teacher 均值和非单位标准差。
保留真实增强、BN 和 Dropout，从各次前向输出独立重算 hard loss、辅助惩罚、AE 重建及 student–AE 损失，
结果与模型返回值一致；逐损失求梯度也验证了 student 两半输出、共享主干及 AE 的更新范围。

另外从本次修改前的 Git HEAD 加载原 backbone 实现，在相同随机种子下对比 layer3：
teacher/student 的所有 state dict 键、张量值、严格加载和前向输出均完全一致。

| 本轮检查 | 结果 |
|---|---|
| 完整 unittest discovery | 125 项通过，155.206 秒，退出码 0 |
| layer1 DDP 两进程、每进程 batch=1 全流程 | 通过，退出码 0；包含恢复后第 3 步优化 |
| layer3 与修改前实现的数值兼容检查 | 完全一致，rtol=0、atol=0 |
| 独立代码复核 | 未发现新的结构或损失逻辑问题 |

DDP 仍使用同一 GPU 的两个 rank 与 Gloo，不等同于双物理卡 NCCL 测试。
本轮仅补充测试与说明，保留现有结构和训练损失。

## 10. 代码位置

- `self_efficientad/backbones.py`：teacher、整体扩宽 student、架构版本选择。
- `self_efficientad/resnet_autoencoder.py`：新版特征 AE。
- `self_efficientad/torch_model.py`：编码器参数化、损失与推理的 AE 尺寸接口。
- `self_efficientad/lightning_model.py`：ResNet 统计和 Lightning 旧模型兼容。
- `efficientad_ccd.py`：CLI 选项及 checkpoint 版本读写。
- `ccd_distributed.py`：DDP 启动文件保留实际架构版本。
- `tests/test_resnet_capacity.py`、`tests/test_backbones.py`：结构、梯度与兼容测试。
- `tests/test_resnet50_stages.py`：ResNet-50 layer1/layer2 的结构、梯度、多尺寸和恢复测试。
- `tests/smoke_pipeline.py`：CLI 全流程测试。
