# 本地 EfficientAD 实现

这个目录保存了项目当前环境中 `anomalib==2.2.0` 的 EfficientAD 实现副本，便于在不修改虚拟环境源码的情况下调整网络结构、损失函数和训练流程。

论文：

《EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies》

https://arxiv.org/pdf/2303.14535.pdf

## 文件说明

- `torch_model.py`：PDN Teacher、Student、Autoencoder、训练损失和异常图计算。
- `lightning_model.py`：Lightning 训练模块、教师权重加载和 ImageNette 辅助数据准备。
- `__init__.py`：导出 `EfficientAd` 类。
- `COPY_INFO.md`：副本来源和版本信息。

## 与 Anomalib 的关系

这里的代码是 Anomalib 2.2.0 中 EfficientAD 模块的本地副本，不是完全独立的深度学习框架。它仍然依赖项目环境中的以下库：

- `anomalib==2.2.0`
- `torch`
- `torchvision`
- `lightning`

本地副本可以直接修改。修改 `torch_model.py` 或 `lightning_model.py` 不会改变 `.venv` 中安装的 Anomalib 文件。

## 直接导入

在项目根目录运行：

```python
from self_efficientad import EfficientAd
from self_efficientad.torch_model import EfficientAdModel, EfficientAdModelSize

model = EfficientAd(
    imagenet_dir="assets/imagenette",
    model_size=EfficientAdModelSize.S,
    pre_processor=False,
    post_processor=False,
    evaluator=False,
    visualizer=False,
)
```

也可以使用字符串指定模型大小：

```python
model = EfficientAd(model_size="small")
```

支持两种模型规模：

- `small`：轻量模型。
- `medium`：较大模型，计算量和显存占用更高。

默认 `batch_size=1`，保持论文和原始 Anomalib 实现的训练语义。需要批量训练时，应同时指定 `batch_size` 和 `hard_loss_mode="per_image"`：

```python
model = EfficientAd(
    imagenet_dir="assets/imagenette",
    model_size="small",
    batch_size=4,
    hard_loss_mode="per_image",
)
```

批量模式会对每张图片单独计算 Student-Teacher hard loss 的 Q99.9，再对 batch 求平均；ImageNette 辅助 loader 使用相同 batch size，并丢弃最后一个不完整 batch。`batch_size=1` 时使用原始的全局 hard loss。

如果通过 Anomalib 的 Lightning/Engine 训练，还必须让外部 DataModule 与模型保持一致：

```python
datamodule = ...
datamodule.train_batch_size = 4
datamodule.eval_batch_size = 1
```

模型会检查这两个值；教师均值和标准差会使用独立的单图统计 loader，确保训练集最后一个不完整 batch 也参与统计。`eval_batch_size=1` 同时适用于验证、测试和可视化流程。

## 输入和输出

`EfficientAdModel` 期望输入为形状 `[batch, 3, height, width]` 的 PyTorch Tensor，像素范围为 `[0, 1]`。不要在外部再次执行 ImageNet `Normalize`，因为模型内部会完成归一化。

训练模式下：

```python
model.train()
loss_st, loss_ae, loss_stae = model.model(
    batch=normal_images,
    batch_imagenet=imagenet_images,
)
loss = loss_st + loss_ae + loss_stae
```

三个损失分别对应：

- Teacher-Student hard loss 和 ImageNet 辅助正则项。
- Teacher-Autoencoder 重建损失。
- Autoencoder-Student 损失。

评估或推理模式下：

```python
model.eval()
prediction = model.model(images)
anomaly_map = prediction.anomaly_map
score = prediction.pred_score
```

`anomaly_map` 是像素级异常图，`pred_score` 默认是异常图中的最大值。教师特征统计量和异常图分位数需要在训练或校准阶段设置后，推理结果才有意义。

## 接入当前 CCD 脚本

当前 `efficientad_ccd.py` 已经使用这个本地副本：

```python
from self_efficientad import EfficientAd
```

权重下载配置也从本地模块导入：

```python
from self_efficientad.lightning_model import WEIGHTS_DOWNLOAD_INFO
```

下面的导入仍然可以保留：

```python
from anomalib.data.utils import download_and_extract
```

因为它只是使用 Anomalib 提供的下载工具，并不决定模型结构。

## 修改建议

- 修改 Teacher、Student 或 Autoencoder 结构：编辑 `torch_model.py`。
- 修改三项训练损失或数据增强：编辑 `EfficientAdModel.compute_losses`。
- 修改批量训练的 hard loss：编辑 `student_teacher_hard_loss`。
- 修改异常图尺寸、补边、插值或归一化：编辑 `EfficientAdModel.compute_maps`。
- 修改教师权重、ImageNette 或 Lightning 生命周期：编辑 `lightning_model.py`。
- 修改当前 CCD 的自定义训练循环、阈值和评分方式：编辑项目根目录的 `efficientad_ccd.py`。

修改网络输出通道数、输入尺寸或模型状态字典后，原有教师权重和 checkpoint 可能无法继续使用，需要重新训练或重新导出权重。

## 来源

本目录的文件复制自：

```text
.venv/Lib/site-packages/anomalib/models/image/efficient_ad/
```

版本：`anomalib==2.2.0`。原始源码中的 Apache-2.0 版权声明保留在 Python 文件中。
