# resnet50_layer1v2 Teacher 最终激活检查与修正

日期：2026-09-29

## 网络行为

仅跳过 Teacher 的 layer1 最后一个 Bottleneck 在残差相加后的 ReLU。
该块 conv1、conv2 后的 ReLU，以及前面所有块的 ReLU 均保留。
Teacher 的卷积、BN 和 shortcut 权重名称与形状不变，继续加载 torchvision
ResNet50 IMAGENET1K_V2 权重。Teacher 始终冻结并处于 eval 模式。

Student、AE 和三个训练损失的定义不变。原有输出裁边规则也不变：
256 输入、valid 模式时，Teacher/Student/AE 输出网格均为 62×62。
`teacher.forward_features(x)` 的输入应已完成 ImageNet mean/std 标准化，
返回裁边前的特征；原始 [0,1] RGB 张量应调用 `teacher(x)`。

## 本次发现的问题

原实现仅按 backbone 名称跳过 ReLU，但没有记录输出激活模式。
由于 ReLU 没有参数，两种模式的 state_dict 键和形状完全一样，严格加载
也不能发现错误。旧模型会使用新的 Teacher 特征搭配旧 Student、AE、统计量
和校准，导致预测行为悄悄改变。

## 修正后的配置和兼容性

- 新字段：`resnet_teacher_output_activation`，值为 `relu` 或 `none`。
- 新建 `resnet50_layer1v2` 默认 `none`，保持这次实验要求。
- 新训练可加 `--resnet-teacher-output-activation relu` 做原激活对照。
- `none` 目前仅开放给 `resnet50_layer1v2`；其他 backbone 保持原有激活。
- CLI checkpoint 保存到 config，Lightning checkpoint 保存到 hyper_parameters。
- 历史 checkpoint 缺少字段时按 `relu` 恢复，不因构造默认值改变行为。
- DDP 启动文件记录父进程实际激活模式和输出裁边模式。
- CLI 续训禁止更改激活模式。更换模式需新训练、重新计算 mean/std、
  anomaly quantiles 和检测阈值。

如果已经用此前“去 ReLU 但没有保存字段”的中间版本训练过模型，它也缺少
模式标记，无法仅凭权重自动判断。这样的文件不能当作历史 ReLU 模型直接
恢复；必须依据确切训练记录补充正确模式，或重新训练，不能猜测。
直接加载裸 state_dict 的调用者也必须显式构建匹配模式。

## 验证范围

`tests/test_teacher_activation.py` 检查：

- 逐次记录 ReLU 输出，确认只少最后一次调用；新输出 ReLU 后与旧输出逐元素一致。
- 两种模式加载预训练参数时不会被权重加载过程切换。
- CLI/Lightning 两种模式的保存恢复，以及无字段旧 ReLU checkpoint 的预测一致性。
- 续训拒绝切换模式，DDP 启动配置保留实际模式。
- batch=1/2 的真实前向、三个损失的数值与梯度路径、优化器更新及 Teacher/BN 冻结。

这些测试用于验证实现和兼容性；去掉最终 ReLU 是否改善缺陷检测，需要重新训练后
在同一数据划分上比较，不能由单元测试或允许负特征直接推出。

本次还运行了 resnet50_layer1v2 的完整合成 CLI 流程：使用缓存的 ImageNet 教师权重，
在单张 GPU 上启动两个 Gloo rank，每 rank batch=1，完成训练、三种 score 报告、
评估、预测和延长一步的实际续训，并检查最终模型及 last.pt 保留激活模式。
该测试不代表两张物理 GPU 的 Linux/NCCL 验证。

## 第二次检查（2026-09-29）

本轮未发现新的生产逻辑错误，新增两项回归测试：

- 使用真实有正负值的 Teacher 输出，包含最后一个不完整 batch，核对裁边后的
  channel mean/std 与直接计算完全遵循同一定义；标准化后的通道均值约为 0、方差约为 1。
- 使用真正的 Lightning Trainer 保存、恢复并继续优化，覆盖无激活字段的旧 ReLU
  模型和新 none 模型。恢复前故意构建相反模式，确认重建后优化器绑定正确参数，
  Student/AE 确实更新，Teacher/BN 和 mean/std 保持不变。

本轮全量回归：145 项测试全部通过，用时 196.373 秒。
日志：项目根目录 `.teacher-activation-second-review.log`。
