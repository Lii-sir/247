# ResNet 最终输出不补边，异常图补零

日期：2026-09-28

## 范围

根据“仅最终特征输出不 padding，中间层允许 padding”的要求实现。
适用于架构版本 2 的全部 ResNet：ResNet18 layer2/layer3，ResNet50
layer1/layer1v2/layer2/layer3。PDN 及架构版本 1 的既有流程保持不变。

不是把所有卷积改成 valid，也不是简单地给完整的原生异常图再加一圈零。

## 处理流程

1. Teacher 和 Student 的 stem、池化、残差块保留原有 padding；不修改残差 forward。
2. Student 最后 `3×3, stride=1` 输出头设置 `padding=0`，网格高宽各减少 2。
3. Teacher 不增加随机投影，只把原生输出四周各裁 1 格，与 Student 窗口中心对齐。
   预训练参数键、形状不变；教师统计也使用裁剪后的输出。
4. AE 在完整主干网格上解码，最后一个 `3×3` 卷积不补边，输出与 Teacher 对齐。
5. 两路平方差沿通道求均值，得到原生异常图。此时还没有补零。
6. `pad_maps=True`（默认）时，两路异常图四周各补 1 格**常数零**，再双线性插值到输入尺寸。
7. 按原流程校准、融合并提取整图分数。校准可能改变边界零的数值；“零边框”指校准前。

| 输入 | ResNet50 分支 | 对齐后的 Teacher / Student / AE 网格 | 补零后网格 |
|---|---|---|---|
| 224×224 | layer1 / layer1v2 | 54×54 | 56×56 |
| 224×224 | layer2 | 26×26 | 28×28 |
| 224×224 | layer3 | 12×12 | 14×14 |
| 256×256 | layer1 / layer1v2 | 62×62 | 64×64 |
| 256×256 | layer2 | 30×30 | 32×32 |
| 256×256 | layer3 | 14×14 | 16×16 |

矩形/奇数输入遵循每维 `ceil(input / feature_stride) - 2`，补边各 1 格。
这保留原来的采样步距，不等于提高空间分辨率，也不消除主干内部 padding 的影响。
`pad_maps=False` 仅关闭分数图补边，不改变特征提取。

低层模型 API 支持上述 224 输入。新 valid 模式下，AE 五次下采样得到 7×7，
会先双线性插值到 8×8，再执行原有 8×8 瓶颈卷积；这只在尺寸不足时启用。
256 及更大输入不走此补偿。PDN 与 native 模式的 AE 行为不变；CLI 的输入尺寸选项未扩展。

## 保存、恢复和训练

- 新建版本 2 模型默认 `resnet_feature_mode="valid"`。
- `resnet_feature_mode="native"` 显式使用历史行为：Teacher 不裁剪，Student/AE 输出头 padding=1，ResNet 异常图不补边。
- CLI 保存实际模式到 config；Lightning 保存实际模式到 hyper_parameters。
- 缺少字段的历史文件在恢复时按 `native` 处理，不根据新的构造默认值猜测。
- 只有裸 state_dict 时无法判断模式，调用者必须选定匹配的结构；参数形状相同不代表边界语义相同。
- 新模式应重新训练并重新计算教师统计、异常图校准和阈值。现有 native 模型可以原样续训。
- 中途实验版本若曾采用“所有层无 padding”且没有保存模式元数据，不能当作原版 native checkpoint 自动兼容。

导出分支热图与在线预测共用 `pad_map_to_feature_grid`，避免补边方式不一致。

## 验证

`tests/test_resnet_valid_output.py` 检查所有 backbone 的 224、256、奇数矩形输入，
主干 padding 保留、Teacher 精确裁剪、Student 真正 valid 卷积、AE 尺寸对齐、
原生异常图内部不变/边框全零、两路热图与导出一致、layer3 224 反向传播、
PDN 行为不变，以及新旧 CLI/Lightning checkpoint 离线恢复。
