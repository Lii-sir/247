# Bridge 配置

默认入口 [`config.json`](config.json) 指向 [`normal`](products/normal/config.json) 型号。
每个型号目录包含以下配置。

- `2D/config.json` 保存圆定位、ROI、YOLO 和各项 2D 算法的参数。
- `3D/config.json` 保存配准、测高、调用端分片和矫正参数。
- `inspection.json` 保存检测项选择、报告优先级和跳过规则。

型号入口引用以上文件。各型号的模型路径相对该型号的 2D 配置文件解析，公共模型存放于 `models/`。 [型号目录说明](products/README.md)

## 阈值与算法参数

2D 和 3D 的每个算法模块都包含 `description`、`thresholds` 和 `parameters`。
`thresholds` 保存候选筛选、数据有效性和缺陷判定门槛。`parameters` 保存 ROI、模型配置、采样与几何处理方式以及并发设置。
某一组没有字段时使用 `{}`。同一字段只填一次，加载时会合并两组配置。修改后重新初始化算法。

## 检测策略

修改 `inspection.json` 后重新初始化算法。例如只检查芯片旋转和引线高度，并在同片 2D 判 NG 后跳过 3D。

```json
{
  "defect_sequence": ["chip_rotation_2d", "wire_height_3d"],
  "skip_3d_on_2d_ng": true
}
```

| 检测项 | 配置名 |
| --- | --- |
| 引线缺失和弯曲 | `wire_bending_2d` |
| 芯片旋转 | `chip_rotation_2d` |
| 芯片外观异常 | `chip_anomaly_2d` |
| 银浆溢出 | `silver_overflow_2d` |
| 银浆断连 | `silver_continuity_2d` |
| 引线高度 | `wire_height_3d` |
| 平台侧焊点和焊脚高度 | `solder_platform_height_3d` |
| 芯片侧焊点和焊脚高度 | `solder_chip_height_3d` |

- 清单只启用列出的项目。同一阶段有多个 NG 时，优先报告清单靠前的项，联合结果优先报告 2D 的 NG。
- `defect_sequence: null` 使用默认检测项。空数组 `[]` 只验证 YOLO 分割是否有有效掩码。
- `skip_3d_on_2d_ng: true` 在同一片 2D 判 NG 时返回 `2d_ng` 并跳过配准和测高。为 `false` 时，有有效掩码的片可继续测高，联合结论仍保留 2D NG。
- 没有对应片的 2D 结果，或该片没有有效掩码时，跳过 3D 并按 NG 返回。只启用 3D 项也需要先提交该片 2D 供 YOLO 分割。

现场使用过旧字段时，把设置迁入 `inspection.json`。独立文件生效后以它为准。自组初始化请求可把文件内容作为顶层 `inspection` 对象传入。

## 上位机提交

使用 `InitAlgorithm("board")` 加载默认配置，或用 `InitAlgorithm("board", configPath)` 加载指定型号的配置。上位机按相同板号、片号分别提交 `detect_2d_piece` 和 `detect_3d_piece`，深度图在调用端完成切片和矫正。C# 和测高结果字段参见[接口说明](../process/OUTPUT_PARAMETERS.md)。
