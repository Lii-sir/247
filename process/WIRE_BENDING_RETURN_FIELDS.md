# 引线弯曲检测新增返回字段说明

更新日期：2026-10-17。

本文说明引线弯曲检测改用“ROI 四交点评估四边形”后，新增的 C# 返回属性和 JSON 字段，供上位机接入与可视化使用。检测方法名、请求参数不变。

## 1. 返回位置

| 内容 | C# 访问路径 | JSON 访问路径 |
| --- | --- | --- |
| 单片 2D 的逐实例弯曲结果 | `response.Results2D.WireBending2D.Measurements[i].Bending2D` | `results_2d.wire_bending_2d.measurements[i].bending_2d` |
| 单片 2D 的 ROI 内逐实例摘要 | `response.Results2D.WireBending2D.RoiChecks[i].BendingChecks[j]` | `results_2d.wire_bending_2d.roi_checks[i].bending_checks[j]` |
| 联合检测的引线弯曲结果 | `response.ResultsBoard.Pieces[i].WireMeasurements[j].Bending2D` | `results_board.pieces[i].wire_measurements[j].bending_2d` |
| 联合检测的 ROI 内逐实例摘要 | `response.ResultsBoard.Pieces[i].WireRoiChecks[j].BendingChecks[k]` | `results_board.pieces[i].wire_roi_checks[j].bending_checks[k]` |

调用时应检查 `Results2D`、`WireBending2D`、`ResultsBoard` 是否存在；表中的 `i/j/k` 是列表索引。

`InstanceId` 用于关联同片内的实例。显示所有引线的 2D 弯曲结果时，应使用 `WireBending2D.Measurements`，再按 `InstanceId` 合并联合结果中的高度。联合 `WireMeasurements` 只包含进入测高流程的引线，不能代替完整的 2D 列表。

## 2. `WireBendingResult` 新增字段

下列属性位于 `Bending2D` 对象内。

| JSON 字段 | C# 属性 | C# 类型 | 含义 |
| --- | --- | --- | --- |
| `status` | `Status` | `string` | 当前实例的测量状态或失败原因，取值见第 4 节。 |
| `roi_id` | `RoiId` | `int?` | 分配的配置 ROI 编号，从 **1** 开始；未分配为 `null`。不是实例编号，也不是从 0 开始的数组索引。 |
| `roi_mask_pixel_count` | `RoiMaskPixelCount` | `int` | 清理后，该实例在分配 ROI 内的像素数，即覆盖率分母。 |
| `covered_mask_pixel_count` | `CoveredMaskPixelCount` | `int` | 上述像素中落在评估四边形内的像素数，即覆盖率分子。 |

同一实例与多个 ROI 重叠时，按**清理后实例掩码的重叠像素数最多**分配给其中一个 ROI；数量相同取配置顺序靠前的 ROI。

有效测量满足：

```text
MaskRatio = CoveredMaskPixelCount / RoiMaskPixelCount
0 ≤ CoveredMaskPixelCount ≤ RoiMaskPixelCount
RoiMaskPixelCount > 0
```

上述像素计数只包含当前实例，不会把同一 ROI 内其他引线、银浆或背景计入分子分母。

## 3. 原有字段的含义变化

| JSON 字段 | C# 属性 | 当前含义 |
| --- | --- | --- |
| `rectangle` | `Rectangle` | 评估四边形的四个有序顶点，沿用原字段名，**不保证是矩形**。 |
| `mask_ratio` | `MaskRatio` | 评估四边形内该实例像素数 / ROI 内该实例像素数。 |
| `is_valid` | `IsValid` | 是否成功得到有效的评估四边形和覆盖率。 |
| `is_ok` | `IsOk` | 有效测量的覆盖率是否达到阈值；无效测量返回 `false`。 |

四边形由实例与 ROI 像素周界上两个连续接触区间的四个端点构成。ROI 使用左闭右开范围 `[x1,x2) × [y1,y2)`，右边、下边的边界像素分别为 `x2-1`、`y2-1`。

`Rectangle` 中各点格式为 `[x, y]`，单位为像素，使用**完整 2D 原图坐标**。在原图上依次连线，并连接最后一点到第一点即可绘制；无需再加 crop 偏移。即使结果来自联合 3D 检测，该字段仍使用 2D 原图坐标。

通过条件为 `MaskRatio >= rectangle_ratio_threshold`，包含等号。当前配置值为 `0.75`，即 75%。由于分母定义已变化，应结合标注样本重新校准该阈值。

## 4. `Bending2D.Status` 状态值

| `Status` | 含义 | `IsValid` | `IsOk` | 展示建议 |
| --- | --- | --- | --- | --- |
| `ok` | 四边形有效，覆盖率达到阈值。 | `true` | `true` | 绿色 OK，显示覆盖率。 |
| `bending_out_of_range` | 四边形有效，覆盖率低于阈值。 | `true` | `false` | 红色 NG，显示覆盖率。 |
| `empty_mask` | 输入为空，或清理后无有效实例掩码。 | `false` | `false` | NG · 无效，显示“掩码清理失败”。 |
| `outside_rois` | 没有可分配的 ROI：ROI 列表为空，或掩码与各 ROI 均无重叠。 | `false` | `false` | 显示“ROI 外/未分配”，不作为单条弯曲超限计入 NG 数。 |
| `insufficient_intersections` | 未形成两个边界接触区间，例如实例完全位于 ROI 内或只到达一端。 | `false` | `false` | NG · 无效，显示“交点不足”。 |
| `ambiguous_intersections` | 接触区间超过两个，或整个 ROI 周界均被占据，无法唯一确定四个交点。 | `false` | `false` | NG · 无效，显示“交点存在歧义”。 |
| `degenerate_shape` | ROI 太窄、端点重合/共线，或四点无法形成有效四边形。 | `false` | `false` | NG · 无效，显示“评估形状退化”。 |

无效测量的 `MaskRatio` 为 `0.0`，`Rectangle` 为 `[]`；该 `0.0` 是无效占位值，界面建议显示“—”，不要解释成测得覆盖率为 0%。

对于 `empty_mask`、`outside_rois`，`RoiId` 为 `null`，两个新增像素计数均为 0。对于已分配 ROI、但交点或形状无效的结果，保留 `RoiId` 和分母 `RoiMaskPixelCount`，分子 `CoveredMaskPixelCount` 为 0。

`outside_rois` 的实例不分配到任何 ROI。即使这些实例单独不计入弯曲 NG 数，整体也不会因此自动通过：配置 ROI 缺线仍判 NG，且未配置任何 ROI 时整体弯曲检查为 NG。

## 5. `WireRoiBendingCheck2D` 新增字段

| JSON 字段 | C# 属性 | C# 类型 | 含义 |
| --- | --- | --- | --- |
| `status` | `Status` | `string` | 当前 ROI 中某个实例的弯曲状态，与该实例 `Bending2D.Status` 一致。 |

位置为 `RoiChecks[].BendingChecks[].Status`，联合结果对应 `WireRoiChecks[].BendingChecks[].Status`。

这里的 `Status` 描述**单个实例**，不同于外层 `RoiChecks[].Status`。外层状态仍沿用以下汇总规则：

| ROI 汇总状态 | 条件 |
| --- | --- |
| `wire_missing` | 没有分配到任何引线实例。 |
| `ok` | 有实例，且全部实例有效、合格。 |
| `bending_unavailable` | 有实例，但至少一个实例测量无效。 |
| `bending_out_of_range` | 全部实例测量有效，但至少一个覆盖率低于阈值。 |

整体弯曲检查要求 ROI 列表非空，并且所有 ROI 的 `IsOk` 均为 `true`。不能只检查有四边形的实例，忽略缺线或测量无效的 ROI。

正常情况下，`RoiChecks[].MaskPixelCount` 汇总该 ROI 中各实例的 `RoiMaskPixelCount`。若某实例清理失败，会用其原始掩码确定故障 ROI，此时该实例对外层 `MaskPixelCount` 的贡献是原始掩码重叠像素数，而其 `Bending2D.RoiMaskPixelCount` 仍为 0；不要把外层计数当成任何单条引线的覆盖率分母。

## 6. JSON 示例

下面是字段示意片段，不是完整的接口响应。示例阈值为 0.75，第一条覆盖率为 0.8，第二条为 0.6。

```json
{
  "wire_bending_2d": {
    "enabled": true,
    "is_ok": false,
    "measurements": [
      {
        "instance_id": 3,
        "class_name": "wire",
        "bending_2d": {
          "is_valid": true,
          "is_ok": true,
          "status": "ok",
          "roi_id": 1,
          "roi_mask_pixel_count": 1000,
          "covered_mask_pixel_count": 800,
          "mask_ratio": 0.8,
          "rectangle": [[100, 200], [300, 205], [300, 225], [100, 220]]
        }
      },
      {
        "instance_id": 9,
        "class_name": "thin",
        "bending_2d": {
          "is_valid": true,
          "is_ok": false,
          "status": "bending_out_of_range",
          "roi_id": 1,
          "roi_mask_pixel_count": 1000,
          "covered_mask_pixel_count": 600,
          "mask_ratio": 0.6,
          "rectangle": [[100, 240], [300, 245], [300, 265], [100, 260]]
        }
      }
    ],
    "roi_checks": [
      {
        "roi_id": 1,
        "box": [100, 180, 301, 281],
        "mask_pixel_count": 2000,
        "instance_ids": [3, 9],
        "is_ok": false,
        "status": "bending_out_of_range",
        "bending_checks": [
          {"instance_id": 3, "mask_ratio": 0.8, "is_valid": true, "is_ok": true, "status": "ok"},
          {"instance_id": 9, "mask_ratio": 0.6, "is_valid": true, "is_ok": false, "status": "bending_out_of_range"}
        ]
      }
    ]
  }
}
```

测量无效但已分配 ROI 时，`Bending2D` 示例：

```json
{
  "is_valid": false,
  "is_ok": false,
  "status": "insufficient_intersections",
  "roi_id": 1,
  "roi_mask_pixel_count": 1000,
  "covered_mask_pixel_count": 0,
  "mask_ratio": 0.0,
  "rectangle": []
}
```

## 7. C# 读取示例

以下代码接在现有单片 2D 调用取得 `response` 之后。`MaskRatio` 是 0～1 的比例；`P1` 格式可显示百分比。

```csharp
var result = response.Results2D?.WireBending2D;
if (result != null)
{
    Console.WriteLine($"引线 ROI 综合判定：{(result.IsOk == true ? "OK" : "NG")}");
    foreach (var wire in result.Measurements)
    {
        var bend = wire.Bending2D;
        if (bend.Status == "outside_rois")
        {
            Console.WriteLine($"引线 #{wire.InstanceId}：ROI 外/未分配");
            continue;
        }
        if (!bend.IsValid)
        {
            Console.WriteLine($"引线 #{wire.InstanceId}：NG，测量无效，原因={bend.Status}");
            continue;
        }
        Console.WriteLine(
            $"引线 #{wire.InstanceId}：{(bend.IsOk ? "OK" : "NG")}，" +
            $"ROI={bend.RoiId}，覆盖率={bend.MaskRatio:P1}，" +
            $"像素={bend.CoveredMaskPixelCount}/{bend.RoiMaskPixelCount}");
        // bend.Rectangle 是四个原图坐标顶点，按顺序连线并闭合。
    }
}
```

## 8. 兼容性与交付

- 检测方法名、请求参数、原有返回字段名保持不变；旧代码仍可读取原有属性。
- 读取新增 C# 属性需要更新 `Bridge.dll` 引用；仅更新 Python 不会给旧版 C# DTO 增加属性。
- 更新后的 C# DTO 读取旧 JSON 时，缺失的 `Status` 默认是空字符串，`RoiId` 为 `null`，两个像素计数为 0。应将其视为旧数据缺少扩展信息，不能仅凭这些默认值推断 `outside_rois` 或测量失败。
- 新旧 `MaskRatio` 的计算含义不同，旧报告数据不会因更换 DLL 或查看器而自动重新计算。跨版本比较时应区分生成结果所用算法版本。
- 本次构建的 Release DLL 位于 `src/Bridge/bin/x64/Release/net48/Bridge.dll`；根目录原有 `Bridge.dll` 未在本次修改中覆盖。

实现依据：[C# DTO](../src/Bridge/BridgeDtos.cs)、[弯曲测量实现](../python/algorithm/2D/defect_geometry.py)、[ROI 汇总实现](../python/algorithm/2D/wire_roi_presence.py)。完整返回结构见 [接口返回参数说明](OUTPUT_PARAMETERS.md)。
