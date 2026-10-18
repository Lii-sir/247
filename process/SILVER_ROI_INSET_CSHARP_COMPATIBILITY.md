# C# 银浆边界显示兼容性说明：ROI 内缩模式

更新日期：2026-10-18。

## 1. 问题概述

银浆允许边界已由模板配准切换为“检测 ROI 四边内缩”。C# 接口方法、请求参数和 DTO 字段没有增加，但以下字符串取值发生变化：

| 属性 | 模板模式常见值 | 当前 ROI 内缩模式 |
| --- | --- | --- |
| `Boundary.Mode` | `polygon` | `roi_inset` |
| `Registration.Method` | `sift_ransac` 或 `template_points_fallback` | `roi_inset` |
| `Registration.Status` | `ok` 或 `fallback` | 成功为 `ok`，构建失败为 `failed` |

如果上位机仅允许旧字符串进入绘图分支，可能出现：**接口已返回边界顶点，但图上没有黄色边界框。**

当前 DTO 中 `Mode`、`Method` 都是 `string`，可以直接接收 `roi_inset`。若上位机另有自定义枚举或反序列化转换器，也需加入该取值，不能把未知值静默当成模板模式。

## 2. 哪些旧代码可能受影响

### 2.1 只允许 polygon 绘制

```csharp
// 旧代码：收到 roi_inset 后不会进入绘图分支。
if (boundary.Mode == "polygon")
{
    // 绘制 boundary.Points
}
```

应将 `roi_inset` 与 `polygon` 都作为闭合多边形处理：

```csharp
bool canDrawPolygon = boundary.Mode == "polygon"
                   || boundary.Mode == "roi_inset";
```

不能把所有模式无条件闭合。旧 `line` 模式的第三点用于指示允许侧，不是三角形顶点，应沿用专门的直线绘制逻辑。

### 2.2 依赖 SIFT 匹配数或模板矩阵

ROI 内缩模式不做模板特征匹配，因此不会返回 `Matches`、`Inliers`、`ReprojectionErrorPx`、模板信息及模板到当前图的矩阵。

以下判断不能用于 ROI 内缩模式：

```csharp
// 若把这些条件作为所有模式的统一门槛，新边界可能被错误隐藏或判失败。
bool useTemplateBranch = registration.Method == "sift_ransac";
bool enoughMatches = registration.Matches >= 8;
bool enoughInliers = registration.Inliers >= 6;
```

应先按 `Registration.Method` 区分边界来源。`roi_inset` 使用生成结果和边界顶点；`sift_ransac` 才使用特征匹配统计。

`Registration.IsValid=true` 在新模式下表示**允许边界构建成功**，不表示执行过模板配准。不要再次用模板矩阵转换 `Boundary.Points`。

### 2.3 用边界有效性代替产品判定

`Registration.Status=ok`、`IsValid=true` 只说明边界生成成功；产品是否银浆溢出，应读取银浆溢出诊断对象的 `IsOk` 和 `Status`。

| 银浆溢出诊断 | 含义 |
| --- | --- |
| `Status=ok, IsOk=true` | 当前检测 ROI 内未发现满足阈值的银浆越界缺陷。 |
| `Status=overflow, IsOk=false` | 检出银浆越界缺陷。 |
| `Status=uncertain, IsOk=null` | 例如内缩过大导致边界无效；不能判产品通过。 |
| `Status=no_silver, IsOk=null` | 未检出银浆掩码；不能判产品通过。 |

边界生成失败时可能没有 `Boundary`，绘制前需要检查对象和点列表。也不要仅凭银浆诊断的 `Enabled` 默认值决定是否显示：部分响应未包含 `enabled` 字段，应结合实际诊断对象、`Status` 和边界判断。

## 3. Boundary.Points 的含义与坐标

`Boundary.Points` 仍是四个有序顶点，例如：

```json
{
  "mode": "roi_inset",
  "points": [[2400, 240], [3800, 240], [3800, 2460], [2400, 2460]],
  "coordinate_system": "image",
  "is_within_crop": true
}
```

依次连接：**第 1 点 → 第 2 点 → 第 3 点 → 第 4 点 → 第 1 点**，即可绘制闭合黄色允许边界。

当前公开的 `CoordinateSystem=image` 表示完整 **2D 原图像素坐标**：

- 画在原图上时，不再加 `CropBox` 偏移。
- 画在单独的检测 crop 上时，先减去 `CropBox` 左上角，再应用 crop 的显示变换。
- 原图缩放、平移或居中留白时，边界须应用与底图相同的显示变换。
- 即使数据来自联合 3D 检测，该银浆边界仍是 2D 坐标，不是 3D 测高点坐标。

原图显示变换为：

```text
屏幕 X = 原图 X × scaleX + offsetX
屏幕 Y = 原图 Y × scaleY + offsetY
```

`scaleX/scaleY/offsetX/offsetY` 必须与当前图片控件实际使用的一致，包含缩放、拖动和居中偏移，不能只按控件宽高盲目换算。

## 4. C# 绘图示例

以下为 WinForms/GDI+ 示例，WPF 等界面可使用同样的顶点检查和坐标变换逻辑。
传入完整 2D 原图的显示比例及偏移；此示例只处理闭合的 `polygon/roi_inset`。

```csharp
using System;
using System.Collections.Generic;
using System.Drawing;
using Bridge;

public static class SilverBoundaryDrawing
{
    public static bool Draw(
        Graphics graphics, Pen pen, BoundaryResult2D boundary,
        double scaleX, double scaleY, double offsetX, double offsetY)
    {
        if (boundary == null || boundary.Points == null
            || boundary.CoordinateSystem != "image")
            return false;

        if (boundary.Mode != "polygon" && boundary.Mode != "roi_inset")
            return false;
        if (boundary.Points.Count < 3)
            return false;
        if (boundary.Mode == "roi_inset" && boundary.Points.Count != 4)
            return false;

        var points = new List<PointF>();
        foreach (var point in boundary.Points)
        {
            if (point == null || point.Count != 2)
                return false;
            float x = (float)(point[0] * scaleX + offsetX);
            float y = (float)(point[1] * scaleY + offsetY);
            if (float.IsNaN(x) || float.IsInfinity(x)
                || float.IsNaN(y) || float.IsInfinity(y))
                return false;
            points.Add(new PointF(x, y));
        }

        // DrawPolygon 会连接最后一点与第一点，无需重复追加第一个顶点。
        graphics.DrawPolygon(pen, points.ToArray());
        return true;
    }
}
```

数据位置：单片 2D 可从 `response.Results2D?.SilverOverflow?.Boundary` 读取；联合结果可从 `response.ResultsBoard.Pieces[i].SilverOverflow2D?.Boundary` 读取。先检查各级对象是否存在，再调用绘图函数。`Pen`、`Graphics` 由界面代码创建和释放。

这个函数的返回值只表示是否执行了绘图，不能用作产品 OK/NG 结果。

## 5. 出现“不显示”或“错位”时如何排查

1. 确认查看的是本次检测响应，而非旧缓存。独立 HTML 报告内嵌旧脚本时，还需重新生成并重新打开报告。
2. 检查银浆诊断的 `Status/IsOk`，以及 `Boundary` 是否存在；边界无效不能当成绘图兼容问题。
3. 检查 `Boundary.Mode` 是否为 `roi_inset`，旧的 `polygon` 分支是否把它过滤掉。
4. 检查 `Registration.Method` 判断是否仍强制依赖 `sift_ransac`、内点数或模板矩阵。
5. 检查 `Points` 是否有四个有效点，当前公开坐标是否为 `image`。
6. 检查是否重复加了 crop 偏移，或图片与边界使用了不同的缩放、平移。
7. 检查图层开关、绘图区域和重绘顺序，避免边界画好后又被底图覆盖。

## 6. 升级范围

- 算法返回结构保持兼容，本次模式切换无需为接收字段重新编译 `Bridge.dll`。
- 若上位机存在上述字符串分支、枚举或显示逻辑，则需要修改并重新构建上位机程序。不能把“无需更新 DTO”理解为“所有上位机逻辑都无需调整”。
- 此结论针对本次银浆 ROI 内缩模式切换；此前引线弯曲新增 DTO 属性对应的 DLL 更新要求仍按原说明执行。

参考：[完整返回参数说明](OUTPUT_PARAMETERS.md)、[ROI 内缩配置说明](../config/2D/README.md)。
