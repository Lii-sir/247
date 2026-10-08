# Silver 银浆溢出检测

## 1. 功能与启动

检查 silver 是否超出模板定义的允许框。与断连检测独立：一张图片可以既断连又溢出。

环境由使用者配置；以下命令不执行依赖同步。在项目根目录执行：

```powershell
Set-Location "D:/python_programs/LXD_project/point-matcher"
# 可先单独检查分割效果
uv run --no-sync python main.py --segment --device 0
# 独立溢出界面
uv run --no-sync python -m silver_overflow --device 0
# 等价入口
uv run --no-sync python main.py --overflow --device 0
```

界面操作：打开模板 → 选闭合允许区域 → 按轮廓顺序点四个角形成框 → 保存标定 → 选待测图 → **开始检测当前图片**。
当前溢出界面一次完成分割、匹配和判定；没有断连界面那样的两阶段按钮。

## 2. 代码位置

| 文件 | 职责 |
| --- | --- |
| [silver_overflow/__main__.py](D:/python_programs/LXD_project/point-matcher/silver_overflow/__main__.py)、[silver_inspection/__main__.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/__main__.py) | 独立入口、参数、GUI/批量路由 |
| [part_segmentation/inference.py](D:/python_programs/LXD_project/point-matcher/part_segmentation/inference.py) | PartSegmenter：银浆分割、CUDA 推理、原图尺寸掩膜 |
| [point_matcher/core.py](D:/python_programs/LXD_project/point-matcher/point_matcher/core.py) | TemplateMatcher / MatchSettings：SIFT 配准、单应矩阵、边界点映射 |
| [silver_inspection/calibration.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/calibration.py) | Calibration：模板路径、边界点的保存和加载 |
| [silver_inspection/geometry.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/geometry.py) | Boundary / OverflowSettings / measure_overflow：距离、容差、面积过滤 |
| [silver_inspection/pipeline.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/pipeline.py) | SilverInspector 串联检测，evaluate 输出状态 |
| [silver_inspection/app.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/app.py) | 模板选点、参数和检测界面 |
| [silver_inspection/visualization.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/visualization.py)、[silver_inspection/export.py](D:/python_programs/LXD_project/point-matcher/silver_inspection/export.py) | 绘图、批量导出 |
| [tests/test_silver_inspection.py](D:/python_programs/LXD_project/point-matcher/tests/test_silver_inspection.py)、[tests/test_silver_gui.py](D:/python_programs/LXD_project/point-matcher/tests/test_silver_gui.py) | 几何、匹配、导出及界面测试 |

## 3. 检测流程

1. **定义允许框**：在模板上标定闭合边界，保存模板路径和边界点；不从 silver 自身推断允许区域。
2. **分割**：加载权重，提取目标图全部 silver 掩膜并取并集，避免实例重叠重复计数。
3. **匹配找点**：模板与目标图进行 SIFT 特征匹配、比值过滤和 RANSAC 单应估计；将模板边界点映射到目标图，连接成允许边界。
4. **计算越界距离**：对每个 silver 像素计算到边界的有符号距离 d：框内为负、边界为 0、框外为正。
5. **距离过滤**：`d > tolerance_px + 1e-6` 的像素作为溢出候选。
6. **面积过滤**：候选按 8 邻域提取连通域，面积 `>= min_area_px` 的区域保留为缺陷。
7. 存在保留区域 → `overflow`；分割与边界有效且无保留区域 → `ok`。

**这里的“找线”是映射标定点后连线，不是额外进行边缘搜索或直线拟合。**

## 4. 边界与标定

- 常规使用四角闭合框，按顺时针或逆时针选点；无需重复首点，不能交叉连线。
- 底层支持至少三点的简单多边形，不强制四点或直角；透视映射后也可能是四边形。
- 兼容 `line` 模式：两个直线端点 + 一个允许侧内点，判断的是无限直线的半平面，不是闭合框。
- 标定 JSON 必需字段：`template_path`、`template_points`；`boundary_mode` 默认 polygon，`schema_version` 为 1。
- `template_path` 若为相对路径，相对于标定 JSON 所在目录解析；移动标定文件时需检查模板路径。
- 模板点和映射点必须在对应图片范围内；匹配失败、不完整或映射越界不能判合格。

## 5. 参数

| CLI 参数 | 默认值 | 作用 |
| --- | --- | --- |
| --weights | D:/python_programs/LXD_project/point-matcher/weights/best.pt | 实例分割权重路径 |
| --source | D:/python_programs/LXD_project/point-matcher/datasets | 待测单图或文件夹 |
| --silver-class | silver | 银浆类别；不会自动用 bond 代替 |
| --calibration | 无 | 已保存标定 JSON；批量检测必填，GUI 可现场创建 |
| --tolerance-px | 0 | 允许越界距离，单位为目标原图像素；增大可容忍轻微偏差 |
| --min-area-px | 1 | 距离过滤后，每个溢出连通域的最小面积；增大可过滤小噪点 |
| --conf / --iou / --imgsz | 0.25 / 0.7 / 640 | 分割置信度、NMS IoU、推理尺寸（32 的倍数） |
| --device | 0 | CUDA GPU 编号；显式 cpu 用于调试，不会自动回退 |

模板匹配使用 `MatchSettings`，下列参数目前只能通过代码配置，不是溢出 CLI 选项：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| ratio_threshold | 0.7 | 最近邻/次近邻距离比过滤，越小越严格 |
| ransac_threshold | 5.0 | 目标原图像素尺度的 RANSAC 重投影阈值，不是银浆越界容差 |
| min_matches / min_inliers | 11 / 8 | 最少良好匹配数 / 最少内点数 |
| min_inlier_ratio | 0.25 | 最低内点占比 |
| max_features / max_image_side | 10000 / 2400 | SIFT 特征数上限 / 配准处理图最长边 |

断连的外扩距离、扇区数量、忽略类别和可见比例不参与溢出判断；当前溢出逻辑没有遮挡排除步骤。

## 6. 状态与输出

| 状态 | 含义 |
| --- | --- |
| ok | 在当前容差与最小面积下未发现溢出 |
| overflow | 检出溢出缺陷 |
| no_silver | 未检出 silver 或银浆掩膜为空，不能视为合格 |
| uncertain | 匹配、映射边界或掩膜无效，无法判定 |
| error | 批量处理中遇到运行异常 |

图示：青色为允许边界；绿色为银浆；橙色为被距离/面积过滤的框外银浆；红色为保留缺陷。

批量示例（先在界面保存标定；输出目录不能放在输入图片目录内）：

```powershell
uv run --no-sync python -m silver_overflow --source "D:/python_programs/LXD_project/point-matcher/datasets" --calibration "D:/python_programs/LXD_project/point-matcher/outputs/boundary.json" --export "D:/python_programs/LXD_project/point-matcher/outputs/overflow-report" --device 0 --tolerance-px 2 --min-area-px 10
```

示例中的 2 px / 10 像素为演示参数，不是默认值或已验证的生产阈值。
输出：每图 `.overlay.png`、`.comparison.jpg`、`.overflow.png`（缺陷二值图）以及 `summary.json`（配置、匹配质量、映射边界、面积与距离统计）。加 `--recursive` 扫描子目录；复用输出目录会覆盖同名结果。

- `outside_area_px`：全部框外银浆面积；`candidate_area_px`：距离过滤后面积；`defect_area_px`：再经面积过滤后面积。
- `max_outside_distance_px`：全部 silver 的最大越界距离，不仅统计最终保留缺陷。
- 无法判定时也可能导出全黑二值图；**以 JSON status 为准，不能看到黑图就认为合格**。
- 批量退出码：只有 ok/overflow 时为 0；存在 no_silver/uncertain/error 时为 1。与断连入口的退出码语义不同。

## 7. 使用边界

- 先检查分割，再检查允许框是否贴合目标；框整体错位时优先修正标定或匹配，不要靠增大容差掩盖。
- 面积阈值作用于单个连通域，不是所有小区域面积之和。
- 当前合并整张图所有 silver 后与一个允许边界比较，不会自动为每个 chip 分配单独允许框。
- 只能判断模型检测到的银浆；遮挡下的真实状态、漏分割部分无法由本流程确认。

