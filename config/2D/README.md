# 2D 配置

2D 配置采用“功能块存在即启用”的结构，不再配置 `enabled`。每个功能块包含
`description` 与 `parameters`；现场只需修改参数内容。

- `common_2d.parameters`：圆搜索、检测裁剪、板级 2D 兜底裁剪/引线 ROI、
  竖向分割、掩码和 YOLO 参数。
- `wire_bending_2d.parameters`：引线/细线 ROI 和弯曲度阈值。

- `silver_continuity_2d.parameters`：银浆断连/联通参数。
- `silver_overflow_2d.parameters`：检测 ROI 四边内缩生成的银浆允许边界、容差和最小缺陷面积。
- `chip_rotation_2d.parameters`：芯片外接矩形角度阈值。
- `chip_anomaly_2d.parameters`：芯片外观异常和 EfficientAD 模型区间映射。

引线弯曲使用清理后的单个 `wire/thin` 实例，在重叠像素最多的 ROI 中寻找两端与边界的接触区间，
各取两个端点连接成评估四边形。ROI 为 `[x1,x2) × [y1,y2)`，右/下边界像素为 `x2-1/y2-1`。
覆盖率为 `该实例在 ROI 和四边形交集内的像素数 / 该实例在 ROI 内的像素数`；
`rectangle_ratio_threshold` 仍为通过下限（含等号），保留当前 0.75 作为初值，需按样本重新校准。
`rectangle` 字段兼容保留，实际为四边形四个有序顶点。交点不足、多个端口或四点退化时直接判无效/NG，
不回退旋转矩形。每个 ROI 必须有引线且其全部实例合格，整体才合格。

芯片长宽比定义为 `long_side / short_side`。`aspect_ratio_models` 使用明确的
数组对象，例如 `{"range": [0.9, 1.1], "model": "chip_short"}`。区间不能
反向或重叠；没有命中区间会报配置错误，不会静默选择错误模型。当前示例中约
1.0 选择 `chip_short`，约 1.5 选择 `chip_long`。

`vertical_split.offset_radius` 控制实验脚本中的竖向分割线：
`split_x = round(circle_x + offset_radius * radius + offset_px)`。ROI 工具保存时
会更新 `common_2d.parameters` 和对应功能块的 `parameters`。
## 银浆允许边界：复用检测 ROI，四边按像素内缩

当前 `silver_overflow_2d.parameters` 为：

```json
{
  "boundary": {
    "mode": "roi_inset",
    "inset_unit": "px",
    "insets": {"left": 376, "top": 146, "right": 34, "bottom": 128}
  },
  "silver_class": "silver",
  "tolerance_px": 0.0,
  "min_area_px": 1
}
```

四边参数是非负有限数，单位为原始 2D 图像像素，可独立调整，支持小数。
当前初值来自原标定模板：先找圆计算该模板的检测 ROI，再取原标定四边形各边中点，
换算为距对应 ROI 边的距离并四舍五入；这只是当前项目初值，需要按现场样本核对。
默认单位为 `px`；可选 `radius_ratio` 则将各内缩量乘以本片圆半径。
修改配置后需重新初始化算法。

已有检测框为 `[x1,y1,x2,y2]`，允许边界顶点依次为：
`[x1+left,y1+top]`、`[x2-right,y1+top]`、`[x2-right,y2-bottom]`、`[x1+left,y2-bottom]`。
使用 ROI 连续坐标边界 `x2/y2`，不减 1；边界线上的像素允许通过，0 内缩覆盖完整检测 ROI。
左右内缩量之和必须小于 ROI 宽度，上下之和必须小于高度；缺参数、负值、非有限值或
内缩过大时返回 `uncertain/is_ok=null`，不自动放宽、不回退模板坐标。
圆定位失败沿用 2D 预处理失败逻辑，不会用旧帧圆或边界继续检测。

此模式直接复用本片的检测 ROI，不重复找圆，不读取模板、不做特征匹配；轴对齐矩形用
向量化距离公式判断越界，角点使用欧氏距离以保留原 `tolerance_px` 语义，连通域过滤规则不变。
它随检测 ROI 平移；使用 px 时内缩距离不随半径缩放。矩形不会跟随工件旋转或透视形变。
只检查检测 ROI 内已有银浆掩码，不能保证 ROI 外没有缺陷。

C# 调用和字段不变，`Boundary.Mode` 和 `Registration.Method` 均为 `roi_inset`，
`Registration.Status=ok`、`IsValid=true` 表示边界构建成功，**不是模板配准成功**；
`FallbackUsed=false`，不产生匹配/内点数。`Boundary.Points` 正常公开输出为 2D 原图坐标。
可视化显示黄色矩形，并注明“ROI 内缩边界”。

旧模板模式仍保留。需要切换时，将 boundary 改为
`{"mode":"polygon","calibration_path":"calibration/silver_boundary.json"}`。
此时才读取原标定模板和四边形，逐张 SIFT/FLANN/RANSAC 配准；
`fallback_to_template_points` 仅对该模板模式生效，不影响 `roi_inset`。
模板标定工具仍为 `visualization/calibrate_silver_boundary.py`；执行其 `--apply` 会安装标定并切回模板模式。

运行 `scripts/roi_configurator.py` 可在同一张 2D 图片上标定 3D 引脚平台参考区：
按 `F` 添加矩形，拖动框内移动、拖动角点拉伸，`V` 清空平台框，`U` 撤销，
`S` 同时保存两份配置。平台框只写入 `config/3D/config.json` 的
`height_measurement_3d.parameters.platform_roi_offsets`，不混入 2D 参数。
引线弧高仅使用 `wire_bending_2d.parameters.roi_offsets` 内的 wire/thin 掩码；
ROI 外部分不参与测高，thin 焊脚仍从完整掩码的两端划分。

例如：

```powershell
.venv\Scripts\python.exe scripts\roi_configurator.py --image test_data\2D\_2_1_20260929221235223_00331.png
```

在完整 2D 图像上按 `F` 框住干净的引脚平台表面（避开芯片、焊点和引线；可添加多个矩形），
按 `S` 保存。该矩形会按本片检测圆保存为相对坐标；板上其它片会各自用检测圆和透视矩阵映射，
不需要给每片单独维护参数。

## 线程配置

在 `common_2d.parameters.threading` 中配置 `postprocess_workers`（当前默认 5）和 `gpu_workers`（当前默认 2）。统一板级检测会为每个 GPU 工作线程加载独立的 YOLO 与 EfficientAD 模型副本；增加 GPU 工作线程时，也要相应提高 3D 配置中的 `worker_budget`。独立 2D/3D 旧接口目前只支持一个 GPU 工作线程。
