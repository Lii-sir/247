# 2D 配置

2D 配置采用“功能块存在即启用”的结构，不再配置 `enabled`。每个功能块包含
`description` 与 `parameters`；现场只需修改参数内容。

- `common_2d.parameters`：圆搜索、检测裁剪、板级 2D 兜底裁剪/引线 ROI、
  竖向分割、掩码和 YOLO 参数。
- `wire_bending_2d.parameters`：引线/细线 ROI 和弯曲度阈值。
- `silver_continuity_2d.parameters`：银浆断连/联通参数。
- `silver_overflow_2d.parameters`：预标定多边形经 SIFT/RANSAC 配准后的银浆允许边界、容差和最小缺陷面积。
- `chip_rotation_2d.parameters`：芯片外接矩形角度阈值。
- `chip_anomaly_2d.parameters`：芯片外观异常和 EfficientAD 模型区间映射。

芯片长宽比定义为 `long_side / short_side`。`aspect_ratio_models` 使用明确的
数组对象，例如 `{"range": [0.9, 1.1], "model": "chip_short"}`。区间不能
反向或重叠；没有命中区间会报配置错误，不会静默选择错误模型。当前示例中约
1.0 选择 `chip_short`，约 1.5 选择 `chip_long`。

`vertical_split.offset_radius` 控制实验脚本中的竖向分割线：
`split_x = round(circle_x + offset_radius * radius + offset_px)`。ROI 工具保存时
会更新 `common_2d.parameters` 和对应功能块的 `parameters`。
银浆溢出边界单独保存在 `calibration/silver_boundary.json`，不再从芯片外接框生成。

运行 `visualization/calibrate_silver_boundary.py --image <模板原图> --open`，在现有
HTML 查看器中按顺序点选真实允许区域顶点并下载 JSON；之后使用同一脚本的
`--image <同一模板原图> --apply <下载的JSON>` 安装标定。
模板点为完整模板原图像素坐标，`template_path` 相对于标定 JSON 解析。真实图像通过
SIFT/FLANN/RANSAC 得到模板到当前图的 Homography，再将边界转到检测 crop 作越界判断。
所有图片使用同一份模板和点集，逐张配准。当前标定启用
`fallback_to_template_points=true`，匹配失败时直接将模板点作为目标原图坐标，
继续进行越界判断，结果中标记 `registration.status=fallback`、`fallback_used=true`。
回退不缩放或移动边界，需要模板与目标图拍摄位置和像素尺寸一致。
未标定、模板校验失败或边界落出检测 crop 时仍返回 `not_configured`/`uncertain`，
`is_ok=null`。设为 `false` 可禁用配准失败回退。

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
