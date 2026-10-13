# 3D 配置

3D 配置采用 `description + parameters` 功能块，不使用 `enabled` 开关。

- `common_3d`：板号关联所需片数。
- `segmentation_3d`：一张整板深度图分成十片的参数。
- `registration_3d`：初始尺度模板匹配参数。
- `dynamic_affine_3d`：每片运行时仿射优化参数。
- `perspective_registration_3d`：左侧框架边缘和圆特征透视拟合参数。
- `height_measurement_3d`：芯片/引脚平台平面、焊脚和引线高度阈值。引脚平台 ROI 在 2D 图片上由 ROI 工具画定，以该片圆心/半径归一化后保存在本文件；每片处理时使用其透视矩阵映射到 3D，再排除所有对齐后的 YOLO 掩码拟合平台平面。未画 ROI 时平台参考面无效，不会退回固定 3D AOI。
- `wire_height_3d`：独立 3D 接口的引线中心线 P95 参数。

板级接口会把 `config/2D/config.json` 的 2D 结果缓存与本文件的 3D 参数组合；
收到同一 `board_id` 的整板 3D 数据后完成配准、测高并释放十张 2D 缓存。
