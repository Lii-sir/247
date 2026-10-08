# 文件结构与依赖边界

## 1. 分层原则

```text
main.py（入口分发）
   ├─ part_segmentation（分割预览）
   ├─ point_matcher（模板找点）
   ├─ silver_continuity（银浆断连）
   └─ silver_overflow（银浆溢出）
                 ↓
common（图片 I/O、参数、分割、配准、控件）
                 ↓
NumPy / OpenCV / Ultralytics / PySide6
```

- 业务包不互相导入；共同能力下沉到 `common`。
- `common` 不反向导入业务，不负责断连/溢出路由。
- 算法、编排、界面、导出各司其职；Qt 仅在 app/控件层使用，YOLO 仅在共享推理适配层加载。
- 各业务的 `__main__.py` 只解析自身参数，`--export` 走批量，否则走自身 GUI。
- `main.py` 只保留快捷分发，默认功能仍是分割。
- 数据、权重、已有导出目录和环境配置不随源码迁移。

## 2. 业务文件职责

| 文件 | 职责 |
| --- | --- |
| geometry.py | 本功能的参数、掩膜/边界测量（断连与溢出分别维护） |
| pipeline.py | 组合共享能力，生成本功能判定结果 |
| app.py | 本功能的 Qt 交互与后台任务 |
| visualization.py | 本功能绘图，不推理、不写盘 |
| export.py | 本功能结果保存/批量导出 |
| calibration.py | 溢出专用的允许框标定 |
| __main__.py | 本功能启动参数与 GUI/批量路由 |

不是所有功能都需要上述全部文件；例如纯分割预览不再重复保存通用推理实现。

## 3. 主要路径迁移

| 旧路径 | 新位置 |
| --- | --- |
| silver_inspection/continuity.py | [silver_continuity/geometry.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/geometry.py) |
| silver_inspection/continuity_app.py、continuity_pipeline.py、continuity_export.py、continuity_visualization.py | [silver_continuity/app.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/app.py)、[silver_continuity/pipeline.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/pipeline.py)、[silver_continuity/export.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/export.py)、[silver_continuity/visualization.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/visualization.py) |
| silver_inspection/app.py、pipeline.py、geometry.py、calibration.py、export.py、visualization.py | 全部归入 [silver_overflow](D:/python_programs/LXD_project/point-matcher/silver_overflow/README.md) |
| part_segmentation/inference.py、models.py、visualization.py | [common/segmentation/inference.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/inference.py)、[common/segmentation/models.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/models.py)、[common/segmentation/visualization.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/visualization.py) |
| part_segmentation/image_io.py；point_matcher/core.py 内重复的 I/O | [common/image_io.py](D:/python_programs/LXD_project/point-matcher/common/image_io.py) |
| point_matcher/core.py 中配准与绘图 | [common/matching.py](D:/python_programs/LXD_project/point-matcher/common/matching.py) |
| point_matcher/core.py 中 CSV/JSON 导出 | [point_matcher/export.py](D:/python_programs/LXD_project/point-matcher/point_matcher/export.py) |
| part_segmentation/widgets.py；point_matcher/app.py 中 ImageView | [common/widgets/image_view.py](D:/python_programs/LXD_project/point-matcher/common/widgets/image_view.py)、[common/widgets/point_view.py](D:/python_programs/LXD_project/point-matcher/common/widgets/point_view.py) |
| 分散在 tests/、point_matcher/tests/ 的测试 | [tests 中按业务分类](D:/python_programs/LXD_project/point-matcher/tests/README.md)；测试数据统一放 tests/support |

旧 Python 导入路径不提供重导出壳，以免保留混杂结构。外部脚本需按表更新，例如：

```python
from common.segmentation.inference import PartSegmenter
from common.segmentation.models import SegmentationSettings
from silver_continuity.geometry import ContinuitySettings, measure_continuity
from silver_overflow.geometry import OverflowSettings, measure_overflow
```

原来的 `main.py --silver` / `--overflow` / `--segment` 仍可用；旧 `silver_inspection` 模块启动需迁移到独立入口，不再使用 `--mode`。

## 4. 回归保护

- [tests/test_structure.py](D:/python_programs/LXD_project/point-matcher/tests/test_structure.py)：检查共享层不导入业务、业务之间不互相导入、独立导入与 CLI 帮助不加载不必要组件。
- [tests/test_entrypoints.py](D:/python_programs/LXD_project/point-matcher/tests/test_entrypoints.py)：快捷路由、参数隔离、默认值、GUI/批量传参。
- 算法和 GUI 测试按功能组织，不再靠修改 sys.path 或导入另一测试模块取得数据。
- `segmentation_transfer` 已更新共享导入，但现有源码缺少 pipeline、diagnostics 等模块；这是重构前已有问题，没有通过跳过测试掩盖。

