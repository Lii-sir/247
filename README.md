# 银浆检测与部件分割

断连、溢出、分割预览和模板找点是独立功能；通用代码只在 `common` 中维护。环境由使用者配置，启动不会自动安装或同步依赖。

## 文件结构

```text
common/                     # 共享层，不导入任何业务模块
  image_io.py               # 图片读写、扫描、文件选择过滤器
  paths.py                  # 权重与数据集默认路径
  cli.py                    # 通用输入和推理参数
  matching.py               # SIFT 配准、坐标变换、匹配结果绘制
  segmentation/             # 推理参数/结果、YOLO 适配、分割绘图
  widgets/                  # 图片缩放、平移、选点控件
part_segmentation/          # 独立分割预览与批量导出
point_matcher/              # 独立模板找点界面与坐标导出
silver_continuity/          # 断连：算法、界面、导出、入口、说明
silver_overflow/            # 溢出：算法、标定、界面、导出、入口、说明
segmentation_transfer/      # 跨模板分割映射（现有源码尚不完整）
tests/                      # 按模块分类的测试；support 为合成数据
datasets/                   # 输入图片，检测不写入
weights/                    # 模型权重，不修改
outputs/                    # 导出结果，不属于源码
main.py                     # 只负责快捷入口分发
```

依赖规则、路径迁移见 [结构说明](D:/python_programs/LXD_project/point-matcher/docs/architecture.md)。

## 启动

在项目根目录执行：

```powershell
Set-Location "D:/python_programs/LXD_project/point-matcher"
uv run --no-sync python -m part_segmentation
uv run --no-sync python -m silver_continuity
uv run --no-sync python -m silver_overflow
uv run --no-sync python -m point_matcher
```

原快捷入口保留：`main.py` 默认分割，`--segment` 分割，`--silver` 断连，`--overflow` 溢出。每次启动一个窗口，同一张图可分别运行两个检测功能。
旧的 `python -m silver_inspection` / `--mode` 已移除，请使用对应独立模块。

- 推理入口默认 `--device 0`（CUDA）；调试可显式指定 `--device cpu`，没有 CUDA 不会自动回退。
- `--no-sync` 禁止 uv 启动时同步项目环境；也可直接用 `.venv/Scripts/python.exe`。不要为了启动运行 `uv sync`。
- 权重、数据集默认位置保持不变；只加载可信 `.pt` 权重。

## 功能说明

| 功能 | 思路与说明 |
| --- | --- |
| 断连 | chip 外接矩形外扩 → 排除遮挡 → 按角度检查 silver；[断连文档](D:/python_programs/LXD_project/point-matcher/silver_continuity/README.md) |
| 溢出 | silver 分割 → 模板允许框映射 → 距离/面积过滤；[溢出文档](D:/python_programs/LXD_project/point-matcher/silver_overflow/README.md) |
| 分割 | 原图掩膜预览、显示参数、独立批量导出；[分割文档](D:/python_programs/LXD_project/point-matcher/part_segmentation/README.md) |
| 找点 | SIFT + RANSAC 配准后映射手选点；[找点文档](D:/python_programs/LXD_project/point-matcher/point_matcher/README.md) |

## 批量示例

```powershell
uv run --no-sync python -m part_segmentation --source "D:/python_programs/LXD_project/point-matcher/datasets" --export "D:/python_programs/LXD_project/point-matcher/outputs/segmentation"
uv run --no-sync python -m silver_continuity --source "D:/python_programs/LXD_project/point-matcher/datasets" --export "D:/python_programs/LXD_project/point-matcher/outputs/continuity" --outward-px 15
uv run --no-sync python -m silver_overflow --source "D:/python_programs/LXD_project/point-matcher/datasets" --calibration "D:/python_programs/LXD_project/point-matcher/outputs/boundary.json" --export "D:/python_programs/LXD_project/point-matcher/outputs/overflow"
```

溢出的标定 JSON 需先在界面保存。各入口 `--help` 查看本功能参数；`--recursive` 扫描子目录。输出不能放入输入目录；复用输出目录会覆盖同名结果。

## 验证

测试命令与已知缺文件问题见 [测试说明](D:/python_programs/LXD_project/point-matcher/tests/README.md)。本次结构重构不改变断连/溢出的判定阈值及算法。
