# Silver 银浆断连检测

## 1. 功能与启动

检查 chip 外接矩形周围是否存在 silver。与溢出检测独立，同一张图片可以分别执行两种检测。

环境由使用者配置；以下命令不执行依赖同步。在项目根目录执行：

```powershell
Set-Location "D:/python_programs/LXD_project/point-matcher"
uv run --no-sync python -m silver_continuity --device 0
# 等价入口
uv run --no-sync python main.py --silver --device 0
```

界面操作：选图 → 设置参数 → **开始分割**，先检查掩膜 → **断连检测** → 保存对比图。
修改断连参数可复用已有分割结果；修改权重、置信度、推理尺寸或设备需重新分割。

## 2. 代码位置

| 文件 | 职责 |
| --- | --- |
| [silver_continuity/__main__.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/__main__.py) | 独立命令行入口、GUI/批量路由；共享参数在 common/cli.py |
| [common/segmentation/inference.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/inference.py) | PartSegmenter：加载权重、推理、返回原图尺寸布尔掩膜 |
| [silver_continuity/geometry.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/geometry.py) | ContinuitySettings、measure_continuity：环带、遮挡、扇区测量 |
| [silver_continuity/pipeline.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/pipeline.py) | evaluate_continuity：汇总判定；SilverContinuityInspector：串联推理 |
| [silver_continuity/app.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/app.py) | 分割预览与断连界面、动态忽略类别图例 |
| [silver_continuity/visualization.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/visualization.py) | 环带、内外框、遮挡及缺失区域绘制 |
| [silver_continuity/export.py](D:/python_programs/LXD_project/point-matcher/silver_continuity/export.py) | 批量图片和 JSON 导出 |
| [tests/silver_continuity/test_continuity.py](D:/python_programs/LXD_project/point-matcher/tests/silver_continuity/test_continuity.py)、[tests/silver_continuity/test_legend.py](D:/python_programs/LXD_project/point-matcher/tests/silver_continuity/test_legend.py) | 算法、图例与 wire 忽略回归测试 |

## 3. 检测流程

1. 分割 silver、chip 和配置的遮挡类别；同类掩膜取并集。
2. 对 chip 并集提取连通域，过滤面积过小的区域；每个有效连通域分别检测。
3. 取 chip 掩膜的**轴对齐外接矩形**，不是模型检测框，也不是旋转矩形。
4. 矩形四边向外扩展指定像素并裁到图像范围，形成环带：

   `环带 = 外扩矩形 − 原矩形 − 所有 chip 像素`

5. 合并遮挡类别掩膜，可额外膨胀；以矩形中心按角度将环带划为 N 个扇区。
6. 每个扇区计算：

   - `R`：环带像素；`O`：遮挡像素；`V = R − O`：可见像素。
   - `S`：可见区域中的 silver 像素。
   - 可见比例 `V/R`；银浆覆盖率 `S/V`。

7. 按以下顺序判定扇区：

| 条件 | 扇区状态 |
| --- | --- |
| V 小于最少有效像素，或存在遮挡且 V/R 小于最小可见比例 | 有遮挡记 ignored；无遮挡记 insufficient；均不参与缺失判断 |
| S 达到最少 silver 像素，且 S/V 达到最低覆盖率 | covered |
| 其余可检查扇区 | missing |

全图存在 missing → `disconnected`；有效扇区为 0 → `uncertain`；否则 → `ok`。
未检出 chip/silver、掩膜或环带无效也为 `uncertain`；批量运行异常记 `error`。

## 4. 参数

默认值按命令行入口（及其启动的界面）列出，长度/面积均使用原图像素。

| CLI 参数 | 默认值 | 作用 |
| --- | --- | --- |
| --weights | D:/python_programs/LXD_project/point-matcher/weights/best.pt | 实例分割权重路径 |
| --source | D:/python_programs/LXD_project/point-matcher/datasets | 待测单图或文件夹 |
| --silver-class / --chip-class | silver / chip | 对应权重类别名 |
| --occlusion-classes | thin,bond,wire | 这些类别的环带像素不参与判断；不是忽略整类银浆实例 |
| --outward-px | 20 | 外接矩形外扩距离；过大可能把远处银浆算入 |
| --sectors | 72 | 角度段数，即默认每段 5°；越多越敏感，也越易受噪声影响 |
| --min-silver-px | 3 | 每个有效扇区至少需要的 silver 像素数 |
| --min-coverage | 0.01 | 有效扇区 silver 覆盖率下限（1%） |
| --min-valid-px | 1 | 扇区参与判断所需最少可见像素 |
| --min-visible-ratio | 0.10 | 有遮挡时，可见比例低于 10% 则跳过扇区；设为 0 关闭比例过滤 |
| --occlusion-dilation-px | 0 | 遮挡掩膜额外膨胀半径；调大可容忍边缘误差，但也会掩盖真实缺口 |
| --conf / --iou / --imgsz | 0.25 / 0.7 / 640 | 分割置信度、NMS IoU、推理尺寸（32 的倍数） |
| --device | 0 | CUDA GPU 编号；显式 cpu 用于调试，不会自动回退 |

代码参数 `min_chip_area_px=100` 用于过滤 chip 小连通域，当前未暴露 CLI/界面控件。
直接创建 `ContinuitySettings()` 时，遮挡类别默认仍为 `thin,bond`；通过 CLI 启动时为 `thin,bond,wire`。调用 API 请显式传入所需类别。

## 5. 输出与排查

- 白框：chip 外接矩形；青框：外扩矩形；紫色：环带；绿色：silver；黄色：配置类别遮挡；红色：缺失扇区内未遮挡且没有 silver 的像素。
- 图例随忽略类别输入框更新；模型未分割出的遮挡像素不会因填写类别名而自动忽略。
- 例如 R=536、O=526、V=10：可见比例约 1.9%，默认跳过该扇区，避免少量边缘像素触发告警。这是容忍规则，不代表已证明遮挡下银浆连续。
- 排查顺序：分割掩膜 → 内外框位置 → 遮挡类别 → 缺失扇区的 V、S、O、visible_ratio → 再调阈值。

批量示例（输出目录不能放在输入图片目录内）：

```powershell
uv run --no-sync python -m silver_continuity --source "D:/python_programs/LXD_project/point-matcher/datasets" --export "D:/python_programs/LXD_project/point-matcher/outputs/continuity-report" --device 0 --outward-px 15 --occlusion-classes thin,bond,wire --min-visible-ratio 0.10
```

输出：每图 `.continuity.png`、`.comparison.jpg` 和汇总 `summary.json`（含配置、状态、每个 chip 的内外框、每扇区统计）。加 `--recursive` 扫描子目录；复用输出目录会覆盖同名结果。
批量退出码：全部 ok 为 0；存在断连、无法判定或错误为 1。

## 6. 当前限制

- 这是**扇区内银浆存在性检查**，不是 silver 掩膜的拓扑连通性证明；小于扇区尺度的缺口可能漏检。
- chip 按掩膜并集的连通域拆分；接触的 chip 可能合并，破碎掩膜可能拆成多个 chip。
- ok 仅表示有效扇区通过，跳过的扇区未被验证。当前未设置全圈最小可检查比例，也未要求每个 chip 均有有效扇区；图像裁切或大面积遮挡需人工复核。
- 全图 silver 面积含遮挡重叠，扇区 S 已排除遮挡；全图 coverage_ratio 是通过扇区数/有效扇区数，不是像素覆盖率。

