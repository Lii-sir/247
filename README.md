# 部件分割与模板找点

两个独立功能，**分割不调用找点，找点不加载 YOLO**：

| 模块 | 职责 |
| --- | --- |
| `part_segmentation/` | 使用 `weights/best.pt` 展示部件实例分割 |
| `point_matcher/` | 原有模板匹配、选点与坐标映射，代码保持不变 |
| `datasets/` | 输入图片，只读，不写入预测结果 |
| `outputs/` | 生成的展示图和统计信息，已加入 Git 忽略 |

## 安装与启动

在项目根目录执行（Python 3.12，依赖由 `uv.lock` 锁定）：

```powershell
uv sync
uv run python main.py
# 等价的独立模块入口
uv run python -m part_segmentation
```

默认加载数据集列表和权重路径；选中图片后点击 **开始分割当前图片**。

- 原图和预测结果并排展示，支持滚轮缩放、拖动平移、适应窗口。
- 每类颜色固定，显示实例编号、类别、置信度和原图掩膜面积。
- 可调整掩膜不透明度、检测框和文字标签；这些显示操作不重新运行模型。
- 修改置信度、推理尺寸或权重后会清除旧结果，需重新分割。
- **保存当前效果** 可选择对比图或叠加图，不允许覆盖本次输入图片和权重。
- 推理在后台线程运行；只保留当前图片结果，复用模型。处理期间不切换输入，关闭窗口会等待当前推理完成。
- 不进行训练、不修改权重、不与找点结果组合。

当前权重的类别名为 `bond`、`wire`、`chip`、`thin`，程序从权重读取类别名，而非硬编码。

## 无界面批量导出

```powershell
uv run python -m part_segmentation --export outputs/segmentation
```

每张图片生成 `原文件名.overlay.png` 和 `原文件名.comparison.jpg`；`summary.json` 包含推理设置、类别、置信度、框坐标、掩膜面积和失败记录。框坐标采用原图像素 `xyxy`；左上角为原点，X 向右、Y 向下。

默认仅扫描当前文件夹，`--recursive` 可包含子文件夹，导出保留子目录结构和原文件扩展名。输出目录不能位于输入目录内部，避免污染输入。重复使用同一输出目录会覆盖同名展示图和摘要，建议不同实验使用不同目录。损坏图片会记录错误并继续；存在失败图片时退出码为 1。

自定义输入、阈值与尺寸：

```powershell
uv run python -m part_segmentation --source datasets/1.bmp --conf 0.35 --imgsz 960
uv run python -m part_segmentation --source datasets --recursive --export outputs/run-960 --imgsz 960
```

默认 `--device cpu`，不需要 GPU；如已安装与本机 CUDA 匹配的 PyTorch，可用 `--device 0`。默认 `--imgsz 640` 与当前权重记录一致，要求 32 的倍数；增大尺寸会增加耗时和内存。`--iou` 默认 0.7，传给模型的 NMS 设置（实际是否应用取决于模型架构）。

这是**预测可视化，不是分割精度评测**；当前数据目录只有图片，没有用于计算 IoU/mAP 的真值标注。空预测会显示原图及“未检测到部件”，不会当作程序失败。只加载可信来源的 `.pt` 权重。

## 解耦结构

```text
part_segmentation/
  models.py         # 普通数据对象与推理参数，无 YOLO / Qt 依赖
  image_io.py       # 图片发现与中文路径读写
  inference.py      # 唯一 YOLO 适配层，返回原图尺寸的 NumPy 掩膜
  visualization.py  # 纯绘图，不推理、不写盘、不依赖 Qt
  export.py         # 无界面批处理与 JSON 摘要
  widgets.py        # 独立图片视图
  app.py            # Qt 交互与后台线程
  __main__.py       # 参数解析，选择桌面 / 批量入口
```

后续若需组合“分割 → 匹配找点”，在上层新增编排模块即可，不需要让两套算法互相导入。

原有找点程序可从根目录单独启动：

```powershell
uv run python -m point_matcher.app
```

## 测试

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s point_matcher/tests -v
```

单元测试使用合成掩膜 / 模拟推理，GUI 测试使用离屏 Qt；真实权重验收使用上面的批量导出命令。
