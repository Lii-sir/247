# 部件分割预览

独立展示实例分割结果，不执行模板匹配、银浆断连或溢出判断。

## 代码位置

- [part_segmentation/__main__.py](D:/python_programs/LXD_project/point-matcher/part_segmentation/__main__.py)：独立入口；[part_segmentation/app.py](D:/python_programs/LXD_project/point-matcher/part_segmentation/app.py)：界面与后台线程；[part_segmentation/export.py](D:/python_programs/LXD_project/point-matcher/part_segmentation/export.py)：批量导出。
- 推理、数据模型、绘图在 [共享层](D:/python_programs/LXD_project/point-matcher/common/README.md)，不在本目录重复实现。

## 启动

在项目根目录、已配置环境中执行：

```powershell
uv run --no-sync python -m part_segmentation
# 等价快捷入口
uv run --no-sync python main.py --segment
```

选图 → 开始分割 → 检查叠加结果 → 保存当前效果。支持缩放、拖动、透明度、检测框/文字开关；显示调整不重新推理。
修改模型/推理参数需重新分割；分割在后台运行，关闭时等待当前任务结束。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --weights | 项目 weights/best.pt | 可信实例分割权重 |
| --source | 项目 datasets | 单图/文件夹 |
| --conf / --iou / --imgsz | 0.25 / 0.7 / 640 | 置信度、NMS IoU、32 倍数的推理尺寸 |
| --device | 0 | CUDA GPU 编号；cpu 需显式指定 |
| --recursive | 关闭 | 扫描子目录 |
| --export | 无 | 设置后无界面导出 |

```powershell
uv run --no-sync python -m part_segmentation --source "D:/python_programs/LXD_project/point-matcher/datasets" --export "D:/python_programs/LXD_project/point-matcher/outputs/segmentation"
```

每图 `.overlay.png`、`.comparison.jpg`，并生成 `summary.json`；保留原文件名扩展名和子目录，损坏图片记录错误后继续。
这是效果展示，不是分割精度评测；空预测不会视为运行错误。输出不能位于输入图片目录内。

