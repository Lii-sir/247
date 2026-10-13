# 独立检测结果查看器

查看器与算法/C# Bridge 独立，不修改现有服务或检测流程。`generate_report.py` 只读取已保存结果，不加载 YOLO；可选的 `detect_and_view.py` 是独立检测客户端，调用现有 Python 服务，保存原始结果并生成报告。

## 最简单：检测并查看现有样本

在项目根目录执行：

```powershell
& .\.venv\Scripts\python.exe .\visualization\detect_and_view.py --open
```

默认读取 `demodatasets/2D` 的 10 张图片以及 `demodatasets/3D` 的唯一深度图，走逐片 2D → 整板 3D 的流式路径，结束后生成并打开报告。需要已安装算法依赖和可用 CUDA GPU；初始化或检测失败也会保存错误 JSON/日志和报告。

输出保存在 `visualization/output/<时间戳>/`：`result.json`、`report.html` 和 `python.log`。报告自包含；查看时不需要保持 Python 服务运行。

指定自己的数据：

```powershell
& .\.venv\Scripts\python.exe .\visualization\detect_and_view.py `
  --images-2d D:\data\2D --depth D:\data\3D\board.png `
  --board-id board-001 --open
```

EfficientAD 评分可以在启动时显式传参：

```powershell
& .\.venv\Scripts\python.exe .\visualization\detect_and_view.py `
  --score-mode pool+top --score-pool-kernel 7 --open
```

同样的参数也可以写入 `config/2D/config.json` 的 `chip_anomaly_2d.parameters`。运行参数优先于模型 checkpoint 中的评分方法；池化归一化优先读取对应 kernel 的校准值，阈值使用 checkpoint 的 `pool_top_threshold`，没有该字段时兼容使用原 `threshold`。产线切换评分方式后应重新校准阈值。

如需与当前 `Bridge.Demo` 的批量路径比较，添加 `--mode board`。批量路径没有执行全部流式 2D 诊断；报告会显示“未提供”，不会将缺失数据标为 OK。这个客户端不改变现有算法的串行调度或合格判定。

## 1. 保存检测结果

查看器输入的是 Bridge 返回的 JSON。当前 `Bridge.Demo` 只打印摘要，没有自动保存完整响应；建议上位机将 `DetectResponse` 序列化为 JSON 保存，例如：

```csharp
File.WriteAllText("D:/results/board-001.json", JsonSerializer.Serialize(response));
```

也可以保存同一块板的逐片响应数组（不能混合多个板号）：

```json
[
  {"cmd":"detect_2d_piece", "board_id":"board-001", "piece_id":1, "results_2d":{}},
  {"cmd":"detect_2d_piece", "board_id":"board-001", "piece_id":2, "results_2d":{}},
  {"cmd":"detect_3d_board", "board_id":"board-001", "results_board":{}}
]
```

## 2. 生成独立 HTML 报告

在项目根目录执行：

```powershell
& .\.venv\Scripts\python.exe .\visualization\generate_report.py `
  --result D:\results\board-001.json `
  --images-2d .\demodatasets\2D `
  --depth .\demodatasets\3D\_2_221241743_00033.png `
  --output D:\results\board-001.report.html `
  --open
```

生成的 HTML 是自包含文件，可以直接双击打开或复制到没有 Python 环境的电脑上查看。

如果只保存 C# `DetectResponse`，部分 Python 原始诊断字段会因 DTO 不完整而缺失；报告不推测这些字段。逐片 2D 的银浆/芯片诊断应一起保存，或使用上述独立客户端保存完整原始响应。

## 查看内容

- 整板 OK/NG、配准状态、耗时；
- 10 个片号的独立状态和配准分数；
- 2D 原图、ROI、圆和 mask 轮廓；
- 3D 深度伪彩、对齐 mask 和测高点；
- 引线弯曲、引线高度、焊脚高度；
- 银浆、芯片旋转、芯片异常等服务返回的诊断；
- 原始 JSON 和配准明细。

2D 图片目录会按照文件名自然排序，对应片号 1～N；如果上位机提交顺序不是文件名顺序，应在 JSON 中提供 `images_2d` 路径映射，或不要加载原图，只查看结果轮廓。

银浆溢出诊断如果返回 `measurement.regions[].box_xyxy`，查看器会在 2D 原图上用醒目的红色框绘制对应区域，并在框旁显示缺陷像素面积。

路径映射示例：

```json
{
  "response": {"ok": true, "results_board": {"pieces": []}},
  "responses_2d": [],
  "images_2d": [{"piece_id": 1, "path": "D:/data/2D/piece-01.png"}],
  "image_3d": "D:/data/3D/board.png"
}
```

## 验证

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s .\visualization\tests -v
```

注意：深度伪彩预览将有效灰度归一化，不能当作毫米高度图；精确毫米数只展示算法返回的测量。原图会缩小嵌入报告，叠加轮廓仍使用原始图像坐标。输入 JSON 应来自可信本地检测程序。
