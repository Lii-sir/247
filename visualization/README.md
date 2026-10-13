# 独立检测结果查看器

## 银浆允许边界：模板标定 → SIFT 映射 → 查看黄色边界

银浆溢出现在使用 `config/2D/calibration/silver_boundary.json` 中的预标定多边形。
SIFT 匹配的是模板与当前图像的特征点，得到 Homography 后映射人工标定的边界顶点；
不会从当前银浆 mask 或芯片框自动生成允许边界。初始标定点为空，未安装真实标定时
诊断为 `not_configured / is_ok=null`，不会回退为芯片外扩或判 OK。

**1. 用现有 HTML 查看器标定。** 在项目根目录运行：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\calibrate_silver_boundary.py `
  --image .\demodatasets\2D\001.png --open
```

建议选择视角、分辨率与生产图一致的标准良品图。页面中勾选“点击添加边界点”，
沿真正允许银浆区域顺时针或逆时针点击至少三个顶点。滚轮缩放，按住 Shift 拖动；
点必须位于检测 crop 内，不能自交。点击“下载标定 JSON”，得到 `silver_boundary.json`。
查看器把屏幕坐标转换成原图像素坐标；记录原图尺寸和 SHA-256，以免把标定点装到另一张图。

**2. 显式安装下载的标定。** `--image` 必须使用标定时同一张原图：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\calibrate_silver_boundary.py `
  --image .\demodatasets\2D\001.png `
  --apply "$env:USERPROFILE\Downloads\silver_boundary.json"
```

脚本校验点序、边界面积、原图尺寸和哈希；复制原图为 `config/2D/calibration/` 下的模板，
保存标定并备份旧 JSON/配置为带时间戳的 `.bak` 文件。仅更新银浆 `boundary` 配置，
不修改其它检测阈值。安装后重启服务或重新初始化算法。

**3. 检测并查看边界：**

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\detect_and_view.py `
  --mode stream --parallel --timeout 900 --open
```

查看器的“银浆标定边界”开关显示黄色多边形及 B1/B2…顶点，紫色表示银浆，红色轮廓是
实际越界像素。银浆边界配准卡片显示匹配数、内点数和重投影误差；详细 JSON 保存
`homography_template_to_image`。匹配不足、模板内容变化、边界落出 crop 时返回
`uncertain / is_ok=null`，不得解释为通过。

也可仅为本次检测选择其它标定文件，不改生产配置：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\detect_and_view.py `
  --mode stream --parallel --silver-calibration D:\calibration\silver_boundary.json --open
```

标定 JSON 中的 `template_path` 相对于标定 JSON 所在目录解析，`template_points` 使用
完整模板原图坐标。运行时先映射到当前原图，再转换到检测 crop；响应中的
`boundary.coordinate_system=image` 表示坐标已经转回原图，查看器不会再叠加 crop 偏移。
普通离线报告只展示算法实际返回的边界，不重新运行 SIFT 或自行推算边界。

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s .\tests -p test_silver_boundary.py -v
```

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
  --score-mode pool+top --score-pool-kernel 7 `
  --threshold "chip_short=0.4,chip_long=2.0" --open
```

两个模型的阈值独立。`--threshold` 推荐按模型名分别传入，示例值 `0.4`、`2.0` 仅演示语法，不是产线推荐阈值。只传 `chip_short=0.4` 不会改变 `chip_long` 的配置或默认阈值。为兼容旧命令，单个数字仍作为显式覆盖两个模型的选项，但不建议用于需要独立阈值的场景。

`--threshold` 会随每张 2D 推理请求传递，不是算法初始化参数。优先级是：本次请求中对应模型的阈值 → 初始化配置中对应模型的阈值 → 该模型 checkpoint 自己的校准阈值。`pool+top` 优先使用各自 checkpoint 的 `pool_top_threshold`；没有该字段时兼容回退到该模型原始 `threshold`。运行覆盖只影响判定和异常框，不改权重或 checkpoint，响应的每个芯片都返回实际 `threshold` 和 `threshold_source`。Python 协议也支持在单片 `detect_2d_piece` 请求的 `threshold` 字段中传数值或每模型字典；C# `DetectStreamingFilesAsync`/`DetectPieceFile` 同样接受单个 `double` 或 `Dictionary<string,double>`。只指定一个模型的阈值时，另一个模型沿用其配置/自身 checkpoint 阈值。

配置形式：在 `config/2D/config.json` 的 `chip_anomaly_2d.parameters` 中添加：

```json
"threshold": {"chip_short": 0.4, "chip_long": 2.0}
```

自定义 Python 协议客户端也可在 `detect_2d_piece` 或独立 `detect(stage=2d)` 请求中传：

```json
"threshold": {"chip_short": 0.4, "chip_long": 2.0}
```

池化归一化仍按 checkpoint 对应 kernel 的校准值读取；更改 kernel 或评分模式后应验证并重新校准阈值。

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
& .\.venv\Scripts\python.exe -B .\visualization\tests_threshold.py -v
```

注意：深度伪彩预览将有效灰度归一化，不能当作毫米高度图；精确毫米数只展示算法返回的测量。原图会缩小嵌入报告，叠加轮廓仍使用原始图像坐标。输入 JSON 应来自可信本地检测程序。
