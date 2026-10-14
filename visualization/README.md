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
`homography_template_to_image`。所有目标图共用同一个标定文件，每张图独立计算变换。
当前标定启用 `fallback_to_template_points=true`：匹配不足或配准不可靠时，直接把
`template_points` 当作目标原图坐标，随后正常检测银浆溢出；不缩放、不平移点集。
此时 `registration.status=fallback`、`fallback_used=true`、`is_valid=false`，
查看器显示“已回退到模板原始点（未配准）”，检测结论仍来自实际越界像素。
该方式适合模板与目标图像素尺寸和拍摄位置一致的场景；目标位置变化时回退边界不会跟随。
设为 `false` 可恢复匹配失败时返回 `uncertain / is_ok=null`。
模板内容或尺寸与标定记录不一致、标定点非法、边界落出 crop 时仍返回不确定结果。

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

### 连续两批：首次启动推理 → 复用已加载模型再次推理

```powershell
$root = "D:\python_programs\LXD_project\247-likai"
Set-Location $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:BRIDGE_MAX_WORKERS = "13"
$env:BRIDGE_BOARD_TIMEOUT_SECONDS = "900"
& "$root\.venv\Scripts\python.exe" -B "$root\visualization\detect_and_view.py" `
  --mode stream --parallel --repeat 2 --timeout 900 --open
```

`--repeat 2` 只启动一个服务进程、只调用一次 `init_algorithm`。第一批完整结束后，
第二批直接推理同一批 10 张 2D 和整板 3D 图片，不重新加载模型。每批使用独立板号
（`-run-01`、`-run-02`），防止缓存串板；每批重新建立共享内存，模拟新的采集批次。
所有批次完成后才释放模型并生成 HTML，避免报告生成干扰后续推理计时。

输出目录包含：

- `startup.json`：服务 PID、初始化响应、模型加载耗时及初始化次数。
- `summary.json`：逐批输入准备、客户端检测耗时、请求时间窗口及是否复用模型。
- `batch-01/result.json`、`batch-01/report.html`：第一批。
- `batch-02/result.json`、`batch-02/report.html`：第二批。
- `python.log`：同一个服务进程的日志。

`batch_wall_ms` 含本批文件读取及共享内存准备，不含初始化、释放服务和报告生成；
`total_with_startup_ms` 仅在第一批额外计入初始化。第二批仍进行真实推理，不复用检测结果。
产品 NG 不阻止下一批；协议/检测请求失败则保留已有结果并停止后续批次。
不指定 `--repeat` 时仍只运行一次，保留原有 `result.json`、`report.html` 输出路径。

### 详细耗时日志：定位排队、GPU 等待和 CPU 后处理

```powershell
$root = "D:\python_programs\LXD_project\247-likai"
Set-Location $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:BRIDGE_MAX_WORKERS = "13"
$env:BRIDGE_BOARD_TIMEOUT_SECONDS = "900"
$env:BRIDGE_TIMING_LOG = "1"
# 可选诊断开关：额外同步 EfficientAD GPU 工作；会影响吞吐，不用于正常性能基准。
$env:BRIDGE_TIMING_CUDA_SYNC = "1"
& "$root\.venv\Scripts\python.exe" -B "$root\visualization\detect_and_view.py" `
  --mode stream --parallel --repeat 2 --timeout 900 --open
```

服务 stderr（客户端保存到 `python.log`）中的 `[TIMING]` 行是 JSON，含 `board_id`、
`piece_id`、`request_id`、`pid`、`thread`、`phase`、`stage`、`elapsed_ms`。
`event=begin/end` 是成对的阶段开始/结束；`duration` 是队列等待；异常也保存结束耗时。
`phase_summary` 汇总一个阶段的所有测量，`efficientad_chip_begin/end` 标明芯片实例，
`gpu_memory` 记录当前显存分配和保留量，`silver_fallback` 记录匹配失败原因。
这些日志不会进入协议 stdout。关闭文本日志可设 `BRIDGE_TIMING_LOG=0`，JSON 计时仍保留。

每批 `result.json` 的 `responses_2d[*].timings` 包含详细 2D 耗时；
`response.results_board.pieces[*].timings` 包含逐片联合 3D 耗时。
每批另外导出 UTF-8 `timings.csv`（`board_id,phase,piece_id,stage,elapsed_ms`），
HTML 中展开“配准、参考平面与阶段耗时”可查看选中片的 2D/联合 3D/整板计时。

主要字段：

| 阶段 | 字段 |
| --- | --- |
| 线程池排队 | `cpu_prepare_queue_ms`, `gpu_queue_ms`, `cpu_postprocess_queue_ms`, `joint_queue_ms` |
| 2D 输入与 ROI | `input_shared_memory_read_ms`, `input_image_decode_ms`, `circle_find_ms`, `roi_copy_ms` |
| YOLO | `cuda_before_yolo_ms`, `yolo_predict_wall_ms`, `cuda_after_yolo_ms`, `mask_extract_to_cpu_ms`，以及框架返回的 `yolo_reported_preprocess_ms`, `yolo_reported_inference_ms`, `yolo_reported_postprocess_ms` |
| 芯片外观 | `efficientad_chip_rectify_ms`, `efficientad_preprocess_ms`, `efficientad_forward_enqueue_ms`, `efficientad_cuda_after_ms`, `efficientad_score_and_response_ms` |
| 银浆边界 | `silver_template_read_decode_ms`, `silver_template_hash_ms`, `silver_template_sift_ms`, `silver_target_sift_ms`, `silver_flann_match_ms`, `silver_ransac_ms` |
| 银浆检测 | `silver_overflow_measure_ms`, `silver_continuity_measure_ms` |
| 联合 3D | `coarse_registration_ms`, `affine_registration_ms`, `perspective_registration_ms`, `mask_alignment_ms`, `height_measurement_ms` |
| 高度子步骤 | `height_fit_platform_plane_ms`, `height_fit_chip_plane_ms`, `height_wire_regions_ms`, `height_solder_ms` |
| 协议输出，仅日志 | `response_json_encode_ms`, `response_write_lock_wait_ms`, `response_pipe_write_ms` |

`matching_input_wait_ms` 表示一方数据已就绪后等待另一方的时间，不占用工作线程。
`shared_memory_read_decode_ms` 现在仅统计实际输入读取/解码，不再把 ROI/平台掩码算入其中。
所有计时单位为毫秒；父阶段包含子阶段、多片可能重叠，不能将全部字段相加。
YOLO 框架返回的三项耗时不一定覆盖首次 warmup/初始化等整个 `predict()` 调用开销；
若 `yolo_predict_wall_ms` 很长而 `yolo_reported_*` 很小，应优先检查框架调用的额外开销。
单片重复操作（如多个芯片的 EfficientAD）在该片 JSON/CSV 中累加，日志保留每次调用。
未开启额外 CUDA 同步时，`efficientad_forward_enqueue_ms` 主要反映 CPU 提交时间，
异步 GPU 等待可能计入 `efficientad_score_and_response_ms`；不能把提交时间当成纯 GPU 推理时间。

查看最新一次运行中最慢的 30 个阶段：

```powershell
$run = Get-ChildItem "$root\visualization\output" -Directory |
  Where-Object { Test-Path (Join-Path $_.FullName "summary.json") } |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem $run.FullName -Filter timings.csv -Recurse |
  ForEach-Object { Import-Csv $_.FullName } |
  Sort-Object { [double]$_.elapsed_ms } -Descending |
  Select-Object -First 30 board_id,phase,piece_id,stage,elapsed_ms |
  Format-Table -AutoSize
```

恢复正常吞吐测试时使用 `$env:BRIDGE_TIMING_CUDA_SYNC = "0"`；额外同步不影响算法判定，
但改变 CPU/GPU 交叠与计时分布，必须在相同诊断设置下比较各批数据。

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

联合 3D 检测的 `height_3d.point` 返回整板原始深度图坐标（`point_coordinate_space: "board_3d"`）：当前按行切片，因此 `x` 不变，`y = 分片内 y + depth_start_row`。引线、焊脚和实例中的高度点一致；`mask_3d_local` 与参考平面仍使用分片坐标。报告显示分片预览时自动减回行偏移；旧报告缺少坐标标记时继续按旧的局部坐标解释。独立 3D 引线测高的 `results_3d.pieces[].points` 原本已经是整图坐标，不再加偏移。
