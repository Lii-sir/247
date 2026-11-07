# Windows 仿射配准：跨机器完整测试流程

更新：正式流式流程已拆成配准池（`registration_workers`）和映射测高池（`joint_workers`）。本文CPU基准仍只测配准函数，不模拟其它阶段竞争；正式复验请分别调整两项，先固定测高2路比较配准4/6/8路，再比较测高2/4路。保持`early_joint_workers`和其它参数一致。每片同时比较`joint_pipeline_ms`、两个池的queue/phase和原仿射子计时，整板节拍以`total_detect_ms`为准。详见config/3D/README.md“线程配置”。

最新正式实现固定为常驻进程配准，默认4进程并启用Windows 4物理核分区。增加registration_workers时须同步增加registration_affinity.physical_cores并给主进程留核。CPU基准脚本仍直接运行求解器，不能用于证明进程隔离收益；实际收益需通过正式整板测试验证，并保存初始化registration_execution（PID/CPU分区/启动时间）和全部片的registration_worker及共享区/返回延迟。没有thread模式开关；需要旧版A/B时使用独立旧提交工作副本，不覆盖当前配置。

目标是解释配准台单片不到200ms、正式整板中单片约1800ms的差异，并验证已有优化开关。先测同一算法在不同并发下的表现，再比较算法开关。不要先修改生产线程数、标定或容差。

`benchmark_affine.py`只执行正式流程中的CPU配准函数，不加载YOLO、EfficientAD、PyTorch或模型。不要求CUDA显卡。它读取当前生产配置，先完成解码、Pitch、固定分片和粗配准；这些只执行一次，单独记录、不计入仿射阶段。每片保留完整2D原图灰度和对应深度片的完整宽度，与正式坐标一致。

测试不会改写生产配置。输出在`visualization/output/affine-benchmark/run-时间戳/`，每次新建目录。提前准备Python 3.10 x64、Git和访问仓库/软件包源的权限。若测试机只有CPU，仍可完成前五步；最终正式检测验收需原有GPU检测环境。

## 1. 获取相同代码和数据

第一次复制到另一台电脑，可在PowerShell中执行（目录按实际修改）：

```powershell
git clone --branch yinxkj --single-branch https://github.com/Lii-sir/247.git D:\affine-test\247-likai
if ($LASTEXITCODE -ne 0) { throw '克隆失败，请先解决仓库权限或网络问题' }
Set-Location D:\affine-test\247-likai
git rev-parse HEAD
git status --short
```

已有仓库则先检查`git status --short`，保留本地修改，再在`yinxkj`上`git pull --ff-only origin yinxkj`。不要使用`reset --hard`或覆盖当前配置。两台电脑对比时务必记录同一commit，脚本还会保存关键源码哈希和工作区状态。

仓库跟踪了示例`001.png`～`010.png`和一张3D深度图。自定义样本需另行复制：一个目录放同一板全部2D片图，另指定一张原始16位单通道深度PNG。支持`001.png`、`006.png`或`2_1.png`、`2_6.png`；同目录不要混板或混命名方式。`2_6.png`表示第2板第6片，数字从1开始。默认测试配置要求的全部片数，也支持`--pieces 1 6`。

文件名与物理片号必须人工核对，脚本不能从图像内容证明对应关系。固定分片框、Pitch参数必须适用于该整板；如果另一批数据需要不同标定，复制一份3D配置到实验目录，使用`--config 实验配置.json`，不要覆盖生产文件。直接传3D配置即可执行本工具；也支持`config/config.json`清单。

## 2. 创建独立CPU测试环境

从项目根目录执行，直接调用环境中的解释器，无需activate：

```powershell
py -3.10 -m venv .venv-affine
if ($LASTEXITCODE -ne 0) { throw '需要Python 3.10 x64，先检查 py -0p' }
& .\.venv-affine\Scripts\python.exe -m pip install -r .\visualization\requirements-affine-benchmark.txt
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
& .\.venv-affine\Scripts\python.exe -B -c "import sys,struct,numpy,scipy,cv2,PIL; print(sys.version); print('bits=',struct.calcsize('P')*8); print(numpy.__version__,scipy.__version__,cv2.__version__,PIL.__version__)"
```

预期Python3.10、64位；NumPy2.2.6、SciPy1.15.3、OpenCV5.0.0、Pillow12.3.0。OpenCV安装包完整版本是5.0.0.93。无需安装项目全部依赖，无需把原机器`.venv`复制过来，也无需为本实验安装Torch。

在同一台电脑、相同电源状态下依次运行各组；笔记本接电。测试期间停下自己正在执行的大型任务和重复启动的检测进程，记录是否还存在其它负载。无需关闭安全软件或更改系统全局设置。首次安装和初始化时间不计入性能结论。

## 3. 检查输入（不执行优化）

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\benchmark_affine.py --dry-run
if ($LASTEXITCODE -ne 0) { throw '输入或标定检查失败，先查看报错' }
```

默认读取项目样本和当前配置。这个命令仍会解码、Pitch补偿、分片与粗配准，以便发现尺寸/标定问题，但不会执行Powell。查看终端给出的输出目录及`results.json`：

- `inputs.pieces`应对应所有片号，检查图像尺寸、分片框、粗配准和context。
- `inputs.pitch`检查本次实际补偿状态、角度、步距；不能把配准台默认关闭Pitch的结果拿来直接比较。
- `config.dynamic_affine_3d.parameters`确认容差、评分区域、preview_scale等。特别保留实际`ftol`，不要默认套用旧例的0.0001。
- 保存输入SHA256、配置哈希；两机同样本实验必须一致。

自定义输入命令示例（后续每条实验命令都附加相同参数）：

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\benchmark_affine.py --dry-run --images-dir D:\samples\board2\2D --depth D:\samples\board2\2.png --config .\config\3D\config.json
```

## 4. 第一轮：只改变片并发数

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\benchmark_affine.py --variants current --workers 1 2 4 8 15 --opencv-threads 1 --warmup 1 --repeats 3
if ($LASTEXITCODE -ne 0) { throw '存在失败，请保留输出目录并检查逐片error' }
```

每个并发组先用同一套片图预热一整轮，再正式测试3轮。组的顺序按固定seed打乱；线程池在组内复用，组之间等所有任务结束。所有组使用相同图像、相同粗配准初值、相同参数。

`current`保留当前配置的全部仿射开关，要求`score_mode=full`。`workers=15`同时最多运行15片；若一板只有10片，实际最多10片，不会凭空构造15片。`workers=1`是相同函数在串行环境中的对照。任务入池排队`queue_ms`与真正执行`affine_ms`分别计时。两者不能混为一谈。

本进程将OpenMP/OpenBLAS/MKL线程设为1，OpenCV也显式设为1；这些设置不写回系统或生产配置。没有依据直接把正式流程认定为“OpenCV默认20线程”：加载Ultralytics也可能改变OpenCV线程数。本轮用于控制变量，不能代替正式服务内的实际线程状态。

看`analysis.md`和`summary.csv`，逐片细节在`pieces.csv`，完整停止原因/矩阵在`results.json`。主要比较：

|指标|含义|
|---|---|
|piece_ms.median / p95|单片进入函数后完成的墙钟耗时，含运行期间的等待|
|batch_ms.median|全部选中片都完成的仿射批次墙钟，含入池排队；不等于正式整板检测总耗时|
|throughput_pieces_per_s|本轮实际处理片数/仿射批次耗时|
|nfev / warps|Powell调用评分次数/真正执行评分变换次数，缓存命中不做warp|
|compute_mean_ms|一次真正评分计算的平均墙钟耗时，先按相同片号比较|
|queue_ms|提交到线程真正开始运行的等待；不能解释函数内部全部运行等待|
|reference差异|同方案、同OpenCV线程下，最小并发首轮为参照，矩阵投影5个图像点的最大偏移、损失和nfev差异|

判断逻辑：

1. **nfev、warp、损失和矩阵相同，单片随并发明显变慢**：优先判断CPU执行/调度、内存和Python运行竞争；不能仅凭这组数字把原因全部归结于GIL。
2. **单片变慢，整批仍更快**：存在吞吐和延迟取舍。按整板完成时间挑候选并发数，而不是只看最短单片。
3. **并发增加，整批也变慢**：继续加片线程没有收益，优先选择实测最短整批耗时的组，再在其附近细测（例如2、3、4、5）。
4. **nfev也改变**：先核对同片输入、参数、初值、损失和停止原因；不要把全部差异当成线程开销。
5. **纯仿射并发仍很快，正式流程很慢**：说明纯仿射实验未复现问题，需要正式流程并行负载下的结果，不能宣称问题已解决。

3轮P95只是小样本描述。结果波动明显时，把同一命令改为`--repeats 5`再执行一次；第二次也会保留独立报告。

## 5. 第二轮：独立比较优化开关

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\benchmark_affine.py --variants baseline cache multiscale budget --workers 1 4 --opencv-threads 1 --warmup 1 --repeats 3
if ($LASTEXITCODE -ne 0) { throw '存在失败，请检查结果与逐片error' }
```

上面的4只是例子：换成第一轮整批耗时较优的并发数；始终保留1作为对照。对另一板也重复执行，不能只测第一片或标定片。

|variant|相对原始单级基线的变化|
|---|---|
|baseline|缓存关闭、标定关闭、多尺度关闭，maxfev/total_maxfev均为null；其它评分、边界、容差保留|
|cache|仅开启精确参数缓存|
|multiscale|缓存+0.5倍粗搜15轮+完整评分分辨率精修10轮，局部窗口0.25；不限制评分预算，标定关闭|
|budget|缓存+单级maxfev=150、total_maxfev=150；多尺度与标定关闭|

“完整评分分辨率”指当前仿射评分画布：仍保留原始配置的`preview_scale`，不是强制2D原图1:1求解。各方案独立构造参数，互不继承上一组修改。

`summary`额外提供`comparison_variant`、`max_comparison_displacement_px`与`max_comparison_loss_increase`，本命令以baseline的最小并发首轮为参照。5个点的最大偏移用于提示矩阵变化，单位是目标3D片像素；不是实际定位误差或产品精度阈值。null表示没有可比较矩阵，不能读成0。完整逐片矩阵、分区损失和回退信息仍应人工检查。

- cache对baseline：最终矩阵、分区损失、nfev应一致；关注实际warp减少和墙钟耗时，不预设缓存命中率必须30%。
- multiscale对cache：比较最终完整损失、所有评分区、实际矩阵、回退比例、单片长尾、批次耗时；频繁回退可能更慢。
- budget对cache：看预算触顶、停止原因和几何变化。次数≤150与`affine_valid=true`均不等于对齐已经满足产品要求。
- 不要求所有方案nfev都下降；低分辨率多评分也可能更快。以实际总墙钟和可接受质量共同判断。

若已有配准台人工确认且带context的“保存矩阵”JSON，可另外试标定初值；它只复用线性四项，不复用旧片平移：

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\benchmark_affine.py --variants baseline cache calibration --profile D:\samples\verified-registration-matrix.json --workers 1 4 --warmup 1 --repeats 3
```

没有可靠标定文件就跳过，不要随手填一个矩阵。检查`affine_initialization.source`是否为calibration_linear以及拒绝原因；有文件不代表本片实际采用了它。

## 6. 叠加与正式整板验收

使用同一对输入，在配准台检查基线和候选参数，确保Pitch、片号、固定分片、评分区、preview_scale、ftol等全部对应。CPU测试环境可启动配准台：

```powershell
& .\.venv-affine\Scripts\python.exe -B .\visualization\registration_lab\server.py --open
```

在页面高级设置，将本轮`results.json`的`variants.方案名`**参数对象**粘贴到“仿射参数”文本框，替换该文本框内容；不要用“导入参数”按钮导入整个实验报告。选择与正式相同的整板分片/Pitch设置，运行相同片号，导出JSON和叠加图；检查三个评分区、孔边、芯片边和引线焊脚附近。完整画面、全部片都需检查，尤其关注损失恶化或矩阵变化最大的片。网页耗时还包含平台掩码和粗配准，单看“配准计算”总值不能直接等同本工具`affine_ms`。

最后在有原项目完整依赖、模型与GPU的环境启动正式验证台：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\validation_lab\server.py --open
```

1. 上传同板所有2D片和同一3D原始深度，核对板号片号。
2. 第一次作为完整检测预热；相同配置再测至少3次，保存每次result.json、timings.csv、python.log和report.html。配置变化会重载模型，因此每组都需独立预热。
3. 每次仅更改已通过独立实验的一项设置，保留原配置副本、明确记录差异，并在完成后恢复。生产文件修改由你决定；基准脚本不会代做。
4. 检查每片`affine_registration_ms`、nfev、真实warp、最终分区损失、有效性；检查整板总耗时而非简单相加所有并行片耗时。
5. 对比所有引线/焊脚数量、掩码映射、采样位置、高度差值、有效性以及阈值附近OK/NG变化。产品可接受的像素/高度差阈值按现有验收标准，不预设损失更低必然更准。
6. 如果独立测试表现好而正式并发退化，回传全部并行阶段计时，下一步再定位竞争的阶段。此时无需先调整YOLO。

验证台是常驻服务，测试完成后在另一终端执行以下命令关闭并释放模型：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\validation_lab\server.py --stop
```

配准台在原终端Ctrl+C结束；关闭浏览器本身不会结束服务。

## 7. 回传材料与分析结论

把本次`run-*`整个目录打包即可，含环境、CPU信息、代码/输入/配置哈希、参数快照、逐次结果、CSV和初步analysis.md。`complete=false`表示中断后只保存了已完成的组；不要当完整实验。终端出错时额外保留报错。报告包含本机路径和主机名，可按需脱敏后提供。

最终分析按以下表述组织：

1. 测试机CPU/内存/版本、commit、输入与标定是否和原机一致。
2. 第一轮并发1/2/4/8/15：单片中位/P95、整批中位、吞吐；最优组是哪一组，是否重复稳定。
3. 同片nfev/warp是否一致，单次评分成本是否放大，矩阵/损失是否一致；明确事实与原因推测。
4. 缓存、多尺度、预算各自的实际收益、退化片、停止原因、预算/回退比例；不承诺一定800ms或150次内合格。
5. 叠加和正式测高是否通过；若只完成CPU实验，明确写“尚未验证实际测高与正式整板负载”。
6. 推荐参数与并发范围、证据、仍需补测的项。不同CPU的最优并发不可直接搬回生产机，生产机仍需复验。

## 可直接交给另一台电脑上Codex的说明

> 请按visualization/AFFINE_CROSS_MACHINE_TESTING.md执行仿射配准实验。先检查工作区并记录当前commit，使用yinxkj最新代码和Python3.10独立CPU环境；不覆盖本地修改，不改生产标定，不启动YOLO或EfficientAD。先dry-run核对同板2D/3D片号、Pitch、分片和输入哈希，再运行current在workers=1/2/4/8/15的并发对照，OpenCV=1、每组预热1轮、正式3轮。随后用workers=1及实测较优并发，对baseline/cache/multiscale/budget独立对照；没有人工确认的矩阵文件时跳过calibration。遇到输入不匹配或错误先保留结果并说明，不擅自换数据或降低质量阈值。分析results.json、pieces.csv、summary.csv与analysis.md：分别报告单片延迟、整批耗时、nfev、真实warp、单次评分成本、损失/矩阵变化、停止原因、预算和回退。区分原因推测和证据；CPU结果不能宣称测高精度已通过。最后给出实测推荐并发和参数、所有输出目录、仍需正式GPU流程验收的内容。无需提交实测输出到Git。
