# 整板检测验证台

在项目根目录双击 `启动检测验证台.cmd`，或运行：

```powershell
& .\.venv\Scripts\python.exe -B .\visualization\validation_lab\server.py --open
```

默认本机地址 `http://127.0.0.1:8770/`，`--port`可修改。只监听本机。需要现有环境中的检测依赖、模型和CUDA GPU；启动网页本身不加载模型，首次点击验证才初始化。

上述命令是前台常驻HTTP服务，打开网页后终端会继续运行，关闭网页不会结束服务。同端口再次启动时会复用已有验证台，不再启动第二个服务；`--open`只负责打开浏览器。

停止当前端口的验证台（可在另一终端执行）：

```powershell
& .\.venv\Scripts\python.exe .\visualization\validation_lab\server.py --stop
```

使用其他端口时加`--port 端口号`。停止会中断正在运行的验证、关闭检测进程并释放模型资源，不等待900秒请求超时；已保存结果保留。本机停止接口只接受同源访问，命令行会先核对验证台身份，不关闭占用同端口的其他应用。

Windows若不希望保留命令窗口，可直接运行：

```powershell
& .\.venv\Scripts\pythonw.exe .\visualization\validation_lab\server.py --open
```

后台运行后用上面的`python.exe ... --stop`停止。检测子进程始终使用隐藏控制台的`python.exe`和私有协议管道，不另外显示CMD。

## 操作

1. 批量添加2D片图、3D整板深度PNG。2D命名`板号_片号.png`，例如`2_6.png`；3D命名`板号.png`，例如`2.png`。其他常见2D图像扩展名也可用。
2. 核对表格里的板号、片号，可直接编辑。数字板号前导零统一，例如02与2视为同一板；2D的06视为第6片。不能识别的文件需要手动编号，不靠选择顺序暗自分配。
3. 选择验证板号及方式，再点“开始验证”。一次只验证一个板号，其他板的文件不提交。
4. 在右侧交互报告切换片号/图层、缩放、点击测量编号定位；可独立打开报告，下载JSON、CSV及查看日志。本次服务会话的历史记录最多保留32条，磁盘结果不会自动删除。

“整板联合”须具有配置要求的全部片号（当前10片，每号恰好一张2D）及恰好一张3D整板图；缺片、重复、越界片号、重复深度均会在提交前提示，后端也会校验。3D须为原始16位单通道整数PNG，不能用伪彩图或截图。匹配错误会导致检测结果错误，编号不能只看文件数量。

3D文件只用于离线样本选择。验证台在提交请求前读取原始16位像素、按小端uint16紧密排列写入共享内存，并发送宽、高及`data_size=宽×高×2`。算法入口不再接收PNG或进行深度PNG解码，因此这部分样本读文件耗时属于输入准备，不计入算法处理时间。实时SDK可直接提交完整`ushort[]`，无需文件保存。

“仅2D”可选少量片图，保留指定片号；不提交深度请求，本次服务副本关闭Pitch，磁盘配置不变。若只想用一张2D检查对应深度片的配准效果，仍使用`visualization/registration_lab`配准台；正式联合协议需要整板片图齐全。

“载入项目样本”明确将旧命名001～010归为demo板；这是显式样本预置，不是自定义文件的自动命名规则。可手动修改预置编号。上传后的原文件名完整显示，不使用临时UUID当展示名。

## 请求提交节奏

页面“03 请求提交”新增两种方式：

- **并发提交**：默认保留现有方式，整板3D与各片2D请求并发发送。
- **按间隔提交：2D依次 → 3D最后**：2D按片号升序发送，最后一张2D发出后，再经过同样间隔发送3D。仅2D验证时不发送3D。

“相邻请求间隔”支持0～3600000ms的有限数值。它表示相邻请求**开始发送时刻**之间的最小间隔，不等待上一张检测返回。例如100ms对应：第1片2D→至少100ms→第2片2D→…→最后2D→至少100ms→3D。第一张立即发送；0ms取消人为等待，仍保证写入协议管道的顺序。Windows/Python调度与写入开销可能使实际间隔更长，不是硬实时相机触发。

本次模式和设定值保存在结果JSON的`submission`，`submission.requests`逐项记录实际发送顺序、cmd、piece_id、request_id、sent_ms及gap_ms。sent_ms以本轮发送阶段起点为零，首项gap_ms为null；原`timeline`以协议客户端启动为零，通过request_id关联，勿直接混减两者时间。界面历史摘要和导出报告标题保留提交时的模式与间隔，修改当前控件不会改写旧报告。

长间隔会按计划跨度加30秒余量，必要时只通过检测子进程环境覆盖板缓存超时，避免先到2D在3D提交前过期；不写回生产配置或全局环境。常见小间隔与并发模式可复用已加载模型，必要缓存超时改变时会重启服务并重新初始化。现有每个已发请求的900秒响应超时保持。

收到请求失败会停止继续提交并释放板缓存；产品NG（接口ok=true）继续正常提交。停止验证台会打断间隔等待与响应等待，映射输入在已发请求结束后释放。

**验证总时长包含人为间隔、输入准备和可能的初始化；算法timings按原处理阶段统计。** 3D最后提交时，`total_detect_ms`起点也相应后移，不能直接把它与整轮总墙钟或原并发模式混为同一口径。

`POST /api/run`在既有files/board_id/mode之外支持`submission_mode`（concurrent/paced）及`request_interval_ms`（数值）；旧调用缺项仍为并发。前后端均校验，非法值在加载模型前拒绝。

## 正式流程与模型生命周期

正式流式固定使用三组常驻CPU进程：配准、2D重后处理、3D映射测高；没有thread回退。针对32物理核，当前三组各8进程/8物理核，给主进程留8核。3D的joint/instance计算不再留在主进程：每个测高进程拥有本地实例执行器，instance_workers默认1，不是全板共享20/64线程池。实际节拍需在目标机复验。

配置：2D组由common_2d.parameters.threading.cpu_process_workers/cpu_process_affinity控制；3D文件的registration_workers/registration_affinity控制配准，joint_workers/joint_affinity控制映射测高，early_joint_workers控制2D诊断未齐时在途联合片数。增加进程时要同时分配对应物理核并给主进程留核。三组核分区关闭仍使用进程，关闭按分配逆序恢复原CPU范围。

查看初始化threading.registration_execution、cpu_2d_execution、joint_measurement_execution各自workers[].cpu_ids及threading.main_cpu_ids，核对集合不重叠。2D响应cpu_2d_worker和联合片joint_measurement_worker记录实际PID/CPU/无Torch/局部实例数。joint_measurement_queue_ms含共享区准备与IPC，phase_ms为测高进程计算；joint_pipeline_ms覆盖配准提交到高度结果完成，不能只看某个阶段变小。

整板补偿深度共享一次，配准与测高复用；测高共享原2D实例mask，保持实例ID、顺序、轮廓、P95和坐标转换。最终在主进程等待所有2D诊断后合并，不因3D先完成而漏判。仅2D验证的内存配置取消空闲配准和测高组的核预留，保留2D CPU策略，不写回磁盘。

复用`detect_and_view.ProtocolClient`、命名共享内存与正式`bridge_server.py`，逐片发送实际`board_id/piece_id`，根据提交模式并发发送，或按间隔依次2D、最后3D；不替换算法或伪造检测结果。模型按当前配置初始化一次，相同配置和模型文件时间/尺寸不变时复用；配置或模型文件变化时关闭旧服务重新初始化。

参数来自`config/2D/config.json`与`config/3D/config.json`。页面显示当前片数和Pitch概况，不写回生产配置。当前正式配准为粗配准+仿射，Pitch只处理深度。模型初始化或检测失败会记录错误并尝试生成失败报告；产品NG与接口失败分别显示。

Pitch已移到提交前预处理：验证台先按配置准备深度，再开始发送检测请求与计时。页面及结果的`total_wall_ms`现在从输入准备完成后计算，不含模型初始化、深度读文件或Pitch矫正；报告另存`excludes_depth_preprocessing=true`。算法响应不再含`smartray_pitch_ms`或`pitch_projection_*`。独立配准测试台仍可单独测量矫正本身。

连续验证复用调用侧Pitch线程池和输出缓冲；改变标定参数时重建预处理器，关闭/停止验证台时释放。每份请求持有独立字节快照，后续帧不会覆盖已提交数据。查表及直接写输出不改变取整、边界和碰撞规则。

2D性能目标按每张请求“发出至收到响应”的端到端时间≤1000ms验收，与时间线条形使用同一口径。结果摘要显示最大值和超标片号，`result.json`的`benchmark.two_d_latency`保存最大值/P95/是否完成/是否达标；检测失败或少了请求不能被标成达标。性能目标与产品OK/NG分开，不会删掉检测项或提前返回。目标14900KF按4P主进程＋2P后处理优先保护2D，3D再分配剩余核；静态配置仍须用实际发送间隔和目标机RTT验证。

`benchmark.threading`保留初始化时的实际核分配，重点核对`main_reserved_cores`、`main_reserved_cpu_ids`、`main_cpu_ids`及各进程组的`workers[].cpu_ids`。请在整板联合模式下用相同输入及发送间隔验收；仅2D模式达标不能替代联合负载验收。更新代码和配置后完整重启验证台/算法服务，才能应用新的快核预留策略。

只在文件格式和配对检查通过后加载模型。任一2D请求失败会释放板缓存，使等待中的整板请求尽快结束；仅2D验证完成后也释放缓存，避免下次同板号混入旧结果。检测服务超时为900秒。运行中不接受第二次验证请求。

输出在`visualization/output/validation-lab/session-*/`：`uploads/`保留上传文件；各次验证UUID子目录含`result.json`、`report.html`、`timings.csv`、`python.log`。结果中的路径引用这些持久输入；关闭/重启后仍可从磁盘打开自包含报告。清空页面清单不删除磁盘文件。

前台使用Ctrl+C或上述`--stop`结束服务，模型随之释放。模型常驻时会占用显存。HTTP仅允许127.0.0.1的同源页面访问，生成文件下载限定到已知结果文件，不开放任意目录浏览。

## 验证

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s visualization/tests -p test_validation_lab.py -v
node --test visualization/tests/test_validation_pairing.cjs visualization/tests/test_validation_workflow.cjs
```

交互脚本测试使用模拟DOM，不代表真实浏览器视觉验收；真实正式服务的两批10片接口运行记录见`visualization/output/validation-lab/verification.json`。
