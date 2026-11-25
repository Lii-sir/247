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
3. 添加「检测」组件，在节点里设置检测类型、板号和片号。片号可写一个数字、逗号列表或范围，
   例如 `1,3-5`。同一个节点可以改参数后重新执行。拖动节点标题可调整顺序。
4. 点击「一键启动流程」后，按画布顺序执行全部节点。一个检测节点有多个片号时，按输入顺序逐片发送并等待结果。
   在检测前放置「延时触发」即可等待指定毫秒数。检测节点也可以单独执行。
   单独启动延时节点时，到时只触发相邻检测节点。加载样本、添加节点和拖拽排序都只更新界面。
5. 一键流程结束后，在结果区上方选择本次流程。报告顶部可切换板号和片号。
   同一片的 2D 和 3D 并排复核，只有一类结果时占用整行，跳过或失败会显示原因。
   同片同阶段执行多次时，界面显示最后一次，各次原始结果都在流程 JSON 内。
   单独执行节点时，其结果可从历史记录中打开。

在画布上方填写名称，点击「保存流程」，即可保存节点顺序、检测参数和延时。
下次在同一个浏览器、同一个验证台地址打开时，从「已保存的流程」选择并载入。
图片清单和检测结果分别管理；载入模板后，添加对应板号的图片即可重新启动。

要指定文件位置，点击「另存为流程文件」，选择目录和文件名，把模板保存成 JSON。
以后用「从文件载入」打开该 JSON。浏览器未提供位置选择时，流程文件按浏览器的下载设置保存。

检测节点可填一片或片号范围。验证台在调用侧从原始 16 位整板样本取片，只提交 `detect_3d_piece`。同片 2D 缺失或过期时，结果会注明跳过原因。旧整板 3D 节点载入时自动转换为当前配置的单片范围。

产品 NG 可以继续执行后面的节点。请求执行失败时流程停在该节点。点击「停止流程」可结束当前倒计时，或等当前检测响应后停止后续请求。

## 批量 2D 验证

「批量 2D 验证」支持并发提交和按片号升序、按间隔提交。间隔为相邻请求开始发送的最小时间，支持 0～3600000 毫秒，请求间隔和实际时间线会保存到报告。批量 2D 完成后释放该板缓存。需要后续 3D 使用同片缓存时，在画布上安排单片 2D 和单片 3D 节点。

## 正式流程与模型生命周期

正式流式固定使用三组常驻CPU进程：配准、2D重后处理、3D映射测高；没有thread回退。针对32物理核，当前三组各8进程/8物理核，给主进程留8核。3D的joint/instance计算不再留在主进程：每个测高进程拥有本地实例执行器，instance_workers默认1，不是全板共享20/64线程池。实际节拍需在目标机复验。

配置：2D组由common_2d.parameters.threading.cpu_process_workers/cpu_process_affinity控制；3D文件的registration_workers/registration_affinity控制配准，joint_workers/joint_affinity控制映射测高。增加进程时要同时分配对应物理核并给主进程留核。三组核分区关闭仍使用进程，关闭按分配逆序恢复原CPU范围。

查看初始化threading.registration_execution、cpu_2d_execution、joint_measurement_execution各自workers[].cpu_ids及threading.main_cpu_ids，核对集合不重叠。2D响应cpu_2d_worker和联合片joint_measurement_worker记录实际PID/CPU/无Torch/局部实例数。joint_measurement_queue_ms含共享区准备与IPC，phase_ms为测高进程计算；joint_pipeline_ms覆盖配准提交到高度结果完成，不能只看某个阶段变小。

同片配准和测高复用深度共享内存，测高使用同片完整 2D 诊断及只读实例掩码。仅2D验证的内存配置取消空闲配准和测高组的核预留，保留2D CPU策略，不写回磁盘。

复用`detect_and_view.ProtocolClient`、命名共享内存与正式`bridge_server.py`，逐片发送实际`board_id/piece_id`，根据提交模式并发发送，或按间隔依次2D、最后3D；不替换算法或伪造检测结果。模型按当前配置初始化一次，相同配置和模型文件时间/尺寸不变时复用；配置或模型文件变化时关闭旧服务重新初始化。

参数来自`config/2D/config.json`与`config/3D/config.json`。页面显示当前片数和Pitch概况，不写回生产配置。当前正式配准为粗配准+仿射，Pitch只处理深度。模型初始化或检测失败会记录错误并尝试生成失败报告；产品NG与接口失败分别显示。

Pitch已移到提交前预处理：验证台先按配置准备深度，再开始发送检测请求与计时。页面及结果的`total_wall_ms`现在从输入准备完成后计算，不含模型初始化、深度读文件或Pitch矫正；报告另存`excludes_depth_preprocessing=true`。算法响应不再含`smartray_pitch_ms`或`pitch_projection_*`。独立配准测试台仍可单独测量矫正本身。

连续验证复用调用侧Pitch线程池和输出缓冲；改变标定参数时重建预处理器，关闭/停止验证台时释放。每份请求持有独立字节快照，后续帧不会覆盖已提交数据。查表及直接写输出不改变取整、边界和碰撞规则。

2D性能目标按每张请求“发出至收到响应”的端到端时间≤1000ms验收，与时间线条形使用同一口径。结果摘要显示最大值和超标片号，`result.json`的`benchmark.two_d_latency`保存最大值/P95/是否完成/是否达标；检测失败或少了请求不能被标成达标。性能目标与产品OK/NG分开，不会删掉检测项或提前返回。目标14900KF按4P主进程＋2P后处理优先保护2D，3D再分配剩余核；静态配置仍须用实际发送间隔和目标机RTT验证。

`benchmark.threading`保留初始化时的实际核分配，重点核对`main_reserved_cores`、`main_reserved_cpu_ids`、`main_cpu_ids`及各进程组的`workers[].cpu_ids`。联合检测节拍使用同样的单片 2D 和 3D 请求序列验证。更新代码和配置后完整重启验证台/算法服务，才能应用新的快核预留策略。

只在文件格式和配对检查通过后加载模型。任一2D请求失败会释放板缓存，使等待中的整板请求尽快结束；仅2D验证完成后也释放缓存，避免下次同板号混入旧结果。检测服务超时为900秒。运行中不接受第二次验证请求。

输出在`visualization/output/validation-lab/session-*/`：`uploads/`保留上传文件；各次验证UUID子目录含`result.json`、`report.html`、`timings.csv`、`python.log`。结果中的路径引用这些持久输入；关闭/重启后仍可从磁盘打开自包含报告。清空页面清单不删除磁盘文件。

前台使用Ctrl+C或上述`--stop`结束服务，模型随之释放。模型常驻时会占用显存。HTTP仅允许127.0.0.1的同源页面访问，生成文件下载限定到已知结果文件，不开放任意目录浏览。

## 验证

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s visualization/tests -p test_validation_lab.py -v
node --test visualization/tests/test_validation_pairing.cjs visualization/tests/test_validation_workflow.cjs
```

交互脚本测试使用模拟DOM，不代表真实浏览器视觉验收；真实正式服务的两批10片接口运行记录见`visualization/output/validation-lab/verification.json`。
