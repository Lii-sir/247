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

2D银浆检测和公开实例几何固定走独立CPU进程，配置位于`common_2d.parameters.threading.cpu_process_workers/cpu_process_affinity`，当前2进程、2物理核；配准仍默认4进程、4核。初始化的`cpu_2d_execution`及每片2D响应的`cpu_2d_worker`可核对实际PID/核范围。读取解码与GPU阶段保持原路径，早期3D不会等待2D重诊断。仅2D验证会在本次内存配置关闭空闲配准池的核预留，保留2D CPU进程的核策略，不写回磁盘。

正式流式联合链路已拆为独立配准池→映射测高池，回调交接、不占线程等待下一阶段。`config/3D/config.json`的`common_3d.parameters.threading.registration_workers`控制配准（当前4），`joint_workers`控制映射和测高（当前2），`early_joint_workers`控制2D诊断尚未完成时在途联合片数（当前2）。修改后重新验证会按配置重建；真实节拍需实测。每片新增`registration_*`、`joint_measurement_*`及`joint_pipeline_ms`，两个池的排队分别报告；原`joint_phase_ms`为两个执行阶段之和，不含中间排队。

配准现在固定使用常驻子进程，无thread模式。当前开启`registration_affinity`，给4个配准进程划分4个真实物理核，主进程使用剩余核；核不足或设置失败会报初始化错误。只想关闭绑核可设`registration_affinity.enabled=false`，仍是进程配准。共享区准备、传输计时与纯子进程计算分开；查看初始化响应`threading.registration_execution`与每片`registration.registration_worker`核对PID、核集合和Torch未加载。配准台的直接求解时间不能代表正式多进程整板节拍。

复用`detect_and_view.ProtocolClient`、命名共享内存与正式`bridge_server.py`，逐片发送实际`board_id/piece_id`，整板3D与2D并发提交；不替换算法或伪造检测结果。模型按当前配置初始化一次，相同配置和模型文件时间/尺寸不变时复用；配置或模型文件变化时关闭旧服务重新初始化。

参数来自`config/2D/config.json`与`config/3D/config.json`。页面显示当前片数和Pitch概况，不写回生产配置。当前正式配准为粗配准+仿射，Pitch只处理深度。模型初始化或检测失败会记录错误并尝试生成失败报告；产品NG与接口失败分别显示。

只在文件格式和配对检查通过后加载模型。任一2D请求失败会释放板缓存，使等待中的整板请求尽快结束；仅2D验证完成后也释放缓存，避免下次同板号混入旧结果。检测服务超时为900秒。运行中不接受第二次验证请求。

输出在`visualization/output/validation-lab/session-*/`：`uploads/`保留上传文件；各次验证UUID子目录含`result.json`、`report.html`、`timings.csv`、`python.log`。结果中的路径引用这些持久输入；关闭/重启后仍可从磁盘打开自包含报告。清空页面清单不删除磁盘文件。

前台使用Ctrl+C或上述`--stop`结束服务，模型随之释放。模型常驻时会占用显存。HTTP仅允许127.0.0.1的同源页面访问，生成文件下载限定到已知结果文件，不开放任意目录浏览。

## 验证

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s visualization/tests -p test_validation_lab.py -v
node --test visualization/tests/test_validation_pairing.cjs visualization/tests/test_validation_workflow.cjs
```

交互脚本测试使用模拟DOM，不代表真实浏览器视觉验收；真实正式服务的两批10片接口运行记录见`visualization/output/validation-lab/verification.json`。
