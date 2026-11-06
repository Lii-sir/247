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

## 正式流程与模型生命周期

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
