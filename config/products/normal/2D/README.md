# 2D 配置

项目启用与 3D 跳过规则参见[检测策略](../../../README.md#检测策略)。

| 模块 | `thresholds` | `parameters` |
| --- | --- | --- |
| `common_2d` | YOLO 置信度与 NMS IoU、最小连通域面积、分割灰度阈值 | 圆搜索、检测裁剪、固定 ROI、竖向分割位置、YOLO 尺寸与执行设置 |
| `wire_bending_2d` | 最小合格覆盖率 | 引线和细线 ROI |
| `silver_continuity_2d` | 扇区银浆像素数与覆盖率、扇区有效像素数、芯片面积、可见扇区比例下限 | 检测类别、外扩长度、扇区数和遮挡膨胀量 |
| `silver_overflow_2d` | 越界容差和最小越界面积 | 允许边界和银浆类别 |
| `chip_rotation_2d` | 芯片旋转角上限和最小面积 | `{}` |
| `chip_anomaly_2d` | 各 EfficientAD 模型的分数阈值 | 模型选择、裁剪留边、矫正角上限和评分方式 |

## YOLO 掩码后处理

板级和独立 2D 共用同一套掩码后处理。同类预测先合并，按 8 邻域拆成独立部件。不同类别的重叠像素保留。每个部件的置信度取实际覆盖该部件的预测中最高的一项。

原尺寸预测只批量传回各检测框内的有效像素。CPU 将可能相连的同类掩码分组，在局部范围内合并和拆分，并把部件边界复用于几何输出、银浆检查和引线清理。`common_2d.thresholds.mask.min_component_area` 按裁剪图中的部件面积过滤。

芯片掩码轮廓不足三个点，或最小外接矩形短边不超过一像素时，跳过该掩码的 EfficientAD 检测。结果在 `chip_anomaly_2d.skipped_chips` 中记录跳过原因，外观结论根据其余有效芯片判定。全部跳过时，外观项返回 `status=skipped` 和 `is_ok=null`。

引线弯曲使用清理后的单个 `wire/thin` 实例，在重叠像素最多的 ROI 中寻找两端与边界的接触区间，
各取两个端点连接成评估四边形。ROI 为 `[x1,x2) × [y1,y2)`，右/下边界像素为 `x2-1/y2-1`。
覆盖率为 `该实例在 ROI 和四边形交集内的像素数 / 该实例在 ROI 内的像素数`；
`rectangle_ratio_threshold` 仍为通过下限（含等号），保留当前 0.75 作为初值，需按样本重新校准。
`rectangle` 字段兼容保留，实际为四边形四个有序顶点。交点不足、多个端口或四点退化时直接判无效/NG，
不回退旋转矩形。每个 ROI 必须有引线且其全部实例合格，整体才合格。

芯片长宽比定义为 `long_side / short_side`。`aspect_ratio_models` 使用明确的
数组对象，例如 `{"range": [0.9, 1.1], "model": "chip_short"}`。区间不能
反向或重叠；没有命中区间会报配置错误，不会静默选择错误模型。当前示例中约
1.0 选择 `chip_short`，约 1.5 选择 `chip_long`。

`vertical_split.offset_radius` 控制实验脚本中的竖向分割线：
`split_x = round(circle_x + offset_radius * radius + offset_px)`。ROI 工具保存时
会更新 `common_2d.parameters` 和对应功能块的 `parameters`。
## 银浆允许边界：复用检测 ROI，四边按像素内缩

Windows x64 可通过 `silver_scan.dll` 加速 `roi_inset` 矩形的银浆像素扫描。
距离、容差和连通域处理仍使用现有算法。DLL 缺失时使用 NumPy 路径。
如需对照，设置 `YINXKJ_DISABLE_RUST_SILVER=1` 后重启检测进程。重建使用 `python/algorithm/vision2d/native/build.ps1`。

当前 `silver_overflow_2d` 的参数分组如下。

```json
{
  "thresholds": {
    "tolerance_px": 0.0,
    "min_area_px": 1
  },
  "parameters": {
    "boundary": {
      "mode": "roi_inset",
      "inset_unit": "px",
      "insets": {"left": 376, "top": 146, "right": 34, "bottom": 128}
    },
    "silver_class": "silver"
  }
}
```

四边参数是非负有限数，单位为原始 2D 图像像素，可独立调整，支持小数。
当前初值来自原标定模板：先找圆计算该模板的检测 ROI，再取原标定四边形各边中点，
换算为距对应 ROI 边的距离并四舍五入；这只是当前项目初值，需要按现场样本核对。
默认单位为 `px`；可选 `radius_ratio` 则将各内缩量乘以本片圆半径。
修改配置后需重新初始化算法。

已有检测框为 `[x1,y1,x2,y2]`，允许边界顶点依次为：
`[x1+left,y1+top]`、`[x2-right,y1+top]`、`[x2-right,y2-bottom]`、`[x1+left,y2-bottom]`。
使用 ROI 连续坐标边界 `x2/y2`，不减 1；边界线上的像素允许通过，0 内缩覆盖完整检测 ROI。
左右内缩量之和必须小于 ROI 宽度，上下之和必须小于高度；缺参数、负值、非有限值或
内缩过大时返回 `uncertain/is_ok=null`，不自动放宽、不回退模板坐标。
圆定位失败沿用 2D 预处理失败逻辑，不会用旧帧圆或边界继续检测。

此模式直接复用本片的检测 ROI，不重复找圆，不读取模板、不做特征匹配；轴对齐矩形用
向量化距离公式判断越界，角点使用欧氏距离以保留原 `tolerance_px` 语义，连通域过滤规则不变。
它随检测 ROI 平移；使用 px 时内缩距离不随半径缩放。矩形不会跟随工件旋转或透视形变。
只检查检测 ROI 内已有银浆掩码，不能保证 ROI 外没有缺陷。

C# 调用和字段不变，`Boundary.Mode` 和 `Registration.Method` 均为 `roi_inset`，
`Registration.Status=ok`、`IsValid=true` 表示边界构建成功，**不是模板配准成功**；
`FallbackUsed=false`，不产生匹配/内点数。`Boundary.Points` 正常公开输出为 2D 原图坐标。
可视化显示黄色矩形，并注明“ROI 内缩边界”。

旧模板模式仍保留。需要切换时，将 boundary 改为
`{"mode":"polygon","calibration_path":"calibration/silver_boundary.json"}`。
此时才读取原标定模板和四边形，逐张 SIFT/FLANN/RANSAC 配准；
`fallback_to_template_points` 仅对该模板模式生效，不影响 `roi_inset`。
模板标定工具仍为 `visualization/calibrate_silver_boundary.py`；执行其 `--apply` 会安装标定并切回模板模式。

运行 `scripts/roi_configurator.py` 可在同一张 2D 图片上标定 3D 引脚平台参考区：
按 `F` 添加矩形，拖动框内移动、拖动角点拉伸，`V` 清空平台框，`U` 撤销，
`S` 同时保存两份配置。平台框只写入同型号的 [3D 配置](../3D/config.json) 的
`height_measurement_3d.parameters.platform_roi_offsets`，不混入 2D 参数。
引线弧高仅使用 `wire_bending_2d.parameters.roi_offsets` 内的 wire/thin 掩码；
ROI 外部分不参与测高，thin 焊脚仍从完整掩码的两端划分。

例如：

```powershell
.venv\Scripts\python.exe scripts\roi_configurator.py --image test_data\2D\_2_1_20260929221235223_00331.png
```

在完整 2D 图像上按 `F` 框住干净的引脚平台表面（避开芯片、焊点和引线；可添加多个矩形），
按 `S` 保存。该矩形会按本片检测圆保存为相对坐标；板上其它片会各自用检测圆和透视矩阵映射，
不需要给每片单独维护参数。

## 线程配置

正式流式的银浆溢出/断连与公开2D实例几何已移入专用常驻CPU进程，不再投递到主进程的3D共享instance池。读图/圆定位/平台掩码、GPU推理、早期弯曲/缺线/旋转和灰度准备仍按原阶段执行；本次没有把整个2D流程迁出主进程，也没有改变检测算法或阈值。

`common_2d.parameters.threading`新增：

```json
"cpu_process_workers": 2,
"cpu_process_affinity": {
  "enabled": true,
  "physical_cores": 2
}
```

`cpu_process_workers`是2D CPU后处理进程数，环境变量`BRIDGE_2D_CPU_PROCESSES`可作为缺项默认值；JSON优先。`cpu_process_affinity`控制该进程组的Windows物理核分区，进程数不能超过保留核数。目标14900KF布局先保护主进程4P，再给2D后处理2P，3D使用剩余2P＋16E（配准4核、测高14核）。`main_reserved_physical_cores=4`位于3D线程配置，防止3D先占完P核。2D以请求RTT不超过1000ms为首要验收目标，实际达标仍需目标机验证。关闭亲和性仍使用进程计算，没有thread回退。

CPU组先从已有可用范围划分，后一个池从剩余范围选择；启用亲和性的池优先启动，关闭时逆序恢复。三组均开启时，配准、2D后处理、3D映射测高与主进程CPU集合互不重叠。检查初始化响应的`threading.registration_execution.workers[].cpu_ids`、`threading.cpu_2d_execution.workers[].cpu_ids`、`threading.joint_measurement_execution.workers[].cpu_ids`与`threading.main_cpu_ids`。只修改Bridge及其子进程，不改系统其它进程；缓存和内存带宽仍然共享。

每片只把实例mask复制到只读共享区，传递必要的标量/公开字段和已准备的银浆Boundary；不传GPU模型、Tensor、整张照片或检测器，不重复SIFT边界准备。子进程OpenCV/BLAS=1，无Torch。进程内按原顺序完成溢出、断连与各实例几何，不修改父进程实例；父进程轻量合并诊断与返回结果。mask实例顺序、编号、轮廓精度、crop偏移及OK/NG保持。

每片实例mask只发布一份共享内存，2D后处理与3D映射测高复用相同描述符，先启动的一方负责首次复制。缓存随板释放；真实在途的2D、3D任务各持租约，全部结束后才回收。允许重传的未认领片使用新一代mask缓存。早期joint_entry仍在GPU后的geometry阶段发布，不必等银浆和公开几何。最终整板继续等待全部2D诊断，不能因3D先完成而漏判。

`postprocess_workers`现在负责共享区准备和轻量finalize，不负责正式重后处理；它仍计入父线程预算。2D CPU进程不计入`worker_budget`，初始化单独记录。旧同步入口保留直接计算接口；正式流式固定进程处理。

每片新增`cpu_2d_worker`记录执行进程的PID/核范围/依赖版本/无Torch。计时：

|字段|口径|
|---|---|
|`cpu_2d_dispatch_queue_ms` / `cpu_2d_shared_memory_ms`|父准备线程排队 / 共享mask与元数据准备|
|`cpu_2d_queue_ms`|提交到子进程开始，包含上述准备和进程入队/IPC|
|`cpu_2d_phase_ms`|子进程挂接mask、银浆及公开几何计算|
|`cpu_2d_result_transfer_ms` / `cpu_2d_log_relay_ms`|结果返回与调度延迟 / 父进程转发日志|
|`cpu_postprocess_finalize_phase_ms`|父进程轻量合并结果|
|`cpu_postprocess_phase_ms`|子进程计算＋父finalize墙钟之和，排除中间排队|
|`cpu_postprocess_pipeline_ms`|从提交2D CPU任务到完整2D后处理结束，包含准备、队列、IPC和finalize|

`silver_*`与`public_2d_geometry_*`保留在原结果字段中，变为子进程计算口径；`public_2d_geometry_work_ms`是本片逐实例累计工作，不代表独立实例线程池。整个2D响应时间仍以`request_pipeline_ms`为准。修改后重新初始化，在另一台用同样输入预热后比较整板与2D响应中位/长尾，并复核银浆、轮廓、编号及OK/NG。尚未验证真实节拍，不承诺固定提升。

`postprocess_workers`、`silver_workers`、`gpu_workers`当前项目均配置15；前两者是父进程准备/汇总和边界准备线程，重后处理由上述CPU进程执行。统一板级检测为每个GPU工作线程加载独立YOLO与EfficientAD资源；增加父线程配置时仍需满足3D配置中的`worker_budget`。独立2D/3D旧接口目前只支持一个GPU工作线程。

统一板级流程的公开实例几何使用3D配置中的`common_3d.parameters.threading.instance_workers`（1为串行），通过独立字典快照提交任务，只读共享掩码，父线程按原顺序收集。`public_2d_geometry_ms`是阶段墙钟，`public_2d_geometry_work_ms`是各实例工作耗时之和；`total_detect_ms`现在包含公开几何生成，完整请求仍可参考`request_pipeline_ms`。

同一次银胶检查共用分类联合掩码及芯片连通域，跨图不复用。芯片连通域在保持全局2×2扫描对齐的有效包围框内计算，每个分量仅在自身包围框比较标签，汇总统计仅扫描所有外扩区域的包围框。保留原全尺寸输出掩码、芯片顺序、扇区统计及合格判定。
