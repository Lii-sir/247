# 3D 配置

3D 配置采用 `description + parameters` 功能块，不使用 `enabled` 开关。

- `common_3d`：板号关联所需片数。
- `segmentation_3d`：一张整板深度图分成十片的参数。
- `registration_3d`：初始尺度模板匹配参数。
- `dynamic_affine_3d`：每片运行时仿射优化参数。
- `smartray_pitch_3d`：分片前的SmartRay Pitch补偿，默认关闭；确认SDK状态和标定后，通过`enabled_3d`启用。
- `height_measurement_3d`：芯片/引脚平台平面、焊脚和引线高度阈值。引脚平台 ROI 在 2D 图片上由 ROI 工具画定，以该片圆心/半径归一化后保存在本文件；每片处理时使用其透视矩阵映射到 3D，再排除所有对齐后的 YOLO 掩码拟合平台平面。未画 ROI 时平台参考面无效，不会退回固定 3D AOI。
- `wire_height_3d`：独立 3D 接口的引线中心线 P95 参数。

板级接口会把 `config/2D/config.json` 的 2D 结果缓存与本文件的 3D 参数组合；
收到同一 `board_id` 的整板 3D 数据后完成配准、测高并释放十张 2D 缓存。

## 联合测高的引线边缘剔除

参数位于 `height_measurement_3d.parameters`，按类别独立设置：

| 参数 | 当前项目初值 | 旧配置缺项默认值 | 作用范围 |
| --- | --- | --- | --- |
| `wire_measurement_erode_px` | 2 | 0 | `wire` 的测高区域 |
| `thin_measurement_erode_px` | 1 | 0 | `thin` 划分两端焊脚后，中间引线的测高区域 |

两者均为非负整数，单位为 **3D 深度图像素半径**。`0` 关闭；半径 `r` 对应
`(2r+1) × (2r+1)` 的椭圆腐蚀核。当前值是可调初值，需按深度分辨率、实际线宽和反光范围校准。
修改后需重新初始化算法。

处理顺序：全部已检测实例的2D掩码清理（thin取中间段）→ 配准映射至深度图 → 按类别收缩测量掩码 →
剔除无效深度 → 原始深度 P95 代表点 → 相对平台参考面计算高度差。
收缩涵盖掩码全部边缘，包括thin中间段划分产生的边缘；测高不再按2D引线ROI筛选或裁剪。
若收缩后没有像素或内部均为无效深度，则返回 `Height3D.IsValid=false`、`IsOk=false`、
`Point=null`，不会回退到原始边缘。平台参考面不可用时仍按原逻辑返回测量无效。

只修改独立映射得到的引线测高掩码；平台/芯片平面采样使用的完整语义掩码不缩小，
该引线参数不控制 `thin` 两端焊脚及 `bond`，两者使用下节的独立参数；2D 弯曲检测和独立 3D 中心线接口不受影响。
C# 返回字段不变：`WireMeasurements[].Mask3DLocal` 是收缩后实际测量区域；
`Height3D.Point` 仍转换为整板深度图坐标；实例 `Masks[].Mask3DLocal` 仍表示完整对齐掩码。

效率处理：关闭时跳过腐蚀；开启时只在实例包围框内原地腐蚀、复用已生成的对齐缓冲区，
形态学核按半径缓存。uint8 测量掩码也只在包围框内枚举像素，避免为每根引线分配整幅深度图
大小的布尔数组；像素顺序不变，关闭时保留原来的 P95 点及并列值选择规则。

## 联合测高的焊点/焊脚边缘剔除

参数同样位于 `height_measurement_3d.parameters`：

| 参数 | 当前项目初值 | 旧配置缺项默认值 | 作用范围 |
| --- | --- | --- | --- |
| `bond_measurement_erode_px` | 1 | 0 | YOLO `bond` 焊点 |
| `thin_solder_measurement_erode_px` | 1 | 0 | 从原始 `thin` 拆出的 start/end 两端焊脚，两端共用此参数 |

两者均为非负整数，单位为 3D 深度图像素半径，0 关闭；椭圆核大小为 `(2r+1) × (2r+1)`。
当前 1 像素只是可调初值，需结合线宽、配准误差和反光范围校准，修改需重新初始化。

`thin_measurement_erode_px` 只控制中间引线，`thin_solder_measurement_erode_px` 只控制两端焊脚。
中间段、两端均从原始 thin 掩码独立拆分后处理，不会先内缩整个 thin 再拆端部，也不会重复内缩。

焊点/焊脚流程：配准映射 → 用原区域确定芯片归属、选择或拟合参考面 → 按类型内缩测量区 →
过滤无效深度 → 原始深度 P95 代表点 → 计算相对参考面高度差。芯片面、平台面、局部平台环带、
完整语义排除掩码均不因内缩参数变化而改变。

thin 两端在映射后仍先使用 `thin_solder_search_dilation_px`（当前 6）向外膨胀搜索，再按焊脚参数内缩。
因此剔除的是最终搜索掩码的外圈，不等同于剔除真实焊料轮廓边缘，膨胀与内缩参数需要配合调整。
若内缩后为空、内部没有有效深度或参考面不可用，则返回 `Height3D.IsValid=false/IsOk=false`，
测点和高度为空，不回退到外圈反光区域。

C# 方法和字段不增加。`SolderMeasurements[].Mask3DLocal` 是实际内缩后的测量区，仍为分片深度坐标；
`Height3D.Point` 仍为整板深度坐标。实例 `Masks[].Mask3DLocal` 和 `Mask2D` 保留原区域。

效率：关闭时直接复用原掩码；开启时仅为包围框内腐蚀生成独立小数组，不复制或回填整幅掩码，
不会原地改写共享 bond/实例掩码。局部像素坐标恢复至分片坐标后再读取深度、计算参考面及输出几何，
最后按原流程转换整板测点。形态学核与引线处理共用缓存。

## 线程配置

2D重后处理现另有独立常驻CPU进程与核组，默认2进程/2物理核，见config/2D/README.md。默认与配准4核合计预留6核，主进程使用剩余CPU；初始化/重载先关闭2D与配准进程并排空旧代工作，按分配的逆序恢复亲和性。两池共用同一进程生命周期和共享区租约实现，正式流程均无thread回退。

`common_3d.parameters.threading`控制正式流式调度。当前配置：

```json
{
  "worker_budget": 80,
  "cpu_workers": 15,
  "registration_workers": 4,
  "registration_affinity": {
    "enabled": true,
    "physical_cores": 4
  },
  "joint_workers": 2,
  "early_joint_workers": 2,
  "instance_workers": 20
}
```

- `registration_workers`：正式流式配准的常驻进程数，同时限制父进程共享区准备线程数；4路是待实测起点。配准子进程执行3D平台掩码、粗配准和仿射，未改变Powell、评分及停止条件。正式流程没有thread模式或静默线程回退。
- `registration_affinity.enabled`：Windows物理核分区，当前开启。读取真实物理核/SMT拓扑，优先选择EfficiencyClass较高的核；同核超线程整体分配。配准进程与主进程使用互不重叠的逻辑CPU集合。关闭该项仍使用进程配准，只取消核分区。
- `registration_affinity.physical_cores`：保留给配准的物理核数，当前4；至少等于进程数，并且必须留下至少一个物理核给主进程。当前每个配准进程对应一个完整物理核；保留更多核时按进程均分。核不足、亲和性设置失败或多处理器组不支持时明确初始化失败，不猜CPU编号或静默降级。
- `joint_workers`：独立后续池，只执行掩码映射、参考面、测高和结果整理；一片配准完成便可进入此池，不要求凑齐其它片。
- `early_joint_workers`：全部2D诊断完成前，允许进入联合链路但尚未完成的片数上限，计数覆盖配准、测高和中间排队；不是配准线程数。独立配准池模式下可以大于单个池的并发数；旧共享模式仍校验不超过joint_workers。全部2D完成后由两个池各自限制并发。
- `cpu_workers`：图像/深度输入准备和2D几何，不占用该池等待配准进程或测高完成。
- `instance_workers`：整个检测器共享的实例任务池，供实例映射、引线、焊脚和公开2D几何复用；不为每片另建一套。
- `worker_budget`：父进程桥接线程池预算，包含CPU、GPU、postprocess、silver、registration共享区准备、joint；别名池只算一次，不包括配准子进程、instance、Pitch和底层库内部线程。当前父进程桥接池总额66，另有4个常驻配准进程；初始化响应的`threading.effective_workers`记录预算内线程总额。

旧配置缺少`registration_workers`时，共享区准备池别名到joint池，不额外消耗父线程预算，进程数继承该数量；仍然执行进程配准。可用`BRIDGE_REGISTRATION_WORKERS`指定，JSON优先。旧配置缺少整个`registration_affinity`时不绑核，新配置已开启。准备线程仅复制/发布共享输入和异步投递，不等待子进程或测高Future；测高池不向自身提交片级父任务。

每板的补偿后深度只复制到共享内存一次，每片gray和平台mask各复制一次；子进程只接收共享区描述、当前片行范围和数值配置，不传模型、检测器、CUDA或整幅数组。一个全宽深度共享区供所有片只读，子进程计算输出独立持有。板失败/释放后不再开始后续测高；已经投递的进程工作可完成，最后一个真实进程Future结束前共享区不会被提前释放，取消公开Future也不能提前释放其输入。重载前排空旧代联合工作，关停/重载恢复原主进程CPU范围。

初始化预先启动全部常驻进程并加载依赖，Windows使用基础解释器的pythonw.exe及虚拟环境配置，避免重定向启动器句柄错误与额外CMD；需要正常的Windows/Python3.10环境。每个子进程OpenCV/OMP/MKL/OpenBLAS均设为1，不加载Torch、YOLO或EfficientAD。启动失败或工作进程异常会明确报错，不回退线程；崩溃后需重新初始化。初始化快照`threading.registration_execution`记录worker PID、CPU集合、版本、实际OpenCV线程、未加载Torch、核分区以及启动耗时；每片`registration.registration_worker`记录执行worker的启动快照。请检查main_cpus与registration_cpus无交集。

核分区仅约束本Bridge及其子进程，内存带宽、末级缓存和系统其它进程仍然共享，不承诺固定单片耗时。只作用于正式`BoardPipeline`流式入口；配准台、同步缓存整板和旧批量接口保持直接求解方式，复用同一函数，不是正式流程的备用thread模式。

修改参数后需要重新初始化；验证台配置变化会重建检测器。先按4进程/4物理核、2路测高、early在途2片测试；增加进程时同步调整physical_cores且给主进程留核，再比较测高2/4路。每组整板预热1次，至少稳定测试3次。比较整板完成时间和长尾，不仅比较单片仿射；检查矩阵、分区损失、掩码及测高结果一致。仅比较算法结果时忽略timings、registration_worker等执行诊断字段。

新增计时字段位于每片`timings`：

|字段|含义|
|---|---|
|`registration_queue_ms` / `registration_phase_ms`|提交到子进程开始的延迟（含准备、复制、IPC、入队）/ 子进程共享区挂接、平台掩码、粗配准及仿射墙钟|
|`registration_dispatch_queue_ms` / `registration_shared_memory_ms`|父准备线程排队 / 本次共享区准备复制；均是registration_queue_ms的子项|
|`registration_result_transfer_ms` / `registration_log_relay_ms`|子进程结束到父进程接收回调的延迟（含清理、传输和调度）/ 父进程转发诊断日志|
|`registration_process_roundtrip_ms`|提交到父进程收到子进程结果，不含后续测高；与上面计时嵌套|
|`joint_measurement_queue_ms` / `joint_measurement_phase_ms`|映射测高池排队 / 映射、测高和整理墙钟|
|`joint_pipeline_ms`|本片提交配准到测高完成，包含两个队列与交接开销|
|`joint_queue_ms`|兼容原字段，等于`registration_queue_ms`，不代表全部排队|
|`joint_phase_ms`|两个执行阶段墙钟之和，不含中间排队；不能当作新流水线端到端延迟|

`affine_registration_ms`、`mask_alignment_ms`和各测高子计时保持原函数边界。不要把这些嵌套字段重复相加。初始化快照新增`threading.opencv_threads`，仅记录已加载OpenCV的实际值，不在初始化快照阶段改变线程设置。

## 正式流程：Pitch → 分片 → 粗配准 → 仿射 → 测高

透视配准阶段已移除，旧`perspective_registration_3d`块不再读取，不再输出对应的求解耗时；`homography_2d_to_3d`仍以3×3齐次矩阵表示仿射映射，底行为[0,0,1]。这不影响芯片外观裁剪等其他功能使用OpenCV的几何映射。

新增可选功能块`smartray_pitch_3d`，只由`enabled_3d`控制是否执行。缺少整个块时关闭；启用后始终按未补偿的原始输入处理。启用前至少确认：

- 正式配置不再使用`input_state`。如果数据已由SDK补偿，请将`enabled_3d`设为`false`，否则会重复处理。旧`input_state`字段会被忽略，不再触发跳过。
- `angle_deg`为实际Pitch角，`scan_axis=rows/columns`对应深度图扫描方向。
- `transform_direction=recover_ideal`使用Z=Z′/cosβ、X=X′−Z′tanβ；`document_forward`使用文档正向公式。按设备导出定义确认方向。
- 外触发`scan_pitch_mm>0`；内触发设置`trigger_mode=internal`、`transport_rate_mm_s>0`、`scan_rate_hz>0`，以速度/频率计算基础步距。
- `gray_offset`、`z_step_mm`必须与测高配置的灰度零点、垂直分辨率一致；`z_origin_mm`是灰度零点对应的设备物理Z，并非旋转中心。
- 此功能块仅包含3D参数；算法名在内部固定，配置不需要`model`、`enabled_2d`、`image_role`或`input_state`。旧配置中的默认2D字段可兼容读取，但尝试启用2D同步映射会报错。正式补偿返回信息也不再带2D映射字段。

整板深度只补偿一次，随后分片、配准、参考面拟合和测高都使用这份数据。流式、同步缓存整板、批量板级和独立3D入口均已接入。无效/越界点与碰撞处理沿用测试算法，不写回共享内存或原始图像。

响应`results_board`/`results_3d`新增`pitch_correction`和`depth_coordinate_space`；联合单片也含对应元数据。关闭时为输入栅格`board_3d`；本工具实际补偿为`pitch_corrected_board_3d`，测高点标签同步，mask/平面仍为该栅格的分片局部坐标。无法把补偿坐标当原图坐标直接画在原始深度PNG上。离线报告会依据补偿元数据重建显示栅格；传入的应为原始采集深度文件。

C# DTO新增元数据属性，现有测高点/掩码字段保持；源码已构建验证。上位机若继续引用旧DLL，新增属性不会出现在旧类型中，需要更新引用的构建产物。不要重复对SDK已矫正数据启用原始数据补偿。

## 固定位置分片（2026-11-06）

`segmentation_3d.parameters`新增：

- `mode`：`auto`（默认，原有效值定位流程）或`fixed`（直接使用预设矩形）。
- `fixed_rois`：从上到下排列的区域列表，每项为整数`[x1,y1,x2,y2]`。实际项目为10片就必须配置10项，与`expected_piece_count`、`common_3d.input.piece_count`一致。

固定模式只验证配置和图像尺寸，不生成整图有效掩码、不统计行列、不扫描阈值，也不复制分片图像；返回与自动模式相同的piece_id/x1/y1/x2/y2/defects结构，调用方继续读取原数组视图。

坐标为 **Pitch处理后的整板深度像素**，左上角为(0,0)，右/下边界不包含。要求非负、正面积、在图内、从上往下且行范围不重叠；越界或数量错误直接报错，不静默裁剪、不自动回退。`padding_x/y`和自动阈值参数在fixed模式不再作用于最终坐标。

联合路径仍保留全宽X，只按y1:y2读取；独立3D测高使用x1:x2，因此应填实际标定X范围，不能随意改成全宽。固定窗口需要覆盖产品且避开相邻片。固定模式不通过像素判断空窗或实体缺片，后续检测仍需判定有效性。

### 获取坐标

可直接使用配准台高级设置中的“自动扫描并生成固定分片”：只选择3D整板，设置与正式流程相同的Pitch，然后扫描一次，即自动生成全部`fixed_rois`并切换本页为fixed模式。无需2D配准，仍按下面第3、4步导出并加载正式配置。按钮会强制扫描，不沿用已有固定坐标；扫描失败不改原配置。

1. 在配准台使用与正式流程相同的Pitch参数，以auto模式运行一张代表性整板。
2. 高级设置中点击“将本板分片设为固定”，本页的分片JSON会填入mode=fixed及全部实际区域；默认正式配置仍为auto，未代填实际坐标。
3. 检查范围后导出参数，把导出文件的`parameters.segmentation_3d`复制到正式文件的`segmentation_3d.parameters`。
4. 重启Python服务或重新初始化算法加载新配置。图尺寸、触发起点、Pitch或位置标定改变后重新标定；切换分片原点后重新计算配准矩阵。

配准台保存的新矩阵会记录分片模式和固定ROI，分片设置不匹配时拒绝复用；老矩阵文件没有该字段时保留原兼容行为并提示核对局部原点。仅配置切换不会验证不同实物板的定位稳定性。

## 实例并行处理

`common_3d.parameters.threading.instance_workers`默认16，必须为正整数；设为1恢复串行。该线程池在检测器初始化时配置、跨片/跨板复用，释放或重新初始化时等待任务结束并关闭，与联合阶段`joint_workers`线程池独立。它是额外的实例任务线程池，不计入桥接层原有worker_budget。

仿射矩阵确定后，各实例独立映射并提取几何，输入掩码和矩阵只读；结果按原实例顺序汇总，编号保持。流式联合路径和旧整板批量路径共用实现。任务失败先取消未运行任务、等待已运行任务结束，再向上报告错误。

`mask_alignment_ms`仍为该片映射阶段的墙钟耗时；新增`instance_alignment_work_ms`为实例任务耗时之和。并行累计工作时间可能超过墙钟耗时，不能将二者相加。子线程使用独立计时字典，最终由父线程汇总，防止多片并发混计。配置修改需重新初始化算法。

引线测高也使用`instance_workers`线程池，逐根独立处理，全部wire/thin按输入顺序连续编号，不再过滤ROI外实例或按ROI裁剪。形态学清理保留完整依赖边界，把局部偏移合入2D至3D矩阵，保持原坐标。清理后为空或缺少有效深度也保留无效记录。高度百分位、无效深度和内缩规则不变。

3D测高已移除目标选择ROI参数及裁剪函数。`wire_rois/roi_offsets`仅供2D弯曲、缺线诊断使用，不决定引线或焊点是否测高；3D内部高度结论仅依据高度结果，整体联合结论仍合并2D检查。高度记录可携带已有`bending_2d`，3D中不再重新计算弯曲。引脚平台的`platform_roi_offsets`用于参考平面采样，配准ROI用于求矩阵，均保留。bond和thin端点继续全量测高。

每个thin在引线阶段仅拆分一次，父线程保存中间区域与两端焊脚区域，后续焊脚复用；每次测量重新准备，避免复用旧比例。`height_wire_regions_ms`仍为整个引线阶段墙钟，包含`height_prepare_thin_ms`；`height_wire_work_ms`为各根引线的累计工作耗时。

芯片外部填充区域及面积在父线程统一准备（`height_prepare_chip_regions_ms`），焊脚归属查询只读，不再在子任务中写缓存。平台/芯片参考面和引线处理完成后，各bond及thin端点使用同一实例线程池并行计算；子任务只返回结果，父线程按原实例及start/end顺序写回字段。即使某任务失败，也先结束所有运行中的兄弟任务，再返回错误。

焊脚计时：`height_solder_wall_ms`为整段焊脚任务墙钟；`height_solder_work_ms`累计完整子任务（含thin端点映射、膨胀）；原`height_solder_ms`保留为其中测高/归属/几何处理的累计耗时。三者相互包含或重叠，不应相加。上述改动不调整仿射参数、参考面选择、测高百分位或合格阈值。

## 仿射评分范围与诊断

`dynamic_affine_3d.parameters.score_mode`为`full`（默认，原生整图评分）或`roi`（仅手动对照的试验路径）。ROI版本虽通过像素一致性检查，但每次评分构造坐标数组、浮点转换和remap可能比原生warpAffine更慢，不能仅凭处理像素减少认定提速；用户日志出现仿射阶段严重回退后，代码缺省值和正式配置均恢复full。ROI模式区域裁剪、分区平均和剪切惩罚不变；保留全局目标坐标和原画幅尾部取整边界。当前逐像素验证的后端为OpenCV5 AVX2；其他后端、关闭优化或超出remap尺寸限制时自动使用full，并在返回值`affine_score_mode`显示实际方式。

配准结果新增`affine_function_evaluations`（优化器nfev）、`affine_optimizer_success/status`、`affine_stop_reason`、`affine_optimizer_ms`，以及`affine_score_calls/total_ms/mean_ms`。score_calls包含初始和最终检查，通常比nfev多2次；平均评分耗时可区分评分次数与单次评分成本。停止原因与最终损失合格判断独立，未改变Powell参数、迭代上限或合格阈值。

## 仿射优化开关与测试

`dynamic_affine_3d.parameters.cache_enabled`当前配置为true，省略时false。缓存仅属于本片/本次优化的固定评分分辨率，以完全相同的6参数作键；不量化参数、不保存变换图、不跨片复用。开关不会改变目标函数或Powell评估序列。

新增`affine_cache_hits/misses/hit_rate`、`affine_warp_evaluations`和`affine_score_compute_ms/compute_mean_ms`。`affine_function_evaluations`仍是SciPy调用目标函数的次数，缓存命中也计入；`affine_score_calls`还包含优化外检查；`affine_warp_evaluations`才是真正执行评分变换的次数。`affine_score_total_ms/mean_ms`包含缓存查询，compute两项只统计实际计算。均在每片`registration`中。

实图测试由使用者执行：同一套10片输入、同配置和并发，分别cache_enabled=false/true，比较每片最终矩阵、最终损失、各区损失、真实变换次数及`affine_registration_ms`。命中率不设30%硬门槛，以墙钟和输出一致性判断。重新初始化后参数生效。

`calibration.enabled`默认false；启用时把配准台“保存矩阵”的完整JSON放进`calibration.profile`。校验原始2D尺寸、深度宽度、坐标来源及Pitch几何设置后，仅复用线性4项，当前片粗配准参考点位置不变，不复用历史平移和分片行偏移。候选必须位于原粗配准搜索边界内，完整分辨率损失严格更低才作为起点；不匹配、非法或不改善均保留粗配准，并在`affine_initialization.reason`说明原因。片高和分片模式可不同，因为未复用历史平移。仅支持独立2D相机坐标。

新增`affine_initial_parameters`、`affine_coarse_parameters`及`affine_initial_score`；`affine_score_before`仍代表原粗配准初值的完整评分。候选检查计入score_calls但不计入SciPy nfev。此功能可能改变收敛结果，需用同一板/不同板对照最终各区损失、映射位置与测高结果后再正式启用。

`multiscale.enabled`默认false。启用时先在`search_scale`（默认0.5）分辨率搜索，再做完整分辨率局部精修；两阶段迭代上限分别由`search_maxiter/refine_maxiter`控制。完整规范掩码先闭运算再降采样，矩阵按源/目标实际尺寸比例换算；6参数及边界始终保持完整分辨率单位，评分框同步floor/ceil缩放且不丢弃某个分区。各分辨率使用独立缓存，低分辨率损失不参与最终合格判断。

精修窗口半径为原搜索半径乘`refine_window_ratio`（默认0.25），且截在原边界内。完整评分退化、超过阈值、碰到新窗口边界或阶段异常时，回退原粗配准起点和原边界的完整搜索。保留实际完整评分候选中较好的解，不能据此保证等同原优化器的最优解。`affine_stages`逐级记录nfev、尺寸、边界、耗时、停止状态，顶层nfev/optimizer_ms为各级之和；`affine_multiscale`说明回退与最终解来源。完整阶段墙钟仍看timings中的affine_registration_ms；新增affine_optimization_pipeline_ms还覆盖级间准备和验证。实际速度和质量须由实图验证，默认不启用。

`maxfev`为每次Powell调用的评估上限；多尺度`search_maxfev/refine_maxfev`为null时继承它。`total_maxfev`限制本次配准所有优化阶段的调用总数（含回退）；均默认null，非null须为正整数。缓存命中也消耗优化器评估预算，初始/候选/最终验收等额外检查不消耗该预算但计入实际耗时。每阶段保存已评估最佳有限参数的副本，仅在预算截断时优先取它，避免未完成线搜索丢掉更好试探点。`affine_budget`与`affine_stages`标记预算消耗、skipped及used_best_observed；总预算耗尽导致回退未运行时有fallback_not_run标记。停止状态与最终质量独立。实图分组步骤见[仿射验收说明](../../visualization/AFFINE_OPTIMIZATION_TESTING.md)。

## Pitch条带并行

`smartray_pitch_3d.parameters.projection_workers`默认配置为8，必须为正整数；设为1串行。省略该项的旧配置按1处理。统一板级和独立3D检测器各自在初始化时配置独立Pitch线程池，跨板复用，释放/重新初始化时关闭；它与joint及instance线程池分开，也不计入桥接层原有worker_budget。

按非扫描轴分条带：rows扫描分列，columns扫描分行。每个条带覆盖整个扫描方向，投影和碰撞聚合仅写自己的缓冲，父线程按条带顺序合并，因此跨行的大位移不会造成并发写冲突。投影公式、np.rint取整、0/65535无效值、越界过滤、碰撞取最大矫正深度保持不变。同栅格强度图复用本次投影点，平局仍取最大的原始源索引，不再二次投影。

`smartray_pitch_ms`仍是整体墙钟；新增`pitch_projection_work_ms`为各条带工作耗时之和（有计时上下文时记录）。补偿元数据中的`projection_workers`是本次实际条带数，受非扫描方向尺寸和可用线程数限制。独立函数/配准台未传入持久池时使用一次性私有池并在结束时关闭；不开补偿或SDK已补偿时仍原样跳过。
