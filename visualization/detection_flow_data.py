"""Reviewed process descriptions for the offline detection flow atlas.

This is documentation data: importing it does not import detection algorithms,
load models, start pools, or run inference.
"""

B = 'python/algorithm/detect_board.py'
P = 'python/algorithm/board_pipeline.py'
T = 'python/transport/bridge_server.py'
R = 'python/algorithm/registration.py'
H = 'python/algorithm/height_measurement.py'
S = 'python/algorithm/2D/silver_inspection.py'
A = 'python/algorithm/2D/chip_anomaly.py'
G = 'python/algorithm/2D/defect_geometry.py'
C = 'python/algorithm/2D/circle_roi.py'
Z = 'python/algorithm/tilt_correction.py'
SEG = 'python/algorithm/3D/segment.py'
I = 'python/algorithm/instance_tasks.py'
V = 'visualization/validation_lab/server.py'
RP = 'python/algorithm/registration_process.py'
RW = 'python/algorithm/registration_worker.py'
D2 = 'python/algorithm/cpu_2d_worker.py'
P2 = 'python/algorithm/cpu_2d_process.py'
IG = 'python/algorithm/2D/instance_geometry.py'
J = 'python/algorithm/joint_geometry.py'
JW = 'python/algorithm/joint_worker.py'
JP = 'python/algorithm/joint_process.py'


def build_flows():
    scenes = []
    current = None

    def scene(key, title, subtitle, notes):
        nonlocal current
        current = dict(id=key, title=title, subtitle=subtitle, notes=notes, nodes=[], edges=[])
        scenes.append(current)

    def n(key, title, sub, col, row, steps, *, pool='', kind='serial', inputs='', outputs='', params=(), times=(), src=()):
        refs = [] if not src else [src] if isinstance(src[0], str) else list(src)
        current['nodes'].append(dict(id=key, title=title, subtitle=sub, col=col, row=row,
            steps=steps.split('|'), pool=pool, kind=kind, inputs=inputs, outputs=outputs,
            parameters=list(params), timings=list(times), sources=refs))

    def e(a, b, label='', kind='serial'):
        current['edges'].append(dict(source=a, target=b, label=label, kind=kind))

    scene('overview', '运行总览', '一板N片2D + 一张3D整板；各路按就绪事件推进', [
        '实线表示前置依赖；横向分支可重叠执行，不表示实际耗时。紫色代表并行投递，橙色代表汇合条件。',
        '联合入口需要本片joint_entry和整板深度；不需要等本片银胶检查完成，也不需要等所有2D片先到齐。',
        '整板最终响应仍需全部N份2D诊断及N份联合结果；单片先完成不等于整板已经返回。'])
    n('o-entry', '协议入口与板缓存', 'board_id / piece_id / request_id', 1, 0,
      '读取一行JSON请求并检查并发容量|按board_id建立或定位板缓存|2D片号由调用方指定，不按到达顺序编号|2D与3D请求可任意先后到达', kind='data', src=(T, '_accept'))
    n('o-2d', '每片2D输入', 'N片可并发准备', 0, 1,
      '校验板号和片号|读取共享内存图像|圆定位并裁剪检测图|构造粗配准平台掩码', pool='cpu', src=(B, 'prepare_2d_piece'))
    n('o-depth', '3D整板输入', '单张原始深度PNG', 2, 1,
      '按board_id接收整板深度|读取共享内存与PNG解码|保留整数深度，禁止用彩色预览替代', pool='cpu', src=(P, 'submit_3d'))
    n('o-fork', '2D准备完成，投递两支', 'GPU支路 + 银胶边界支路', 0, 2,
      '同一prepared对象由两个只读/受控分支消费|投递silver_prepare和gpu任务|回调计数器保护分支生命周期|不占用工作线程等待尚未到达的输入', pool='cpu', kind='parallel', src=(P, 'submit_2d'))
    n('o-gpu', 'YOLO → 掩码 → EfficientAD', '同一资源内顺序调用', 0, 3,
      '借用GPU模型资源槽位|YOLO分割后批量传回CPU掩码|逐个芯片进行EfficientAD|归还该资源的CUDA流和模型槽位', pool='gpu', src=(B, 'infer_2d_piece'))
    n('o-boundary', '银胶允许边界准备', '当前：检测crop四边内缩', 1, 3,
      '仅依赖图像/crop/圆定位，不依赖YOLO掩码|当前roi_inset模式生成允许矩形|可与YOLO和EfficientAD重叠|可选旧模板模式另走SIFT标定分支', pool='silver', src=(B, 'prepare_silver_boundary'))
    n('o-pitch', 'Pitch补偿', '非扫描轴条带并行', 2, 2,
      '启用时先矫正整板，再分片|按不发生位移的轴切条带|各条带独立投影、处理碰撞|父线程合并，更新深度坐标空间', pool='pitch', kind='parallel', times=('smartray_pitch_ms',), src=(Z, 'correct_smartray_pitch'))
    n('o-slices', '3D整板分片', '当前fixed；上到下1…N', 2, 3,
      'fixed模式直接验证并读取配置矩形|不扫描整图有效值|联合路径只按y1:y2取全宽深度视图|发布depth和pieces', pool='cpu', times=('depth_segmentation_ms',), src=(SEG, 'segment_depth_image'))
    n('o-geometry', '2D几何 + joint_entry', '弯曲 / 缺线 / 旋转 / 灰度', 0, 4,
      'GPU完成后在CPU准备联合输入|依次计算2D弯曲、ROI存在性和芯片旋转|缓存完整原图灰度|发布joint_entry，允许联合工作提前启动', pool='cpu', times=('geometry_phase_ms',), src=(B, 'prepare_joint_piece'))
    n('o-postgate', '2D后处理汇合', '等待边界与几何均完成', 0, 5,
      '分支计数归零后才投递2D CPU任务准备|此处是事件汇合，不是额外计算线程|若某分支失败，释放输入并返回片级失败', kind='gate', src=(P, 'submit_2d'))
    n('o-joingate', '本片联合就绪门', 'depth + 本片joint_entry', 2, 5,
      '整板深度就绪且该片joint_entry已发布|同片未被claimed，防止重复计算|受early_joint_workers约束|不等待银胶检查结束', kind='gate', src=(P, '_schedule_ready'))
    n('o-post', '2D共享mask与异步派发', '不传模型、GPU Tensor或整张照片', 0, 6,
      '只复制实例mask到只读共享区|携带必要公开字段与已准备Boundary|父准备线程不等待worker完成|不影响早期joint_entry发布', pool='postprocess', src=(P2, 'submit_piece'))
    n('o-cpu2d', '2D独立CPU进程', '银浆溢出 → 断连 → 公开几何', 0, 7,
      '独立于配准和主进程的核组|无Torch，OpenCV/BLAS为1|本片内按原实例顺序处理，不用3D instance池|返回诊断与公开几何，原始实例只读', pool='cpu_2d_process', src=(D2, 'compute'), times=('cpu_2d_phase_ms',))
    n('o-finalize2d', '父进程轻量合并', '不重复银浆与公开几何计算', 0, 8,
      '回填完整entry的银浆诊断|汇总原2D检测结论|构造响应并释放原图owner|整板最终仍需等待本次完整2D诊断', pool='postprocess', src=(B, 'finish_2d_piece'))
    n('o-shared', '共享输入与异步派发', '每板一份深度；每片gray/mask', 2, 6,
      '父准备线程复制并发布共享区|整板补偿后深度所有片复用|只向常驻子进程发送描述和数值配置|租约保护取消/释放期间的真实任务输入', pool='registration', times=('registration_shared_memory_ms',), src=(RP, 'prepare'))
    n('o-registration', '常驻进程独立配准', '粗配准 → 仿射；专用物理核组', 2, 7,
      '按registration_workers限制进程数|不加载Torch和模型，原生线程均为1|主进程与配准使用不同物理核组|只读共享区，每片拟合一个全局矩阵|矩阵和诊断返回后立即投递测高', pool='registration_process', times=('registration_phase_ms',), src=(RW, 'register_arrays'))
    n('o-jointprep', '测高共享输入与派发', '复用整板深度＋原2D实例mask', 2, 8,
      '深度复用配准发布的同一共享块|追加当前片只读原2D实例mask|仅发送高度配置、几何元数据和当前矩阵|父joint线程不等待测高进程', pool='joint', src=(JP,'submit_piece'),times=('joint_measurement_shared_memory_ms',))
    n('o-joint', '独立进程映射与全实例测高', '第三核组；本进程实例执行器', 2, 9,
      '按joint_workers限制进程数|映射、参考面、引线及焊脚都在worker完成|instance_workers仅属于当前测高进程，默认1|不加载Torch，主进程不执行实例或测高任务|全部wire/thin/bond参与，CPU结果回主进程最后合并2D诊断', pool='joint_measurement_process', times=('joint_measurement_phase_ms',), src=(J, 'measure_registered_piece'))
    n('o-2ddone', '本片2D响应 / entry就绪', '2D请求独立返回', 0, 9,
      '输出2D检查、公开掩码和阶段耗时|记录完整entry|尝试推进整板汇合', kind='data', src=(P, 'submit_2d'))
    n('o-3ddone', '本片联合结果就绪', '几何 / 平面 / 高度 / 配准', 2, 10,
      '保存结果到state.results[piece_id]|将高度点转换到整板坐标|掩码轮廓仍为片局部坐标|后续仍会合并完整2D诊断', kind='data', src=(P, '_measured'))
    n('o-all', '整板最终汇合', 'N个entries + N个results + 无inflight', 1, 11,
      '必须收齐所有2D诊断和联合结果|按片号排序|再次合并银胶、外观等晚到诊断|单片NG与接口错误分别处理', kind='gate', src=(P, '_complete_if_ready'))
    n('o-return', '整板JSON响应与释放', 'request_id匹配；stdout单行', 1, 12,
      '整板registration_ok与is_ok分别汇总|返回released板信息|删除本板缓存并完成Future|桥接层写JSON响应并归还请求容量', kind='data', src=(T, '_accept'))
    for a,b,label,kind in [('o-entry','o-2d','逐片提交','fork'),('o-entry','o-depth','整板提交','fork'),
        ('o-2d','o-fork','','serial'),('o-fork','o-gpu','','fork'),('o-fork','o-boundary','','fork'),
        ('o-depth','o-pitch','','serial'),('o-pitch','o-slices','','serial'),('o-gpu','o-geometry','','serial'),
        ('o-geometry','o-postgate','','wait'),('o-boundary','o-postgate','','wait'),
        ('o-geometry','o-joingate','提前发布','data'),('o-slices','o-joingate','','wait'),
        ('o-postgate','o-post','','serial'),('o-joingate','o-shared','','serial'),('o-shared','o-registration','跨进程描述','data'),('o-registration','o-jointprep','回调交接','data'),('o-jointprep','o-joint','共享输入','data'),('o-post','o-cpu2d','共享mask','data'),('o-cpu2d','o-finalize2d','结果回传','data'),('o-finalize2d','o-2ddone','','serial'),
        ('o-joint','o-3ddone','','serial'),('o-2ddone','o-all','','wait'),('o-3ddone','o-all','','wait'),('o-all','o-return','','serial')]:e(a,b,label,kind)

    scene('initialization', '初始化与资源', '协议初始化、算法初始化、模型副本和线程池', [
        '构造YOLO对象不等于完成首次predictor设置和预热；首次predict仍可能产生额外耗时。',
        '当前线程配置只是快照，真实服务还可能使用环境变量覆盖；初始化响应threading才是该进程实际设置。'])
    n('i-init','init：协议就绪','轻量响应，不加载全部模型',1,0,
      '启动bridge_server并设置原生线程默认值|构造默认执行器与请求容量|init只返回协议状态和当前线程信息|日志走stderr，stdout仅协议JSON',src=(T,'handle_init'))
    n('i-config','合并配置与计算线程设置','config/config.json → 2D + 3D',1,1,
      '读取manifest所指2D/3D文件|合并feature.description/parameters对象|解析线程预算和共享池关系|参数校验失败返回初始化错误',src=(T,'handle_init_algorithm'),params=('common_3d.parameters.threading','common_2d.parameters.threading'))
    n('i-board','构造统一检测器与解析算法参数','stage=board',1,2,
      '设置GPU副本数量G|读取圆定位、YOLO、银胶、外观、配准和高度参数|要求分片数量与input.piece_count一致|配置独立instance与Pitch任务池',src=(B,'_load_config'))
    n('i-cuda','检查CUDA与主YOLO构造','模型未初始化时不接受实际推理',1,3,
      '检查CUDA可用和YOLO device配置|根据YOLO模型路径构造主模型|未通过检查时释放已创建资源并返回错误',pool='gpu',src=(B,'handle_init'))
    n('i-ad','加载主EfficientAD资源','按长宽比规则加载所需模型',1,4,
      '解析aspect_ratio_models所引用模型名|逐一加载chip_short / chip_long等模型|读取各自图像尺寸、校准分数和阈值|没有配置外观功能时返回disabled',pool='gpu',src=(A,'load'))
    n('i-replicas','建立其余GPU工作资源','worker 2…G；初始化循环串行',1,5,
      '为每个GPU工作资源构造独立YOLO|为该资源加载独立EfficientAD预测器|建立独立锁和可用CUDA流|资源进入槽位队列，推理完成后归还',pool='gpu',src=(B,'_load_gpu_worker_replicas'))
    n('i-pools','应用正式执行器并建立pipeline','只有算法初始化成功才切换',1,6,
      '校验各独立池总数不超过worker_budget，独立registration池也计入|缺省silver池可别名到postprocess，joint可别名到cpu，registration可别名到joint|只有空闲时允许重配线程池|BoardPipeline持有这些执行器并按事件调度',kind='parallel',src=(T,'_configure_thread_pools'))
    n('i-process','预热三组CPU进程','32核起点：8＋8＋8，主进程留8',1,7,
      '分别初始化配准、2D后处理和3D测高全部worker，无Torch|按真实物理核/SMT拓扑划分三组独立范围|启用亲和性的池优先分配，其余从剩余CPU启动|每个测高进程有自己的实例执行器，默认1工作位|主进程使用剩余范围，关停逆序恢复|任一池失败清理全部已创建池',kind='parallel',pool='registration_process',src=(T,'_configure_cpu_processes'),params=('common_3d.parameters.threading.registration_affinity','common_2d.parameters.threading.cpu_process_affinity','common_3d.parameters.threading.joint_affinity'))
    n('i-ready','返回算法初始化成功','模型驻留，等待输入',1,8,
      '返回模型加载耗时和线程配置|同配置/同模型的验证台任务可复用检测进程|配置或模型文件指纹改变后重新初始化|退出时关闭自有池并释放模型',kind='data',src=(V,'_ensure'))
    for a,b in zip(['i-init','i-config','i-board','i-cuda','i-ad','i-replicas','i-pools','i-process'],['i-config','i-board','i-cuda','i-ad','i-replicas','i-pools','i-process','i-ready']):e(a,b)

    scene('two-d','2D主处理','本片顺序与两个阶段分支；不同片可同时处在不同阶段',[
        'YOLO与EfficientAD在同一资源内顺序执行；silver_prepare是独立并发支路。',
        'joint_entry发布后即可参与3D配准，完整2D响应需等后续银胶和公开几何。'])
    n('d-input','校验与读取图像','encoded 或 raw BGR8',0,0,
      '验证board_id非空、piece_id范围|同片inflight或已claimed时拒绝重复|encoded路径读取字节后解码|raw BGR8路径建立只读SharedImage视图并持有owner',pool='cpu',src=(B,'prepare_2d_piece'),times=('input_shared_memory_read_ms','input_image_decode_ms','input_read_decode_ms'))
    n('d-circle','在搜索框内找圆','灰度 → 高斯 → Hough',0,1,
      '验证circle_search_roi在原图范围内|裁搜索图并转灰度，9×9高斯平滑|固定半径±容差，按param2=35/28/22逐级尝试|优先选接近搜索区中心与期望半径的候选|加回搜索框偏移，输出完整2D圆心与半径',pool='cpu',src=(C,'find_circle'),params=('common_2d.parameters.crop',),times=('circle_find_ms',))
    n('d-crop','圆相对裁剪与2D诊断ROI','原图 → 检测crop',0,2,
      '使用圆心、半径和detection_roi_offsets生成检测crop|依据wire_bending_2d.roi_offsets生成弯曲/缺线框|将诊断框转换成crop相对坐标|crop.copy建立后续推理自有内存|缺少圆配置的兼容路径才使用固定crop_2d.roi',pool='cpu',src=(B,'_stream_crop'),params=('common_2d.parameters.crop.detection_roi_offsets','wire_bending_2d.parameters.roi_offsets','common_2d.parameters.crop_2d'),times=('roi_prepare_ms','roi_copy_ms'))
    n('d-platform','预备2D粗配准平台掩码','完整原图按scale缩放',0,3,
      '完整原图转灰度|按registration_3d.scale缩小|按threshold_2d二值化并闭运算|与检测crop不同，保留原图坐标对应关系|prepared保存图像、crop、圆、ROI和平台掩码',pool='cpu',src=(R,'platform_mask_2d'),params=('registration_3d.parameters',),times=('platform_mask_2d_ms',))
    n('d-boundary','并行：银胶边界准备','只依赖图像与crop',1,4,
      '在silver池执行prepare_silver_boundary|当前roi_inset模式无需等待分割|模板模式可借用同一灰度缓存锁|完成后写回prepared.silver_boundary',pool='silver',kind='parallel',src=(B,'prepare_silver_boundary'),times=('silver_prepare_phase_ms',))
    n('d-slot','并行：借用GPU模型资源','槽位 → 资源锁 → CUDA流',0,4,
      '从可用资源队列借出独立模型副本|记录槽位等待与锁等待|进入该资源CUDA流|同资源内保持YOLO、掩码提取和外观检测顺序',pool='gpu',kind='parallel',src=(B,'infer_2d_piece'),times=('gpu_model_slot_wait_ms','model_lock_wait_ms'))
    n('d-yolo','YOLO分割预测','一次调用，不等待凑满多张',0,5,
      '以当前单片crop调用model.predict，retina_masks=True|应用imgsz/conf/iou/max_det/device等配置|首次调用可能设置predictor并预热|返回实例掩码、类别与置信度|记录predict墙钟和模型reported三段耗时，口径不同',pool='gpu',src=(B,'_predict_batch'),params=('common_2d.parameters.yolo',),times=('yolo_predict_wall_ms','yolo_reported_inference_ms'))
    n('d-masks','整片掩码批量转换至CPU','GPU阈值化 / uint8 / 一次传输',0,6,
      '要求掩码为CUDA张量|尺寸不匹配时用nearest插值到crop大小|全部实例一起执行>0.5，再转uint8并乘255|一次cpu().numpy()传回本片所有掩码|读取classes/confidences并按min_component_area过滤|实例编号连续，CPU掩码切片保留各自区域',pool='gpu',src=(B,'_extract_instances'),params=('common_2d.parameters.mask.min_component_area',),times=('mask_extract_to_cpu_ms',))
    n('d-anomaly','芯片外观检测','每个chip顺序进入EfficientAD',0,7,
      '按chip掩码提取轮廓和最小外接矩形|从原始彩色图拉正芯片局部crop|按长宽比选择模型，超范围是接口错误|逐chip前向、分数校准和异常区域映射|结果写入chip_anomaly_2d',pool='gpu',src=(A,'inspect'),params=('chip_anomaly_2d.parameters',),times=('efficientad_total_ms',))
    n('d-gpudone','GPU资源归还','prepared.instances可供CPU消费',0,8,
      '保存实例、推理耗时和外观诊断|同步本资源流并归还槽位|GPU分支完成后投递geometry任务|原图owner仍保留，直到2D完整后处理结束',pool='gpu',src=(B,'infer_2d_piece'),times=('gpu_worker_stream_sync_ms',))
    n('d-bending','2D引线弯曲','wire / thin原始实例',0,9,
      '局部清理实例并保留偏移|按最大交集选择一个2D诊断ROI|找ROI周界的两个连续接触段|四个端点凸包形成评估四边形|覆盖率=四边形覆盖前景/本实例ROI内前景|与rectangle_ratio_threshold比较，ROI外标记outside_rois',pool='cpu',src=(G,'measure_wire_bending'),params=('wire_bending_2d.parameters',),times=('wire_bending_ms',))
    n('d-presence','2D缺线与ROI存在性','每个配置框独立给诊断',0,10,
      '使用已清理的弯曲结果选择ROI归属|缺少候选引线时给wire_missing|掩码被清理拒绝时给bending_unavailable|返回ROI框、关联实例和OK/NG|此ROI不限制后续3D测高目标',pool='cpu',src=('python/algorithm/2D/wire_roi_presence.py','check_wire_roi_presence'),times=('wire_roi_presence_ms',))
    n('d-rotation','芯片旋转 + 灰度缓存','geometry支路顺序收尾',0,11,
      '各chip轮廓求最小面积矩形|长边角度折算到最近坐标轴偏差[-45,45)|abs偏差与max_angle_deg比较|原图灰度通过gray_lock只转换一次|构建joint_entry并对实例字典做浅拷贝',pool='cpu',src=(B,'prepare_joint_piece'),params=('chip_rotation_2d.parameters',),times=('chip_rotation_ms','cache_gray_image_ms'))
    n('d-ready','提前发布联合输入','不等待银胶检查',0,12,
      '保存joint_entries[piece_id]|记录_2d_ready_at|立即尝试_schedule_ready|如depth已就绪，可投递本片联合任务',kind='data',src=(P,'submit_2d'))
    n('d-join','边界 + geometry汇合','两支都成功才进入2D最终处理',1,13,
      '回调remaining计数归零|合并各支路计时|先处理支路错误并释放图像owner|成功时投递postprocess池，不在原线程阻塞等待',kind='gate',src=(P,'submit_2d'))
    n('d-dispatch','只读mask准备与进程派发','每片独立共享区租约',1,14,
      '只复制实例mask和必要公开字段|复用已准备Boundary，不重复SIFT|不传整张图像或GPU对象|真实worker完成前共享区保持有效',pool='postprocess',src=(P2,'submit_piece'),times=('cpu_2d_shared_memory_ms','cpu_2d_queue_ms'))
    n('d-silver','进程内银胶检查','溢胶 → 断连，顺序不变',1,15,
      '建立本次检查的联合掩码缓存|先处理银浆溢出|随后处理银浆连通/断连|把局部几何加回crop原点|诊断返回父进程统一合并',pool='cpu_2d_process',src=(D2,'compute'),times=('silver_mask_checks_ms','silver_geometry_offset_ms'))
    n('d-public','进程内公开2D几何','本片串行，不同片跨进程并行',1,16,
      '逐实例读取只读mask|使用相同面积、包围框、轮廓函数|按原实例顺序输出|不占用3D instance池，也不修改父进程实例',pool='cpu_2d_process',src=(IG,'public_instance'),times=('public_2d_geometry_ms','public_2d_geometry_work_ms'))
    n('d-finalize','父进程轻量finalize','合并诊断与原有2D检查',1,17,
      '将worker诊断回填完整entry|不重复银浆和公开几何计算|构造实例与总OK/NG|记录子进程计算和父finalize之和，以及完整后处理链路',pool='postprocess',src=(B,'finish_2d_piece'),times=('cpu_postprocess_finalize_phase_ms','cpu_postprocess_pipeline_ms'))
    n('d-return','返回本片2D结果','保留缓存等待整板最终汇合',1,18,
      '汇总实例、缺线、旋转、外观与银胶结论|响应ok表示接口成功，不等于产品is_ok|写入完整state.entries并清除inflight|归还原图共享内存owner|尝试整板完成门',kind='data',src=(P,'submit_2d'),times=('request_pipeline_ms',))
    chain=['d-input','d-circle','d-crop','d-platform','d-slot','d-yolo','d-masks','d-anomaly','d-gpudone','d-bending','d-presence','d-rotation','d-ready','d-join','d-dispatch','d-silver','d-public','d-finalize','d-return']
    for a,b in zip(chain,chain[1:]):e(a,b)
    e('d-platform','d-boundary','并发投递','fork');e('d-boundary','d-join','等待边界','wait')

    scene('silver','银胶与芯片外观','各检测项内部步骤；本场景左右两列可处于并发支路',[
        '银胶溢出与断连在inspect_masks中按顺序执行，当前没有并行化这两个检测项。',
        'EfficientAD内部的局部芯片透视拉正仍存在；被移除的是2D→3D配准的透视求解，两者不同。'])
    n('s-boundary','银胶边界模式','当前roi_inset',0,0,
      '沿用检测crop坐标系|left/top/right/bottom独立内缩|检查矩形正面积与图内有效性|无效边界给不确定/未标定，不静默当作通过|旧polygon/calibration模式才加载模板及SIFT映射',pool='silver',src=(S,'prepare_boundary'),params=('silver_overflow_2d.parameters.boundary',))
    n('s-union','本次检查共享联合掩码','silver / chip / occlusion',0,1,
      '按归一化类别名称建立union缓存|同一类组合只阈值化合并一次|溢胶与断连共享silver/chip联合掩码|缓存仅属于本次图像，不跨图沿用|缺少掩码与尺寸不匹配有独立错误状态',pool='postprocess',src=(S,'_MaskUnionCache'),times=('silver_mask_union_ms',))
    n('s-overflow','银浆溢出测量','前景点到允许边界的距离',0,2,
      '枚举silver前景像素|分块计算到边界的signed_distance|大于容差的点成为候选|8连通域统计面积并按min_area_px筛选|输出缺陷面积、框、轮廓及最大越界距离|缺陷面积非零时is_ok=False',pool='postprocess',src=(S,'measure_overflow'),params=('silver_overflow_2d.parameters',),times=('silver_overflow_measure_ms',))
    n('s-labels','断连：芯片分量与外扩环','保持全局2×2扫描对齐',0,3,
      '复用chip联合掩码及连通域结果|在有效包围框内做连通域，原点向偶数对齐|每个达面积阈值的chip分量只在自身bbox比较标签|构造轴对齐外扩矩形，去掉芯片矩形与其它chip掩码|线/焊脚遮挡掩码可按配置膨胀',pool='postprocess',src=(S,'measure_continuity'),params=('silver_continuity_2d.parameters',))
    n('s-sectors','断连：环形区域角度分区','每个chip独立统计扇区',0,4,
      '只对局部外环前景计算atan2角度|按sector_count归入角度区间|bincount统计环、银胶、遮挡和覆盖像素|silver或遮挡都计入覆盖，避免把遮挡误认为断胶|有效像素不足为insufficient，否则判断covered/missing',pool='postprocess',src=(S,'_measure_chip_component'),times=('silver_continuity_measure_ms',))
    n('s-summary','银胶诊断汇总','保持原全尺寸掩码接口',0,5,
      '合并各chip局部掩码到公开全尺寸输出|面积计数只扫描所有外扩框的联合包围框|存在missing扇区则判断连|没有有效扇区则结论不确定|将诊断几何转换到完整2D图坐标',pool='postprocess',src=(S,'measure_continuity'),times=('silver_continuity_summary_ms','silver_geometry_offset_ms'))
    n('a-chip','外观：逐个chip处理','循环内部串行',2,0,
      '未配置则disabled；无chip则no_chip|局部提取掩码外部轮廓|加回crop原点得到原图轮廓|按原实例顺序处理各chip',pool='gpu',src=(A,'inspect'),times=('efficientad_chip_contour_ms',))
    n('a-rectify','外观：旋转矩形与局部拉正','保留局部透视裁剪',2,1,
      '计算最小外接矩形及角度修正|按padding_ratio扩展|把修正角clip到±max_rotation_deg，不把该上限当旋转NG阈值|四角透视拉正得到芯片crop|保留逆矩阵用于异常框映回原图',pool='gpu',src=(A,'rectified_chip_crop'),params=('chip_anomaly_2d.parameters.padding_ratio','chip_anomaly_2d.parameters.max_rotation_deg'),times=('efficientad_chip_rectify_ms',))
    n('a-route','外观：按长宽比选模型','闭区间规则逐项匹配',2,2,
      '用拉正矩形的长宽比匹配aspect_ratio_models|选择对应的chip_short/chip_long等模型|未匹配任何区间会抛ValueError|这属于请求/输入失败，不是普通产品NG',pool='gpu',src=(A,'_model_rule'),params=('chip_anomaly_2d.parameters.aspect_ratio_models',))
    n('a-infer','外观：预处理和前向','缩放归一化 → 张量 → 模型',2,3,
      '缩放到模型自身image_size并规范化|进入预测器锁和torch.inference_mode|执行模型得到anomaly_map|forward_enqueue为主机调用耗时，不能直接等同CUDA执行时间|后续取分数/响应图可能等待设备结果',pool='gpu',src=(A,'predict'),times=('efficientad_preprocess_ms','efficientad_forward_enqueue_ms'))
    n('a-score','外观：分数、校准和阈值','当前pool+top；严格大于阈值为NG',2,4,
      '按模型校准及配置选择评分方法|pool+top先池化并做高响应分数聚合|应用对应模型阈值：运行时覆盖优先于配置和checkpoint|score > threshold为NG，等于阈值为OK|仅在异常分数触发时生成二值定位响应',pool='gpu',src=(A,'_scores_and_response'),params=('chip_anomaly_2d.parameters.score_mode','chip_anomaly_2d.parameters.score_pool_kernel','chip_anomaly_2d.parameters.threshold'),times=('efficientad_score_and_response_ms',))
    n('a-boxes','外观：异常区域返回','形态学 / 连通域 / 坐标映射',2,5,
      '对异常二值图开闭运算|按最小面积筛选连通域框|响应图坐标换算到拉正crop|用逆透视矩阵映射到原始2D图|写入实例外观诊断并汇总所有chip结论',pool='gpu',src=(A,'_component_boxes'),times=('efficientad_component_boxes_ms','efficientad_total_ms'))
    for node in current['nodes']:
        if node['pool'] == 'postprocess':
            node['pool'] = 'cpu_2d_process'
    for chain in [['s-boundary','s-union','s-overflow','s-labels','s-sectors','s-summary'],['a-chip','a-rectify','a-route','a-infer','a-score','a-boxes']]:
        for a,b in zip(chain,chain[1:]):e(a,b)

    scene('depth','3D深度与分片','先整板Pitch补偿，后按片号给出深度视图',[
        'Pitch只矫正深度，同栅格强度图可在独立测试函数里同步；正式2D相机照片不使用Pitch公式。',
        'fixed_rois是已补偿整板上的固定分片矩形；它和已删除的“测高目标选择ROI”是不同用途。'])
    n('z-input','深度共享内存与解码','保留单通道整数数据',1,0,
      '校验shm_name、data_size与板号|读取完整编码深度字节|decode_depth_png验证PNG与单通道整数类型|颜色预览、截图或raw uint16缺尺寸输入不能替代原深度',pool='cpu',src=(B,'_read_3d'),times=('depth_shared_memory_read_ms','depth_image_decode_ms','depth_read_decode_ms'))
    n('z-switch','是否启用正式Pitch','enabled_3d开关',1,1,
      '正式API按未补偿原始深度解释输入|enabled_3d=False时原数组返回|启用时要求原始uint16、有效角度和正扫描步距|z_step_mm/gray_offset须与测高校准一致|原input_state不是正式配置开关；测试台另有SDK已补偿状态',kind='decision',src=(Z,'production_pitch_parameters'),params=('smartray_pitch_3d.parameters',))
    n('z-strips','沿非扫描轴分条带','rows分列；columns分行',1,2,
      '条带数取配置线程数、可用池容量和非扫描轴尺寸的最小值|每条带覆盖完整扫描轴|独立Pitch池运行条带任务|输入只读，每条带分配独立输出|大位移仍不会跨条带写同一目标像素',pool='pitch',kind='parallel',src=(Z,'correct_smartray_pitch'),params=('smartray_pitch_3d.parameters.projection_workers',))
    n('z-project','每条带：投影深度点','有效值 → 毫米 → 平移与Z修正',1,3,
      '仅枚举0 < raw < 65535的输入点|设备Z=(raw−gray_offset)×z_step_mm+z_origin_mm|还原公式：Z=Z′/cosβ，扫描坐标位移=−Z′tanβ|文档正向：Z′=Zcosβ，扫描位移=Zsinβ|位移除scan_pitch_mm后加到扫描行/列并np.rint|Z换回整数灰度，再过滤空间越界和灰度1…65534范围',pool='pitch',src=(Z,'correct_smartray_pitch'))
    n('z-collision','每条带：解决投影碰撞','np.maximum.at；深度越大获胜',1,4,
      '同一目标像素多个投影时保留最大矫正深度|统计输入、投影、输出点数量|独立测试的强度图复用同一批投影点|矫正深度平局时取最大源索引对应强度|不插值填洞，空位保持0',pool='pitch',src=(Z,'correct_smartray_pitch'),times=('pitch_projection_work_ms',))
    n('z-merge','父线程合并补偿结果','统一补偿后整板栅格',1,5,
      '按条带顺序回填独立缓冲|累加discarded_points和collision_points等统计|启用时坐标空间变为pitch_corrected_board_3d|正式2D照片保持原样|后续配准和参考面全部使用此补偿后深度',pool='cpu',src=(Z,'depth_coordinate_space'),times=('smartray_pitch_ms',))
    n('z-mode','选择分片模式','当前配置见参数表',1,6,
      '读取segmentation_3d.parameters|expected_piece_count必须和板片数一致|fixed和auto走不同边界定位路径|配置或边界错误直接抛错，不静默改变片数',kind='decision',src=(SEG,'segment_depth_image'),params=('segmentation_3d.parameters',))
    n('z-fixed','fixed：直接取标定矩形','不读像素有效性统计',0,7,
      '矩形数必须等于expected_piece_count|每项为整数[x1,y1,x2,y2]，右/下边界不含|校验正面积、图内、由上到下且行范围不重叠|不加padding、不裁越界、不回退auto|按列表顺序分配piece_id=1…N',pool='cpu',src=(SEG,'segment_depth_image'),times=('depth_segmentation_ms',))
    n('z-auto','auto：有效像素定位','行占比 → 带区间 → 列范围',2,7,
      'depth > min_valid_depth生成有效掩码|统计每行有效占比并填短断口|筛选满足最小片高的带区间|初始阈值不满足片数时尝试观测到的更高占比|每片按列有效占比取X边界并加padding|检测片数不符返回piece_count_mismatch',pool='cpu',src=(SEG,'segment_depth_image'))
    n('z-ready','发布整板深度和片描述','depth_ready_at',1,8,
      '整板深度仍只有一份|联合路径实际取depth[y1:y2, :]，X保留全宽|不复制每片深度数组|记录本板深度就绪时刻|尝试为已有joint_entry调度对应片',kind='data',src=(P,'submit_3d'))
    e('z-input','z-switch');e('z-switch','z-strips','启用');e('z-switch','z-mode','关闭：原样跳过','optional')
    for a,b in zip(['z-strips','z-project','z-collision','z-merge'],['z-project','z-collision','z-merge','z-mode']):e(a,b)
    e('z-mode','z-fixed','fixed');e('z-mode','z-auto','auto');e('z-fixed','z-ready');e('z-auto','z-ready')

    scene('registration','配准与掩码映射','每片一套矩阵；评分区不代表分别变换',[
        '当前正式路径是粗配准+6参数Powell仿射，没有额外透视求解。R1/R2/R3共同约束同一个矩阵。',
        'score_mode默认full，仍只在所选ROI计算损失。ROI自建坐标映射为手动试验路径，曾发生耗时回退。',
        '停止优化不等于对齐合格；当前合格逻辑不要求optimizer.success=True。'])
    n('r-gate','本片联合工作启动','depth + joint_entry已就绪',1,0,
      '父准备线程构造共享输入并异步派发|整板深度仅复制一次，每片gray/mask各一次|只发送当前片行范围和数值配置，不发送检测器或模型|无需等银胶最终诊断',pool='registration',src=(B,'submit_registration'),times=('registration_dispatch_queue_ms','registration_shared_memory_ms','matching_input_wait_ms'))
    n('r-worker','子进程挂接只读共享区','CPU专用核组；不加载模型',1,1,
      '常驻worker取到本片描述|挂接gray/mask/整板深度并设只读|按y1:y2取全宽深度视图|执行与直接求解相同的register_arrays|此处开始记录registration_phase_ms',pool='registration_process',src=(RW,'run'),times=('registration_queue_ms','registration_phase_ms'))
    n('r-platform','构造3D平台掩码','depth_piece > 0并闭运算',1,1,
      '对片深度的非零像素二值化|按registration_3d.morphology_kernel_size闭运算|2D平台掩码此前已在准备阶段生成|两幅掩码进入固定尺度模板匹配',pool='joint_measurement_process',src=(R,'platform_mask_3d'),times=('platform_mask_3d_ms',))
    n('r-coarse','粗配准模板匹配','TM_CCOEFF_NORMED',1,2,
      '从3D掩码裁roi_3d作模板，底部保留安全边界|从缩放2D掩码裁search_2d_scaled作搜索区|matchTemplate取最高相关位置|结合模板与搜索框偏移得到tx/ty|构造原始2D→片局部矩阵：scale对角项+平移|粗分数与minimum_score给粗有效标记',pool='joint_measurement_process',src=(R,'register_platform_masks'),params=('registration_3d.parameters',),times=('coarse_registration_ms',))
    n('r-source','仿射评分准备','灰度阈值化；源2D再按preview_scale缩放',1,3,
      '原2D灰度>threshold_2d后变为0/1|源掩码按preview_scale做nearest缩放|目标depth>0保持原始3D分片画幅|两侧按仿射morphology_kernel_size闭运算|裁出所选R1/R2/R3目标掩码供评分复用',pool='joint_measurement_process',src=(R,'refine_affine_registration'),params=('dynamic_affine_3d.parameters',))
    n('r-initial','粗矩阵换算6参数初值','sx sy kxy kyx cx cy',1,4,
      '矩阵线性项除preview_scale得到内部sx/sy/交叉项|内部参考点固定为缩小2D中的(1000,400)|cx/ cy是该参考点映到3D后的坐标|默认使用粗配准；可选标定profile只复用线性四项，保留本片中心位置|始终保留原粗配准的scale/shear/translation边界，候选须在范围内且完整评分改善',pool='joint_measurement_process',src=(R,'_affine_parameters_from_matrix'),params=('dynamic_affine_3d.parameters.scale_delta','dynamic_affine_3d.parameters.shear_limit','dynamic_affine_3d.parameters.translation_delta'))
    n('r-score','单次评分','原生warpAffine → 区域差异 → 惩罚',1,5,
      '由6参数生成缩小源→3D片局部仿射矩阵|full路径用原生warpAffine生成完整目标图|只比较选定评分框的二值不一致比例|分区等权平均，不按区域面积再加权|加penalty_weight×[(kxy/shear_limit)²+(kyx/shear_limit)²]|可启用精确6参数缓存，仅保留标量和各区损失；每个分辨率独立缓存并分别统计实际变换次数',pool='joint_measurement_process',src=(R,'refine_affine_registration'),times=('affine_score_total_ms',))
    n('r-powell','Powell方向搜索','单条优化链内串行依赖',1,6,
      '依次沿各方向做有界线搜索|每次尝试参数都调用评分函数|后一方向使用前一方向更新后的参数|一轮可评分多次，maxiter不是评分次数|当前没有并行候选搜索；可选multiscale按低分辨率搜索→完整精修顺序执行，默认关闭',pool='joint_measurement_process',src=(R,'refine_affine_registration'))
    n('r-stop','是否停止优化','改善足够小 / 迭代上限 / 数值异常',1,7,
      '整轮改善按ftol判定：2×(旧损失−新损失) ≤ ftol×(abs旧+abs新)+极小量|达到maxiter也停止|xtol用于方向搜索精度，不直接表示像素误差|maxfev可限制每级调用，total_maxfev限制总调用；均默认null，无毫秒超时|单级触边不自动回退；多级精修窗口触边会尝试原范围回退，总预算耗尽时明确记录未运行',pool='joint_measurement_process',kind='decision',src=(R,'refine_affine_registration'),params=('dynamic_affine_3d.parameters.maxiter','dynamic_affine_3d.parameters.xtol','dynamic_affine_3d.parameters.ftol'))
    n('r-verdict','完整评分与最终质量标记','独立于优化器停止状态',1,8,
      '对最终参数再评分并输出各区损失|换回原始2D→3D片局部2×3矩阵及3×3表示|矩阵全部有限且最终损失≤max_score才affine_valid|未配置max_score时默认0.25|不要求optimizer.success=True；迭代触顶仍可能质量达标|记录缓存/真实warp/逐级nfev/均耗/预算/回退；预算截断时保留本阶段最佳有限试探，最终仍用完整评分',pool='joint_measurement_process',src=(R,'refine_affine_registration'),times=('affine_registration_ms',))
    # Only input preparation runs in parent threads; all solver steps run in
    # persistent model-free processes, before returning to the joint executor.
    for node in current['nodes']:
        if node['id'] != 'r-gate':
            node['pool'] = 'registration_process'
            if node['id'] != 'r-worker':
                node['row'] += 1
    n('r-handoff','返回矩阵并提交第三组进程','复用深度；不占父线程等待',1,10,
      '父进程接收矩阵和评分诊断|复用整板深度与本片实例mask共享块，2D后处理与3D测高只发布一份mask|只向测高worker发送描述和高度参数|真实Future结束才释放深度及mask租约|已释放或失败的板不再开始测高',kind='data',src=(P,'_registered'),times=('registration_result_transfer_ms','joint_measurement_queue_ms'))
    n('r-map','逐实例并行映射','共享只读H；独立目标缓冲',1,11,
      '原crop局部掩码先加crop左上角变成原图坐标|实际矩阵为H_full×T_crop_to_full|nearest映射到完整片局部深度画幅|各实例提取2D/3D几何，字典和输出由任务独立持有|父线程按原实例顺序汇总|一片左右及全部类别都使用同一个矩阵',pool='instance',kind='parallel',src=(R,'align_crop_mask_to_depth'),times=('mask_alignment_ms','instance_alignment_work_ms'))
    n('r-height','进入参考面与全实例测高','已对齐实例 + 深度片',1,12,
      '语义掩码用于参考面排除|完整实例用于chip/bond归属等|wire/thin测高会另按清理/分段构造测量掩码|不按2D引线ROI筛选测高对象',pool='joint_measurement_process',src=(J, '_measure_piece'))
    chain=['r-gate','r-worker','r-platform','r-coarse','r-source','r-initial','r-score','r-powell','r-stop','r-verdict','r-handoff','r-map','r-height']
    for a,b in zip(chain,chain[1:]):e(a,b)
    e('r-stop','r-score','继续搜索','loop')

    scene('height','全实例测高','测高进程内：参考面 → 引线 → 焊脚；默认串行，不同片跨进程并行',[
        '最新代码已删除测高目标选择ROI：全部已检测wire/thin/bond参与，清理后为空也保留无效引线记录。',
        '映射与测高都在独立进程；引线、焊脚按先后阶段使用本进程实例执行器，默认1工作位。芯片面在当前进程逐chip准备，不占主进程。',
        '代表点按原始深度P95选取，不是对“相对参考面的高度差”求P95，也不是简单取最大值。'])
    n('h-union','建立完整语义排除掩码','所有对齐YOLO实例的union',1,0,
      '按像素合并所有完整对齐实例|按semantic_exclusion_dilation_px膨胀|此处不使用内缩后的测高掩码|防止线、焊点等进入平台平面采样',pool='joint_measurement_process',src=(J, '_semantic_exclusion_mask'),times=('height_semantic_exclusion_ms',))
    n('h-platformroi','映射引脚平台采样ROI','2D圆相对标定 → 深度片',1,1,
      '圆心/半径结合platform_roi_offsets生成原图矩形|使用当前片仿射矩阵映射多边形|采样区域减去完整语义排除掩码|排除invalid_values深度|没有标定、圆或有效配准时平台面标记invalid',pool='joint_measurement_process',src=(H,'map_platform_rois'),params=('height_measurement_3d.parameters.platform_roi_offsets','height_measurement_3d.parameters.invalid_values'))
    n('h-platformfit','拟合全局引脚平台面','平面z=a·x+b·y+c',1,2,
      '提取有效平台采样点|按stride=max(1,N//30000)降采样，非严格3万点上限|先保留plane_percentile对应的较低深度点|最多5轮最小二乘和残差MAD剔除|残差阈值取max(50灰度,4.4478×MAD)|点数不足时标记无效，不伪造平面',pool='joint_measurement_process',src=(H,'_fit_plane_points'),params=('height_measurement_3d.parameters.plane_percentile','height_measurement_3d.parameters.plane_min_valid_points'),times=('height_fit_platform_plane_ms',))
    n('h-chipcache','准备芯片归属缓存','父线程统一构造，只读共享',1,3,
      '对各chip的原2D掩码取填充外部区域|填充芯片轮廓内孔洞供归属判断|缓存区域面积|后续焊脚工作线程只读，不懒写共享缓存|此缓存不作为芯片平面实际采样掩码',pool='joint_measurement_process',src=(J, '_prepare_chip_regions'),times=('height_prepare_chip_regions_ms',))
    n('h-chipfit','逐chip拟合芯片参考面','保留本chip，排除其它实例',1,4,
      '为当前chip重建“除它以外”的语义union|不能在全局union上清空chip区域，否则会恢复重叠焊点|chip掩码内排除其它语义和无效深度|检查chip_min_valid_points|采用同一稳健平面拟合，但不先做低百分位筛选|失败则该chip平面None，不悄悄替换参考',pool='joint_measurement_process',src=(H,'fit_chip_plane'),params=('height_measurement_3d.parameters.chip_min_valid_points',),times=('height_fit_chip_plane_ms',))
    n('h-thin','thin一次分段准备','主轴SVD：两端 + 中段',1,5,
      '在原2D thin掩码前景坐标上做主轴SVD|按投影长度划分start/end端部|中间部分作为引线，端部分别作为焊脚|父线程保存本次分段结果供两个阶段复用|不使用2D引线ROI裁剪这些区域',pool='joint_measurement_process',src=(J, '_prepare_thin_parts'),params=('height_measurement_3d.parameters.thin_solder_end_ratio',),times=('height_prepare_thin_ms',))
    n('h-wirejobs','全部wire/thin逐根并行','ROI外目标也必须返回记录',1,6,
      '枚举每个wire和thin，保持输入顺序|wire使用完整掩码；thin使用准备好的中间段|在包围框和形态学halo内清理|不做ROI存在性判断，也不再裁剪ROI|已有bending_2d仅携带展示，3D不重新计算弯曲',pool='instance',kind='parallel',src=(J, '_measure_wire_regions'),times=('height_wire_regions_ms','height_wire_work_ms'))
    n('h-wiremask','每根：映射、内缩与有效点','3D像素半径定义',1,7,
      '将局部清理偏移合入crop到原图变换|映射得到独立引线测高掩码|按wire或thin_measurement_erode_px内缩|只修改自有测量掩码，不修改语义掩码|空/无有效深度或无平台面时保留无效记录',pool='instance',src=(J, '_measure_wire_region'),params=('height_measurement_3d.parameters.wire_measurement_erode_px','height_measurement_3d.parameters.thin_measurement_erode_px'))
    n('h-wirepoint','每根：P95点与平台高度差','原始灰度分位 → 最近实际像素',1,8,
      '枚举测量掩码内有效深度|计算height_percentile分位值|选与该分位值最接近的实际像素，按行序处理平局|在同一个(x,y)计算参考平面深度|高度差=(点灰度−平面灰度)×vertical_resolution_mm|严格小于max_wire_platform_height_mm才OK',pool='instance',src=(H,'measure_mask_height'),params=('height_measurement_3d.parameters.height_percentile','height_measurement_3d.parameters.max_wire_platform_height_mm'))
    n('h-wirejoin','引线结果有序汇合','等待该片全部引线任务',1,9,
      '按原输入顺序连续分配region_id|每个wire/thin有独立height_3d记录|高度有效性与2D弯曲结论分开|准备焊脚任务列表',kind='gate',src=(J, '_measure_wire_regions'))
    n('h-solderjobs','全部bond与thin端点并行','每个端点独立任务',1,10,
      'thin的start/end使用已分好的原2D端部掩码|端部映射到3D后先膨胀搜索范围|bond直接使用完整对齐实例掩码|保持实例顺序以及start/end顺序|任务只返回结果，结束前不写共享实例字段',pool='instance',kind='parallel',src=(J, '_measure_solder_task'),params=('height_measurement_3d.parameters.thin_solder_search_dilation_px',),times=('height_solder_wall_ms','height_solder_work_ms'))
    n('h-ownership','每焊脚：归属芯片或平台','2D掩码重心，不按3D最高点判断',1,11,
      '计算焊脚原2D掩码前景平均坐标|判断其是否位于填充chip外部区域内|重叠多chip时选包含它且面积最小的chip|在chip上时使用该chip平面和芯片焊脚阈值|chip平面不可用则测高无效，不退成平台焊脚',pool='instance',kind='decision',src=(J, '_matching_chip'))
    n('h-localplane','平台焊脚：局部环形平面','局部失败才回退全局平台面',0,12,
      '对原焊脚区域构造外膨胀环减内膨胀区|去掉完整语义排除和无效深度|仅在bbox+halo内计算环形采样|有效点足够时拟合local_wire_platform|局部拟合失败时才回退全局wire_platform平面',pool='instance',src=(H,'fit_local_platform_plane'),params=('height_measurement_3d.parameters.platform_local_ring_px',))
    n('h-chipplane','芯片焊脚：使用既有芯片面','不另拟合局部平台',2,12,
      '从chip_planes取归属芯片平面|reference=chip，保存reference_instance_id|使用max_solder_chip_height_mm|缺少有效芯片平面时保留无效height_3d',pool='instance',src=(J, '_measure_solder'))
    n('h-solderpoint','每焊脚：内缩、P95与高度差','归属先确定；测量再内缩',1,13,
      '按bond或thin_solder_measurement_erode_px内缩独立bbox|不修改共享完整实例掩码|使用与引线相同的实际P95代表点方法|按芯片/平台对应阈值判定高度|记录reference/location/measurement_id及2D/3D测量区域',pool='instance',src=(J, '_measure_solder'),params=('height_measurement_3d.parameters.bond_measurement_erode_px','height_measurement_3d.parameters.thin_solder_measurement_erode_px','height_measurement_3d.parameters.max_solder_chip_height_mm','height_measurement_3d.parameters.max_solder_platform_height_mm'),times=('height_solder_ms',))
    n('h-merge','父线程汇总高度与实例字段','所有子任务结束后再写回',1,14,
      '将引线结果按instance_id关联|将bond高度和芯片归属写回实例|chip保存plane_3d，非测高类别height_3d=None|高度组件is_ok只看高度，不再AND弯曲|整体联合结论仍会合并2D ROI与诊断|失败任务先取消未运行任务、等待运行中的兄弟任务再抛错',pool='joint_measurement_process',src=(J, '_measure_piece'),times=('height_measurement_ms',))
    chain=['h-union','h-platformroi','h-platformfit','h-chipcache','h-chipfit','h-thin','h-wirejobs','h-wiremask','h-wirepoint','h-wirejoin','h-solderjobs','h-ownership']
    for a,b in zip(chain,chain[1:]):e(a,b)
    e('h-ownership','h-localplane','不在chip');e('h-ownership','h-chipplane','在chip')
    e('h-localplane','h-solderpoint');e('h-chipplane','h-solderpoint');e('h-solderpoint','h-merge','有序汇合','wait')

    scene('output','坐标与结果发布','跨坐标系的变换只做一次；整板合格与接口成功分开',[
        '仿射矩阵映射完整原图2D坐标到本片局部深度坐标；不是crop局部矩阵，也未包含整板行偏移。',
        'Pitch开启后，整板高度点属于补偿后栅格；不能直接叠到原始未矫正深度PNG上。'])
    n('u-crop','crop局部 → 完整2D','加crop左上角',0,0,
      'YOLO掩码在检测crop局部坐标中|输出mask_2d几何时加crop原点|若清理用bbox，再额外加mask_offset|不要再把同一crop偏移加第二次',kind='data',src=(J, '_align_instance'))
    n('u-affine','完整2D → 3D片局部','全片一个H',1,1,
      '完整原图点乘H_full得到片局部点|crop掩码使用H_full×T_crop_to_full|输出mask_3d_local、参考平面和采样区均保留片局部坐标|不因为存在多个评分ROI而分开扭曲图像',kind='data',src=(R,'align_crop_mask_to_depth'))
    n('u-global','片局部高度点 → 整板高度点','x不变，y加depth_start_row',1,2,
      '联合路径每片保留全宽，故X已是整板X|只给高度point.y加分片起始行|为每条测高记录标记坐标空间|已有整板坐标的点不重复偏移|完整掩码轮廓仍是片局部，不能跟着全部平移',kind='data',src=(H,'height_points_to_board'),times=('height_points_to_board_ms',))
    n('u-quality','本片联合质量','高度 + 2D检查 + 晚到诊断',1,3,
      '高度组件合格来自所有已测高度结果|流式片结果同时检查2D wire_roi_checks|银胶/旋转/外观诊断齐全后再合并|disabled项不强行当作失败|产品NG仍可返回ok=True',kind='gate',src=(B,'finalize_joint_piece'))
    n('u-all','收齐整板N片','entries/results/inflight门',1,4,
      'len(results)=N且len(entries)=N|不能还有inflight片|按片号升序合并|再次使用完整entry合并晚到2D诊断|registration_ok要求各片registration.is_valid',kind='gate',src=(P,'_complete_if_ready'))
    n('u-json','整板响应与缓存释放','ok / registration_ok / is_ok分层',1,5,
      'ok：协议/算法请求是否成功|registration_ok：全部配准质量是否通过|is_ok：配准且所有片最终产品检查通过|返回片数组、耗时、Pitch和released信息|从板缓存删除本板并完成Future',kind='data',src=(P,'_complete_if_ready'))
    n('u-client','调用方关联响应','request_id，不按响应先后配对',1,6,
      '桥接层stdout只写一行JSON|ProtocolClient按request_id唤醒对应等待者|2D请求可先于或后于3D完成|日志写stderr不进入JSON解析|子进程退出时唤醒全部未完成请求',kind='data',src=('visualization/detect_and_view.py','ProtocolClient'))
    n('u-report','验证台报告与交互','展示不重新求解、不改判定',1,7,
      '保存原响应JSON、阶段CSV和日志|按实际矩阵叠加原彩色2D和深度伪彩|按测量编号关联全部高度与2D弯曲|彩色预览只用于观察，毫米值读取算法结果|历史输入/结果保留，不因清空页面清单删除文件',kind='data',src=(V,'run'))
    for a,b in zip(['u-crop','u-affine','u-global','u-quality','u-all','u-json','u-client'],['u-affine','u-global','u-quality','u-all','u-json','u-client','u-report']):e(a,b)

    scene('lifecycle','异常与生命周期','正常NG、输入错误、失败传播、超时和停止',[
        '90秒是板缓存等待缺失输入的空闲期限，不是单片计算时间上限；活跃计算期间不按此规则清理。',
        '验证台request超时900秒与Powell的停止条件无关。停止验证台会先中断子进程，再等待工作池退出。'])
    n('f-check','输入与请求校验','不合法请求不等于产品NG',0,0,
      '片号越界/重复inflight/已claimed时拒绝|超过26个挂起协议请求时server_busy|超过最大挂起板数拒绝新板|验证台另检查每板片号齐全、无重复、深度唯一',src=(T,'_accept'))
    n('f-piece','2D支路失败','圆/模型范围/编码/分支异常',0,1,
      '清理该片inflight与joint_entry|未被联合认领时先返回片级失败，允许调用方处理|若该片已claimed，终止相应整板|释放原图owner，避免共享内存被永久持有|不把异常转换成正常产品OK',src=(P,'submit_2d'))
    n('f-board','深度或联合失败','整板Future进入失败终态',0,2,
      'PNG/补偿/分片错误返回3d_board_failed|联合函数异常返回joint_detection_failed|删除本板缓存并完成等待中的整板Future|已在运行的任务用_current检查避免污染下一板',src=(P,'_terminate'))
    n('f-validate','验证台看到任一失败','显式release_board',0,3,
      '记录最初失败原因，不用后续board released覆盖|发送release_board解除等待|仅2D模式结束也释放本板缓存|验证报告区分产品NG和请求失败',src=(V,'run'))
    n('f-expire','等待缺失输入的空闲超时','当前90秒；活跃任务不误删',0,4,
      '周期检查pending boards|有inflight/depth_inflight/未完成claimed任务则仍为活跃|只有不活跃且touched超过期限才超时|返回missing_piece_ids并终止该板',kind='decision',src=(P,'expire_pending'))
    n('f-request','客户端请求超时或进程退出','验证台每个请求900秒',0,5,
      '900秒超时会kill检测子进程，防止晚响应串到下个请求|协议stdout EOF会通知全部pending等待队列|后续请求遇closed立即失败|正常close先尝试release，强制stop跳过长等待',src=('visualization/detect_and_view.py','request'))
    n('f-start','验证台启动/重复启动','--open只负责打开网页',2,0,
      '前台python命令是常驻HTTP服务|同端口健康身份匹配时复用已有服务|同时启动导致bind竞态时再检查一次|pythonw.exe可无窗口运行|只绑定127.0.0.1',src=(V,'main'))
    n('f-model','模型复用与重新初始化','配置/模型指纹',2,1,
      '同配置、模型时间与大小不变时复用|配置或模型文件指纹改变先关闭旧进程|重新发送init和init_algorithm|改变参数后需重新初始化才进入运行配置',src=(V,'_ensure'))
    n('f-stop','停止验证台','Ctrl+C 或 --stop',2,3,
      '停止命令先核对本机服务身份|同源shutdown接口结束HTTP循环|先强制关闭自有检测进程并唤醒请求|再取消未开始验证、等待当前线程结束|不能先join任务池再close子进程，否则可能等满900秒',src=(V,'server_close'))
    n('f-resources','释放资源与保留报告','实例池 / Pitch池 / 模型 / 缓存',2,4,
      '关闭pipeline并终止未完成板Future|关闭桥接执行器及检测器自有池|释放模型、槽位和缓存|保留已保存输入与报告|关闭浏览器本身不等于停止后台服务',src=(B,'release'))
    for a,b in zip(['f-check','f-piece','f-board','f-validate','f-expire'],['f-piece','f-board','f-validate','f-expire','f-request']):e(a,b,'可能路径','optional')
    e('f-start','f-model');e('f-model','f-stop','手动停止','optional');e('f-stop','f-resources')
    return scenes
