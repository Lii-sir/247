"""Build the self-contained detection-flow.html; never run detection/models."""
from __future__ import annotations
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import subprocess

from detection_flow_data import build_flows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_ref(path, symbol):
    file = ROOT / path
    tree = ast.parse(file.read_text(encoding='utf-8-sig'))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol]
    if not matches:
        raise ValueError(f'Missing source anchor: {path}:{symbol}')
    return dict(path=path, symbol=symbol, line=matches[0].lineno)


def make_document():
    two = load_json(ROOT / 'config/2D/config.json')
    three = load_json(ROOT / 'config/3D/config.json')
    config = {**two, **{key:value for key,value in three.items() if key != 'model_path'}}
    scenes = build_flows()
    seen = set()
    for scene in scenes:
        local = set()
        for node in scene['nodes']:
            if node['id'] in seen:
                raise ValueError('Duplicate node: ' + node['id'])
            seen.add(node['id']);local.add(node['id'])
            if not node['steps'] or any(not step.strip() for step in node['steps']):
                raise ValueError('Empty processing step: ' + node['id'])
            node['sources'] = [source_ref(*ref) for ref in node['sources']]
            node['x'] = 42 + node['col'] * 354
            node['y'] = 54 + node['row'] * 146
            node['w'], node['h'] = 294, 92
        for edge in scene['edges']:
            if edge['source'] not in local or edge['target'] not in local:
                raise ValueError('Dangling flow edge: ' + repr(edge))
        scene['width'] = 1130
        scene['height'] = max(n['y'] + n['h'] for n in scene['nodes']) + 64
    try:
        revision = subprocess.run(['git','rev-parse','--short','HEAD'],cwd=ROOT,
                                  check=True,capture_output=True,text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    t2 = two['common_2d']['parameters']['threading']
    t3 = three['common_3d']['parameters']['threading']
    budget = int(t3.get('worker_budget',13))
    gpu = int(t2.get('gpu_workers',1));post = int(t2.get('postprocess_workers',5))
    cpu = int(t3.get('cpu_workers',max(1,budget-gpu-post)))
    silver = int(t2.get('silver_workers',post));joint = int(t3.get('joint_workers',cpu))
    registration = int(t3.get('registration_workers',joint))
    instance = int(t3.get('instance_workers',16))
    pitch = int(three.get('smartray_pitch_3d',{}).get('parameters',{}).get('projection_workers',1))
    effective = cpu+gpu+post+(silver if 'silver_workers' in t2 else 0)+(joint if 'joint_workers' in t3 else 0)
    effective += registration if 'registration_workers' in t3 else 0
    pools = [
        ['cpu',cpu,'2D读取/圆定位/平台掩码；geometry；3D读取/Pitch调用/分片','不同片与深度输入可重叠；一个任务内部按函数顺序执行'],
        ['gpu',gpu,'每槽独立YOLO + EfficientAD资源和流','同资源内YOLO→掩码→EfficientAD串行；各资源可重叠'],
        ['silver',silver,'图像侧银胶允许边界准备','缺少silver_workers时别名到postprocess；当前是否独立见配置'],
        ['postprocess',post,'银胶溢出/断连、公开几何汇总和完整2D响应','溢胶与断连当前串行；公开几何投递到instance池'],
        ['registration',registration,'每片3D平台掩码、粗配准和仿射','完成后回调提交joint；不等待映射测高；缺项别名到joint'],
        ['joint',joint,'每片掩码映射、参考面与高度阶段','独立限制并发；接收同片求得的矩阵；不重复配准'],
        ['instance',instance,'实例映射、公开2D几何、各根引线及各个焊脚','额外池，不计入桥接worker_budget；父任务等子任务，子任务不嵌套提交同池'],
        ['pitch',pitch,'独立条带投影与碰撞处理','额外池；分非扫描轴，输出缓冲互不写冲突']]
    coords = [
        ['原始2D','整张相机照片','圆心、灰度图、H_full输入','照片不做SmartRay Pitch'],
        ['crop局部2D','检测ROI左上角为(0,0)','YOLO原始掩码','公开mask_2d加crop偏移；bbox清理还需加mask_offset'],
        ['缩小2D','原图按preview_scale缩放','仿射内部sx/sy/kxy/kyx/cx/cy','参考点(1000,400)属于此坐标系；最后换回原图矩阵'],
        ['原始整板深度','原始PNG像素栅格','输入深度','Pitch关闭时与输出板坐标相同'],
        ['补偿后整板深度','Pitch矫正后栅格','fixed_rois、分片起止行','Pitch开启时coordinate_space=pitch_corrected_board_3d'],
        ['3D片局部','depth[y1:y2, :]','mask_3d_local、参考面、内部测高点','X保留整板全宽；局部Y从0开始'],
        ['整板高度点','与当前深度坐标空间一致','公开Height3D.Point','x不变，y加depth_start_row；掩码轮廓不一起平移']]
    roi = [
        ['圆搜索ROI','common_2d.crop.circle_search_roi','限定Hough找圆范围','保留'],
        ['检测crop','common_2d.crop.detection_roi_offsets','限定YOLO看到的图像；圆相对偏移','保留'],
        ['2D弯曲/缺线ROI','wire_bending_2d.roi_offsets；兼容wire_rois','定义2D弯曲四边形/缺线检查','保留，不限制3D测高'],
        ['粗配准ROI','registration_3d.roi_3d / search_2d_scaled','模板与搜索范围','保留'],
        ['仿射评分区','dynamic_affine_3d.rois / selected_regions','多区域共同评分同一个矩阵','保留；未配rois时用代码3个默认区'],
        ['平台采样ROI','height_measurement_3d.platform_roi_offsets','选干净平台样本拟合高度基准','保留，决定参考面'],
        ['固定分片ROI','segmentation_3d.fixed_rois','从整板直接取各片边界','保留，与测高目标选择不同'],
        ['测高目标选择ROI','旧引线ROI筛选/裁剪函数','曾跳过框外wire/thin中段','已删除：全部检测到的目标参与测高']]
    endpoints = [
        ['统一流式','init_algorithm(stage=board)','加载全套模型和配置','当前验证台走此模式'],
        ['统一流式','detect_2d_piece','一片2D、明确board_id/piece_id','返回本片2D结果，同时发布joint_entry'],
        ['统一流式','detect_3d_board','一张3D整板、board_id','允许先到；Future等整板所有2D诊断及联合结果'],
        ['缓存管理','release_board','明确板号','取消/失败/仅2D验证结束后释放，避免混板'],
        ['旧整板批量','detect_board','N张2D + 整板深度','同类算法；批量YOLO后逐片处理，不是相同事件调度图'],
        ['旧独立2D','stage=2d','独立2D接口','不自动进入BoardPipeline；桥接层只允许单GPU工作资源配置'],
        ['旧独立3D','stage=3d','纯深度中心线测高','按substrate_aois/wire_x_range处理，不是YOLO全实例联合测高'],
        ['配准台','registration_lab','一张2D + 整板/单片深度','CPU配准测试；可复用保存矩阵，正式流式仍动态求解'],
        ['验证台仅2D','validation_lab(mode=2d)','可少量2D片；手动板/片号','不提交深度；本次内存配置关闭Pitch，不写生产文件']]
    classes = [
        ['wire','弯曲/存在性','完整清理后的实例测高','全局引脚平台；按wire半径内缩'],
        ['thin','完整原始thin用于2D弯曲','中间段当引线，两端各当焊脚','只分段一次；中段/端部各自内缩'],
        ['bond','作为语义遮挡等参与检查','每个bond全量测高','先判芯片归属，后选芯片面或局部/全局平台'],
        ['chip','旋转与EfficientAD外观','拟合芯片平面及焊脚归属','不作为wire高度点；公开plane_3d'],
        ['silver','溢胶与断连','完整语义掩码用于参考面排除','不是wire/bond目标，不伪造高度点']]
    timing_rows = [
        ['2D','cpu_prepare_queue_ms','提交到CPU线程开始的排队','不含计算'],
        ['2D','cpu_prepare_phase_ms','读图、圆/crop、平台掩码整个准备阶段','包含其子计时'],
        ['2D','gpu_phase_ms','借资源、YOLO、掩码提取、EfficientAD和流收尾','包含reported和预测器外工作'],
        ['2D','yolo_predict_wall_ms','完整model.predict调用','首次设置、预热等可不在reported三段内'],
        ['2D','mask_extract_to_cpu_ms','批量阈值化/转换/传输及实例组装','不是mask_postprocess_ms'],
        ['2D','efficientad_total_ms','逐chip外观检测总工作','forward_enqueue不是纯GPU内核时间'],
        ['2D','geometry_phase_ms','弯曲、缺线、旋转、灰度准备','完成后发布joint_entry'],
        ['2D','cpu_postprocess_phase_ms','银胶检查和公开实例输出','与本片联合工作可重叠'],
        ['2D','silver_mask_checks_ms','inspect_masks溢胶+断连','内部两个主要检测项目前串行'],
        ['2D','public_2d_geometry_ms','全部公开几何完成的墙钟','work_ms是子任务时间总和，不相加'],
        ['2D','request_pipeline_ms','本片请求受理到完整2D响应','不同片不能简单相加当整板耗时'],
        ['3D整板','depth_phase_ms','深度读取、Pitch、分片总墙钟','与2D输入路重叠'],
        ['3D整板','smartray_pitch_ms','整板Pitch整体墙钟','pitch_projection_work_ms是条带工作和'],
        ['3D整板','depth_segmentation_ms','真实分片函数耗时','segmentation_3d_ms为外层包裹计时'],
        ['联合','matching_input_wait_ms','两路就绪时刻之差的绝对值','不说明谁先到；不是额外计算或完整等待'],
        ['联合','joint_queue_ms','兼容字段，等于registration_queue_ms','不代表测高池排队'],
        ['联合','registration_queue_ms / registration_phase_ms','配准池排队 / 配准阶段墙钟','平台掩码、粗配准和仿射'],
        ['联合','joint_measurement_queue_ms / joint_measurement_phase_ms','测高池排队 / 执行墙钟','映射、测高和结果整理'],
        ['联合','joint_pipeline_ms','本片提交配准到测高完成','含两个队列和交接开销'],
        ['联合','affine_registration_ms','准备、全部评分和优化器整段','nfev/均耗/停止原因在registration对象中'],
        ['联合','mask_alignment_ms','全部实例映射与几何完成墙钟','instance_alignment_work_ms是子任务工作和'],
        ['联合','height_measurement_ms','参考面、引线、焊脚等整体','包含其各项子计时和未细分操作'],
        ['联合','height_wire_regions_ms','thin准备及所有引线完成墙钟','包含height_prepare_thin_ms；work不另加'],
        ['联合','height_solder_wall_ms','全部焊脚任务完成墙钟','work包含端点映射/膨胀；height_solder_ms是其内部累计工作'],
        ['联合','joint_phase_ms','配准与映射测高执行墙钟之和','不含中间排队，端到端看joint_pipeline_ms；各片不能相加当整板耗时'],
        ['整板','total_detect_ms','3D整板请求受理到所有结果齐全','不一定从第一张2D采集开始；不等于2D+深度+联合简单相加']]
    historical = {'request_pipeline_ms':2348.8998,'cpu_prepare_phase_ms':529.793,'gpu_phase_ms':1447.2737,
        'yolo_predict_wall_ms':1251.9292,'mask_extract_to_cpu_ms':78.8593,'efficientad_total_ms':112.7374,
        'geometry_phase_ms':48.2525,'silver_mask_checks_ms':277.4228,'public_2d_geometry_ms':34.7887,
        'depth_phase_ms':476.4813,'smartray_pitch_ms':252.3124,'depth_segmentation_ms':.0336,
        'affine_registration_ms':1863.0133,'mask_alignment_ms':54.6932,'height_measurement_ms':300.7756,
        'height_wire_regions_ms':33.6691,'height_solder_wall_ms':58.1687,'joint_phase_ms':2238.7497}
    return dict(meta=dict(revision=revision,node_count=len(seen),step_count=sum(len(n['steps']) for s in scenes for n in s['nodes']),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),piece_count=three['common_3d']['parameters']['input']['piece_count'],
        scope='统一板级正式流式检测；旧接口单列说明',timing_note='用户此前提供的历史日志，早于全实例测高ROI删除；当前版本未重测，不作为当前性能承诺。'),
        scenes=scenes,config=config,pools=pools,pool_budget=budget,pool_effective=effective,
        early_joint=t3.get('early_joint_workers',joint),coords=coords,roi=roi,endpoints=endpoints,classes=classes,timings=timing_rows,historical=historical)


def static_book(data):
    esc=lambda value:html.escape(str(value),quote=True)
    chunks=[]
    for scene in data['scenes']:
        cards=[]
        for node in scene['nodes']:
            refs=' · '.join(esc(f"{r['path']}:{r['line']} · {r['symbol']}") for r in node['sources'])
            cards.append(f'<details class="step-card" data-node="{esc(node["id"])}"><summary><span class="node-index">{esc(node["id"])}</span>{esc(node["title"])}</summary>'
                +'<div class="step-body"><p class="card-sub">'+esc(node['subtitle'])+'</p><ol>'
                +''.join('<li>'+esc(step)+'</li>' for step in node['steps'])+'</ol><p class="source-ref">'+refs+'</p></div></details>')
        chunks.append(f'<section class="book-scene" data-scene="{esc(scene["id"])}"><h2>{esc(scene["title"])}</h2>'+''.join(cards)+'</section>')
    return ''.join(chunks)


def build(output):
    data=make_document()
    template=(HERE/'detection_flow_template.html').read_text(encoding='utf-8')
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e')
    result=template.replace('__FLOW_DATA__',encoded).replace('__STATIC_BOOK__',static_book(data))
    if '__FLOW_DATA__' in result or '__STATIC_BOOK__' in result:
        raise ValueError('Unexpanded HTML placeholder')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(result,encoding='utf-8')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'detection-flow.html')
    args=parser.parse_args()
    doc=build(args.output)
    print(f"{args.output.resolve()} | {len(doc['scenes'])} scenes | {doc['meta']['node_count']} nodes | {doc['meta']['step_count']} steps")
