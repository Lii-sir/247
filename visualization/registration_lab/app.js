'use strict';
const $ = id => document.getElementById(id);
const sections = {registration_3d:'粗配准参数', dynamic_affine_3d:'仿射参数', segmentation_3d:'整板分片参数'};
let savedMatrix=null, matrixRevision=0;
let initial, current, pinned, showingA=false, stageKey='affine', loaded={}, zoom=1, pan={x:0,y:0}, drag=null, uploading=0, running=false, ready=false;
const canvas=$('canvas'), ctx=canvas.getContext('2d');
async function api(url,options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);return d;}
function status(text,error=false){$('status').textContent=text;$('status').className=error?'error':'';}
function params(){const p={};for(const k of Object.keys(sections)){try{p[k]=JSON.parse($('param-'+k).value);}catch{throw Error(sections[k]+'不是有效JSON');}if(!p[k]||typeof p[k]!=='object'||Array.isArray(p[k]))throw Error(sections[k]+'必须是对象');}if($('alignmentMode').value!=='reuse'&&Array.isArray(p.dynamic_affine_3d.selected_regions)&&!p.dynamic_affine_3d.selected_regions.length)throw Error('请至少选择一个仿射评分区域');return p;}
function setParams(p){for(const k of Object.keys(sections)){if(!p[k]||typeof p[k]!=='object'||Array.isArray(p[k]))throw Error('缺少参数对象：'+k);}for(const k of Object.keys(sections))$('param-'+k).value=JSON.stringify(p[k],null,2);syncRegionChoices();}
let regionCheckboxes=[];
function syncRegionChoices(){
 const area=$('regionChoices');area.replaceChildren();regionCheckboxes=[];
 let config;try{config=JSON.parse($('param-dynamic_affine_3d').value);}catch{area.textContent='请先修正仿射参数JSON';return;}
 const count=Array.isArray(config.rois)?config.rois.length:3;
 for(let id=1;id<=count;id++){
  const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id='scoreRegion'+id;
  input.checked=config.selected_regions==null||Array.isArray(config.selected_regions)&&config.selected_regions.includes(id);
  input.onchange=()=>{
   let latest;try{latest=JSON.parse($('param-dynamic_affine_3d').value);}catch{status('请先修正仿射参数JSON',true);return;}
   const selected=regionCheckboxes.filter(item=>item.input.checked).map(item=>item.id);
   if(selected.length===regionCheckboxes.length)delete latest.selected_regions;else latest.selected_regions=selected;
   $('param-dynamic_affine_3d').value=JSON.stringify(latest,null,2);markDirty();
   if(!selected.length)status('请至少选择一个仿射评分区域',true);
   else status('已选择 '+selected.map(i=>'R'+i).join('、')+'，请重新运行配准');
  };
  const text=document.createElement('span');text.textContent='R'+id;label.append(input,text);area.append(label);regionCheckboxes.push({id,input});input.disabled=$('alignmentMode').value==='reuse';
 }
}
let submittedSignature=null, calibrationSignature=null;
const formIds=['alignmentMode','path2','path3','piece','depthMode','compensationMode','imageRole','scanAxis','direction','trigger','angle','pitch','speed','frequency','zstep','grayOffset','zorigin','affine',...Object.keys(sections).map(k=>'param-'+k)];
function formSignature(){return JSON.stringify([$('alignmentMode').value==='reuse'?matrixRevision:0,formIds.map(id=>[id,$(id).value,$(id).checked])]);}
function markDirty(){$('pendingChanges').hidden=!current||formSignature()===submittedSignature;if(calibrationSignature!==null)$('sliceCalibration').hidden=formSignature()!==calibrationSignature;}
function updateMatrixUI(){
 const reuse=$('alignmentMode').value==='reuse';$('matrixReuseFields').hidden=!reuse;
 const source=savedMatrix?.source;
 $('matrixSource').textContent=savedMatrix?'已载入：第'+(source?.piece_id??'?')+'片 · '+(source?.stage_name||'保存矩阵')+' · 2D '+savedMatrix.context.image_size?.join('×')+' → 深度片 '+savedMatrix.context.depth_size?.join('×'):'尚未载入；先配准一片并保存，或载入矩阵文件。';
 $('affine').disabled=reuse;
 regionCheckboxes.forEach(({input})=>input.disabled=reuse);
}
$('alignmentMode').onchange=()=>{updateMatrixUI();markDirty();};
$('loadMatrix').onclick=()=>$('matrixFile').click();
$('matrixFile').onchange=async e=>{
 try{
  const file=e.target.files?.[0];if(!file)return;
  savedMatrix=LabPreview.validateMatrixProfile(JSON.parse(await file.text()));matrixRevision++;
  $('alignmentMode').value='reuse';updateMatrixUI();markDirty();status('矩阵已载入，选择下一片后运行即可');
 }catch(err){status('矩阵载入失败：'+err.message,true);}finally{e.target.value='';}
};
$('saveMatrix').onclick=()=>{
 try{
  savedMatrix=LabPreview.buildMatrixProfile(activeResult(),stageKey);matrixRevision++;
  download(savedMatrix,'registration-matrix-'+stageKey+'.json');updateMatrixUI();markDirty();
  status('当前阶段矩阵已保存；换片后选择“复用已保存矩阵”即可');
 }catch(err){status(err.message,true);}
};
function updateTiltUI(){
 const mode=$('compensationMode').value;
 $('compensationFields').hidden=mode!=='raw';
 const internal=$('trigger').value==='internal';
 $('pitchLabel').hidden=internal;$('internalFields').hidden=!internal;
 $('modeHint').textContent=mode==='raw'?'仅用于确认未做过SDK补偿的原始深度；填写真实角度和步距。':mode==='sdk'?'将跳过本工具补偿，直接配准。':'普通配准直接运行即可。';
 $('intensityHint').textContent=$('imageRole').value==='smartray_intensity'?'启用补偿时，强度图自动随深度同步映射；两图必须同栅格。':'独立照片保持原样。';
 $('tiltHint').textContent='按'+($('scanAxis').value==='rows'?'图像行':'图像列')+'扫描，使用'+($('direction').value==='recover_ideal'?'还原':'文档正向')+'公式；可在高级设置修改。';
}
function tiltSettings(){
 const mode=$('compensationMode').value,enabled=mode!=='off';
 const t={model:'smartray_pitch',enabled_2d:enabled&&$('imageRole').value==='smartray_intensity',enabled_3d:enabled,
 input_state:mode==='raw'?'uncorrected':mode==='sdk'?'sdk_corrected':'unknown',
 image_role:$('imageRole').value,scan_axis:$('scanAxis').value,
 transform_direction:$('direction').value,trigger_mode:$('trigger').value,
 angle_deg:Number($('angle').value),scan_pitch_mm:Number($('pitch').value),
 transport_rate_mm_s:Number($('speed').value),scan_rate_hz:Number($('frequency').value),
 z_step_mm:Number($('zstep').value),gray_offset:Number($('grayOffset').value),z_origin_mm:Number($('zorigin').value)};
 if(mode==='raw'){
  if(!Number.isFinite(t.angle_deg)||Math.abs(t.angle_deg)>=60)throw Error('Pitch角必须在-60到60度之间');
  if(t.trigger_mode==='external'&&!(t.scan_pitch_mm>0))throw Error('请填写标定扫描步距(mm/profile)');
  if(t.trigger_mode==='internal'&&!(t.transport_rate_mm_s>0&&t.scan_rate_hz>0))throw Error('请填写正数扫描速度与触发频率');
 }
 return t;
}
function setTilt(t={}){
 const enabled=t.enabled_2d||t.enabled_3d;
 const invalid=enabled&&(t.model!=='smartray_pitch'||!t.enabled_3d||!['uncorrected','sdk_corrected'].includes(t.input_state));
 if(invalid)t={};
 $('compensationMode').value=t.enabled_3d?(t.input_state==='sdk_corrected'?'sdk':'raw'):'off';
 for(const [id,key,value] of [['imageRole','image_role','independent_camera'],['scanAxis','scan_axis','rows'],['direction','transform_direction','recover_ideal'],['trigger','trigger_mode','external'],['angle','angle_deg',0],['pitch','scan_pitch_mm',''],['speed','transport_rate_mm_s',''],['frequency','scan_rate_hz',''],['zstep','z_step_mm',.00105],['grayOffset','gray_offset',32768],['zorigin','z_origin_mm',0]])$(id).value=t[key]??value;
 updateTiltUI();markDirty();
 if(invalid)throw Error('旧版或状态未确认的补偿参数已关闭，请重新选择补偿方式并核对标定值');
}
for(const id of ['compensationMode','imageRole','trigger','scanAxis','direction'])$(id).onchange=()=>{updateTiltUI();markDirty();};
updateTiltUI();
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const [key,label] of Object.entries(sections)){const d=document.createElement('details');d.id='section-'+key;const s=document.createElement('summary');s.textContent=label;const t=document.createElement('textarea');t.id='param-'+key;t.setAttribute('aria-label',label);t.spellcheck=false;d.append(s,t);$('parameters').append(d);}
$('param-dynamic_affine_3d').oninput=()=>{syncRegionChoices();markDirty();};
function busy(){ $('run').disabled=!ready||running||uploading>0;$('scanFixedSlices').disabled=!ready||running||uploading>0; }
const fileEntries={2:new Map(),3:new Map()},manualOptions={};
function showFile(n,path,name,manual=false){
 const select=$('samples'+n);$('path'+n).value=path;
 if(!path){select.value='';select.title='';$('name'+n).textContent='请选择图像';return;}
 let entry=fileEntries[n].get(path);
 if(!entry){
  let option=manual?manualOptions[n]:null;
  if(!option){option=document.createElement('option');select.append(option);if(manual)manualOptions[n]=option;}
  option.value=path;option.textContent=name;option.title=name;
  entry={name,project:false,option};if(!manual)fileEntries[n].set(path,entry);
 }
 select.value=path;select.title=entry.name;$('name'+n).textContent=entry.name;$('name'+n).title=entry.name;
}
function updatePairHint(automatic=false){
 $('pairHint').textContent=$('depthMode').value==='piece'?'单片深度图不使用片号。':
  (automatic?'已按项目2D样本匹配第':'当前第')+$('piece').value+'片，'+(automatic?'可手动修改。':'请确认与当前2D图对应。');
}
for(const n of [2,3]){
 $('samples'+n).onchange=()=>{
  const path=$('samples'+n).value,entry=fileEntries[n].get(path);showFile(n,path,entry?.name||path.split(/[\\/]/).at(-1),!entry);
  let automatic=false;
  if(n===2&&entry?.project&&$('depthMode').value==='board'){
   const match=path.split(/[\\/]/).at(-1).match(/^(\d+)\.[^.]+$/);
   if(match&&Number(match[1])>0){$('piece').value=Number(match[1]);automatic=true;}
  }
  if(n===2)updatePairHint(automatic);
  markDirty();
 };
 $('file'+n).onchange=async e=>{const file=e.target.files[0];if(!file)return;uploading++;busy();status('正在读取 '+file.name);try{const d=await api('/api/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});showFile(n,d.path,file.name);if(n===2)updatePairHint();status('文件已就绪：'+file.name);}catch(err){status(err.message,true);}finally{uploading--;busy();markDirty();e.target.value='';}};
 $('path'+n).oninput=()=>{const path=$('path'+n).value;showFile(n,path,path.split(/[\\/]/).at(-1)||'请选择图像',true);if(n===2)updatePairHint();markDirty();};
}
$('piece').oninput=()=>{updatePairHint();markDirty();};
$('depthMode').onchange=()=>{$('piece').disabled=$('depthMode').value==='piece';updatePairHint();markDirty();};
$('controls').addEventListener('input',markDirty);$('controls').addEventListener('change',markDirty);
$('reset').onclick=()=>{setParams(initial.parameters);setTilt();$('affine').checked=true;$('alignmentMode').value='estimate';updateMatrixUI();markDirty();status('已恢复项目参数和完整配准阶段，Pitch补偿已关闭');};
$('exportParams').onclick=()=>{try{download({parameters:params(),tilt:tiltSettings()},'registration-parameters.json');}catch(e){status(e.message,true);}};
$('import').onclick=()=>$('paramsFile').click();
$('paramsFile').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());setParams(d.parameters||d);setTilt(d.tilt);status('参数已导入');}catch(err){status('导入失败：'+err.message,true);}finally{e.target.value='';}};
const image = (src,label) => LabPreview.loadImage(src,label);
function activeResult(){return showingA?pinned:current;}
async function renderResult(){
 const d=activeResult();if(!d)return;
 $('resultSummary').textContent=(showingA?'对照 A':'当前结果')+' · '+LabPreview.describeResult(d);markDirty();
 $('stages').replaceChildren();if(!d.stages.some(s=>s.key===stageKey))stageKey=d.stages.at(-1).key;
 for(const s of d.stages){const b=document.createElement('button');b.textContent=s.name;b.className=s.key===stageKey?'active':'';b.onclick=()=>{stageKey=s.key;renderResult().catch(e=>status(e.message,true));};$('stages').append(b);}
 const s=d.stages.find(s=>s.key===stageKey);
 const [depth,warped,edges,depthColor,warpedColor]=await Promise.all([image(d.depth_preview,'深度灰度图'),image(s.warped,'2D灰度图'),image(s.edges,'边缘图'),image(d.depth_color_preview,'3D彩色图层'),image(s.warped_color,'2D原图彩色图层')]);loaded={depth,warped,edges,depthColor,warpedColor};
 $('depthScale').textContent=d.depth_color_limits_raw?'3D 色阶：蓝 → 红（低 → 高），当前片 P1～P99 深度值 '+d.depth_color_limits_raw.map(v=>v.toFixed(0)).join('～')+'；无效区透明':'3D 无有效深度像素';
 updateRoiInfo();
 const r=s.registration, final=r;
 $('empty').style.display='none';$('canvasHint').hidden=false;$('badge').textContent=`${showingA?'对照 A':'当前 B'} · ${s.name} · ${d.depth_size.join(' × ')}`;
 $('metrics').replaceChildren();
 for(const [label,value,unit] of [['配准计算',d.algorithm_ms.toFixed(1),'ms'],['粗配准匹配分',final.score==null?'—':Number(final.score).toFixed(4),''],['仿射损失',final.affine_score_after==null?'—':final.affine_score_after.toFixed(5),''],['仿射迭代',final.affine_iterations==null?'—':String(final.affine_iterations),'轮']]){const div=document.createElement('div'),small=document.createElement('small'),strong=document.createElement('strong'),em=document.createElement('em');small.textContent=label;strong.textContent=value+' ';em.textContent=unit;strong.append(em);div.append(small,strong);$('metrics').append(div);}
 $('timings').replaceChildren();const max=Math.max(...Object.values(d.timings));
 for(const [name,ms] of Object.entries(d.timings)){const row=document.createElement('div');row.className='time-row';const label=document.createElement('span');label.textContent=name;const bar=document.createElement('span');bar.className='bar';const fill=document.createElement('i');fill.style.width=(ms/max*100)+'%';bar.append(fill);const val=document.createElement('b');val.textContent=ms.toFixed(1)+' ms';row.append(label,bar,val);$('timings').append(row);}
 const corrected=d.correction&&(d.correction.enabled_2d||d.correction.enabled_3d);
 $('metadata').textContent=`2D ${d.image_size.join('×')} · 深度 ${d.depth_dtype} · ${d.correction?.enabled_3d?'矫正后':'原始'}整板行 [${d.depth_rows.join(', ')})。矩阵将${d.correction?.enabled_2d?'同栅格强度图':'2D原图'}映射至${d.correction?.enabled_3d?'矫正后':''}深度片局部坐标。`;
 $('matrix').textContent=r.homography_2d_to_3d.map(row=>row.map(v=>v.toFixed(7).padStart(13)).join(' ')).join('\n');
 const flag=stageKey==='affine'?r.affine_valid:r.is_valid;
 $('warning').textContent=`当前阶段算法有效标记：${flag?'true':'false'}。${r.affine_error||''}此标记不是独立精度验收。${corrected?' 已应用SmartRay Pitch '+d.correction.angle_deg+'°。3D丢弃点 '+(d.correction.discarded_points??0)+'，碰撞点 '+(d.correction.collision_points??0)+'；空洞未填补。':''}`;
 if(d.correction?.reason==='already_sdk_corrected')$('warning').textContent+=' 输入已由Studio/SDK矫正，本工具已跳过重复补偿。';
 if(d.matrix_reuse)$('warning').textContent='已复用保存矩阵，跳过配准求解及评分；当前片精度未自动验收。'+(d.matrix_reuse.warnings||[]).join('；');
 $('useFixedSlices').disabled=!(d.pieces?.length);
 $('saveMatrix').disabled=false;
 $('source').src=d.source_preview;$('pin').disabled=showingA;$('showA').textContent=showingA?'返回当前 B':'查看 A';$('exportResult').disabled=false;draw();
}
function fit(){zoom=1;pan={x:0,y:0};draw();}
function updateRoiInfo(){
 const scoring=activeResult()?.affine_scoring;
 $('roiInfo').hidden=!$('showAffineRois').checked||!activeResult();
 const ids=scoring?.region_ids||scoring?.rois?.map((_,i)=>i+1)||[];
 const selected=scoring?.selected_region_ids||ids;
 $('roiInfo').textContent=scoring?.rois?.length?
  (scoring.enabled?'本次仿射参与评分：'+selected.map(i=>'R'+i).join('、'):'参考评分区（本次未执行仿射）')+
  ' · 实线为参与区域，虚线为未参与区域 · 深度片局部像素 · '+scoring.rois.map((box,i)=>'R'+ids[i]+' ['+box.join(', ')+']').join(' / '):
  '本结果未包含有效评分区；若是更新前的结果，请重新运行。';
}

function draw(){
 const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#172b34';ctx.fillRect(0,0,w,h);if(!loaded.depth)return;
 const iw=loaded.depth.width,ih=loaded.depth.height,scale=Math.min(w/iw,h/ih)*zoom,x=(w-iw*scale)/2+pan.x,y=(h-ih*scale)/2+pan.y;
 ctx.save();ctx.translate(x,y);ctx.scale(scale,scale);const fraction=Number($('opacity').value)/100,mode=$('view').value;
 if(mode==='edges')ctx.drawImage(loaded.edges,0,0);
 else if(mode==='color'||mode==='wipe'){
  ctx.drawImage(loaded.warpedColor,0,0);
  if(mode==='wipe'){ctx.save();ctx.beginPath();ctx.rect(iw*fraction,0,iw*(1-fraction),ih);ctx.clip();ctx.drawImage(loaded.depthColor,0,0);ctx.restore();ctx.strokeStyle='#46d3bc';ctx.lineWidth=1/scale;ctx.beginPath();ctx.moveTo(iw*fraction,0);ctx.lineTo(iw*fraction,ih);ctx.stroke();}
  else{ctx.globalAlpha=fraction;ctx.drawImage(loaded.depthColor,0,0);}
 }else{ctx.drawImage(loaded.depth,0,0);if(mode==='difference'){ctx.globalCompositeOperation='difference';ctx.drawImage(loaded.warped,0,0);}else{ctx.globalAlpha=fraction;ctx.drawImage(loaded.warped,0,0);}}
 if($('showAffineRois').checked)LabPreview.drawAffineRegions(ctx,activeResult()?.affine_scoring?.rois||[],scale,{regionIds:activeResult()?.affine_scoring?.region_ids,selectedIds:activeResult()?.affine_scoring?.enabled===false?[]:activeResult()?.affine_scoring?.selected_region_ids});
 ctx.restore();
}
canvas.addEventListener('wheel',e=>{e.preventDefault();const old=zoom;zoom=Math.min(12,Math.max(.3,zoom*Math.exp(-e.deltaY*.001)));const r=canvas.getBoundingClientRect(),x=e.clientX-r.left-canvas.clientWidth/2,y=e.clientY-r.top-canvas.clientHeight/2;pan={x:x-(x-pan.x)*zoom/old,y:y-(y-pan.y)*zoom/old};draw();},{passive:false});
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(drag){pan={x:drag.px+e.clientX-drag.x,y:drag.py+e.clientY-drag.y};draw();}};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;canvas.ondblclick=fit;
$('fit').onclick=fit;$('view').onchange=draw;$('opacity').oninput=draw;new ResizeObserver(draw).observe($('viewport'));
$('showAffineRois').onchange=()=>{updateRoiInfo();draw();};
$('scanFixedSlices').onclick=async()=>{
 if(running||uploading>0)return;
 try{
  if($('depthMode').value!=='board')throw Error('请使用整板深度图进行固定分片标定');
  const image_3d=$('path3').value.trim();if(!image_3d)throw Error('请先选择3D整板深度图');
  let segmentation;try{segmentation=JSON.parse($('param-segmentation_3d').value);}catch{throw Error('整板分片参数不是有效JSON');}
  if(!segmentation||typeof segmentation!=='object'||Array.isArray(segmentation))throw Error('整板分片参数必须是对象');
  const request={image_3d,depth_mode:'board',tilt:tiltSettings(),parameters:{segmentation_3d:segmentation}},signature=formSignature();
  running=true;busy();$('sliceCalibration').hidden=true;status('正在自动扫描3D整板…');
  const job=await api('/api/calibrate-segmentation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
  while(true){
   await new Promise(r=>setTimeout(r,500));const state=await api('/api/jobs/'+job.id);
   if(state.status==='failed')throw Error(state.error);
   if(state.status==='done'){
    if(formSignature()!==signature)throw Error('扫描期间输入或参数发生变化，未覆盖当前配置，请重新扫描');
    const d=state.result,p=d.segmentation_3d;
    $('param-segmentation_3d').value=JSON.stringify(p,null,2);$('advancedSettings').open=true;$('section-segmentation_3d').open=true;markDirty();
    calibrationSignature=formSignature();$('sliceCalibration').textContent='整板 '+d.board_size.join(' × ')+' · 从上到下 '+p.fixed_rois.length+' 片\n'+p.fixed_rois.map((box,i)=>'第'+(i+1)+'片：['+box.join(', ')+']').join('\n');$('sliceCalibration').hidden=false;
    status('已自动扫描并生成'+p.fixed_rois.length+'片固定配置，仅更新本页；可导出参数用于正式流程。');break;
   }
   status('标定中 · '+state.phase);
  }
 }catch(err){status(err.message,true);}finally{running=false;busy();}
};
$('useFixedSlices').onclick=()=>{
 try{
  if(running||uploading>0)throw Error('请等待当前任务完成');
  if(showingA||!current?.pieces?.length||formSignature()!==submittedSignature)throw Error('请先按当前输入和参数完成整板配准，再固定本板分片');
  const p=JSON.parse($('param-segmentation_3d').value);
  p.mode='fixed';p.expected_piece_count=current.pieces.length;
  p.fixed_rois=current.pieces.map(box=>[box.x1,box.y1,box.x2,box.y2]);
  $('param-segmentation_3d').value=JSON.stringify(p,null,2);$('advancedSettings').open=true;$('section-segmentation_3d').open=true;markDirty();
  status('已固定本板'+p.expected_piece_count+'片坐标，仅更新本页。正式配置须使用相同Pitch设置；可导出参数。');
 }catch(err){status(err.message,true);}
};
$('pin').onclick=()=>{pinned=current;$('showA').disabled=false;status('已保留当前结果为 A；修改参数后再次运行可比较');};
$('showA').onclick=()=>{showingA=!showingA;renderResult().catch(e=>status(e.message,true));};
$('exportResult').onclick=()=>{const d=activeResult();if(!d)return;const {depth_preview,depth_color_preview,source_preview,...record}=d;record.stages=d.stages.map(({warped,warped_color,edges,...s})=>s);download(record,'registration-test.json');};
$('run').onclick=async()=>{
 try{
  const request={parameters:params(),tilt:tiltSettings(),image_2d:$('path2').value.trim(),image_3d:$('path3').value.trim(),depth_mode:$('depthMode').value,piece_id:Number($('piece').value),affine:$('affine').checked,registration_mode:$('alignmentMode').value};
  if(request.registration_mode==='reuse'){if(!savedMatrix)throw Error('请先保存或载入一个变换矩阵');request.matrix_profile=savedMatrix;}
  if(!request.image_2d||!request.image_3d)throw Error('请先选择2D图像和深度图');
  const signature=formSignature(),imageLabel=$('name2').textContent;
  running=true;busy();status('正在准备测试…');const job=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
  while(true){await new Promise(r=>setTimeout(r,500));const state=await api('/api/jobs/'+job.id);if(state.status==='failed')throw Error(state.error);if(state.status==='done'){current=state.result;current.display_image_2d=imageLabel;submittedSignature=signature;showingA=false;await renderResult();fit();status(`测试完成 · 配准 ${current.algorithm_ms.toFixed(1)} ms · 含读取/预览 ${current.total_ms.toFixed(1)} ms`);break;}status('运行中 · '+state.phase);}
 }catch(e){status(e.message,true);}finally{running=false;busy();}
};
updateMatrixUI();busy();
api('/api/defaults').then(d=>{LabPreview.validateServer(d);initial=d;ready=true;busy();setParams(d.parameters);for(const n of [2,3]){const list=d['images_'+(n===2?'2d':'3d')],select=$('samples'+n);const blank=document.createElement('option');blank.value='';blank.textContent='选择样本 / 自定义文件';select.append(blank);for(const path of list){const o=document.createElement('option');o.value=path;o.textContent=path.split(/[\\/]/).at(-1);select.append(o);fileEntries[n].set(path,{name:o.textContent,project:true,option:o});}if(list.length){select.value=list[0];select.onchange();}}status('就绪 · 选好图像后直接运行');}).catch(e=>status(e.message,true));
