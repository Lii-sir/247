'use strict';
const $=id=>document.getElementById(id);
let files=[],config=null,busy=false,uploading=false,revision=0;
async function api(url,options){const r=await fetch(url,options),d=await r.json();if(!r.ok)throw Error(d.error||'HTTP '+r.status);return d;}
function status(text,error=false){$('progress').textContent=text;$('progress').className=error?'error':'';}
function assess(){
 const d=Pairing.check(files,$('board').value,$('mode').value,config?.piece_count||0);
 const paced=$('submissionMode').value==='paced',raw=$('requestInterval').value.trim(),interval=Number(raw),max=config?.max_request_interval_ms??3600000;
 const valid=!paced||(raw!==''&&Number.isFinite(interval)&&interval>=0&&interval<=max);
 $('checks').textContent=valid?d.message:'请求间隔须为0～'+max+'毫秒的有限数值';$('checks').className=d.ok&&valid?'good':'bad';$('run').disabled=!config||!d.ok||!valid||busy||uploading;
 $('requestInterval').disabled=busy||uploading||!paced;
 $('submissionHint').textContent=paced?'按片号升序发送2D'+($('mode').value==='joint'?'，最后发送3D':'')+'；不等待上一张检测完成。间隔按相邻请求开始发送计，0ms仍保持顺序；实际发送时刻保存到结果JSON。':'并发发送所选板的请求，间隔设置不生效。';
 $('demo').disabled=busy||uploading;$('clear').disabled=busy||uploading;
 for(const id of ['upload2','upload3','board','mode','submissionMode'])$(id).disabled=busy||uploading;
 return {...d,ok:d.ok&&valid};
}
function boardList(){const old=$('board').value;$('board').replaceChildren();for(const id of Pairing.boards(files)){const o=document.createElement('option');o.value=id;o.textContent='板 '+id;$('board').append(o);}if(Pairing.boards(files).includes(old))$('board').value=old;else $('board').value=Pairing.boards(files)[0]||'';$('counts').textContent='共 '+files.length+'个文件 · '+Pairing.boards(files).length+'个已编号板号'+(files.some(r=>!Pairing.board(r.board_id))?' · 有文件尚未指定板号':'');assess();}
function renderFiles(){
 $('files').replaceChildren();$('empty').hidden=files.length>0;
 files.forEach((row,index)=>{
  const tr=document.createElement('tr'),name=document.createElement('td');name.className='filename';name.textContent=row.name;name.title=row.name;
  const kind=document.createElement('span');kind.className='kind';kind.textContent=row.kind==='2d'?'2D片图':'3D整板';name.append(kind);tr.append(name);
  for(const field of ['board_id','piece_id']){const td=document.createElement('td'),input=document.createElement('input');input.type=field==='piece_id'?'number':'text';input.value=row[field]??'';input.setAttribute('aria-label',row.name+' '+(field==='board_id'?'板号':'片号'));input.disabled=busy||row.kind==='3d'&&field==='piece_id';if(field==='piece_id'){input.min=1;input.max=config?.piece_count||10;}input.oninput=()=>{row[field]=field==='piece_id'?(input.value===''?null:Number(input.value)):input.value;revision++;boardList();};td.append(input);tr.append(td);}
  const td=document.createElement('td'),remove=document.createElement('button');remove.textContent='×';remove.title='移除 '+row.name;remove.disabled=busy;remove.onclick=()=>{files.splice(index,1);revision++;renderFiles();};td.append(remove);tr.append(td);$('files').append(tr);
 });
 $('counts').textContent='共 '+files.length+'个文件 · '+Pairing.boards(files).length+'个已编号板号'+(files.some(r=>!Pairing.board(r.board_id))?' · 有文件尚未指定板号':'');boardList();
}
for(const [id,kind] of [['upload2','2d'],['upload3','3d']])$(id).onchange=async e=>{
 uploading=true;assess();try{
  const selected=Array.from(e.target.files||[]);
  for(let i=0;i<selected.length;i++){const f=selected[i];status('正在读入 '+(i+1)+'/'+selected.length+'：'+f.name);const row=await api('/api/upload?kind='+kind+'&name='+encodeURIComponent(f.name),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:f});files.push(row);revision++;}
  renderFiles();status('文件已读入，核对编号后开始验证');
 }catch(err){renderFiles();status(err.message,true);}finally{uploading=false;e.target.value='';assess();}
};
$('demo').onclick=()=>{files=structuredClone(config.samples);revision++;renderFiles();status('项目旧命名样本明确归为 demo 板，片号为1～'+config.piece_count+'，可手动修改。');};
$('clear').onclick=()=>{files=[];revision++;renderFiles();status('清单已清空，磁盘文件与已生成报告保留。');};
$('board').onchange=$('mode').onchange=assess;
$('requestInterval').value=$('requestInterval').value||'100';
$('submissionMode').onchange=$('requestInterval').oninput=()=>{revision++;assess();};
function links(id){for(const [key,file] of [['reportLink','report.html'],['jsonLink','result.json'],['csvLink','timings.csv'],['logLink','python.log']])$(key).href='/runs/'+id+'/'+file;$('links').hidden=false;}
function showResult(id,job){
 links(id);$('resultEmpty').hidden=true;$('report').hidden=false;$('report').src='/runs/'+id+'/report.html';
 const r=job.result;
 $('summary').textContent=r?'板 '+r.board_id+' · '+(r.ok?(r.product_is_ok?'检测完成 · OK':'检测完成 · NG（算法判定）'):'检测请求失败')+' · '+r.pieces+'片 · '+(r.total_wall_ms/1000).toFixed(2)+'秒 · '+(r.model_reused?'复用已加载模型':'本次初始化模型')+(r.submission_mode==='paced'?' · 按间隔提交 '+r.request_interval_ms+'ms':r.submission_mode==='concurrent'?' · 并发提交':'')+(r.error?'\n'+r.error:'')+'\n输出：'+r.output_dir:'验证失败：'+(job.error||'未知错误');
 const latency=r?.two_d_latency;
 if(latency&&Number.isFinite(latency.max_ms))$('summary').textContent+='\n2D请求≤'+latency.target_ms+'ms：'+(latency.meets_target?'达标':'未达标')+' · 最大 '+latency.max_ms.toFixed(1)+'ms'+(latency.over_target_piece_ids?.length?' · 超标片号 '+latency.over_target_piece_ids.join(', '):'');
}
async function history(){const list=await api('/api/history');const old=$('history').value;$('history').replaceChildren();const blank=document.createElement('option');blank.value='';blank.textContent='历史验证记录';$('history').append(blank);for(const j of list){const o=document.createElement('option');o.value=j.id;o.textContent='板 '+j.board_id+' · '+(j.status==='done'?'完成':j.status==='failed'?'失败':'运行中');$('history').append(o);}if(list.some(j=>j.id===old))$('history').value=old;}
$('history').onchange=async()=>{if(!$('history').value)return;try{const id=$('history').value,j=await api('/api/jobs/'+id);if(j.status==='running'){status('该记录仍在运行');return;}showResult(id,j);status('正在查看历史结果；当前文件清单不会改变该报告。');}catch(e){status(e.message,true);}};
$('run').onclick=async()=>{
 if(!assess().ok||busy||uploading||!config)return;
 const snapshot=revision,request={files:structuredClone(files),board_id:$('board').value,mode:$('mode').value,submission_mode:$('submissionMode').value,request_interval_ms:$('submissionMode').value==='paced'?Number($('requestInterval').value):0};
 busy=true;renderFiles();assess();status('正在提交验证…');
 try{
  const {id}=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
  while(true){await new Promise(r=>setTimeout(r,800));const job=await api('/api/jobs/'+id);if(job.status==='running'){status(job.phase);continue;}links(id);if(job.result)showResult(id,job);else $('summary').textContent=job.error||'验证失败';await history();$('history').value=id;const latencyMiss=job.result?.two_d_latency?.meets_target===false;status(job.status==='done'?(latencyMiss?'验证完成；2D请求未满足1秒目标，请查看超标片号。':'验证完成，可在右侧交互查看。'):job.error||'验证失败，请检查报告与服务日志。',job.status!=='done'||latencyMiss);if(snapshot!==revision)status('验证已完成；清单有变化，报告仍对应提交时的文件。');break;}
 }catch(e){status(e.message,true);}finally{busy=false;renderFiles();assess();}
};
api('/api/defaults').then(d=>{config=d;$('pieceCount').textContent=d.piece_count;const p=d.pitch;$('config').textContent='每板 '+d.piece_count+'片 · 粗配准 + 仿射 · Pitch '+(p.enabled_3d?'启用，'+p.angle_deg+'°，步距'+p.scan_pitch_mm+' mm/profile':'关闭');renderFiles();status('就绪，请添加文件或载入项目样本。');return history();}).catch(e=>status('配置加载失败：'+e.message,true));
