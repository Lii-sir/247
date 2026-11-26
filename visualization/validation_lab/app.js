'use strict';
const $=id=>document.getElementById(id);
let files=[],config=null,busy=false,uploading=false,revision=0;
const workflow=[];
const pitchMode='pitch_piece_3d';
const detectionTypes={manual_2d:'单片 2D',manual_piece_3d:'单片 3D',[pitchMode]:'Pitch 就绪送检 3D'};
const actions={...detectionTypes,delay:'延时触发'};
const palette=[['addDetection','manual_2d'],['addDelay','delay']];
const savedWorkflowKey='validation-lab.workflow-templates.v1';
const workflowFileFormat='validation-lab-workflow';
const workflowVersion=2;
const requestClock=new Intl.DateTimeFormat('zh-CN',{hour:'2-digit',minute:'2-digit',second:'2-digit',
 fractionalSecondDigits:3,hourCycle:'h23'});
let nodeID=0, activeRun=null, cancelCountdown=null, refreshCards=[];
const pendingPitchRuns=new Set();
async function api(url,options){const r=await fetch(url,options),d=await r.json();if(!r.ok)throw Error(d.error||'HTTP '+r.status);return d;}
const post=(url,body)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function status(text,error=false){$('progress').textContent=text;$('progress').className=error?'error':'';}
function assess(){
 const d=Pairing.check(files,$('board').value,config?.piece_count||0);
 const paced=$('submissionMode').value==='paced',raw=$('requestInterval').value.trim(),interval=Number(raw),max=config?.max_request_interval_ms??3600000;
 const valid=!paced||(raw!==''&&Number.isFinite(interval)&&interval>=0&&interval<=max);
 $('checks').textContent=valid?d.message:'请求间隔须为0～'+max+'毫秒的有限数值';$('checks').className=d.ok&&valid?'good':'bad';$('run').disabled=!config||!d.ok||!valid||busy||uploading;
 $('requestInterval').disabled=busy||uploading||!paced;
 $('submissionHint').textContent=paced?'按片号升序发送 2D 请求。间隔按相邻请求开始发送计，实际发送时刻保存到结果 JSON。':'并发发送所选板的 2D 请求。';
 $('demo').disabled=busy||uploading;$('clear').disabled=busy||uploading;
 for(const id of ['upload2','upload3','board','submissionMode'])$(id).disabled=busy||uploading;
 $('releaseBoard').disabled=busy||uploading||!$('board').value;
 updateWorkflow();
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
$('board').onchange=assess;
$('requestInterval').value=$('requestInterval').value||'100';
$('submissionMode').onchange=$('requestInterval').oninput=()=>{revision++;assess();};
function links(id,workflow=false){
 for(const [key,file] of [['reportLink','report.html'],['jsonLink','result.json'],['csvLink','timings.csv'],['logLink','python.log']])$(key).href='/runs/'+id+'/'+file;
 $('links').hidden=false;$('logLink').hidden=workflow;
}
const workflowStatus={done:'完成',stopped:'已停止',failed:'失败'};
function showWorkflow(id,group){
 links(id,true);
 $('resultEmpty').hidden=true;$('report').hidden=false;$('report').src='/runs/'+id+'/report.html';
 $('summary').textContent=`${group.title} · ${workflowStatus[group.status]} · ${group.boards} 板 · ${group.runs} 个检测请求`+
   (Number.isFinite(group.total_wall_ms)?` · 总耗时 ${(group.total_wall_ms/1000).toFixed(2)} 秒`:'')+
  (group.error?'\n'+group.error:'');
}
function showResult(id,job){
 links(id);$('resultEmpty').hidden=true;$('report').hidden=false;$('report').src='/runs/'+id+'/report.html';
 const r=job.result;
 $('summary').textContent=r?'板 '+r.board_id+' · '+(r.ok?(r.product_is_ok?'检测完成 · OK':'检测完成 · NG（算法判定）'):'检测请求失败')+' · '+r.pieces+'片 · '+(r.total_wall_ms/1000).toFixed(2)+'秒 · '+(r.model_reused?'复用已加载模型':'本次初始化模型')+(r.submission_mode==='paced'?' · 按间隔提交 '+r.request_interval_ms+'ms':r.submission_mode==='concurrent'?' · 并发提交':'')+(r.error?'\n'+r.error:'')+'\n输出：'+r.output_dir:'验证失败：'+(job.error||'未知错误');
 if(r?.submission_mode==='manual'){
  const skipped={missing_2d_result:'缺少或已过期的2D结果',no_2d_result:'2D无有效掩码','2d_ng':'2D已判为NG',no_3d_checks:'未启用3D检测项'};
  const target=r.piece_id==null?'':' · 第'+r.piece_id+'片';
  $('summary').textContent='板 '+r.board_id+target+' · '+r.action+' · '+(r.ok?(r.product_is_ok?'OK':'NG'):'请求失败')+
   (r.skip_3d_reason?' · '+(skipped[r.skip_3d_reason]||r.skip_3d_reason):'')+
   (r.error?'\n'+r.error:'')+'\n用时 '+(r.total_wall_ms/1000).toFixed(2)+'秒';
  if(r.note)$('summary').textContent+='\n'+r.note;
 }
 const latency=r?.two_d_latency;
 if(latency&&Number.isFinite(latency.max_ms))$('summary').textContent+='\n2D请求≤'+latency.target_ms+'ms：'+(latency.meets_target?'达标':'未达标')+' · 最大 '+latency.max_ms.toFixed(1)+'ms'+(latency.over_target_piece_ids?.length?' · 超标片号 '+latency.over_target_piece_ids.join(', '):'');
}
async function history(){
 const list=await api('/api/history'),old=$('history').value;
 $('history').replaceChildren();
 appendOption($('history'),'','选择流程或单次检测');
 for(const job of list){
  const label=job.kind==='workflow'
   ?`${job.title} · ${workflowStatus[job.status]} · ${job.boards} 板`
   :'板 '+job.board_id+(job.piece_id==null?'':' · 第'+job.piece_id+'片')+
     (actions[job.mode]?' · '+actions[job.mode]:'')+' · '+(workflowStatus[job.status]||'运行中');
  appendOption($('history'),job.id,label);
 }
 if(list.some(job=>job.id===old))$('history').value=old;
}
$('history').onchange=async()=>{
 if(!$('history').value)return;
 try{
  const id=$('history').value,job=await api('/api/jobs/'+id);
  if(job.kind==='workflow'){showWorkflow(id,job);return;}
  if(job.status==='running'){status('该记录仍在运行');return;}
  if(!job.result){showJobError(job);status(job.error||'该记录没有报告',true);return;}
  showResult(id,job);status('正在查看历史结果；报告对应当时选用的文件。');
 }catch(e){status(e.message,true);}
};
async function waitForJob(id,onUpdate){
 while(true){
  await new Promise(r=>setTimeout(r,800));
  const job=await api('/api/jobs/'+id);
  onUpdate?.(job);
  if(job.status!=='running')return job;
 }
}
async function runValidation(request){
 const snapshot=revision;
 status('正在提交验证…');
 try{
  const {id}=await post('/api/run',request);
  const job=await waitForJob(id,job=>{if(job.status==='running')status(job.phase);});
  if(job.result)showResult(id,job);else showJobError(job);
  try{await history();$('history').value=id;}catch(error){console.warn('历史验证记录刷新失败',error);}
  const latencyMiss=job.result?.two_d_latency?.meets_target===false;
  status(job.status==='done'?(latencyMiss?'验证完成；2D请求未满足1秒目标，请查看超标片号。':'验证完成，可在下方复核。'):job.error||'验证失败，请检查报告与服务日志。',job.status!=='done'||latencyMiss);
  if(snapshot!==revision)status('验证已完成；报告对应提交时的文件。');
  return job;
 }catch(e){status(e.message,true);return {status:'failed',error:e.message};}
}
async function runExclusive(work,kind=null){
 if(busy||uploading||!config)return;
 busy=true;
 activeRun=kind?{kind,stopped:false,runIds:[],pitchIds:[],results:[]}:null;
 renderFiles();
 try{return await work(activeRun);}
 finally{activeRun=null;busy=false;renderFiles();}
}
function showJobError(job){
 $('summary').textContent=job.error||'验证失败';
 $('links').hidden=true;$('report').hidden=true;$('resultEmpty').hidden=false;
}
$('run').onclick=async()=>{
 if(!assess().ok||busy||uploading||!config)return;
 return runExclusive(()=>runValidation({files:structuredClone(files),board_id:$('board').value,mode:'2d',
  submission_mode:$('submissionMode').value,request_interval_ms:$('submissionMode').value==='paced'?Number($('requestInterval').value):0}));
};

function target(mode){
 if(mode==='delay')return {mode,delay_ms:'1000'};
 return {mode,board_id:'',piece_spec:'1'};
}
function nodeBoardId(node){
 return Pairing.board(node.board_id)||$('board').value;
}
function nodeCheck(node){
 if(node.mode==='delay'){
  const max=config?.max_request_interval_ms??3600000;
  const text=String(node.delay_ms??'').trim(), ms=Number(text);
  return text!==''&&Number.isFinite(ms)&&ms>=0&&ms<=max
   ?{ok:true,message:''}:{ok:false,message:'等待时间须为0～'+max+'毫秒的有限数值'};
 }
 const count=config?.piece_count||0;
 const pieces=Pairing.pieceNumbers(node.piece_spec,count);
 if(!pieces.length)return {ok:false,message:'请填写 1～'+count+' 的片号，支持用逗号和范围'};
 const board_id=nodeBoardId(node);
 for(const piece of pieces){
  const check=Pairing.checkManual(files,board_id,node.mode,piece,count);
  if(!check.ok)return {ok:false,message:'第'+piece+'片：'+check.message};
 }
 return {ok:true,message:'',pieces,board_id};
}
function delayCheck(node,index){
 const duration=nodeCheck(node);
 if(!duration.ok)return duration;
 const next=workflow[index+1];
 if(!next||next.mode==='delay')return {ok:false,message:'请将延时节点放在要触发的检测节点前'};
 return nodeCheck(next);
}
function addNode(node,index=workflow.length){
 if(busy||uploading||!config)return;
 workflow.splice(index,0,{...node,id:++nodeID,status:'待执行'});
 updateWorkflow();
}
function moveNode(id,index){
 const current=workflow.findIndex(node=>node.id===id);
 if(current<0||index===current||busy||uploading)return;
 const [node]=workflow.splice(current,1);
 workflow.splice(index,0,node);
 updateWorkflow();
}
async function runDetection(node,run){
 const {ok,message,pieces,board_id}=nodeCheck(node);
 if(!ok){status(message,true);return false;}
 const {mode}=node;
 node.runBoardId=board_id;
 node.requestTimes=[];
 const ids=[],results=[];
 let error='';
 for(const piece_id of pieces){
  if(run.stopped)break;
  const timing={piece_id,phase:'submitting'};
  node.requestTimes.push(timing);
  node.status=`正在提交第 ${piece_id} 片 · ${ids.length+1}/${pieces.length}`;
  node.outcome='running';updateWorkflow();
  const snapshot=files.filter(row=>Pairing.board(row.board_id)===board_id&&
   (row.kind==='3d'?mode!=='manual_2d':row.piece_id===piece_id));
  try{
   const accepted=await post('/api/run',{mode,board_id,piece_id,files:structuredClone(snapshot)});
   ids.push(accepted.id);run.runIds.push(accepted.id);
   if(mode===pitchMode){run.pitchIds.push(accepted.id);pendingPitchRuns.add(run);}
   timing.submittedAt=accepted.submitted_at;
   timing.phase=accepted.submitted===false?'unsent':accepted.pitch_waiting?'pitch_pending':'pending';
   results.push(trackRequest(accepted.id,timing));
   refreshWorkflowControls();
   if(accepted.submitted===false){
    error=accepted.error||'检测请求未发出';
    break;
   }
  }catch(e){
   timing.phase='unsent';
   error=e.message;
   break;
  }
 }
 node.message=error;
 node.outcome=ids.length?'running':error?'failed':'';
 node.status=ids.length?`${mode===pitchMode?'已登记':'已提交'} ${ids.length}/${pieces.length} · 等待结果`:error?'请求失败':'已停止';
 updateWorkflow();
 run.results.push(collectNode(node,workflow.indexOf(node),ids,results,pieces.length,error));
 return !error&&!run.stopped;
}
function trackRequest(id,timing){
 return waitForJob(id,job=>{
  const receivedAt=job.received_at??null;
  const submittedAt=job.submitted_at??timing.submittedAt;
  const pitchReadyAt=job.pitch_ready_at??null;
  let phase=timing.phase;
  if(Number.isFinite(receivedAt))phase='received';
  else if(phase!=='unsent'){
   if(job.status==='done')phase='unavailable';
   else if(job.status!=='running'||job.request_finished_at!=null)phase=Number.isFinite(submittedAt)?'unreceived':'unsent';
   else if(Number.isFinite(submittedAt))phase='pending';
  }
  if(timing.phase===phase&&timing.receivedAt===receivedAt&&timing.submittedAt===submittedAt&&
   timing.pitchReadyAt===pitchReadyAt)return;
  Object.assign(timing,{phase,receivedAt,submittedAt,pitchReadyAt});
  refreshWorkflowControls();
 }).catch(error=>{
  if(timing.phase==='pending'||timing.phase==='pitch_pending')timing.phase='unreceived';
  refreshWorkflowControls();
  return {status:'failed',error:error.message};
 });
}
async function collectNode(node,index,ids,results,count,submissionError){
 const jobs=await Promise.all(results);
 const failure=jobs.find(job=>job.status!=='done');
 const error=submissionError||failure?.error||(failure?'检测请求失败':'');
 const finished=jobs.filter(job=>job.status==='done').length;
 const ng=jobs.filter(job=>job.status==='done'&&!job.result?.product_is_ok).length;
 node.outcome=error?'failed':finished===count?'done':'';
 node.status=error?'请求失败':finished<count?`已停止 · ${finished}/${count}`:
  count===1?(ng?'完成 · NG':'完成 · OK'):`完成 ${finished} 片 · ${ng} 片 NG`;
 node.message=error;
 updateWorkflow();
 return {index,error,id:ids.at(-1),job:jobs.at(-1)};
}
function waitDelay(node,next){
 const duration=Math.ceil(Number(node.delay_ms)),deadline=Date.now()+duration;
 const tick=()=>{
  const remaining=Math.max(0,deadline-Date.now());
  status('等待 '+(remaining<1000?remaining+' ms':(remaining/1000).toFixed(1)+' 秒')+
   ' · 板 '+nodeBoardId(next)+' · '+actions[next.mode]);
 };
 node.status='计时中';node.outcome='running';
 return new Promise(resolve=>{
  const finish=elapsed=>{
   clearTimeout(timeout);clearInterval(update);cancelCountdown=null;
   node.status=elapsed?'延时完成':'已取消';node.outcome=elapsed?'done':'';
   updateWorkflow();resolve(elapsed);
  };
  const update=setInterval(tick,250),timeout=setTimeout(()=>finish(true),duration);
  cancelCountdown=()=>finish(false);
  updateWorkflow();tick();
 });
}
async function runSingleNode(node,index){
 const isDelay=node.mode==='delay',next=isDelay?workflow[index+1]:node;
 if(next?.outcome==='running')return;
 const check=isDelay?delayCheck(node,index):nodeCheck(node);
 if(!check.ok){status(check.message,true);return;}
 const run=await runExclusive(async run=>{
  if(!isDelay||await waitDelay(node,next))await runDetection(next,run);
  if(run.stopped)status(isDelay&&!run.runIds.length?'计时已取消。':'后续请求已停止。');
  return run;
 },isDelay?'delay':'detection');
 if(!run)return;
 if(run.stopped&&run.pitchIds.length)await finishPitchSubmissions(run).catch(err=>status(err.message,true));
 const [result]=await Promise.all(run.results);
 pendingPitchRuns.delete(run);refreshWorkflowControls();
 if(!result)return;
 if(!busy){
  if(result.job?.result)showResult(result.id,result.job);
  else if(result.error)showJobError({error:result.error});
  status(result.error||(run.stopped?'后续请求已停止。':'验证完成，可在下方复核。'),!!result.error);
 }
 try{await history();if(!busy&&result.id)$('history').value=result.id;}
 catch(error){console.warn('历史验证记录刷新失败',error);}
}
function workflowIssue(){
 for(const [index,node] of workflow.entries()){
  const check=node.mode==='delay'?delayCheck(node,index):nodeCheck(node);
  if(!check.ok)return `节点 ${index+1}：${check.message}`;
 }
 return '';
}
function finishPitchSubmissions(run){
 return post('/api/pitch/finish',{run_ids:run.pitchIds,stopped:run.stopped});
}
$('startWorkflow').onclick=async()=>{
 const issue=workflowIssue();
 if(issue){status(issue,true);return;}
 if(!workflow.length||workflow.some(node=>node.outcome==='running'))return;
 const submission=await runExclusive(async run=>{
  for(const node of workflow){node.outcome='';node.status='待执行';node.message='';node.requestTimes=[];}
  let error='';
  // Arm the automatic requests before starting 2D. An adjacent delay still
  // controls when its following trigger is armed.
  const armed=new Set();
  for(const [index,node] of workflow.entries()){
   if(run.stopped||error)break;
   if(node.mode!==pitchMode||workflow[index-1]?.mode==='delay')continue;
   armed.add(node.id);
   if(!await runDetection(node,run)&&!run.stopped)error=`流程停在节点 ${index+1}：${node.message||'登记失败'}`;
  }
  for(const [index,node] of workflow.entries()){
   if(run.stopped||error)break;
   if(armed.has(node.id))continue;
   const finished=node.mode==='delay'?await waitDelay(node,workflow[index+1]):await runDetection(node,run);
   if(!finished){
    if(!run.stopped)error=`流程停在节点 ${index+1}：${node.message||'检测请求失败'}`;
    break;
   }
  }
  if(run.pitchIds.length){
   try{await finishPitchSubmissions(run);}
   catch(err){error=error||err.message;}
  }
  return {run,error};
 },'workflow');
 if(!submission)return;
 const {run}=submission;
 if(!busy)status(submission.error||(run.stopped?'后续提交已停止，等待已提交的检测结果…':'请求已提交，等待检测结果…'),!!submission.error);
 const results=await Promise.all(run.results);
 pendingPitchRuns.delete(run);refreshWorkflowControls();
 const failure=results.find(result=>result.error);
 const error=submission.error||(failure?`节点 ${failure.index+1}：${failure.error}`:'');
 const outcome=run.stopped?'stopped':error?'failed':'done';
 try{
  const group=await post('/api/workflows',{run_ids:run.runIds,outcome,error});
  if(!busy)showWorkflow(group.id,group);
  await history();if(!busy)$('history').value=group.id;
  if(!busy)status(error||(run.stopped?'流程已停止。':'流程执行完成。'),!!error);
 }catch(err){
  await history().catch(()=>{});
  if(!busy)status([error,'流程结果汇总失败：'+err.message].filter(Boolean).join('\n'),true);
 }
};
function refreshWorkflowControls(){
 const locked=busy||uploading||!config;
 for(const [id] of palette)$(id).disabled=locked;
 const issue=config&&workflow.length?workflowIssue():'';
 $('flowEmpty').hidden=workflow.length>0;
 $('clearFlow').disabled=locked||!workflow.length;
 $('startWorkflow').disabled=locked||!workflow.length||!!issue||workflow.some(node=>node.outcome==='running');
 $('workflowName').disabled=locked;
 $('savedWorkflow').disabled=locked;
 $('saveWorkflow').disabled=locked||!workflow.length||!$('workflowName').value.trim();
 $('loadWorkflow').disabled=locked||!$('savedWorkflow').value;
 $('saveWorkflowFile').disabled=locked||!workflow.length;
 $('openWorkflowFile').disabled=locked;
 $('workflowIssue').textContent=issue;$('workflowIssue').hidden=!issue||busy;
 const stoppable=activeRun||Array.from(pendingPitchRuns).at(-1);
 $('stopWorkflow').hidden=!stoppable;
 $('stopWorkflow').disabled=stoppable?.stopped===true;
 $('stopWorkflow').textContent=stoppable?.kind==='workflow'?'停止流程':cancelCountdown?'取消计时':'停止后续请求';
 for(const refresh of refreshCards)refresh();
}
function addField(group,text,control){
 const label=document.createElement('label');
 label.textContent=text;label.append(control);group.append(label);
 return label;
}
function appendOption(select,value,text){
 const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);
}
function nodeSetup({mode,board_id,piece_spec,delay_ms}){
 if(mode==='manual_board_3d'){
  const count=config?.piece_count||1;
  return {mode:'manual_piece_3d',board_id,piece_spec:count>1?'1-'+count:'1'};
 }
 return mode==='delay'?{mode,delay_ms}:{mode,board_id,piece_spec};
}
function workflowSetup(){
 return workflow.map(nodeSetup);
}
function validWorkflow(entry){
 const validNode=node=>node&&(node.mode==='delay'?typeof node.delay_ms==='string':
   typeof node.board_id==='string'&&(node.mode==='manual_board_3d'||
    Object.hasOwn(detectionTypes,node.mode)&&typeof node.piece_spec==='string'));
 return entry&&(entry.version===undefined||entry.version===1||entry.version===workflowVersion)&&
  typeof entry.name==='string'&&entry.name.trim()&&
  Array.isArray(entry.nodes)&&entry.nodes.length&&entry.nodes.every(validNode);
}
function restoreWorkflow(entry){
 const savedBoards=new Set(entry.nodes.map(node=>Pairing.board(node.board_id)).filter(Boolean));
 // Older single-board templates saved the current board even when it was just a default.
 const followCurrent=entry.version!==workflowVersion&&savedBoards.size===1;
 const nodes=entry.nodes.map(node=>{
  const setup=nodeSetup(node);
  if(followCurrent&&setup.mode!=='delay')setup.board_id='';
  return setup;
 });
 return {...entry,version:workflowVersion,nodes};
}
function readSavedWorkflows(){
 const entries=JSON.parse(window.localStorage.getItem(savedWorkflowKey)||'[]');
 if(!Array.isArray(entries)||!entries.every(validWorkflow))throw Error('已保存的流程格式有误');
 return entries.map(restoreWorkflow);
}
function showSavedWorkflows(entries,selected=$('savedWorkflow').value){
 const list=$('savedWorkflow');
 list.replaceChildren();appendOption(list,'','选择已保存的流程');
 for(const entry of entries)appendOption(list,entry.name,entry.name);
 list.value=entries.some(entry=>entry.name===selected)?selected:'';
 refreshWorkflowControls();
}
$('workflowName').oninput=refreshWorkflowControls;
$('savedWorkflow').onchange=refreshWorkflowControls;
$('saveWorkflow').onclick=()=>{
 if($('saveWorkflow').disabled)return;
 try{
  const name=$('workflowName').value.trim(),nodes=workflowSetup();
  const entries=readSavedWorkflows(),index=entries.findIndex(entry=>entry.name===name);
  const entry={version:workflowVersion,name,nodes};
  if(index<0)entries.push(entry);
  else{
   if(!window.confirm('更新已保存的流程「'+name+'」？'))return;
   entries[index]=entry;
  }
  window.localStorage.setItem(savedWorkflowKey,JSON.stringify(entries));
  $('workflowName').value=name;
  showSavedWorkflows(entries,name);status('已保存流程「'+name+'」。');
 }catch(error){status('保存流程失败：'+error.message,true);}
};
function applySavedWorkflow(entry,stored=[]){
 const current=JSON.stringify(workflowSetup()),{nodes}=restoreWorkflow(entry);
 if(workflow.length&&current!==JSON.stringify(nodes)&&
  !stored.some(item=>JSON.stringify(item.nodes)===current)&&
  !window.confirm('载入「'+entry.name+'」将替换当前画布，是否继续？'))return;
 workflow.splice(0,workflow.length,...nodes.map(node=>({...node,id:++nodeID,status:'待执行'})));
 $('workflowName').value=entry.name;
 updateWorkflow();
 status('已载入流程「'+entry.name+'」。');
}
$('loadWorkflow').onclick=()=>{
 if($('loadWorkflow').disabled)return;
 try{
  const entries=readSavedWorkflows();
  const entry=entries.find(item=>item.name===$('savedWorkflow').value);
  if(!entry){showSavedWorkflows(entries);status('找不到这份流程。',true);return;}
  applySavedWorkflow(entry,entries);
 }catch(error){status('载入流程失败：'+error.message,true);}
};
$('saveWorkflowFile').onclick=()=>{
 if($('saveWorkflowFile').disabled)return;
 return runExclusive(async()=>{
  const name=$('workflowName').value.trim()||'检测流程';
  const body=JSON.stringify({format:workflowFileFormat,version:workflowVersion,name,nodes:workflowSetup()},null,2)+'\n';
  const filename='流程-'+name.replace(/[<>:"/\\|?*\u0000-\u001f]/g,'_').slice(0,100)+'.json';
  try{
   if(typeof window.showSaveFilePicker==='function'){
    const file=await window.showSaveFilePicker({suggestedName:filename,
     types:[{description:'检测流程',accept:{'application/json':['.json']}}]});
    const stream=await file.createWritable();
    try{await stream.write(body);await stream.close();}
    catch(error){await stream.abort().catch(()=>{});throw error;}
    status('已保存流程文件 '+file.name+'。');
   }else{
    const url=URL.createObjectURL(new Blob([body],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download=filename;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    status('已交给浏览器下载 '+filename+'。');
   }
  }catch(error){if(error.name!=='AbortError')status('保存流程文件失败：'+error.message,true);}
 });
};
$('openWorkflowFile').onclick=()=>{if(!$('openWorkflowFile').disabled)$('workflowFileInput').click();};
$('workflowFileInput').onchange=event=>{
 const input=event.target,file=input.files?.[0];
 input.value='';
 if(!file)return;
 return runExclusive(async()=>{
  try{
   const document=JSON.parse(await file.text());
   if(document?.format!==workflowFileFormat||!Number.isInteger(document.version)||!validWorkflow(document))
    throw Error('请选择验证台导出的流程 JSON');
   applySavedWorkflow(document);
  }catch(error){status('载入流程文件失败：'+error.message,true);}
 });
};
function editNode(node,changes){
 Object.assign(node,changes,{status:'待执行',outcome:'',message:'',requestTimes:[]});
 refreshWorkflowControls();
}
function detectionFields(node,boards,locked){
 const fields=document.createElement('div');fields.className='node-fields';
 const type=document.createElement('select');type.disabled=locked;
 for(const [value,title] of Object.entries(detectionTypes))appendOption(type,value,title);
 type.value=node.mode;
 addField(fields,'检测类型',type);
 const board=document.createElement('select');board.disabled=locked;
 const currentBoard=$('board').value;
 appendOption(board,'',currentBoard?'自动 · 板 '+currentBoard:'自动 · 等待载入文件');
 for(const id of boards)appendOption(board,id,'板 '+id);
 if(node.board_id&&!boards.includes(node.board_id))appendOption(board,node.board_id,'板 '+node.board_id+'（未载入）');
 board.value=node.board_id;
 addField(fields,'板号',board);
 const pieces=document.createElement('input');pieces.type='text';pieces.value=node.piece_spec;
 pieces.placeholder='如 1 或 1,3-5';pieces.disabled=locked;
  addField(fields,'片号',pieces);
  type.onchange=()=>editNode(node,{mode:type.value});
 board.onchange=()=>editNode(node,{board_id:board.value});
 pieces.oninput=()=>editNode(node,{piece_spec:pieces.value});
 return fields;
}
function renderRequestTimes(table,timings,pitch){
 table.hidden=!timings?.length;
 table.replaceChildren();
 if(table.hidden)return;
 const header=document.createElement('thead'),columns=document.createElement('tr');
 const names=pitch?['片号','Pitch 就绪','3D 提交','3D 返回']:['片号','提交时间','收到结果'];
 for(const name of names){
  const cell=document.createElement('th');cell.scope='col';cell.textContent=name;columns.append(cell);
 }
 header.append(columns);
 const body=document.createElement('tbody');
 for(const timing of timings){
  const row=document.createElement('tr');
  const submitted=Number.isFinite(timing.submittedAt)?requestClock.format(timing.submittedAt*1000):
   timing.phase==='submitting'?'提交中':timing.phase==='unsent'?'未提交':timing.phase==='pitch_pending'?'等待中':'—';
  const received=Number.isFinite(timing.receivedAt)?requestClock.format(timing.receivedAt*1000):
   timing.phase==='pending'?'等待中':timing.phase==='unreceived'?'未收到':'—';
  const values=[timing.piece_id];
  if(pitch)values.push(Number.isFinite(timing.pitchReadyAt)?requestClock.format(timing.pitchReadyAt*1000):'—');
  values.push(submitted,received);
  for(const value of values){
   const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);
  }
  body.append(row);
 }
 table.append(header,body);
}
function updateWorkflow(){
 const locked=busy||uploading||!config;
 const boards=Pairing.boards(files);
 refreshCards=[];
 $('flowNodes').replaceChildren();
 for(const [index,node] of workflow.entries()){
  const isDelay=node.mode==='delay';
  const card=document.createElement('li');card.className=isDelay?'flow-node delay-node':'flow-node';card.dataset.flowId=String(node.id);
  const grip=document.createElement('div');grip.className='flow-grip';grip.draggable=!locked;grip.tabIndex=0;
  grip.title='拖动排序，也可用方向键移动';
  grip.ondragstart=event=>event.dataTransfer.setData('text/plain',JSON.stringify({id:node.id}));
  grip.onkeydown=event=>{
   const offset={ArrowLeft:-1,ArrowUp:-1,ArrowRight:1,ArrowDown:1}[event.key];
   if(!offset)return;
   event.preventDefault();moveNode(node.id,Math.max(0,Math.min(workflow.length-1,index+offset)));
  };
  const footer=document.createElement('div');footer.className='flow-footer';
  const times=isDelay?null:document.createElement('table');
  if(times){times.className='node-request-times';times.setAttribute('aria-label','各片请求时间');}
  const state=document.createElement('span');
  const execute=document.createElement('button');
  const refresh=()=>{
   grip.textContent=(index+1)+' · '+actions[node.mode];
    const otherBoard=!isDelay&&node.outcome&&node.runBoardId&&node.runBoardId!==nodeBoardId(node);
    state.className='flow-status '+(node.outcome||'');
    state.textContent=otherBoard?'板 '+node.runBoardId+' · '+node.status:node.status;state.title=node.message||'';
   execute.textContent=isDelay?'开始计时':node.outcome==='done'?'重跑':'执行';
   const check=isDelay?delayCheck(node,index):nodeCheck(node);
    execute.disabled=locked||node.outcome==='running'||!check.ok;execute.title=check.message;
   if(times){
    const pitch=node.mode===pitchMode;
    times.className=pitch?'node-request-times pitch-request-times':'node-request-times';
    renderRequestTimes(times,node.requestTimes,pitch);
   }
  };
  let description;
  if(isDelay){
   description=document.createElement('label');
   description.className='delay-setting';description.textContent='等待时间（ms）';
   const input=document.createElement('input');input.type='number';input.min=0;
   input.max=config?.max_request_interval_ms??3600000;input.step='any';input.value=node.delay_ms;
   input.disabled=locked;
   input.setAttribute('aria-label','节点 '+(index+1)+' 等待时间（毫秒）');
   input.oninput=()=>editNode(node,{delay_ms:input.value});
   description.append(input);
   }else description=detectionFields(node,boards,locked||node.outcome==='running');
  refreshCards.push(refresh);
  execute.onclick=()=>runSingleNode(node,index);
  const remove=document.createElement('button');remove.textContent='×';remove.title='移除节点';remove.disabled=locked;
  remove.setAttribute('aria-label','移除节点 '+(index+1));
  remove.onclick=()=>{workflow.splice(index,1);updateWorkflow();};
  footer.append(state,execute,remove);card.append(grip,description,footer);
  if(times)card.append(times);
  $('flowNodes').append(card);
 }
 refreshWorkflowControls();
}
for(const [id,mode] of palette){
 $(id).onclick=()=>addNode(target(mode));
 $(id).ondragstart=event=>{if(!$(id).disabled)event.dataTransfer.setData('text/plain',JSON.stringify({node:target(mode)}));};
}
$('flowCanvas').ondragover=event=>{if(!busy&&!uploading)event.preventDefault();};
$('flowCanvas').ondrop=event=>{
 event.preventDefault();
 if(busy||uploading)return;
 let data;
 try{data=JSON.parse(event.dataTransfer.getData('text/plain'));}catch{return;}
 const targetID=Number(event.target.closest?.('[data-flow-id]')?.dataset.flowId);
 const index=workflow.findIndex(node=>node.id===targetID);
 if(data?.node&&Object.hasOwn(actions,data.node.mode))addNode(data.node,index<0?workflow.length:index);
 else if(Number.isInteger(data?.id))moveNode(data.id,index<0?workflow.length-1:index);
};
$('clearFlow').onclick=()=>{if(!busy&&!uploading){workflow.length=0;updateWorkflow();}};
$('stopWorkflow').onclick=()=>{
 const run=activeRun||Array.from(pendingPitchRuns).at(-1);
 if(!run||run.stopped)return;
 run.stopped=true;
 if(run.pitchIds.length)finishPitchSubmissions(run).catch(err=>status(err.message,true));
 if(cancelCountdown)cancelCountdown();
 else{refreshWorkflowControls();status('已停止后续提交，已经发出的请求继续检测。');}
};
$('releaseBoard').onclick=async()=>{
 if(busy||uploading||!$('board').value)return;
 const board_id=$('board').value;busy=true;assess();
 try{await post('/api/release-board',{board_id});status('板 '+board_id+' 的检测缓存已释放。');}
 catch(error){status(error.message,true);}finally{busy=false;assess();}
};
api('/api/defaults').then(d=>{
 config=d;
 const p=d.pitch;$('config').textContent='每板 '+d.piece_count+'片 · 粗配准 + 仿射 · Pitch '+(p.enabled_3d?'启用，'+p.angle_deg+'°，步距'+p.scan_pitch_mm+' mm/profile':'关闭');
 renderFiles();status('就绪，请添加文件或载入项目样本。');
 try{showSavedWorkflows(readSavedWorkflows());}catch(error){status('读取已保存流程失败：'+error.message,true);}
 return history();
}).catch(e=>status('配置加载失败：'+e.message,true));
