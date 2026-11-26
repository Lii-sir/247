const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const dir=path.join(__dirname,'../validation_lab'),html=fs.readFileSync(path.join(dir,'index.html'),'utf8');
function browserStorage(){
 const values=new Map();
 return {getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value)};
}
async function boot(latency=null,timers={},failRequest=()=>false,browser={},hooks={}){
 const nodes=new Map(),requests=[],history=[],groups=[];
 class Node{
  constructor(){this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.children=[];this.dataset={};}
  set id(v){this._id=v;nodes.set(v,this);}get id(){return this._id;}
  append(...c){this.children.push(...c);}replaceChildren(...c){this.children=c;}setAttribute(k,v){this[k]=v;}
 }
 for(const m of html.matchAll(/<[a-z]+\b([^>]*\bid="([^"]+)"[^>]*)>/g)){const n=new Node();n.id=m[2];n.hidden=/\bhidden\b/.test(m[1]);n.disabled=/\bdisabled\b/.test(m[1]);}
 for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>(.*?)<\/select>/gs))nodes.get(m[1]).value=m[2].match(/<option\b[^>]*value="([^"]*)"/)?.[1]||'';
 const config={piece_count:2,samples:[{kind:'2d',name:'2_1.png',path:'/2_1.png',board_id:'2',piece_id:1},{kind:'2d',name:'2_2.png',path:'/2_2.png',board_id:'2',piece_id:2},{kind:'3d',name:'2.png',path:'/2.png',board_id:'2',piece_id:null}],pitch:{enabled_3d:false}};
 const ctx=vm.createContext({console,structuredClone,setTimeout:fn=>setImmediate(fn),...timers,
  window:{localStorage:browserStorage(),confirm:()=>true,...browser},document:{getElementById:id=>{if(!nodes.has(id))throw Error(id);return nodes.get(id);},createElement:()=>new Node()},
  fetch:async(url,opt)=>{
   let data;
   if(url==='/api/defaults')data=config;
   else if(url==='/api/history')data=history;
   else if(url.startsWith('/api/upload?')){const kind=new URL(url,'http://local').searchParams.get('kind'),name=opt.body.name,m=name.match(/^(.+)_(\d+)\./);data={kind,name,path:'/uploads/'+name,board_id:kind==='3d'?name.split('.')[0]:m?.[1]||'',piece_id:kind==='2d'&&m?Number(m[2]):null};}
   else if(url==='/api/run'){const q=JSON.parse(opt.body);requests.push(q);const id=String(requests.length).padStart(32,'0');
    const submitted=hooks.submit?.(q)!==false;
    if(!submitted||failRequest(q))history.unshift({id,board_id:q.board_id,status:'failed',error:'synthetic request failure'});
    else history.unshift({id,board_id:q.board_id,status:'done',result:{ok:true,product_is_ok:false,board_id:q.board_id,pieces:2,total_wall_ms:1000,output_dir:'/output',model_reused:false,submission_mode:q.submission_mode,request_interval_ms:q.request_interval_ms,two_d_latency:latency}});
    data={id,submitted,error:submitted?null:'synthetic request failure'};}
   else if(url==='/api/workflows'){
    const q=JSON.parse(opt.body);groups.push(q);
    data={id:'a'.repeat(32),kind:'workflow',title:'流程 '+groups.length,status:q.outcome,boards:2,runs:q.run_ids.length,error:q.error};
    for(const id of q.run_ids){const index=history.findIndex(job=>job.id===id);if(index>=0)history.splice(index,1);}
    history.unshift(data);
   }
   else if(url.startsWith('/api/jobs/')){
    await hooks.beforePoll?.(url);
    data=history.find(job=>job.id===url.split('/').at(-1));
   }
   else throw Error(url);
   return {ok:true,json:async()=>data};
  }});
 vm.runInContext(fs.readFileSync(path.join(dir,'pairing.js'),'utf8'),ctx);vm.runInContext(fs.readFileSync(path.join(dir,'app.js'),'utf8'),ctx);await new Promise(setImmediate);
 return {nodes,requests,config,groups};
}
const execute=card=>card.children[2].children[1];
function setDetection(card,{mode,board,pieces}){
 const [typeField,boardField,pieceField]=card.children[1].children;
 if(mode!==undefined){const type=typeField.children[0];type.value=mode;type.onchange();}
 if(board!==undefined){const select=boardField.children[0];select.value=board;select.onchange();}
 if(pieces!==undefined){const input=pieceField.children[0];input.value=pieces;input.oninput();}
}
test('demo -> validation passes exact board/piece manifest and opens interactive report',async()=>{
 const {nodes,requests}=await boot();assert.equal(nodes.get('run').disabled,true);nodes.get('demo').onclick();
 assert.equal(nodes.get('run').disabled,false);await nodes.get('run').onclick();
 assert.equal(requests[0].board_id,'2');assert.deepEqual(requests[0].files.filter(f=>f.kind==='2d').map(f=>f.piece_id),[1,2]);
 assert.equal(nodes.get('report').hidden,false);assert.match(nodes.get('report').src,/report.html$/);assert.match(nodes.get('summary').textContent,/NG/);
 assert.match(nodes.get('jsonLink').href,/result.json$/);
});

test('2D latency violation is visible without changing product NG',async()=>{
 const {nodes}=await boot({target_ms:1000,max_ms:1079,meets_target:false,over_target_piece_ids:[10]});
 nodes.get('demo').onclick();await nodes.get('run').onclick();
 assert.match(nodes.get('summary').textContent,/2D请求≤1000ms：未达标/);
 assert.match(nodes.get('summary').textContent,/1079\.0ms/);
 assert.match(nodes.get('summary').textContent,/超标片号 10/);
 assert.match(nodes.get('summary').textContent,/NG/);
 assert.equal(nodes.get('progress').className,'error');
 assert.match(nodes.get('progress').textContent,/未满足1秒目标/);
});
test('manual duplicate piece blocks submission and partial2D mode permits missing piece',async()=>{
 const {nodes,requests}=await boot();nodes.get('demo').onclick();
 const input=nodes.get('files').children[1].children[2].children[0];input.value='1';input.oninput();
 assert.equal(nodes.get('run').disabled,true);await nodes.get('run').onclick();assert.equal(requests.length,0);
 nodes.get('demo').onclick();nodes.get('files').children[1].children[3].children[0].onclick();
 assert.equal(nodes.get('run').disabled,false);await nodes.get('run').onclick();assert.equal(requests[0].mode,'2d');
});
test('batch file upload displays original names and permits manual board editing',async()=>{
 const {nodes}=await boot();
 await nodes.get('upload2').onchange({target:{files:[{name:'2_1.png'},{name:'2_2.png'}],value:'files'}});
 await nodes.get('upload3').onchange({target:{files:[{name:'2.png'}],value:'files'}});
 assert.equal(nodes.get('run').disabled,false);assert.equal(nodes.get('files').children[0].children[0].textContent,'2_1.png');
 const board=nodes.get('files').children[0].children[1].children[0];board.value='3';board.oninput();
 assert.match(nodes.get('counts').textContent,/2个已编号板号/);
 nodes.get('board').value='2';nodes.get('board').onchange();assert.equal(nodes.get('run').disabled,false);
});
test('paced submission carries interval and keeps report snapshot',async()=>{
 const {nodes,requests}=await boot();nodes.get('demo').onclick();
 assert.equal(nodes.get('requestInterval').disabled,true);
 nodes.get('submissionMode').value='paced';nodes.get('submissionMode').onchange();
 assert.equal(nodes.get('requestInterval').disabled,false);
 nodes.get('requestInterval').value='125.5';nodes.get('requestInterval').oninput();
 await nodes.get('run').onclick();
 assert.equal(requests[0].submission_mode,'paced');assert.equal(requests[0].request_interval_ms,125.5);
 assert.match(nodes.get('summary').textContent,/125\.5ms/);
 nodes.get('requestInterval').value='900';nodes.get('requestInterval').oninput();
 assert.match(nodes.get('summary').textContent,/125\.5ms/);
});
test('invalid paced intervals block run and concurrent ignores disabled value',async()=>{
 const {nodes,requests}=await boot();nodes.get('demo').onclick();
 nodes.get('submissionMode').value='paced';nodes.get('submissionMode').onchange();
 for(const value of ['', '-1', 'Infinity', '3600001']){
  nodes.get('requestInterval').value=value;nodes.get('requestInterval').oninput();
  assert.equal(nodes.get('run').disabled,true);await nodes.get('run').onclick();
 }
 assert.equal(requests.length,0);
 nodes.get('submissionMode').value='concurrent';nodes.get('submissionMode').onchange();
 assert.equal(nodes.get('run').disabled,false);await nodes.get('run').onclick();
 assert.equal(requests[0].submission_mode,'concurrent');assert.equal(requests[0].request_interval_ms,0);
});

test('loading, dragging and reordering never submit; each execution sends one selected action', async()=>{
 const {nodes,requests}=await boot();
 nodes.get('demo').onclick();
 const dataTransfer={data:'',setData(_,value){this.data=value;},getData(){return this.data;}};
 const canvas=nodes.get('flowCanvas'),cards=nodes.get('flowNodes');
 const drop=target=>canvas.ondrop({target,dataTransfer,preventDefault(){}});
 nodes.get('addDetection').ondragstart({dataTransfer});
 drop(canvas);
 setDetection(cards.children[0],{pieces:'2'});
 nodes.get('addDetection').onclick();
 setDetection(cards.children[1],{mode:'manual_piece_3d'});
 nodes.get('addDetection').onclick();
 setDetection(cards.children[2],{mode:'manual_piece_3d',pieces:'2'});
 cards.children[1].children[0].ondragstart({dataTransfer});
 drop({closest:()=>cards.children[0]});
 assert.equal(requests.length,0);
 assert.match(cards.children[0].children[0].textContent,/单片 3D/);

 await execute(cards.children[0]).onclick();
 assert.equal(requests.length,1);
 assert.equal(requests[0].mode,'manual_piece_3d');
 assert.equal(requests[0].piece_id,1);
 assert.deepEqual(requests[0].files.map(row=>row.name),['2_1.png','2.png']);
 await execute(cards.children[1]).onclick();
 assert.equal(requests.length,2);
 assert.equal(requests[1].mode,'manual_2d');
 assert.equal(requests[1].piece_id,2);
 assert.deepEqual(requests[1].files.map(row=>row.name),['2_2.png']);
 await execute(cards.children[2]).onclick();
 assert.equal(requests.length,3);
 assert.equal(requests[2].mode,'manual_piece_3d');
 assert.equal(requests[2].piece_id,2);
});

test('saved whole-board 3D nodes load as a range of single-piece requests',async()=>{
 const localStorage=browserStorage();
 localStorage.setItem('validation-lab.workflow-templates.v1',JSON.stringify([
  {name:'旧流程',nodes:[{mode:'manual_board_3d',board_id:'2',piece_spec:'1'}]}
 ]));
 const {nodes,requests}=await boot(null,{},()=>false,{localStorage});
 nodes.get('demo').onclick();
 nodes.get('savedWorkflow').value='旧流程';nodes.get('savedWorkflow').onchange();
 nodes.get('loadWorkflow').onclick();
 const card=nodes.get('flowNodes').children[0];
 assert.equal(card.children[1].children[2].children[0].value,'1-2');
 await execute(card).onclick();
 assert.deepEqual(requests.map(({mode,piece_id})=>[mode,piece_id]),[
  ['manual_piece_3d',1],['manual_piece_3d',2]
 ]);
});

function deferredTimers(){
 const pending=new Map(),delays=[];
 let id=0;
 const timers={
  setTimeout(fn,ms){
   if(ms===800)return setImmediate(fn);
   pending.set(++id,fn);delays.push(ms);return id;
  },
  clearTimeout(handle){pending.delete(handle);},
  setInterval(){return 123;},
  clearInterval(){}
 };
 return {timers,pending,delays};
}

test('a configurable delay only triggers its adjacent detection when the timer elapses',async()=>{
 const {timers,pending,delays}=deferredTimers();
 const {nodes,requests}=await boot(null,timers);
 nodes.get('demo').onclick();
 nodes.get('addDelay').onclick();
 const cards=nodes.get('flowNodes');
 const start=()=>execute(cards.children[0]);
 assert.equal(start().disabled,true);
 nodes.get('addDetection').onclick();
 setDetection(cards.children[1],{pieces:'2'});
 nodes.get('addDetection').onclick();
 setDetection(cards.children[2],{mode:'manual_piece_3d'});
 const input=cards.children[0].children[1].children[0];
 input.value='';input.oninput();assert.equal(start().disabled,true);
 input.value='1500.5';input.oninput();assert.equal(start().disabled,false);
 const canceled=start().onclick();
 assert.deepEqual(delays,[1501]);
 assert.equal(requests.length,0);
 assert.equal(nodes.get('stopWorkflow').hidden,false);
 nodes.get('stopWorkflow').onclick();
 await canceled;
 assert.equal(pending.size,0);
 assert.equal(requests.length,0);
 const execution=start().onclick();
 const fire=pending.values().next().value;
 fire();await execution;
 assert.equal(requests.length,1);
 assert.equal(requests[0].mode,'manual_2d');
 assert.equal(requests[0].piece_id,2);
 assert.equal(nodes.get('stopWorkflow').hidden,true);
});

test('one click runs a configured range and delay, then opens one grouped workflow report',async()=>{
 const {timers,pending,delays}=deferredTimers();
 let releaseResults;
 const results=new Promise(resolve=>{releaseResults=resolve;});
 const {nodes,requests,config,groups}=await boot(null,timers,()=>false,{}, {beforePoll:()=>results});
 config.samples.push({kind:'2d',name:'3_1.png',path:'/3_1.png',board_id:'3',piece_id:1});
 nodes.get('demo').onclick();
 const cards=nodes.get('flowNodes');
 nodes.get('addDetection').onclick();
 setDetection(cards.children[0],{pieces:'3'});
 assert.equal(nodes.get('startWorkflow').disabled,true);
 assert.match(nodes.get('workflowIssue').textContent,/片号/);
 setDetection(cards.children[0],{pieces:'2,1-2'});
 nodes.get('addDelay').onclick();
 nodes.get('addDetection').onclick();
 setDetection(cards.children[2],{board:'3',pieces:'1'});
 assert.equal(nodes.get('startWorkflow').disabled,false);
 assert.equal(requests.length,0);
 const run=nodes.get('startWorkflow').onclick();
 for(let n=0;!pending.size&&n<10;n++)await new Promise(setImmediate);
 assert.deepEqual(requests.map(q=>[q.board_id,q.piece_id]),[['2',2],['2',1]]);
 assert.deepEqual(delays,[1000]);
 pending.values().next().value();
 for(let n=0;requests.length<3&&n<10;n++)await new Promise(setImmediate);
 assert.deepEqual(requests.map(q=>[q.board_id,q.piece_id]),[['2',2],['2',1],['3',1]]);
 assert.equal(groups.length,0);
 releaseResults();
 await run;
 assert.deepEqual(groups,[{run_ids:[1,2,3].map(id=>String(id).padStart(32,'0')),outcome:'done',error:''}]);
 assert.equal(nodes.get('history').children.length,2);
 assert.equal(nodes.get('report').src,'/runs/'+'a'.repeat(32)+'/report.html');
 assert.match(nodes.get('summary').textContent,/流程 1.*3 个检测请求/);
 assert.match(nodes.get('progress').textContent,/流程执行完成/);
});

test('stopping during an active request prevents remaining pieces from being sent',async()=>{
 const {nodes,requests}=await boot();
 nodes.get('demo').onclick();
 nodes.get('addDetection').onclick();
 const cards=nodes.get('flowNodes');
 setDetection(cards.children[0],{pieces:'1-2'});
 const run=nodes.get('startWorkflow').onclick();
 assert.equal(requests.length,1);
 nodes.get('stopWorkflow').onclick();
 await run;
 assert.equal(requests.length,1);
 assert.match(nodes.get('progress').textContent,/流程已停止/);
});

test('an error before sending stops subsequent submissions',async()=>{
 const {nodes,requests}=await boot(null,{},()=>false,{}, {submit:q=>q.mode!=='manual_2d'});
 nodes.get('demo').onclick();
 nodes.get('addDetection').onclick();
 nodes.get('addDetection').onclick();
 const cards=nodes.get('flowNodes');
 setDetection(cards.children[0],{pieces:'1-2'});
 setDetection(cards.children[1],{mode:'manual_piece_3d'});
 await nodes.get('startWorkflow').onclick();
 assert.equal(requests.length,1);
 assert.match(nodes.get('progress').textContent,/节点 1.*synthetic request failure/);
});

test('failed detection results are collected without delaying later submissions',async()=>{
 const {nodes,requests,groups}=await boot(null,{},q=>q.mode==='manual_2d');
 nodes.get('demo').onclick();
 nodes.get('addDetection').onclick();
 nodes.get('addDetection').onclick();
 setDetection(nodes.get('flowNodes').children[1],{mode:'manual_piece_3d'});
 await nodes.get('startWorkflow').onclick();
 assert.deepEqual(requests.map(q=>q.mode),['manual_2d','manual_piece_3d']);
 assert.equal(groups[0].outcome,'failed');
 assert.match(groups[0].error,/节点 1.*synthetic request failure/);
});

test('another node can be clicked while a submitted result is pending',async()=>{
 let releaseResults;
 const results=new Promise(resolve=>{releaseResults=resolve;});
 const {nodes,requests}=await boot(null,{},()=>false,{}, {beforePoll:()=>results});
 nodes.get('demo').onclick();
 nodes.get('addDetection').onclick();
 nodes.get('addDetection').onclick();
 const cards=nodes.get('flowNodes');
 setDetection(cards.children[1],{pieces:'2'});
 const first=execute(cards.children[0]).onclick();
 await new Promise(setImmediate);
 assert.equal(execute(cards.children[0]).disabled,true);
 assert.equal(execute(cards.children[1]).disabled,false);
 const second=execute(cards.children[1]).onclick();
 await new Promise(setImmediate);
 assert.deepEqual(requests.map(q=>q.piece_id),[1,2]);
 releaseResults();
 await Promise.all([first,second]);
});

test('a named setup survives reload, restores its node parameters and sends no requests',async()=>{
 const localStorage=browserStorage();
 const first=await boot(null,{},()=>false,{localStorage});
 const {nodes,requests}=first;
 nodes.get('demo').onclick();
 const cards=nodes.get('flowNodes');
 nodes.get('addDetection').onclick();setDetection(cards.children[0],{pieces:'2,1'});
 nodes.get('addDelay').onclick();
 const delay=cards.children[1].children[1].children[0];delay.value='725';delay.oninput();
 nodes.get('addDetection').onclick();
 setDetection(cards.children[2],{mode:'manual_piece_3d',pieces:'2'});
 assert.equal(nodes.get('saveWorkflow').disabled,true);
 const name=nodes.get('workflowName');name.value='A 板逐片复核';name.oninput();
 nodes.get('saveWorkflow').onclick();
 const key='validation-lab.workflow-templates.v1';
 assert.deepEqual(JSON.parse(localStorage.getItem(key)),[{name:'A 板逐片复核',nodes:[
  {mode:'manual_2d',board_id:'2',piece_spec:'2,1'},
  {mode:'delay',delay_ms:'725'},
  {mode:'manual_piece_3d',board_id:'2',piece_spec:'2'}
 ]}]);
 nodes.get('clearFlow').onclick();
 assert.equal(cards.children.length,0);
 assert.equal(requests.length,0);
 const reloaded=await boot(null,{},()=>false,{localStorage});
 const page=reloaded.nodes;
 const saved=page.get('savedWorkflow');
 assert.equal(saved.children[1].value,'A 板逐片复核');
 saved.value='A 板逐片复核';saved.onchange();
 page.get('loadWorkflow').onclick();
 const [twoD,recoveredDelay,threeD]=page.get('flowNodes').children;
 assert.equal(twoD.children[1].children[2].children[0].value,'2,1');
 assert.equal(recoveredDelay.children[1].children[0].value,'725');
 assert.match(threeD.children[0].textContent,/单片 3D/);
 assert.equal(threeD.children[2].children[0].textContent,'待执行');
 assert.equal(page.get('startWorkflow').disabled,true);
 page.get('demo').onclick();
 assert.equal(page.get('startWorkflow').disabled,false);
 assert.equal(reloaded.requests.length,0);
});

test('declining a template overwrite or replacement preserves both saved and unsaved setups',async()=>{
 const localStorage=browserStorage();
 const {nodes}=await boot(null,{},()=>false,{localStorage,confirm:()=>false});
 nodes.get('demo').onclick();nodes.get('addDetection').onclick();
 const name=nodes.get('workflowName');
 name.value='复核';name.oninput();nodes.get('saveWorkflow').onclick();
 const key='validation-lab.workflow-templates.v1',previous=localStorage.getItem(key);
 setDetection(nodes.get('flowNodes').children[0],{pieces:'2'});
 nodes.get('saveWorkflow').onclick();
 assert.equal(localStorage.getItem(key),previous);
 nodes.get('loadWorkflow').onclick();
 assert.equal(nodes.get('flowNodes').children[0].children[1].children[2].children[0].value,'2');
});

test('workflow JSON uses the save picker and can be loaded in a different browser profile',async()=>{
 let selected,content='',closed=false;
 const browser={showSaveFilePicker:async options=>{
  selected=options;
  return {name:'选定流程.json',createWritable:async()=>({
   write:async data=>{content=data;},close:async()=>{closed=true;},
  })};
 }};
 const {nodes,requests}=await boot(null,{},()=>false,browser);
 nodes.get('demo').onclick();nodes.get('addDetection').onclick();nodes.get('addDelay').onclick();
 setDetection(nodes.get('flowNodes').children[0],{pieces:'1-2'});
 nodes.get('workflowName').value='逐片流程';
 await nodes.get('saveWorkflowFile').onclick();
 assert.equal(selected.suggestedName,'流程-逐片流程.json');
 assert.equal(closed,true);
 assert.deepEqual(JSON.parse(content),{format:'validation-lab-workflow',version:1,name:'逐片流程',nodes:[
  {mode:'manual_2d',board_id:'2',piece_spec:'1-2'},{mode:'delay',delay_ms:'1000'}
 ]});
 assert.equal(requests.length,0);
 const other=await boot(),file=other.nodes.get('workflowFileInput');
 await file.onchange({target:{value:'chosen.json',files:[{text:async()=>content}]}});
 assert.equal(other.nodes.get('workflowName').value,'逐片流程');
 assert.equal(other.nodes.get('flowNodes').children.length,2);
 assert.equal(other.nodes.get('flowNodes').children[0].children[1].children[2].children[0].value,'1-2');
 assert.equal(other.requests.length,0);
 await file.onchange({target:{value:'report.json',files:[{text:async()=>'{"mode":"workflow","runs":[]}' }]}});
 assert.match(other.nodes.get('progress').textContent,/请选择验证台导出的流程 JSON/);
 assert.equal(other.nodes.get('flowNodes').children.length,2);
});
