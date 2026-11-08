const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const dir=path.join(__dirname,'../validation_lab'),html=fs.readFileSync(path.join(dir,'index.html'),'utf8');
async function boot(){
 const nodes=new Map(),requests=[],history=[];
 class Node{
  constructor(){this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.children=[];}
  set id(v){this._id=v;nodes.set(v,this);}get id(){return this._id;}
  append(...c){this.children.push(...c);}replaceChildren(...c){this.children=c;}setAttribute(k,v){this[k]=v;}
 }
 for(const m of html.matchAll(/<[a-z]+\b([^>]*\bid="([^"]+)"[^>]*)>/g)){const n=new Node();n.id=m[2];n.hidden=/\bhidden\b/.test(m[1]);n.disabled=/\bdisabled\b/.test(m[1]);}
 for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>(.*?)<\/select>/gs))nodes.get(m[1]).value=m[2].match(/<option\b[^>]*value="([^"]*)"/)?.[1]||'';
 const config={piece_count:2,samples:[{kind:'2d',name:'2_1.png',path:'/2_1.png',board_id:'2',piece_id:1},{kind:'2d',name:'2_2.png',path:'/2_2.png',board_id:'2',piece_id:2},{kind:'3d',name:'2.png',path:'/2.png',board_id:'2',piece_id:null}],pitch:{enabled_3d:false}};
 const ctx=vm.createContext({console,structuredClone,setTimeout:fn=>setImmediate(fn),document:{getElementById:id=>{if(!nodes.has(id))throw Error(id);return nodes.get(id);},createElement:()=>new Node()},
  fetch:async(url,opt)=>{
   let data;
   if(url==='/api/defaults')data=config;
   else if(url==='/api/history')data=history;
   else if(url.startsWith('/api/upload?')){const kind=new URL(url,'http://local').searchParams.get('kind'),name=opt.body.name,m=name.match(/^(.+)_(\d+)\./);data={kind,name,path:'/uploads/'+name,board_id:kind==='3d'?name.split('.')[0]:m?.[1]||'',piece_id:kind==='2d'&&m?Number(m[2]):null};}
   else if(url==='/api/run'){const q=JSON.parse(opt.body);requests.push(q);const id='a'.repeat(32);history.unshift({id,board_id:q.board_id,status:'done',result:{ok:true,product_is_ok:false,board_id:q.board_id,pieces:2,total_wall_ms:1000,output_dir:'/output',model_reused:false,submission_mode:q.submission_mode,request_interval_ms:q.request_interval_ms}});data={id};}
   else if(url.startsWith('/api/jobs/'))data=history[0];
   else throw Error(url);
   return {ok:true,json:async()=>data};
  }});
 vm.runInContext(fs.readFileSync(path.join(dir,'pairing.js'),'utf8'),ctx);vm.runInContext(fs.readFileSync(path.join(dir,'app.js'),'utf8'),ctx);await new Promise(setImmediate);
 return {nodes,requests,config};
}
test('demo -> validation passes exact board/piece manifest and opens interactive report',async()=>{
 const {nodes,requests}=await boot();assert.equal(nodes.get('run').disabled,true);nodes.get('demo').onclick();
 assert.equal(nodes.get('run').disabled,false);await nodes.get('run').onclick();
 assert.equal(requests[0].board_id,'2');assert.deepEqual(requests[0].files.filter(f=>f.kind==='2d').map(f=>f.piece_id),[1,2]);
 assert.equal(nodes.get('report').hidden,false);assert.match(nodes.get('report').src,/report.html$/);assert.match(nodes.get('summary').textContent,/NG/);
 assert.match(nodes.get('jsonLink').href,/result.json$/);
});
test('manual duplicate piece blocks submission and partial2D mode permits missing piece',async()=>{
 const {nodes,requests}=await boot();nodes.get('demo').onclick();
 const input=nodes.get('files').children[1].children[2].children[0];input.value='1';input.oninput();
 assert.equal(nodes.get('run').disabled,true);await nodes.get('run').onclick();assert.equal(requests.length,0);
 nodes.get('demo').onclick();nodes.get('files').children[1].children[3].children[0].onclick();
 assert.equal(nodes.get('run').disabled,true);nodes.get('mode').value='2d';nodes.get('mode').onchange();
 assert.equal(nodes.get('run').disabled,false);await nodes.get('run').onclick();assert.equal(requests[0].mode,'2d');
});
test('batch file upload displays original names and permits manual board editing',async()=>{
 const {nodes}=await boot();
 await nodes.get('upload2').onchange({target:{files:[{name:'2_1.png'},{name:'2_2.png'}],value:'files'}});
 await nodes.get('upload3').onchange({target:{files:[{name:'2.png'}],value:'files'}});
 assert.equal(nodes.get('run').disabled,false);assert.equal(nodes.get('files').children[0].children[0].textContent,'2_1.png');
 const board=nodes.get('files').children[0].children[1].children[0];board.value='3';board.oninput();
 assert.match(nodes.get('counts').textContent,/2个已编号板号/);
 nodes.get('board').value='2';nodes.get('board').onchange();assert.equal(nodes.get('run').disabled,true);
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
