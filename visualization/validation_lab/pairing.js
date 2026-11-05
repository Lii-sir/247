(function(root){
 'use strict';
 function board(value){const s=String(value??'').trim();return /^[0-9]+$/.test(s)?s.replace(/^0+(?=\d)/,''):s;}
 function boards(files){return [...new Set(files.map(r=>board(r.board_id)).filter(Boolean))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
 function check(files,id,mode,count){
  id=board(id);if(!id)return {ok:false,message:'请选择验证板号'};
  const rows=files.filter(r=>board(r.board_id)===id),images=rows.filter(r=>r.kind==='2d'),depths=rows.filter(r=>r.kind==='3d');
  if(!images.length)return {ok:false,message:'该板没有2D片图'};
  const used=new Set();for(const r of images){
   if(!Number.isInteger(r.piece_id)||r.piece_id<1||r.piece_id>count)return {ok:false,message:'片号须为1～'+count+'的整数：'+r.name};
   if(used.has(r.piece_id))return {ok:false,message:'第'+r.piece_id+'片重复，请修改编号或移除重复图'};used.add(r.piece_id);
  }
  if(mode==='joint'){
   const missing=Array.from({length:count},(_,i)=>i+1).filter(i=>!used.has(i));
   if(missing.length)return {ok:false,message:'缺少2D片号：'+missing.join('、')};
   if(depths.length!==1)return {ok:false,message:'每板须恰好一张整板深度图，目前'+depths.length+'张'};
  }
  return {ok:true,message:'板 '+id+' · '+images.length+'张2D'+(mode==='joint'?' + 1张整板3D · 配对齐全':' · 仅验证这些片的2D结果')};
 }
 const api={board,boards,check};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.Pairing=api;
})(typeof globalThis!=='undefined'?globalThis:this);
