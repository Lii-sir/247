(function(root){
 'use strict';
 function board(value){const s=String(value??'').trim();return /^[0-9]+$/.test(s)?s.replace(/^0+(?=\d)/,''):s;}
 function boards(files){return [...new Set(files.map(r=>board(r.board_id)).filter(Boolean))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));}
 function check(files,id,count){
  id=board(id);if(!id)return {ok:false,message:'请选择验证板号'};
  const images=files.filter(r=>board(r.board_id)===id&&r.kind==='2d');
  if(!images.length)return {ok:false,message:'该板没有2D片图'};
  const used=new Set();for(const r of images){
   if(!Number.isInteger(r.piece_id)||r.piece_id<1||r.piece_id>count)return {ok:false,message:'片号须为1～'+count+'的整数：'+r.name};
   if(used.has(r.piece_id))return {ok:false,message:'第'+r.piece_id+'片重复，请修改编号或移除重复图'};used.add(r.piece_id);
  }
  return {ok:true,message:'板 '+id+' · '+images.length+' 张 2D'};
 }
 function checkManual(files,id,mode,piece,count){
  if(!['manual_2d','manual_piece_3d','pitch_piece_3d'].includes(mode))return {ok:false,message:'未知检测方式'};
  id=board(id);
  if(!id)return {ok:false,message:'请选择板号'};
  if(!Number.isInteger(piece)||piece<1||piece>count)return {ok:false,message:'请选择有效片号'};
  const rows=files.filter(r=>board(r.board_id)===id);
  const images=rows.filter(r=>r.kind==='2d'&&r.piece_id===piece);
  if(images.length>1)return {ok:false,message:'该片有重复的2D文件'};
  if((mode==='manual_2d'||mode==='pitch_piece_3d')&&images.length!==1)return {ok:false,message:'该片需要一张2D图片'};
  if(mode!=='manual_2d'&&rows.filter(r=>r.kind==='3d').length!==1)
   return {ok:false,message:'单片3D需要一张整板深度图'};
  return {ok:true,message:''};
 }
 function checkPitch(files,id){
  if(!id)return {ok:false,message:'请选择验证板号'};
  if(files.filter(row=>row.kind==='3d'&&board(row.board_id)===id).length!==1)
   return {ok:false,message:'启动 Pitch 需要一张整板深度图'};
  return {ok:true,message:''};
 }
 function pieceNumbers(value,count){
  const ids=new Set();
  for(const part of String(value??'').trim().split(/[,，]/)){
   const range=/^(\d+)(?:\s*-\s*(\d+))?$/.exec(part.trim());
   if(!range)return [];
   const first=Number(range[1]),last=Number(range[2]??range[1]);
   if(first<1||last<first||last>count)return [];
   for(let n=first;n<=last;n++)ids.add(n);
  }
  return [...ids];
 }
 const api={board,boards,check,checkManual,checkPitch,pieceNumbers};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.Pairing=api;
})(typeof globalThis!=='undefined'?globalThis:this);
