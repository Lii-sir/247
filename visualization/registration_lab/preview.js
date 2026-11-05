/* Preview response contract shared by rendering and regression tests. */
(function(root){
 'use strict';
 const restart='请关闭旧的测试台命令窗口，重新双击“启动配准测试台.cmd”，再刷新页面。';
 function validateServer(data){
  if(data.preview_version!==5)throw Error('检测到旧版配准服务，与矩阵复用界面版本不匹配。'+restart);
 }
 function loadImage(src,label,ImageType){
  if(typeof src!=='string'||!src.startsWith('data:image/png;base64,'))
   return Promise.reject(Error('服务未返回有效的'+label+'。'+restart));
  return new Promise((resolve,reject)=>{
   const img=new (ImageType||root.Image)();
   img.onload=()=>resolve(img);
   img.onerror=()=>reject(Error(label+'解码失败，请重新运行配准；若仍失败，请重启测试台。'));
   img.src=src;
  });
 }
 function describeResult(result){
  const inputs=result.inputs||{},c=result.correction;
  const name=result.display_image_2d||(inputs.image_2d||'2D图像').split(/[\\/]/).pop();
  const piece=inputs.depth_mode==='piece'?'单片深度':'第'+(inputs.piece_id??'?')+'片';
  let effect='补偿状态未记录';
  if(c?.applied)effect='Pitch已补偿 '+c.angle_deg+'°（'+(c.transform_direction==='document_forward'?'文档正向':'还原')+'）';
  else if(c?.reason==='already_sdk_corrected')effect='SDK已补偿 · 本工具已跳过';
  else if(c?.reason==='disabled')effect='未启用Pitch补偿';
  const alignment=result.registration_mode==='reuse'?' · 复用已保存矩阵（未重新评分）':'';
  return name+' ↔ '+piece+' · '+effect+alignment;
 }
 function validateMatrixProfile(profile){
  if(profile?.format!=='registration_lab_matrix'||profile.version!==1||!profile.context||
     !Array.isArray(profile.matrix)||profile.matrix.length!==3||
     profile.matrix.some(row=>!Array.isArray(row)||row.length!==3||row.some(v=>typeof v!=='number'||!Number.isFinite(v))))
   throw Error('不是有效的测试台矩阵文件，请载入“保存当前阶段矩阵”导出的JSON');
  if(profile.matrix[2].slice(0,2).some(v=>Math.abs(v)>1e-12))throw Error('透视变换已移除，请载入仿射矩阵');
  if(profile.context.target_space!=='depth_piece_local'||
     !['independent_2d_image','smartray_intensity_depth_piece'].includes(profile.context.source_space)||
     !profile.context.preprocessing||['image_size','depth_size'].some(k=>!Array.isArray(profile.context[k])||profile.context[k].length!==2||profile.context[k].some(v=>!Number.isInteger(v)||v<=0)))
   throw Error('矩阵文件的坐标或尺寸信息不完整，请重新保存');
  return profile;
 }
 function buildMatrixProfile(result,stageKey){
  const stage=result?.stages?.find(s=>s.key===stageKey);
  if(!stage||!result.matrix_context)throw Error('本结果缺少矩阵坐标信息，请重新运行后保存');
  const r=stage.registration;
  const profile={format:'registration_lab_matrix',version:1,matrix:r.homography_2d_to_3d,
   context:result.matrix_context,source:{stage_key:stage.key,stage_name:stage.name,
    image_2d:result.inputs?.image_2d,piece_id:result.inputs?.piece_id,
    algorithm_valid:stage.key==='affine'?r.affine_valid:r.is_valid}};
  validateMatrixProfile(profile);return JSON.parse(JSON.stringify(profile));
 }
 function drawAffineRegions(ctx,rois,scale,options={}){
  const colors=['#ffe16b','#64f2d3','#e6a5ff'];
  ctx.save();ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';
  rois.forEach(([x0,y0,x1,y1],index)=>{
   const id=options.regionIds?.[index]??index+1;
   const selected=options.selectedIds==null||options.selectedIds.includes(id);
   const color=colors[(id-1)%colors.length];
   ctx.globalAlpha=selected?1:.55;
   ctx.setLineDash(selected?[]:[7/scale,5/scale]);
   ctx.strokeStyle='#10242b';ctx.lineWidth=4/scale;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
   ctx.strokeStyle=color;ctx.lineWidth=1.5/scale;ctx.strokeRect(x0,y0,x1-x0,y1-y0);
   ctx.fillStyle='#10242be6';ctx.fillRect(x0,y0,Math.min(x1-x0,(selected?38:105)/scale),20/scale);
   ctx.fillStyle=color;ctx.font=`bold ${12/scale}px sans-serif`;
   ctx.textBaseline='top';ctx.fillText('R'+id+(selected?'':' 未参与'),x0+5/scale,y0+4/scale);
  });
  ctx.restore();
 }
 const api={validateServer,loadImage,describeResult,drawAffineRegions,validateMatrixProfile,buildMatrixProfile};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 else root.LabPreview=api;
})(typeof globalThis!=='undefined'?globalThis:this);
