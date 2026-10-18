/* Execute the real viewer script against a minimal DOM/canvas test double. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'viewer.js'), 'utf8');
const template = fs.readFileSync(path.join(root, 'viewer.html'), 'utf8');

function createViewer(payload) {
  const texts = [], strokes = [], transforms = [];
  const context = new Proxy({
    measureText: (value) => ({width: String(value).length * 7}),
    fillText(text, x, y) { texts.push({text, color: this.fillStyle, x, y}); },
    stroke() { strokes.push({color: this.strokeStyle, width: this.lineWidth}); },
    scale(x, y) { transforms.push([x, y]); },
  }, {get(target, key) { return key in target ? target[key] : () => {}; }});
  class Element {
    constructor(id='') { this.id=id; this.children=[]; this.dataset={}; this.style={}; this.listeners={}; this.textContent=''; this.className=''; }
    appendChild(child) { this.children.push(child); }
    append(...children) { this.children.push(...children); }
    replaceChildren() { this.children=[]; this.innerHTML=''; }
    setAttribute(key, value) { this[key]=value; }
    addEventListener(type, handler) { this.listeners[type]=handler; }
    getBoundingClientRect() { return {width: 850, height: 470, left: 0, top: 0}; }
    getContext() { return context; }
    scrollIntoView() { this.scrolled=true; }
    click() { this.listeners.click?.({target:this}); }
  }
  const nodes = Object.fromEntries([...template.matchAll(/id="([^"]+)"/g)].map((match) => [match[1],new Element(match[1])]));
  nodes['report-data'].textContent=JSON.stringify(payload);
  vm.runInNewContext(source, {
    document: {getElementById(id) { assert.ok(nodes[id], `Missing template element: ${id}`); return nodes[id]; },createElement: () => new Element()},
    window: {devicePixelRatio:1,addEventListener() {}},
    Image: class { constructor() { this.complete=true; } },
  });
  return {nodes,texts,strokes,transforms};
}
const geometry = {bounding_box:[30,40,130,60],contours:[[[30,40],[130,40],[130,60],[30,60]]]};
function wire(id, status='ok') {
  return {instance_id:id,class_name:'wire',mask_2d:geometry,bending_2d:{
    is_valid:!['insufficient_intersections','empty_mask'].includes(status),is_ok:status==='ok',status,
    mask_ratio:status==='ok'?.95:.25,rectangle:[[30,40],[130,40],[130,60],[30,60]],
  }};
}
function payload(wires, extra={}) {
  return {response:{ok:true},pieces:[{piece_id:1,image_2d_size:[200,150],results_2d:{wire_bending_2d:{measurements:wires}},...extra}]};
}
function rows(view) {return view.nodes.measurements.children.filter((row)=>row.dataset.kind==='wire');}

test('2D-only results expose per-wire OK/NG, coverage and cause',()=>{
  const view=createViewer(payload([wire(1),wire(2,'bending_out_of_range'),wire(3,'insufficient_intersections'),wire(4,'outside_rois')]));
  assert.equal(rows(view).length,4);
  assert.equal(rows(view)[0].children[2].textContent,'OK');
  assert.equal(rows(view)[1].children[2].textContent,'NG');
  assert.equal(rows(view)[1].children[3].textContent,'25.0%');
  assert.equal(rows(view)[1].children[4].textContent,'覆盖率低于阈值');
  assert.equal(rows(view)[2].children[2].textContent,'NG · 无效');
  assert.equal(rows(view)[2].children[3].textContent,'—');
  assert.equal(rows(view)[2].children[4].textContent,'交点不足');
  assert.equal(rows(view)[3].children[2].textContent,'ROI 外');
  assert.match(view.nodes['wire-bending-summary'].textContent,/OK 1 · NG 2 · ROI 外 1/);
});

test('2D wires omitted from joint heights are retained and heights are merged by ID',()=>{
  const view=createViewer(payload([wire(1),wire(2,'bending_out_of_range')],{
    wire_measurements:[{...wire(1),height_3d:{height_diff_mm:.42,is_valid:true,is_ok:true}}],
  }));
  assert.equal(rows(view).length,2);
  assert.equal(rows(view)[0].children[5].textContent,'0.4200');
  assert.equal(rows(view)[1].children[5].textContent,'—');
});

test('canvas draws red/green verdict labels and evaluation outlines',()=>{
  const view=createViewer(payload([wire(1),wire(2,'bending_out_of_range')]));
  assert.ok(view.texts.some((item)=>item.text==='wire #1 OK · 95.0%' && item.color==='#65d8b1'));
  assert.ok(view.texts.some((item)=>item.text==='wire #2 NG · 25.0%' && item.color==='#ff6b7a'));
  assert.ok(view.strokes.some((item)=>item.color==='#ff6b7a' && item.width>0));
  const count=view.texts.length;
  view.nodes['show-bending'].listeners.change({target:{checked:false}});
  assert.ok(!view.texts.slice(count).some((item)=>item.text.includes(' NG · ')));
});

test('clicking an NG wire highlights it, zooms the image and scrolls to the canvas',()=>{
  const view=createViewer(payload([wire(1),wire(2,'bending_out_of_range')]));
  const count=view.transforms.length;
  rows(view)[1].children[0].children[0].click();
  assert.match(rows(view)[1].className,/selected/);
  assert.equal(view.nodes.canvas2d.scrolled,true);
  assert.ok(view.transforms.length>count);
});

test('a missing wire has its own ROI NG label and summary count',()=>{
  const view=createViewer(payload([], {wire_roi_checks:[{roi_id:1,status:'wire_missing',box:[20,20,150,80]}]}));
  assert.match(view.nodes['wire-bending-summary'].textContent,/缺线 ROI 1/);
  assert.ok(view.texts.some((item)=>item.text==='ROI #1 NG · 缺线'));
});

test('image-size mismatch prevents canvas overlays while retaining table verdicts',()=>{
  const view=createViewer(payload([wire(1)],{overlay_2d_valid:false}));
  assert.equal(rows(view)[0].children[2].textContent,'OK');
  assert.ok(!view.texts.some((item)=>item.text.includes('wire #1 OK')));
});

test('legacy joint-only reports still show bending and missing data stays unknown',()=>{
  const view=createViewer({response:{ok:true},pieces:[{piece_id:1,wire_measurements:[wire(1),{instance_id:2,class_name:'thin',mask_2d:geometry}]}]});
  assert.equal(rows(view).length,2);
  assert.equal(rows(view)[1].children[2].textContent,'未提供');
  assert.equal(rows(view)[1].children[3].textContent,'—');
});

test('ROI-inset silver boundaries show their source instead of template registration',()=>{
  const view=createViewer({response:{ok:true},pieces:[{piece_id:1,results_2d:{silver_overflow_2d:{
    is_ok:true,status:'ok',boundary:{mode:'roi_inset',coordinate_system:'image',points:[[20,20],[150,20],[150,80],[20,80]]},
    registration:{method:'roi_inset',status:'ok',is_valid:true,fallback_used:false},
  }}}]});
  assert.ok(view.texts.some((item)=>item.text==='ROI 内缩边界'));
  assert.ok(!view.texts.some((item)=>item.text.includes('calibrated points')));
  assert.ok(view.nodes.checks.children.some((card)=>card.children[0].textContent==='银浆边界生成'));
});

module.exports={createViewer,rows};
