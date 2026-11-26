// Run the real viewer renderer with a minimal DOM/canvas to verify plot coordinates.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'viewer.js'), 'utf8');

function loadViewer(data) {
  const nodes = new Map();
  const arcs = [];
  const makeNode = (id = '') => ({
    id, textContent: '', style: {}, dataset: {}, children: [],
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); },
    replaceChildren(...items) { this.children = items; },
    handlers: {}, addEventListener(name,handler) { this.handlers[name]=handler; }, setAttribute() {},
    getBoundingClientRect() { return {width: 600, height: 400}; },
    getContext() {
      return new Proxy({
        measureText(text) { return {width: text.length * 6}; },
        arc(x, y) { if (id === 'canvas3d') arcs.push([x, y]); },
      }, {get(target, key) { return key in target ? target[key] : () => {}; }});
    },
  });
  const document = {
    body: {dataset:{}},
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, makeNode(id));
      return nodes.get(id);
    },
    createElement() { return makeNode(); },
  };
  document.getElementById('report-data').textContent = JSON.stringify(data);
  vm.runInNewContext(source, {document, window: {devicePixelRatio: 1, addEventListener() {}}});
  return {arcs,nodes};
}
function plotted(piece) { return loadViewer({response:{ok:true},pieces:[piece],warnings:[]}).arcs; }

function piece(space, row = 1000) {
  return {piece_id: 2, depth_start_row: row, depth_end_row: row + 200,
    height_point_coordinate_space: space, depth_origin_row: row,
    wire_measurements: [{instance_id: 1, height_3d: {
      point: {x: 75, y: ['board_3d','pitch_corrected_board_3d'].includes(space) ? row + 31 : 31}, is_ok: true}}],
    solder_measurements: [],
  };
}

assert.deepEqual(plotted(piece('board_3d')), [[75, 31]], 'Full-board point is plotted relative to preview');
assert.deepEqual(plotted(piece(undefined)), [[75, 31]], 'Legacy point stays local');
assert.deepEqual(plotted(piece('board_3d', 0)), [[75, 31]], 'First piece at row zero');
const tagged = piece(undefined);
tagged.wire_measurements[0].height_3d.point = {x: 75, y: 1031};
tagged.wire_measurements[0].height_3d.point_coordinate_space = 'board_3d';
assert.deepEqual(plotted(tagged), [[75, 31]], 'Height-specific tag survives C# response mapping');
const absent = piece('board_3d');
absent.wire_measurements[0].height_3d.point = null;
assert.deepEqual(plotted(absent), [], 'Invalid height has no fabricated point');
assert.deepEqual(plotted(piece('pitch_corrected_board_3d')), [[75,31]], 'Corrected full-board point is plotted relative to corrected preview');
console.log('6 real-viewer coordinate checks passed');

const done={status:'done',is_ok:true};
const board=(board_id,pieces)=>({board_id,pieces,response:{ok:true},warnings:[]});
const {nodes}=loadViewer({mode:'workflow',title:'Test flow',response:{mode:'workflow',total_wall_ms:1750},boards:[
  board('A',[
    {piece_id:1,workflow_stages:{'2d':done,'3d':done}},
    {piece_id:2,workflow_stages:{'2d':done,'3d':{status:'skipped',reason:'2d_ng'}}},
  ]),
  board('B',[{piece_id:1,workflow_stages:{'3d':done}}])
]});
assert.equal(nodes.get('panel-2d').hidden,false);
assert.equal(nodes.get('panel-3d').hidden,false);
assert.equal(nodes.get('elapsed-label').textContent,'流程总耗时');
assert.equal(nodes.get('elapsed').textContent,'1.75 秒');
nodes.get('pieces').children[1].handlers.click();
assert.equal(nodes.get('panel-2d').hidden,false);
assert.equal(nodes.get('panel-3d').hidden,true);
assert.match(nodes.get('piece-stage-warnings').textContent,/3D 跳过/);
nodes.get('workflow-board').value='B';
nodes.get('workflow-board').handlers.change();
assert.equal(nodes.get('panel-2d').hidden,true);
assert.equal(nodes.get('panel-3d').hidden,false);
assert.equal(nodes.get('piece-stage-warnings').hidden,true);
assert.equal(nodes.get('elapsed').textContent,'1.75 秒');
console.log('Workflow board/piece switches and stage visibility passed');
