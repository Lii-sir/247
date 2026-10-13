// Run the real viewer renderer with a minimal DOM/canvas to verify plot coordinates.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'viewer.js'), 'utf8');

function plotted(piece) {
  const nodes = new Map();
  const arcs = [];
  const makeNode = (id = '') => ({
    id, textContent: '', style: {}, children: [],
    append(...items) { this.children.push(...items); },
    appendChild(item) { this.children.push(item); },
    replaceChildren(...items) { this.children = items; },
    addEventListener() {}, setAttribute() {},
    getBoundingClientRect() { return {width: 600, height: 400}; },
    getContext() {
      return new Proxy({
        measureText(text) { return {width: text.length * 6}; },
        arc(x, y) { if (id === 'canvas3d') arcs.push([x, y]); },
      }, {get(target, key) { return key in target ? target[key] : () => {}; }});
    },
  });
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, makeNode(id));
      return nodes.get(id);
    },
    createElement() { return makeNode(); },
  };
  document.getElementById('report-data').textContent = JSON.stringify({
    response: {ok: true}, pieces: [piece], warnings: [],
  });
  vm.runInNewContext(source, {document, window: {devicePixelRatio: 1, addEventListener() {}}});
  return arcs;
}

function piece(space, row = 1000) {
  return {piece_id: 2, depth_start_row: row, depth_end_row: row + 200,
    height_point_coordinate_space: space, depth_origin_row: row,
    wire_measurements: [{instance_id: 1, height_3d: {
      point: {x: 75, y: space === 'board_3d' ? row + 31 : 31}, is_ok: true}}],
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
console.log('5 real-viewer coordinate checks passed');
