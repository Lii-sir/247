(function () {
  "use strict";

  const data = JSON.parse(document.getElementById("report-data").textContent);
  const state = {
    selected: Number(data.pieces?.[0]?.piece_id || 1),
    showMasks: true,
    showRois: true,
    showPoints: true,
    showLabels: true,
    highlight: null,
    views: {
      "canvas2d": { scale: 1, x: 0, y: 0, fitted: false, drag: null },
      "canvas3d": { scale: 1, x: 0, y: 0, fitted: false, drag: null },
    },
  };

  const $ = (id) => document.getElementById(id);
  const pieces = () => data.pieces || [];
  const selectedPiece = () => pieces().find((piece) => Number(piece.piece_id) === state.selected) || pieces()[0] || null;
  const number = (value, digits = 2) => value == null || typeof value === "boolean" || value === "" || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
  const statusText = (value) => value === true ? "OK" : value === false ? "NG" : "未提供";
  const statusClass = (value) => value === true ? "status-ok" : value === false ? "status-ng" : "status-unknown";
  const boardResult = data.response?.results_board || {};
  const result2dOfPiece = (piece) => piece?.results_2d || {};
  const classColor = (name) => {
    const key = String(name || "").toLowerCase();
    if (key === "chip") return "#87abff";
    if (key === "wire") return "#4ed3ca";
    if (key === "thin") return "#f4cb76";
    if (key === "bond") return "#ffa588";
    if (key === "silver") return "#c291e9";
    return "#a7b5c7";
  };

  function setText(id, value, className) {
    const node = $(id);
    node.textContent = value;
    node.className = className || "";
  }

  function updateOverview() {
    const response = data.response || {};
    setText("report-title", data.title || "检测结果复核");
    setText("board-id", String(data.board_id || "未提供板号"));
    setText("service-status", statusText(response.ok), statusClass(response.ok));
    const verdict = boardResult.is_ok ?? response.results_2d?.is_ok ?? response.results_3d?.is_ok;
    $("verdict-label").textContent = response.results_board ? "算法整板结论" : "当前响应结论";
    setText("board-status", statusText(verdict), statusClass(verdict));
    setText("registration-status", statusText(boardResult.registration_ok), statusClass(boardResult.registration_ok));
    setText("elapsed", response.elapsed_ms == null ? "—" : `${number(response.elapsed_ms, 1)} ms`);
    $("piece-count").textContent = `${pieces().length} 片 · 点击片号查看细节`;
    const warningText = (data.warnings || []).join("\n");
    const notice = $("notice");
    notice.hidden = !warningText;
    notice.textContent = warningText;
    const error = $("error");
    error.hidden = !!response.ok || !response.error;
    error.textContent = response.error ? `${response.error}${response.message ? `\n${response.message}` : ""}` : "";
  }

  function renderPieceButtons() {
    const root = $("pieces");
    root.replaceChildren();
    for (const piece of pieces()) {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-current", String(Number(piece.piece_id) === state.selected));
      const id = document.createElement("strong"); id.textContent = `#${piece.piece_id}`;
      const verdict = document.createElement("span"); verdict.className = statusClass(piece.is_ok); verdict.textContent = statusText(piece.is_ok);
      const score = document.createElement("span"); score.textContent = `配准 ${number(piece.registration?.score, 3)}`;
      button.append(id, verdict, score);
      button.addEventListener("click", () => {
        state.selected = Number(piece.piece_id);
        state.highlight = null;
        state.views.canvas2d.fitted = false;
        state.views.canvas3d.fitted = false;
        renderAll();
      });
      root.appendChild(button);
    }
  }

  function polygon(ctx, contour, color, lineWidth = 1.6, fill = false) {
    if (!Array.isArray(contour) || contour.length < 2) return;
    ctx.beginPath();
    contour.forEach((point, index) => {
      const x = Number(point?.[0]), y = Number(point?.[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth / (ctx.viewerScale || 1);
    ctx.stroke();
    if (fill) {
      ctx.fillStyle = `${color}26`;
      ctx.fill();
    }
  }

  function box(ctx, values, color, lineWidth = 1.4) {
    if (!Array.isArray(values) || values.length < 4) return;
    const [x1, y1, x2, y2] = values.map(Number);
    if (![x1, y1, x2, y2].every(Number.isFinite)) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth / (ctx.viewerScale || 1);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  function drawGeometry(ctx, geometry, color, selected = false) {
    if (!geometry) return;
    const contours = geometry.contours || [];
    contours.forEach((contour) => polygon(ctx, contour, color, selected ? 3 : 1.5, selected));
    if (!contours.length) box(ctx, geometry.bounding_box, color, selected ? 3 : 1.5);
  }

  function drawText(ctx, x, y, text, color) {
    if (!state.showLabels) return;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(1 / (ctx.viewerScale || 1), 1 / (ctx.viewerScale || 1));
    ctx.font = "12px Segoe UI, Microsoft YaHei, sans-serif";
    ctx.fillStyle = "rgba(6, 10, 16, .82)";
    const width = ctx.measureText(text).width + 8;
    ctx.fillRect(0, -15, width, 18);
    ctx.fillStyle = color;
    ctx.fillText(text, 4, -2);
    ctx.restore();
  }

  function renderCanvas(canvas, piece, kind) {
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#070c13";
    ctx.fillRect(0, 0, rect.width, rect.height);

    const view = state.views[canvas.id];
    const preview = kind === "2d" ? piece?.preview_2d : piece?.preview_3d;
    const crop = piece?.crop_2d || result2dOfPiece(piece).crop_box || [];
    const fallbackWidth = kind === "2d" ? (piece?.image_2d_size?.[0] || crop[2] || 1600) : 1600;
    const fallbackHeight = kind === "2d" ? (piece?.image_2d_size?.[1] || crop[3] || 1000) : (piece?.depth_end_row && piece?.depth_start_row != null ? piece.depth_end_row - piece.depth_start_row : 1000);
    const sourceWidth = Number(preview?.width || fallbackWidth);
    const sourceHeight = Number(preview?.height || fallbackHeight);
    if (!view.fitted) {
      view.scale = Math.min((rect.width - 28) / sourceWidth, (rect.height - 28) / sourceHeight);
      view.scale = Math.max(0.005, Math.min(view.scale, 4));
      view.x = (rect.width - sourceWidth * view.scale) / 2;
      view.y = (rect.height - sourceHeight * view.scale) / 2;
      view.fitted = true;
    }
    ctx.save();
    ctx.translate(view.x, view.y);
    ctx.scale(view.scale, view.scale);
    ctx.viewerScale = view.scale;
    if (preview?.url) {
      if (!preview._image) {
        preview._image = new Image();
        preview._image.onload = () => { if (selectedPiece() === piece) renderCanvas(canvas, piece, kind); };
        preview._image.src = preview.url;
      }
      if (preview._image.complete) ctx.drawImage(preview._image, 0, 0, sourceWidth, sourceHeight);
    } else {
      ctx.strokeStyle = "#2a374a";
      ctx.strokeRect(0, 0, sourceWidth, sourceHeight);
      ctx.fillStyle = "#97a6ba";
      drawText(ctx, 18, 28, kind === "2d" ? "未加载 2D 原图，仅显示结果轮廓" : "未加载 3D 深度图，仅显示结果轮廓", "#97a6ba");
    }
    const masks = Array.isArray(piece?.masks) ? piece.masks : [];
    const instances = Array.isArray(result2dOfPiece(piece)?.instances) ? result2dOfPiece(piece).instances : [];
    const allMasks = masks.length ? masks : instances;
    const canOverlay = kind !== "2d" || piece?.overlay_2d_valid !== false;
    if (state.showMasks && canOverlay) {
      for (const item of allMasks) {
        const name = item.class_name || "instance";
        const id = Number(item.instance_id);
        const selected = state.highlight && state.highlight.id === id &&
          (state.highlight.kind === "mask" ||
            (state.highlight.kind === "wire" && ["wire", "thin"].includes(String(name).toLowerCase())) ||
            (state.highlight.kind === "solder" && ["thin", "bond"].includes(String(name).toLowerCase())));
        const color = classColor(name);
        drawGeometry(ctx, kind === "2d" ? item.mask_2d : item.mask_3d_local, color, selected);
        const geometry = kind === "2d" ? item.mask_2d : item.mask_3d_local;
        const point = geometry?.bounding_box;
        if (point) drawText(ctx, point[0], point[1], `${name} #${id}`, color);
      }
    }
    if (kind === "2d" && state.showRois && canOverlay) {
      box(ctx, crop, "#79b3fb", 1.5);
      const origin = crop.length ? crop : [0, 0];
      for (const roi of piece.wire_rois_2d || result2dOfPiece(piece).wire_rois_2d || []) {
        if (roi.length >= 4) box(ctx, [roi[0] + origin[0], roi[1] + origin[1], roi[2] + origin[0], roi[3] + origin[1]], "#65d8b1", 1.4);
      }
      const circle = result2dOfPiece(piece).circle;
      if (circle) {
        ctx.strokeStyle = "#f4cb76";
        ctx.lineWidth = 1.5 / view.scale;
        ctx.beginPath(); ctx.arc(Number(circle.x), Number(circle.y), Number(circle.radius), 0, Math.PI * 2); ctx.stroke();
      }
      const platform = piece?.reference_planes?.platform_roi;
      for (const values of platform?.boxes_2d || []) box(ctx, values, "#79b3fb", 1.5);
      const anomaly = piece?.chip_anomaly_2d || result2dOfPiece(piece).chip_anomaly_2d;
      for (const chip of anomaly?.chips || []) {
        for (const region of chip.anomaly_boxes || []) polygon(ctx, region.image_polygon, "#ff8d91", 2, true);
      }
      const overflow = result2dOfPiece(piece).silver_overflow_2d || piece?.silver_overflow_2d;
      for (const region of overflow?.measurement?.regions || []) {
        box(ctx, region.box_xyxy, "#ff4d5f", 3);
        drawText(ctx, region.box_xyxy?.[0] || 0, region.box_xyxy?.[1] || 0, `silver overflow ${number(region.area_px, 0)} px`, "#ff4d5f");
      }
    }
    if (kind === "3d" && state.showRois) {
      const platform = piece?.reference_planes?.platform_roi;
      for (const contour of platform?.polygons_3d_local || []) polygon(ctx, contour, "#79b3fb", 1.5);
    }
    if (kind === "3d" && state.showPoints) {
      const points = [];
      (piece.wire_measurements || []).forEach((item) => (item.height_3d?.point ? points.push({ ...item.height_3d.point, label: `wire #${item.instance_id}`, ok: item.height_3d.is_ok }) : null));
      (piece.solder_measurements || []).forEach((item) => (item.height_3d?.point ? points.push({ ...item.height_3d.point, label: `solder ${item.endpoint}`, ok: item.height_3d.is_ok }) : null));
      (piece.points || []).forEach((item, index) => points.push({ x: item.x, y: item.y, label: `point #${index + 1} ${number(item.height_diff_mm, 3)} mm`, ok: null }));
      for (const point of points) {
        const color = point.ok === true ? "#65d8b1" : point.ok === false ? "#ff8d91" : "#f4cb76";
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(Number(point.x), Number(point.y), 5 / Math.max(view.scale, 0.25), 0, Math.PI * 2); ctx.fill();
        drawText(ctx, Number(point.x) + 7, Number(point.y), point.label, color);
      }
    }
    ctx.restore();
  }

  function renderChecks(piece) {
    const root = $("checks");
    root.replaceChildren();
    const two = result2dOfPiece(piece);
    const checks = [
      ["片级结论", piece?.is_ok],
      ["配准", piece?.registration?.is_valid],
      ["引线 ROI", Array.isArray(piece?.wire_roi_checks) ? (piece.wire_roi_checks.length ? piece.wire_roi_checks.every((x) => x.is_ok === true) : null) : two.wire_bending_2d?.is_ok],
      ["芯片旋转", two.chip_rotation_2d?.is_ok],
      ["芯片外观", piece?.chip_anomaly_2d?.is_ok ?? two.chip_anomaly_2d?.is_ok],
      ["银浆连通", two.silver_continuity_2d?.is_ok],
      ["银浆溢出", two.silver_overflow_2d?.is_ok],
      ["引线高度", Array.isArray(piece?.wire_measurements) && piece.wire_measurements.length ? piece.wire_measurements.every((x) => x.height_3d?.is_ok === true) : null],
      ["焊脚高度", Array.isArray(piece?.solder_measurements) && piece.solder_measurements.length ? piece.solder_measurements.every((x) => x.height_3d?.is_ok === true) : null],
      ["配准分数", piece?.registration?.score == null ? "—" : number(piece.registration.score, 4)],
    ];
    checks.forEach(([name, value]) => {
      const card = document.createElement("div"); card.className = "check";
      const valueClass = typeof value === "boolean" || value == null ? statusClass(value) : "";
      const label = document.createElement("div"); label.textContent = name;
      const result = document.createElement("strong"); result.className = valueClass; result.textContent = typeof value === "boolean" || value == null ? statusText(value) : String(value);
      card.append(label, result);
      root.appendChild(card);
    });
  }

  function renderMeasurements(piece) {
    const root = $("measurements"); root.replaceChildren();
    const rows = [];
    (piece?.wire_measurements || []).forEach((item) => rows.push({ id: Number(item.instance_id), type: "wire", location: "引线", bend: item.bending_2d, height: item.height_3d }));
    (piece?.solder_measurements || []).forEach((item) => rows.push({ id: Number(item.source_instance_id), type: "solder", location: `${item.endpoint || "焊脚"} / ${item.location || "—"}`, bend: null, height: item.height_3d }));
    if (!rows.length) { root.innerHTML = `<tr><td colspan="7" class="muted">没有返回引线或焊脚测量结果</td></tr>`; return; }
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const ok = row.height?.is_ok;
      tr.className = state.highlight?.id === row.id && state.highlight.kind === row.type ? "selected" : "";
      const values = [
        `#${row.id}`,
        `${row.type} · ${row.location}`,
        row.bend ? `${statusText(row.bend.is_ok)} / ${number(row.bend.mask_ratio * 100, 1)}%` : "—",
        row.bend ? `${number(row.bend.mask_ratio * 100, 1)}%` : "—",
        row.height?.is_valid === false ? "无效" : number(row.height?.height_diff_mm, 4),
        row.height?.reference || "—",
        statusText(ok),
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        if (index === 0) {
          const button = document.createElement("button"); button.type = "button"; button.className = "row-button"; button.textContent = value;
          button.addEventListener("click", () => { state.highlight = { id: row.id, kind: row.type }; renderAll(); });
          cell.appendChild(button);
        } else {
          cell.textContent = value;
          if (index === 6) cell.className = statusClass(ok);
        }
        tr.appendChild(cell);
      });
      root.appendChild(tr);
    });
  }

  function renderDetails(piece) {
    $("detail-json").textContent = JSON.stringify({ registration: piece?.registration, reference_planes: piece?.reference_planes, timings: data.response?.timings }, null, 2);
    const two = result2dOfPiece(piece);
    $("diagnostics-json").textContent = JSON.stringify({ chip_rotation_2d: two.chip_rotation_2d || null,
      chip_anomaly_2d: piece?.chip_anomaly_2d || two.chip_anomaly_2d || null,
      silver_continuity_2d: two.silver_continuity_2d || null, silver_overflow_2d: two.silver_overflow_2d || null }, null, 2);
    $("raw-json").textContent = JSON.stringify(data.response, null, 2);
  }

  function renderAll() {
    updateOverview(); renderPieceButtons();
    const piece = selectedPiece();
    if (!piece) { $("piece-title").textContent = "没有片数据"; return; }
    $("piece-title").textContent = `片 ${piece.piece_id}`;
    $("piece-subtitle").textContent = `2D 尺寸 ${piece.image_2d_size?.join(" × ") || "—"} · 深度行 ${piece.depth_start_row ?? "—"}–${piece.depth_end_row ?? "—"} · 配准平移 ${number(piece.registration?.tx, 1)}, ${number(piece.registration?.ty, 1)}`;
    $("image2d-info").textContent = piece.preview_2d ? `${piece.preview_2d.width} × ${piece.preview_2d.height}` : "未加载原图";
    const range = piece.preview_3d?.gray_range;
    $("image3d-info").textContent = piece.preview_3d ? `${piece.preview_3d.width} × ${piece.preview_3d.height}${range ? ` · 灰度 ${number(range[0], 0)}–${number(range[1], 0)}` : ""}` : "未加载深度图";
    $("image2d-note").textContent = piece.overlay_2d_valid === false ? "原图尺寸与检测结果不一致，已停用叠加。" : "轮廓使用检测返回的原始像素坐标。";
    renderCanvas($("canvas2d"), piece, "2d"); renderCanvas($("canvas3d"), piece, "3d");
    renderChecks(piece); renderMeasurements(piece); renderDetails(piece);
  }

  function bindCanvas(canvas) {
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const view = state.views[canvas.id];
      const rect = canvas.getBoundingClientRect();
      const old = view.scale;
      const factor = event.deltaY < 0 ? 1.12 : 0.89;
      const next = Math.max(0.005, Math.min(old * factor, 12));
      const px = event.clientX - rect.left, py = event.clientY - rect.top;
      view.x = px - (px - view.x) * (next / old); view.y = py - (py - view.y) * (next / old); view.scale = next; view.fitted = true; renderAll();
    }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => { canvas.setPointerCapture(event.pointerId); state.views[canvas.id].drag = { x: event.clientX, y: event.clientY }; });
    canvas.addEventListener("pointermove", (event) => { const view = state.views[canvas.id]; if (!view.drag) return; view.x += event.clientX - view.drag.x; view.y += event.clientY - view.drag.y; view.drag = { x: event.clientX, y: event.clientY }; view.fitted = true; renderAll(); });
    canvas.addEventListener("pointerup", () => { state.views[canvas.id].drag = null; });
    canvas.addEventListener("pointercancel", () => { state.views[canvas.id].drag = null; });
  }

  const checkboxState = { "show-masks": "showMasks", "show-rois": "showRois", "show-points": "showPoints", "show-labels": "showLabels" };
  Object.entries(checkboxState).forEach(([id, key]) => $(id).addEventListener("change", (event) => { state[key] = event.target.checked; renderAll(); }));
  $("fit-views").addEventListener("click", () => { state.views.canvas2d.fitted = false; state.views.canvas3d.fitted = false; renderAll(); });
  $("export-json").addEventListener("click", () => { const blob = new Blob([JSON.stringify(data.response, null, 2)], { type: "application/json" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${data.board_id || "detection"}-response.json`; link.click(); URL.revokeObjectURL(link.href); });
  bindCanvas($("canvas2d")); bindCanvas($("canvas3d"));
  window.addEventListener("resize", () => { state.views.canvas2d.fitted = false; state.views.canvas3d.fitted = false; renderAll(); });
  renderAll();
})();
