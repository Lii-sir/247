"""通过简单图形界面选择 ROI、检测圆并保存二值圆形 mask。"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from circle_mask import (
    CIRCLE_DETECTION_DEFAULTS,
    dark_region_mask,
    detect_circle,
    effective_mask_radius,
    make_mask,
    overlay_diagnostics,
    read_rgb,
)


IMAGE_TYPES = [
    ("图片", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff *.webp"),
    ("所有文件", "*.*"),
]
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def find_image_files(folder: Path) -> list[Path]:
    """递归返回文件夹中的受支持图片，按相对路径稳定排序。"""
    return sorted(
        (
            path.resolve()
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(folder)).casefold(),
    )


class CircleMaskApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("圆检测与 Mask 生成")
        self.root.geometry("1380x980")
        self.root.minsize(1050, 680)

        self.image_path: Path | None = None
        self.image_rgb: np.ndarray | None = None
        self.overlay_rgb: np.ndarray | None = None
        self.dark_mask: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.result: dict | None = None
        self.params: dict | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.tk_image: ImageTk.PhotoImage | None = None
        self.roi_item: int | None = None
        self.view_mode = "original"
        self.folder_path: Path | None = None
        self.image_paths: list[Path] = []
        self.image_index = -1
        self.batch_records: dict[Path, dict] = {}
        self.batch_running = False
        self.batch_queue: queue.Queue[tuple] = queue.Queue()

        self._build_ui()
        self.canvas.bind("<ButtonPress-1>", self._start_roi)
        self.canvas.bind("<B1-Motion>", self._drag_roi)
        self.canvas.bind("<ButtonRelease-1>", self._finish_roi)
        self.canvas.bind("<Configure>", lambda _event: self.root.after_idle(self._render))
        self.root.bind("<Left>", lambda event: self._handle_navigation_key(event, -1))
        self.root.bind("<Right>", lambda event: self._handle_navigation_key(event, 1))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(outer, background="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        controls = ttk.Frame(outer, padding=(8, 4), width=300)
        controls.grid(row=0, column=1, sticky="ns")
        controls.grid_propagate(False)
        controls.columnconfigure(1, weight=1)

        open_buttons = ttk.Frame(controls)
        open_buttons.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        open_buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(open_buttons, text="打开单图", command=self.open_image).grid(
            row=0, column=0, sticky="ew", padx=(0, 2)
        )
        ttk.Button(open_buttons, text="打开文件夹", command=self.open_folder).grid(
            row=0, column=1, sticky="ew", padx=(2, 0)
        )

        navigation = ttk.Frame(controls)
        navigation.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        navigation.columnconfigure(1, weight=1)
        ttk.Button(navigation, text="◀ 上一张", command=self.show_previous_image).grid(
            row=0, column=0, sticky="ew"
        )
        self.position_text = tk.StringVar(value="0 / 0")
        ttk.Label(navigation, textvariable=self.position_text, anchor=tk.CENTER).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(navigation, text="下一张 ▶", command=self.show_next_image).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Label(controls, text="2. 在左侧按住鼠标拖框选择 ROI", foreground="#1f5f99").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        defaults = CIRCLE_DETECTION_DEFAULTS
        self.min_radius = tk.DoubleVar(value=100.0 * float(defaults["min_radius_ratio"]))
        self.max_radius = tk.DoubleVar(value=100.0 * float(defaults["max_radius_ratio"]))
        self.dark_threshold_offset = tk.DoubleVar(value=float(defaults["dark_threshold_offset"]))
        self.morph_kernel = tk.IntVar(value=int(defaults["morph_kernel"]))
        self.outer_min_axis_ratio = tk.DoubleVar(
            value=100.0 * float(defaults["outer_min_axis_ratio"])
        )
        self.inner_radius_min = tk.DoubleVar(
            value=100.0 * float(defaults["inner_radius_min_ratio"])
        )
        self.inner_radius_max = tk.DoubleVar(
            value=100.0 * float(defaults["inner_radius_max_ratio"])
        )
        self.black_ring_width = tk.DoubleVar(
            value=100.0 * float(defaults["black_ring_width_ratio"])
        )
        self.min_black_ring_coverage = tk.DoubleVar(
            value=100.0 * float(defaults["min_black_ring_coverage"])
        )
        self.min_inner_angular_coverage = tk.DoubleVar(
            value=100.0 * float(defaults["min_inner_angular_coverage"])
        )
        self.param1 = tk.DoubleVar(value=float(defaults["param1"]))
        self.blur_kernel = tk.IntVar(value=int(defaults["blur_kernel"]))
        self.mask_scale = tk.DoubleVar(value=float(defaults["mask_radius_scale"]))
        self.mask_margin = tk.IntVar(value=int(defaults["mask_margin"]))
        self.require_inside = tk.BooleanVar(value=bool(defaults["require_circle_inside_roi"]))

        row = 3
        ttk.Label(
            controls,
            text="检测方式：外圆定位 + 黑环内圆拟合",
            foreground="#1f5f99",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1
        row = self._spin(controls, row, "最小半径（ROI短边%）", self.min_radius, 1, 95, 1)
        row = self._spin(controls, row, "最大半径（ROI短边%）", self.max_radius, 1, 100, 1)
        row = self._spin(
            controls, row, "暗区阈值偏移", self.dark_threshold_offset, -100, 100, 1,
        )
        row = self._spin(
            controls, row, "轮廓形态学窗口", self.morph_kernel, 1, 31, 2,
        )
        row = self._spin(
            controls, row, "外圆最小轴比（%）", self.outer_min_axis_ratio, 30, 100, 1,
        )
        row = self._spin(
            controls, row, "内圆半径下限（外圆%）", self.inner_radius_min, 1, 95, 1,
        )
        row = self._spin(
            controls, row, "内圆半径上限（外圆%）", self.inner_radius_max, 2, 99, 1,
        )
        row = self._spin(
            controls, row, "黑环宽度（外圆%）", self.black_ring_width, 1, 50, 1,
        )
        row = self._spin(
            controls, row, "最小黑环覆盖率（%）", self.min_black_ring_coverage, 0, 100, 1,
        )
        row = self._spin(
            controls, row, "最小内圆弧覆盖率（%）", self.min_inner_angular_coverage, 0, 100, 1,
        )
        row = self._spin(
            controls, row, "内圆 Canny 高阈值", self.param1, 1, 500, 1,
        )
        row = self._spin(controls, row, "中值滤波窗口", self.blur_kernel, 1, 31, 2)
        row = self._spin(controls, row, "Mask半径比例", self.mask_scale, 0.1, 2.0, 0.01)
        row = self._spin(controls, row, "Mask额外像素", self.mask_margin, -100, 200, 1)
        ttk.Checkbutton(controls, text="要求整个圆位于 ROI 内", variable=self.require_inside).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(5, 8)
        )
        row += 1

        ttk.Button(controls, text="3. 检测圆 / 更新 Mask", command=self.detect).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        row += 1
        self.batch_button = ttk.Button(
            controls, text="批量检测当前文件夹", command=self.detect_folder
        )
        self.batch_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        row += 1

        views = ttk.Frame(controls)
        views.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for index, (text, mode) in enumerate(
            (("原图", "original"), ("暗区", "dark"), ("检测", "overlay"), ("填白", "preview"), ("Mask", "mask"))
        ):
            views.columnconfigure(index, weight=1)
            ttk.Button(views, text=text, command=lambda selected=mode: self.show_view(selected)).grid(
                row=0, column=index, sticky="ew", padx=1
            )
        row += 1

        ttk.Button(controls, text="4. 保存二值 Mask", command=self.save_mask).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10)
        )
        row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1
        self.status = tk.StringVar(value="请先打开一张图片。")
        ttk.Label(controls, textvariable=self.status, wraplength=280, justify=tk.LEFT).grid(
            row=row, column=0, columnspan=2, sticky="nw"
        )

    def _spin(
        self,
        parent: ttk.Frame,
        row: int,
        text: str,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
        increment: float,
    ) -> int:
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=0, sticky="w", pady=3)
        widget = ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=minimum,
            to=maximum,
            increment=increment,
            width=12,
        )
        widget.grid(row=row, column=1, sticky="ew", pady=3)
        return row + 1

    def open_image(self) -> None:
        if self.batch_running:
            messagebox.showinfo("正在检测", "请等待当前批量检测完成后再打开其他图片。")
            return
        selected = filedialog.askopenfilename(title="选择参考图片", filetypes=IMAGE_TYPES)
        if not selected:
            return
        path = Path(selected).resolve()
        self.folder_path = None
        self.image_paths = [path]
        self.image_index = 0
        self.batch_records.clear()
        if not self._load_image_at(0):
            self.image_paths = []
            self.image_index = -1
            self.position_text.set("0 / 0")

    def open_folder(self) -> None:
        if self.batch_running:
            messagebox.showinfo("正在检测", "请等待当前批量检测完成后再打开其他文件夹。")
            return
        selected = filedialog.askdirectory(title="选择待检测图片文件夹")
        if not selected:
            return
        folder = Path(selected).resolve()
        try:
            paths = find_image_files(folder)
        except Exception as exc:
            messagebox.showerror("扫描文件夹失败", str(exc))
            return
        if not paths:
            messagebox.showwarning("没有图片", "所选文件夹及其子文件夹中没有受支持的图片。")
            return
        self.folder_path = folder
        self.image_paths = paths
        self.image_index = 0
        self.batch_records.clear()
        if self._load_image_at(0):
            self.status.set(
                f"{self._position_prefix()}\n已载入文件夹，共 {len(paths)} 张图片。\n"
                "请在首张图片上选择 ROI、调整参数，然后点击“批量检测当前文件夹”。"
            )

    def _load_image_at(
        self, index: int, roi_ratio: list[float] | tuple[float, ...] | None = None
    ) -> bool:
        if not 0 <= index < len(self.image_paths):
            return False
        path = self.image_paths[index]
        try:
            image = read_rgb(path)
        except Exception as exc:
            messagebox.showerror("读取失败", f"{path}\n\n{exc}")
            return False
        self.image_index = index
        self.image_path = path
        self.image_rgb = image
        self.overlay_rgb = None
        self.dark_mask = None
        self.mask = None
        self.result = None
        self.params = None
        height, width = image.shape[:2]
        record = self.batch_records.get(path)
        if record is not None:
            roi_ratio = record["params"]["roi"]
        if roi_ratio is None:
            self.roi = (0, 0, width, height)
        else:
            x, y, roi_width, roi_height = roi_ratio
            self.roi = (
                int(round(x * width)),
                int(round(y * height)),
                int(round((x + roi_width) * width)),
                int(round((y + roi_height) * height)),
            )
        self.view_mode = "original"
        self.position_text.set(f"{index + 1} / {len(self.image_paths)}")
        if record is None:
            self.status.set(
                f"{self._position_prefix()}\n尺寸：{width} × {height}\n尚未检测。"
            )
        elif record.get("error"):
            self.status.set(
                f"{self._position_prefix()}\n批量检测失败：{record['error']}"
            )
        else:
            try:
                self._apply_detection_result(record["result"], record["params"])
            except Exception as exc:
                self.status.set(f"{self._position_prefix()}\n显示检测结果失败：{exc}")
        self._render()
        return True

    def _normalized_roi(self) -> list[float] | None:
        if self.image_rgb is None or self.roi is None:
            return None
        height, width = self.image_rgb.shape[:2]
        x0, y0, x1, y1 = self.roi
        return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]

    def _position_prefix(self) -> str:
        if self.image_path is None:
            return "尚未打开图片"
        if len(self.image_paths) > 1:
            return f"[{self.image_index + 1}/{len(self.image_paths)}] {self.image_path.name}"
        return self.image_path.name

    def _change_image(self, step: int) -> None:
        if len(self.image_paths) < 2:
            return
        roi_ratio = self._normalized_roi()
        target = (self.image_index + step) % len(self.image_paths)
        self._load_image_at(target, roi_ratio)

    def show_previous_image(self) -> None:
        self._change_image(-1)

    def show_next_image(self) -> None:
        self._change_image(1)

    def _handle_navigation_key(self, event: tk.Event, step: int) -> str | None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return None
        self._change_image(step)
        return "break"

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int] | None:
        if self.image_rgb is None:
            return None
        offset_x, offset_y = self.display_offset
        image_x = int(round((x - offset_x) / self.display_scale))
        image_y = int(round((y - offset_y) / self.display_scale))
        height, width = self.image_rgb.shape[:2]
        if image_x < 0 or image_y < 0 or image_x >= width or image_y >= height:
            return None
        return image_x, image_y

    def _start_roi(self, event: tk.Event) -> None:
        if self.batch_running:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.drag_start = point
        if self.roi_item is not None:
            self.canvas.delete(self.roi_item)
        self.roi_item = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ffcc00", width=2, dash=(6, 3)
        )

    def _drag_roi(self, event: tk.Event) -> None:
        if self.drag_start is None or self.roi_item is None:
            return
        offset_x, offset_y = self.display_offset
        start_x = offset_x + self.drag_start[0] * self.display_scale
        start_y = offset_y + self.drag_start[1] * self.display_scale
        self.canvas.coords(self.roi_item, start_x, start_y, event.x, event.y)

    def _finish_roi(self, event: tk.Event) -> None:
        if self.drag_start is None or self.image_rgb is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            height, width = self.image_rgb.shape[:2]
            offset_x, offset_y = self.display_offset
            point = (
                min(width - 1, max(0, int(round((event.x - offset_x) / self.display_scale)))),
                min(height - 1, max(0, int(round((event.y - offset_y) / self.display_scale)))),
            )
        x0, x1 = sorted((self.drag_start[0], point[0]))
        y0, y1 = sorted((self.drag_start[1], point[1]))
        self.drag_start = None
        if x1 - x0 < 10 or y1 - y0 < 10:
            self._draw_roi()
            return
        self.roi = (x0, y0, x1 + 1, y1 + 1)
        self.overlay_rgb = None
        self.dark_mask = None
        self.mask = None
        self.result = None
        self.params = None
        self.batch_records.clear()
        self.view_mode = "original"
        self.status.set(
            f"{self._position_prefix()}\n"
            f"ROI：x={x0}, y={y0}, width={x1 + 1 - x0}, height={y1 + 1 - y0}\n"
            "点击单图检测，或重新批量检测当前文件夹。"
        )
        self._render()

    def _current_params(self) -> dict:
        if self.image_rgb is None or self.roi is None:
            raise ValueError("请先打开图片并选择 ROI。")
        minimum = float(self.min_radius.get()) / 100.0
        maximum = float(self.max_radius.get()) / 100.0
        if not 0 < minimum < maximum <= 1:
            raise ValueError("半径范围必须满足 0 < 最小半径 < 最大半径 <= 100%。")
        blur = int(self.blur_kernel.get())
        if blur < 1:
            raise ValueError("中值滤波窗口必须大于 0。")
        morph_kernel = int(self.morph_kernel.get())
        if morph_kernel < 1:
            raise ValueError("轮廓形态学窗口必须大于 0。")
        outer_min_axis_ratio = float(self.outer_min_axis_ratio.get()) / 100.0
        if not 0 < outer_min_axis_ratio <= 1:
            raise ValueError("外圆最小轴比必须位于 (0, 100%]。")
        inner_minimum = float(self.inner_radius_min.get()) / 100.0
        inner_maximum = float(self.inner_radius_max.get()) / 100.0
        if not 0 < inner_minimum < inner_maximum < 1:
            raise ValueError("内圆半径范围必须满足 0 < 下限 < 上限 < 100%。")
        min_inner_angular_coverage = float(self.min_inner_angular_coverage.get()) / 100.0
        if not 0 <= min_inner_angular_coverage <= 1:
            raise ValueError("最小内圆弧覆盖率必须位于 [0, 100%]。")
        height, width = self.image_rgb.shape[:2]
        x0, y0, x1, y1 = self.roi
        return {
            **CIRCLE_DETECTION_DEFAULTS,
            "enabled": True,
            "detection_method": "outer_inner_ring",
            "require_circle_inside_roi": bool(self.require_inside.get()),
            "roi": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
            "min_radius_ratio": minimum,
            "max_radius_ratio": maximum,
            "dark_threshold_offset": float(self.dark_threshold_offset.get()),
            "morph_kernel": morph_kernel,
            "outer_min_axis_ratio": outer_min_axis_ratio,
            "inner_radius_min_ratio": inner_minimum,
            "inner_radius_max_ratio": inner_maximum,
            "black_ring_width_ratio": float(self.black_ring_width.get()) / 100.0,
            "min_black_ring_coverage": float(self.min_black_ring_coverage.get()) / 100.0,
            "min_inner_angular_coverage": min_inner_angular_coverage,
            "param1": float(self.param1.get()),
            "blur_kernel": blur,
            "mask_radius_scale": float(self.mask_scale.get()),
            "mask_margin": int(self.mask_margin.get()),
        }

    def detect(self) -> None:
        if self.image_rgb is None:
            messagebox.showwarning("尚未打开图片", "请先打开一张参考图片。")
            return
        try:
            params = self._current_params()
            result = detect_circle(self.image_rgb, params)
            self._apply_detection_result(result, params)
        except Exception as exc:
            messagebox.showerror(
                "圆检测失败",
                f"{exc}\n\n可调整内圆半径范围、内圆 Canny 高阈值、黑环宽度、"
                "最小黑环覆盖率和最小内圆弧覆盖率。",
            )
            self.status.set(f"检测失败：{exc}")
            return
        if self.image_path is not None:
            self.batch_records[self.image_path] = {
                "result": result,
                "params": params,
                "error": None,
            }
        self._render()

    def _apply_detection_result(self, result: dict, params: dict) -> None:
        if self.image_rgb is None:
            raise ValueError("当前没有可显示的图片。")
        selected = result["selected"]
        mask = make_mask(self.image_rgb.shape[:2], selected, params)
        if not np.any(mask):
            raise ValueError("检测结果没有生成有效 mask。")
        overlay = overlay_diagnostics(self.image_rgb, result, params)
        dark_mask = dark_region_mask(self.image_rgb, params)
        self.params = params
        self.result = result
        self.mask = mask
        self.overlay_rgb = overlay
        self.dark_mask = dark_mask
        self.view_mode = "overlay"
        detector = result["detection"].get("detector_used", "unknown")
        confidence = selected.get("candidate_score")
        confidence_text = f"，置信度 {confidence:.3f}" if confidence is not None else ""
        ring_text = (
            f"\n黑环覆盖率 {selected['black_ring_coverage']:.3f}，"
            f"边缘覆盖率 {selected['edge_coverage']:.3f}"
            if "black_ring_coverage" in selected else ""
        )
        group_text = "按黑环评分选择（内外圆允许偏心）"
        outer = result["detection"].get("outer_circle")
        outer_text = (
            f"外圆：圆心 ({outer['center_x']}, {outer['center_y']})，半径 {outer['radius']}\n"
            if outer is not None else ""
        )
        target_name = "内圆" if outer is not None else "检测圆"
        self.status.set(
            f"{self._position_prefix()}\n"
            f"{outer_text}"
            f"{target_name}：圆心 ({selected['center_x']}, {selected['center_y']})，半径 {selected['radius']}\n"
            f"Mask 半径 {effective_mask_radius(selected, params)}\n"
            f"候选圆 {selected['candidate_count']}，{group_text}\n"
            f"实际检测器 {detector}{confidence_text}{ring_text}\n"
            f"检测耗时 {result['detection_ms']:.2f} ms"
        )

    def detect_folder(self) -> None:
        if self.folder_path is None or not self.image_paths:
            messagebox.showwarning("尚未打开文件夹", "请先点击“打开文件夹”。")
            return
        if self.batch_running:
            return
        try:
            params = self._current_params()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        while True:
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break
        self.batch_records.clear()
        self.batch_running = True
        self.batch_button.configure(state="disabled")
        self.status.set(f"准备批量检测 {len(self.image_paths)} 张图片……")
        worker = threading.Thread(
            target=self._batch_worker,
            args=(list(self.image_paths), params),
            daemon=True,
        )
        worker.start()
        self.root.after(50, self._poll_batch_queue)

    def _batch_worker(self, paths: list[Path], params: dict) -> None:
        records: dict[Path, dict] = {}
        success_count = 0
        for index, path in enumerate(paths, start=1):
            record_params = {**params, "roi": list(params["roi"])}
            try:
                image = read_rgb(path)
                result = detect_circle(image, record_params)
                records[path] = {
                    "result": result,
                    "params": record_params,
                    "error": None,
                }
                success_count += 1
                error = None
            except Exception as exc:
                error = str(exc)
                records[path] = {
                    "result": None,
                    "params": record_params,
                    "error": error,
                }
            self.batch_queue.put(
                ("progress", index, len(paths), path.name, error)
            )
        self.batch_queue.put(
            ("done", records, params, success_count, len(paths) - success_count)
        )

    def _poll_batch_queue(self) -> None:
        done = False
        while True:
            try:
                event = self.batch_queue.get_nowait()
            except queue.Empty:
                break
            if event[0] == "progress":
                _, index, total, name, error = event
                state = "成功" if error is None else f"失败：{error}"
                self.status.set(f"批量检测 {index}/{total}\n{name}\n{state}")
            elif event[0] == "done":
                _, records, params, success_count, failure_count = event
                self.batch_records = records
                self.batch_running = False
                self.batch_button.configure(state="normal")
                done = True
                self._load_image_at(self.image_index, params["roi"])
                messagebox.showinfo(
                    "批量检测完成",
                    f"共 {len(records)} 张图片\n成功：{success_count}\n失败：{failure_count}\n\n"
                    "可使用上一张/下一张按钮或键盘左右方向键查看。",
                )
        if self.batch_running and not done:
            self.root.after(100, self._poll_batch_queue)

    def show_view(self, mode: str) -> None:
        if self.image_rgb is None:
            return
        if mode != "original" and self.mask is None:
            messagebox.showinfo("尚未检测", "请先点击“检测圆 / 更新 Mask”。")
            return
        self.view_mode = mode
        self._render()

    def _view_image(self) -> np.ndarray | None:
        if self.image_rgb is None:
            return None
        if self.view_mode == "overlay" and self.overlay_rgb is not None:
            return self.overlay_rgb
        if self.view_mode == "dark" and self.dark_mask is not None:
            return np.repeat(self.dark_mask[:, :, None], 3, axis=2)
        if self.view_mode == "mask" and self.mask is not None:
            return np.repeat(self.mask[:, :, None], 3, axis=2)
        if self.view_mode == "preview" and self.mask is not None:
            preview = self.image_rgb.copy()
            preview[self.mask > 0] = 255
            return preview
        return self.image_rgb

    def _render(self) -> None:
        image = self._view_image()
        if image is None or not self.canvas.winfo_exists():
            return
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        height, width = image.shape[:2]
        self.display_scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(round(width * self.display_scale)))
        display_height = max(1, int(round(height * self.display_scale)))
        offset_x = (canvas_width - display_width) / 2
        offset_y = (canvas_height - display_height) / 2
        self.display_offset = (offset_x, offset_y)

        resized = Image.fromarray(image).resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW)
        self.roi_item = None
        self._draw_roi()

    def _draw_roi(self) -> None:
        if self.roi is None or self.image_rgb is None:
            return
        x0, y0, x1, y1 = self.roi
        offset_x, offset_y = self.display_offset
        self.roi_item = self.canvas.create_rectangle(
            offset_x + x0 * self.display_scale,
            offset_y + y0 * self.display_scale,
            offset_x + x1 * self.display_scale,
            offset_y + y1 * self.display_scale,
            outline="#ffcc00",
            width=2,
            dash=(6, 3),
        )

    def save_mask(self) -> None:
        if self.mask is None or self.result is None or self.params is None:
            messagebox.showwarning("尚无 Mask", "请先完成圆检测。")
            return
        initial_dir = str(self.image_path.parent) if self.image_path else str(Path.cwd())
        selected = filedialog.asksaveasfilename(
            title="保存二值圆 Mask",
            initialdir=initial_dir,
            initialfile="default_mask.png",
            defaultextension=".png",
            filetypes=[("无损 PNG", "*.png")],
        )
        if not selected:
            return
        mask_path = Path(selected)
        if mask_path.suffix.lower() != ".png":
            messagebox.showerror("格式错误", "Mask 必须保存为 PNG。")
            return
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", np.where(self.mask > 0, 255, 0).astype(np.uint8))
            if not ok:
                raise ValueError("OpenCV 无法编码 PNG。")
            encoded.tofile(mask_path)
            metadata = {
                "input": str(self.image_path),
                "mask_path": str(mask_path.resolve()),
                "mask_semantics": "255=ignored circle, 0=valid image region",
                "params": self.params,
                **self.result,
            }
            metadata_path = mask_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status.set(f"Mask 已保存：\n{mask_path}\n参数记录：\n{metadata_path}")
        messagebox.showinfo("保存成功", f"二值 Mask：\n{mask_path}\n\n检测参数：\n{metadata_path}")


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    CircleMaskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
