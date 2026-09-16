"""通过简单图形界面选择 ROI、检测圆并保存二值圆形 mask。"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from circle_mask import (
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

        self._build_ui()
        self.canvas.bind("<ButtonPress-1>", self._start_roi)
        self.canvas.bind("<B1-Motion>", self._drag_roi)
        self.canvas.bind("<ButtonRelease-1>", self._finish_roi)
        self.canvas.bind("<Configure>", lambda _event: self.root.after_idle(self._render))

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

        ttk.Button(controls, text="1. 打开参考图片", command=self.open_image).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        ttk.Label(controls, text="2. 在左侧按住鼠标拖框选择 ROI", foreground="#1f5f99").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        self.circle_target = tk.StringVar(value="outer")
        self.group_target = tk.StringVar(value="best_score")
        self.detection_method = tk.StringVar(value="outer_inner_ring")
        self.min_radius = tk.DoubleVar(value=15.0)
        self.max_radius = tk.DoubleVar(value=55.0)
        self.dark_threshold_offset = tk.DoubleVar(value=0.0)
        self.morph_kernel = tk.IntVar(value=5)
        self.min_axis_ratio = tk.DoubleVar(value=65.0)
        self.min_contour_score = tk.DoubleVar(value=0.40)
        self.inner_radius_min = tk.DoubleVar(value=15.0)
        self.inner_radius_max = tk.DoubleVar(value=75.0)
        self.black_ring_width = tk.DoubleVar(value=6.0)
        self.min_black_ring_coverage = tk.DoubleVar(value=45.0)
        self.param1 = tk.DoubleVar(value=100.0)
        self.param2 = tk.DoubleVar(value=24.0)
        self.blur_kernel = tk.IntVar(value=5)
        self.mask_scale = tk.DoubleVar(value=0.95)
        self.mask_margin = tk.IntVar(value=2)
        self.require_inside = tk.BooleanVar(value=True)

        row = 2
        row = self._combo(
            controls, row, "检测方式", self.detection_method,
            ("outer_inner_ring", "hybrid", "dark_contour", "hough"),
        )
        row = self._combo(controls, row, "旧模式圆选择", self.circle_target, ("outer", "inner", "best_contrast"))
        row = self._combo(controls, row, "旧模式圆组选择", self.group_target, ("best_score", "largest", "strongest"))
        row = self._spin(controls, row, "最小半径（ROI短边%）", self.min_radius, 1, 95, 1)
        row = self._spin(controls, row, "最大半径（ROI短边%）", self.max_radius, 1, 100, 1)
        row = self._spin(controls, row, "暗区阈值偏移", self.dark_threshold_offset, -100, 100, 1)
        row = self._spin(controls, row, "轮廓形态学窗口", self.morph_kernel, 1, 31, 2)
        row = self._spin(controls, row, "最小圆度轴比（%）", self.min_axis_ratio, 30, 100, 1)
        row = self._spin(controls, row, "最小轮廓置信度", self.min_contour_score, 0, 1, 0.01)
        row = self._spin(controls, row, "内圆半径下限（外圆%）", self.inner_radius_min, 1, 95, 1)
        row = self._spin(controls, row, "内圆半径上限（外圆%）", self.inner_radius_max, 2, 99, 1)
        row = self._spin(controls, row, "黑环采样宽度（外圆%）", self.black_ring_width, 1, 50, 1)
        row = self._spin(controls, row, "最小黑环覆盖率（%）", self.min_black_ring_coverage, 0, 100, 1)
        row = self._spin(controls, row, "Canny阈值 param1", self.param1, 1, 500, 1)
        row = self._spin(controls, row, "Hough阈值 param2", self.param2, 1, 200, 1)
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

    @staticmethod
    def _combo(parent: ttk.Frame, row: int, text: str, variable: tk.StringVar, values: tuple[str, ...]) -> int:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=16).grid(
            row=row, column=1, sticky="ew", pady=3
        )
        return row + 1

    @staticmethod
    def _spin(
        parent: ttk.Frame,
        row: int,
        text: str,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
        increment: float,
    ) -> int:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=minimum,
            to=maximum,
            increment=increment,
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=3)
        return row + 1

    def open_image(self) -> None:
        selected = filedialog.askopenfilename(title="选择参考图片", filetypes=IMAGE_TYPES)
        if not selected:
            return
        try:
            image = read_rgb(Path(selected))
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return
        self.image_path = Path(selected).resolve()
        self.image_rgb = image
        self.overlay_rgb = None
        self.dark_mask = None
        self.mask = None
        self.result = None
        self.params = None
        height, width = image.shape[:2]
        self.roi = (0, 0, width, height)
        self.view_mode = "original"
        self.status.set(f"已打开：{self.image_path.name}\n尺寸：{width} × {height}\n请拖框选择圆所在区域。")
        self._render()

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
        self.view_mode = "original"
        self.status.set(f"ROI：x={x0}, y={y0}, width={x1 + 1 - x0}, height={y1 + 1 - y0}\n点击“检测圆”。")
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
        min_axis_ratio = float(self.min_axis_ratio.get()) / 100.0
        min_contour_score = float(self.min_contour_score.get())
        if not 0 < min_axis_ratio <= 1:
            raise ValueError("最小圆度轴比必须位于 (0, 100%]。")
        if not 0 <= min_contour_score <= 1:
            raise ValueError("最小轮廓置信度必须位于 [0, 1]。")
        inner_minimum = float(self.inner_radius_min.get()) / 100.0
        inner_maximum = float(self.inner_radius_max.get()) / 100.0
        if not 0 < inner_minimum < inner_maximum < 1:
            raise ValueError("内圆半径范围必须满足 0 < 下限 < 上限 < 100%。")
        height, width = self.image_rgb.shape[:2]
        x0, y0, x1, y1 = self.roi
        return {
            "enabled": True,
            "detection_method": self.detection_method.get(),
            "require_circle_inside_roi": bool(self.require_inside.get()),
            "roi": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
            "circle_target": self.circle_target.get(),
            "group_target": self.group_target.get(),
            "min_radius_ratio": minimum,
            "max_radius_ratio": maximum,
            "dark_threshold_offset": float(self.dark_threshold_offset.get()),
            "morph_kernel": morph_kernel,
            "min_axis_ratio": min_axis_ratio,
            "min_contour_score": min_contour_score,
            "inner_radius_min_ratio": inner_minimum,
            "inner_radius_max_ratio": inner_maximum,
            "black_ring_width_ratio": float(self.black_ring_width.get()) / 100.0,
            "min_black_ring_coverage": float(self.min_black_ring_coverage.get()) / 100.0,
            "min_inner_angular_coverage": 0.35,
            "ransac_iterations": 180,
            "ransac_tolerance_ratio": 0.03,
            "min_ransac_inlier_ratio": 0.45,
            "min_angular_coverage": 0.50,
            "dp": 1.2,
            "min_dist_ratio": 0.12,
            "param1": float(self.param1.get()),
            "param2": float(self.param2.get()),
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
            selected = result["selected"]
            mask = make_mask(self.image_rgb.shape[:2], selected, params)
            if not np.any(mask):
                raise ValueError("检测结果没有生成有效 mask。")
            overlay = overlay_diagnostics(self.image_rgb, result, params)
            dark_mask = (
                dark_region_mask(self.image_rgb, params)
                if params["detection_method"] != "hough"
                else np.zeros(self.image_rgb.shape[:2], dtype=np.uint8)
            )
        except Exception as exc:
            messagebox.showerror(
                "圆检测失败",
                f"{exc}\n\n双圆黑环模式可调整内圆半径范围、黑环宽度和最小覆盖率；"
                "旧模式可调整暗区阈值或 Hough param2。",
            )
            self.status.set(f"检测失败：{exc}")
            return

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
        group_text = (
            "按黑环评分选择（内外圆允许偏心）"
            if params["detection_method"] == "outer_inner_ring"
            else f"同心组内 {selected['group_size']}"
        )
        self.status.set(
            f"检测成功：圆心 ({selected['center_x']}, {selected['center_y']})\n"
            f"检测半径 {selected['radius']}，Mask 半径 {effective_mask_radius(selected, params)}\n"
            f"候选圆 {selected['candidate_count']}，{group_text}\n"
            f"实际检测器 {detector}{confidence_text}{ring_text}\n"
            f"检测耗时 {result['detection_ms']:.2f} ms"
        )
        self._render()

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
