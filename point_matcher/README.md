# 模板点定位（独立桌面程序）

在模板图中点击需要定位的位置，自动在目标图片或文件夹中的图片上找到对应点。程序独立于 PatchCore，不使用深度学习模型，也不修改项目原有脚本。

## 启动

在项目根目录、使用者已配置的环境中执行（不自动安装或同步依赖）：

```powershell
uv run --no-sync python -m point_matcher
# 原模块入口也保留
uv run --no-sync python -m point_matcher.app
```

界面仅依赖 NumPy、OpenCV 和 PySide6，不加载 YOLO/Torch，不需要 GPU。
共享算法与控件已迁到 common，本目录不再作为单独复制即可运行的独立工程。

## 使用步骤

1. 点击 **打开模板图**。
2. 在左侧模板上 **左键点击** 添加选点，按顺序编号 P1、P2……，至少一个即可。无需自己选择 SIFT 特征点，也不要求手工选四个角。
   - 滚轮缩放；按住左键拖动平移。
   - 点击 **适应窗口** 恢复整图显示。
   - **撤销上一点** / **清空选点** 用于修正。
3. 点击 **选择目标图** 或 **选择文件夹**，也可以把路径粘贴到目标输入框。
   - 默认只读取当前目录；需要递归时勾选 **包含子文件夹**。
   - 支持 JPG / JPEG / PNG / BMP / TIF / TIFF / WebP，支持中文路径。
4. 点击 **开始找点**。
5. 在下方图片列表中点击一行，右侧查看对应的定位结果。黄色十字是选点位置，蓝色虚线是模板在目标图中的边界；坐标表显示各点的原始像素坐标。
6. 可选：**导出坐标 CSV / JSON**，或 **保存当前标注图**。

批量处理在后台线程中进行。停止会等待当前图片处理完成，并保留已有结果；导出只包含已经处理的图片，尚未处理的图片不会冒充失败结果。修改模板、选点、目标路径或匹配参数后，旧结果会清除，避免导出不一致的坐标。

## 坐标与结果

- 所有坐标都是**解码后的原图像素**，不受窗口缩放影响：左上角为 `(0, 0)`，X 向右，Y 向下。
- 界面显示两位小数；CSV 保存四位小数；JSON 保留计算精度及单应性矩阵。
- CSV 使用 UTF-8 BOM，便于 Excel 读取中文，每张图片的每个选点占一行。失败图片也保留行，目标坐标留空。
- `ok`：匹配成功且所有选点在图内。
- `partial` / `outside`：模板匹配成功，但部分 / 全部选点投影到目标图外；不会把越界坐标强行裁到边缘。
- `error`：读取失败、特征不足或匹配不可靠；不会输出看似成功的坐标。
- 保存标注图不允许覆盖本次加载的模板或目标原图。
- 程序不自动写入目标文件夹；结果只有在主动导出时才写盘。

## 原理和使用范围

沿用原项目 `img_tr/viomatch.py` 的思路：SIFT 提取特征 → FLANN 最近邻匹配 → 0.70 比值筛选 → RANSAC 估计单应性矩阵 → 将手选点投影到目标图。

模板特征在一批任务中只计算一次。默认要求至少 11 个去重的可靠匹配点、8 个 RANSAC 内点、25% 内点比例，并检查特征分布和变换几何。大图特征提取最长边缩到 2400，输出坐标会恢复到原图尺寸。界面可调整比值阈值，越小筛选越严格。

**注意：这不是根据所选点的局部外观逐点搜索，而是先将整张模板与目标图配准，再映射所选点。** 适合纸张、板件、印刷图案等同一平面上的目标。模板应包含足够纹理或文字。纯色区域、严重遮挡、重复图案、较大非刚性形变或不同深度的物体可能失败或误匹配，需要人工核对。若一张目标图内出现多个相同模板，目前只返回一组对应位置，不枚举全部实例。匹配内点数是几何一致性指标，不是精度保证。

## 文件与测试

- [app.py](D:/python_programs/LXD_project/point-matcher/point_matcher/app.py)：本功能 Qt 界面和后台匹配线程。
- [export.py](D:/python_programs/LXD_project/point-matcher/point_matcher/export.py)：本功能坐标 CSV/JSON 导出。
- [common/matching.py](D:/python_programs/LXD_project/point-matcher/common/matching.py)：共享配准、坐标变换及结果绘图。
- [common/image_io.py](D:/python_programs/LXD_project/point-matcher/common/image_io.py)：共享图片读写。
- [common/widgets/point_view.py](D:/python_programs/LXD_project/point-matcher/common/widgets/point_view.py)：共享选点控件。

在项目根目录执行：

```powershell
uv run --no-sync python -m unittest discover -s tests/common -t . -v
uv run --no-sync python -m unittest discover -s tests/point_matcher -t . -v
```
