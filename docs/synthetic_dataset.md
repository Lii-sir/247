# 随机划分正常图并生成测试 NG

脚本 `run.py make-dataset` 仅依赖 Pillow，不需要 GPU。输入必须是正常图片目录。

```powershell
uv run python D:\python_programs\LXD_project\247\run.py make-dataset `
  --input-dir "D:\data\normal" `
  --output-dir "D:\data\CCD1" `
  --train-ratio 0.7 --test-good-ratio 0.15 --test-ng-ratio 0.15 `
  --defect-types occlusion,dirt --seed 2026
```

输出：`train/good`、`test/good`、`test/ng` 以及记录源图和划分的 `manifest.json`。

- 三个比例之和为 1；比例以去重后的图片数为基准，余数按最大余数法分配。图片较少时某些目录可能为空。
- 先划分三个互斥源图集合，只对 NG 集合生成缺陷；NG 的干净源图不再出现在任一 good 目录。
- 自动去除解码后 RGB 像素完全相同的重复图，包含改名或转换无损格式的副本。不保证识别 JPEG 重压缩、裁剪、连拍等近重复图；同一工件的多张照片应提前人工分组，避免数据泄漏。
- 默认递归读取，`--no-recursive` 仅读取顶层。good 图保持原文件，NG 使用无损 PNG，保持 EXIF 校正后的尺寸。
- 多帧图像、宽高不足 2 像素或损坏图像会报错；不静默跳过。
- 固定源目录内容、参数、依赖版本和 seed 可以复现。源文件始终不修改。

## 缺陷参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--defect-types` | `occlusion,dirt` | 遮挡/脏污随机选择；可只选一种 |
| `--min-defects` / `--max-defects` | 1 / 2 | 每张 NG 缺陷个数 |
| `--min-size` / `--max-size` | 0.04 / 0.22 | 缺陷框宽高相对图片短边的比例，不是面积比例 |
| `--occlusion-opacity` | 0.82 | 遮挡不透明度，大于 0、不超过 1 |
| `--dirt-dir` | 无 | 可选脏污贴图目录，推荐透明 PNG；否则合成斑点脏污 |

缺陷随机放置在整张图片内，可能遮挡背景或边框；不自动识别工件区域。脏污贴图应使用独立素材，不要使用训练集图像作为测试贴图。

默认拒绝非空输出目录。`--overwrite` 只允许重建带本脚本 manifest、文件清单一致且无链接/额外目录的输出目录。生成中断可能留下不完整目录，此时请换一个新输出目录，不要强行复用。输入、输出目录不可互相包含。

合成 NG 适合调试流程，不能替代真实缺陷评测。本脚本不生成像素级 ground truth 掩码。
