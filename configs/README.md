# 配置目录

`circle_config.json` 从根目录移动到这里，内容保持不变。训练时使用 `--circle-config configs/circle_config.json`。

配置中的 default_mask 如果是绝对路径，仍须在本机存在；不要因整理目录改用不同 mask。
新配置可将相对路径指向本目录下的 masks，例如 `masks/CCD1.png`。
模型 checkpoint 若保存了旧绝对路径，推理时通过 `--mask` 或 `--circle-config` 显式重定位。
