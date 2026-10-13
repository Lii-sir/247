# 银浆边界标定文件

`silver_boundary.json` 初始点集为空，不能作为生产标定。不会填入示例坐标或按 YOLO
芯片框自动生成边界。请使用 `visualization/calibrate_silver_boundary.py` 在现有查看器中
标定标准图，再通过 `--apply` 安装真实点集及对应模板图。

边界顶点须有序、无自交，定义“允许银浆存在”的区域。SIFT 特征点是用来估计
模板到当前图像变换的辅助点，不等同于这些业务边界顶点。

安装脚本记录原图尺寸和 SHA-256，按模板内容生成文件名，自动备份旧配置。
模板路径相对于本文件夹，顶点坐标是模板原图坐标。

算法返回的黄色边界点已经转回当前原图坐标；`registration` 中的 Homography、
内点数、匹配数等诊断不做坐标平移。详细步骤见 `visualization/README.md`。
