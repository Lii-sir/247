# 工具目录

推荐从项目根目录使用统一入口；`python run.py --help` 查看所有命令。

| 子目录 | 工具 | 统一命令 |
|---|---|---|
| data | VisA 下载和辅助数据整理 | `python run.py download-visa` |
| data | 互斥划分正常图片并合成 test/ng | `python run.py make-dataset` |
| masks | mask 图形界面 | `python run.py mask-gui` |
| masks | 参考图生成圆形 mask | `python run.py generate-mask` |
| masks | 单图/目录圆检测诊断 | `python run.py detect-circle` |
| diagnostics | Teacher/Student/AE 分支异常图导出 | `python run.py branch-maps` |
| 根目录 | 推理交付包源码同步 | `python tools/build_inference_delivery.py` |

数据工具保持独立，合成数据仍只依赖 Pillow。核心包不自动导入所有工具，不需要为了数据划分加载 PyTorch 或 GUI。
