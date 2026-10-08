# 共享层

`common` 只提供可复用能力，不选择检测模式、不启动业务窗口、不安装依赖。

| 文件 | 对外能力 |
| --- | --- |
| [common/image_io.py](D:/python_programs/LXD_project/point-matcher/common/image_io.py) | collect_images、read_image、write_image、IMAGE_FILTER |
| [common/paths.py](D:/python_programs/LXD_project/point-matcher/common/paths.py) | PROJECT_ROOT、DEFAULT_WEIGHTS、DEFAULT_SOURCE；按源码位置定位 |
| [common/cli.py](D:/python_programs/LXD_project/point-matcher/common/cli.py) | 添加通用 CLI 参数、创建 SegmentationSettings |
| [common/segmentation/models.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/models.py) | Segment、SegmentationResult、SegmentationSettings |
| [common/segmentation/inference.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/inference.py) | PartSegmenter；唯一 YOLO 加载/推理适配层，延迟导入 Torch/Ultralytics |
| [common/segmentation/visualization.py](D:/python_programs/LXD_project/point-matcher/common/segmentation/visualization.py) | class_color、render_overlay、render_comparison |
| [common/matching.py](D:/python_programs/LXD_project/point-matcher/common/matching.py) | MatchSettings、MatchResult、TemplateMatcher、transform_points、annotate_image；不依赖分割/Qt |
| [common/widgets/image_view.py](D:/python_programs/LXD_project/point-matcher/common/widgets/image_view.py) | ImageView：普通图片预览，支持保留缩放 |
| [common/widgets/point_view.py](D:/python_programs/LXD_project/point-matcher/common/widgets/point_view.py) | PointImageView：选点、标记、坐标提示；不包含匹配算法 |

两种控件交互不同，保留为独立控件；不从某个业务的 app.py 获取通用控件。
业务代码只依赖所需的共享模块；模型数据、纯绘图和命令行帮助不需要打开 GUI。
新增共享代码前先确认确实有多个功能复用；银浆判断、标定格式及各功能输出保持在各自目录。

