# 产品型号配置

每个型号用一个目录。验证台从本目录下各一级子目录的 `config.json` 读取型号列表，目录名就是型号。
`normal` 是默认型号。新增型号时可复制 `normal` 目录并改名，例如 `A100/`。

```text
A100/
├── config.json
├── 2D/
│   └── config.json
├── 3D/
│   └── config.json
└── inspection.json
```

按型号修改 ROI、分片范围、判定阈值及启用的检测项。默认型号配置见 [normal/config.json](normal/config.json)。
共用模型时，把型号 2D 配置的 `model_path` 改为相对于该 2D 文件的路径，例如 `../../../models/yolo_2d.pt`。
2D 银浆标定路径相对该型号的 2D 文件，3D 仿射标定路径相对该型号的 3D 文件。
需要标定文件时，将它们放在对应的 `calibration/` 目录，或填入指向共用标定的相对路径。
`common_3d.parameters.input.piece_count` 与 `segmentation_3d.parameters.expected_piece_count` 应使用同一片数。

从项目根目录运行上位机时，可以这样初始化。

```csharp
bridge.InitAlgorithm("board", @"config\products\normal\config.json");
```

相对路径以进程的工作目录为基准，也可以传绝对路径。在开板前初始化，初始化成功后再提交单片请求。换型前先等检测请求完成，对上一型号的各板调用 `ReleaseBoard(boardId)`，再初始化新型号。
上位机按该型号的 `segmentation_3d` 和 `smartray_pitch_3d` 设置准备单片深度。
验证台通过左侧「产品型号」切换，切换时清理检测缓存和已矫正的深度片。首次检测时再加载模型。
结果记录型号和配置摘要。要重新加载同一型号的配置，可以切换到另一项再切回来，或重启验证台。
