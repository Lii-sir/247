# Python 检测代码目录

```text
python/
├── transport/                Bridge 请求入口、协议和共享内存
└── algorithm/
    ├── application/
    │   ├── board/             整板输入、按片调度、片级及整板结果汇总
    │   └── standalone/        独立 2D、独立 3D 入口
    ├── vision2d/              圆定位、引线、芯片、银浆和模型实现
    ├── vision3d/
    │   ├── depth/             深度分片、提交前 Pitch 预处理
    │   ├── alignment/         粗配准、仿射和坐标映射
    │   └── measurement/       参考面拟合、引线和焊点测高
    ├── execution/
    │   ├── processes/         主进程投递任务、准备共享输入
    │   └── workers/           CPU 子进程入口和任务执行
    └── runtime/               配置读取、运行计时
```

- 调整 2D 检测顺序或首项 NG 的规则，查看 [defect_checks.py](algorithm/application/defect_checks.py) 和 [2D 配置](../config/products/normal/2D/README.md)。
- 调整整板或单片 3D 输入，查看 [board_depth.py](algorithm/application/board/board_depth.py)。两种入口在 [board_pipeline.py](algorithm/application/board/board_pipeline.py) 共用配准和测高调度。
- 调整跳过原因、片级结论或整板汇总字段，查看 [board_results.py](algorithm/application/board/board_results.py) 和 [上位机返回参数](../process/OUTPUT_PARAMETERS.md)。

C# Bridge 通过 [bridge_server.py](transport/bridge_server.py) 启动检测服务。运行配置位于项目根目录的 [config](../config/README.md)。
