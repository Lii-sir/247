
## Streaming thread budget optimization

Split CPU postprocessing into a bounded independent pool; default 9 preparation/joint + 3 postprocess + 1 GPU workers. Native library defaults are set before imports. 63 tests and default service init/exit passed. See `visualization/output/thread-comparison-20261015.json` for all nine real batches. Warm sample batch improved from 42.427 s to 37.439/37.036 s; fallback fluctuations remain.

## CPU后处理降排队优化

- 文件：`python/algorithm/2D/silver_inspection.py`、`python/algorithm/2D/defect_geometry.py`、`python/algorithm/detect_board.py`、`python/transport/bridge_server.py`、`tests/test_postprocess_equivalence.py`。
- 决策：连续性扇区统计改为一次 `bincount` 聚合；引线清理改为包围框局部处理；几何轮廓改为包围框提取；默认线程预算改为 7 个准备/联合3D、5 个后处理、1 个 GPU。
- 证据：67 项测试通过；50 组连续性逐字段/掩码差分通过；相同样本输出字段无变化；热批次 36.152 秒降至 30.759/28.877 秒，后处理队列平均 6.698 秒降至 1.826/1.892 秒。
- 限制：不同批次存在 YOLO 首次调用和 SIFT 波动；5 路对本机样本有效，其他硬件需重新基准。

## 线程数量配置化

- 文件：`config/2D/config.json`、`config/3D/config.json`、`python/transport/bridge_server.py`、`tests/test_concurrent_pipeline.py`。
- 配置：2D `common_2d.parameters.threading.postprocess_workers=5/gpu_workers=1`；3D `common_3d.parameters.threading.worker_budget=13/cpu_workers=7`。
- 行为：Bridge 初始化阶段读取两份合并配置并重建空闲线程池；环境变量仍可临时覆盖；预算超限或 GPU 线程不为 1 时拒绝配置。
- 证据：70 项测试通过，两份 JSON 和 Python 语法检查通过。

## GPU 推理线程可配置与预算扩展

- 配置：2D `gpu_workers=2`、`postprocess_workers=5`；3D `worker_budget=14`、`cpu_workers=7`。删除 13 的硬编码上限；线程池合计仍受配置预算约束。
- 实现：统一板级检测按 GPU 工作线程加载独立 YOLO / EfficientAD 副本；流按 YOLO 配置设备创建，兼容模型首次推理前仍在 CPU 的延迟设备迁移。同步更新配置说明和流程页；独立 2D/3D 旧接口仍限 1 路 GPU。
- 验证：Python `unittest discover` 75 项通过，包含模拟 GPU 模型副本隔离、并发、显式/auto 设备流选择和 13 以上线程预算测试；未运行真实样本、GPU 推理或硬件压测。
- 风险：多副本峰值显存、硬件吞吐和检测一致性尚未实机验证。
## [23:50] 动作: Python 并行优化第一阶段
- 文件/决策: board_pipeline.py 增加 2D 诊断与 3D 联合计算分支；silver_inspection.py 增加边界预处理；height_measurement.py 使用局部参考面区域；detect_board.py 合并最终诊断结论并缓存芯片区域；线程配置新增 early_joint_workers 读取。
- 验证: Python tests 91 项通过，visualization tests 26 项通过；最终 4 批真实流式测试成功，批次墙钟 25.402/27.577/31.378/26.334 秒，10/10 片、配准有效。
- 限制: 真实样本存在运行波动，未宣称固定加速比例；上位机未改；下一步按受控线程配比继续基准。
