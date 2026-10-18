
## 联合测高按 wire / thin 分别剔除反光边缘
- 完成：HeightConfig 新增 wire_measurement_erode_px / thin_measurement_erode_px，非负整数、旧配置缺项为0；当前配置初值2/1。配准后对独立引线测量mask在包围框内原地椭圆腐蚀，再选原始深度P95代表点。thin只作用中间段；空/无有效内区返回无效NG，不回退边缘。
- 效率：复用已映射uint8缓冲，缓存核；measure_mask_height对uint8按bbox枚举，保留原行序/并列值选择。完整语义掩码、平台/芯片面、焊脚、2D弯曲和独立3D接口不变；Mask3DLocal返回实际收缩后测量区域，C#字段未变。
- 文件：python/algorithm/height_measurement.py、detect_board.py、config/3D/config.json、README.md、process/OUTPUT_PARAMETERS.md、tests/test_wire_height_erosion.py。
- 验证：144项Python测试通过（新增10项），含类别独立、反光点剔除、映射后像素单位、图像边界、零开销关闭、内区耗尽、平台/芯片/焊脚不变与uint8采样等价。真实10片+整板3D请求成功，52个有效测点（wire42/thin10）坐标范围、原始深度、参考面差及阈值一致性通过。
- 性能：上一轮52个公开测量区域多边形重建+真实深度回放，9次中位：旧测高核心127.817ms，新关闭15.911ms，新开启2/1为23.014ms；关闭结果逐字段完全一致。该对照不包含配准、推理、公开几何生成，不能当作整板加速结论。
- 实测产物：visualization/output/wire-height-erosion-20261018/live-report/report.html；benchmark.json、live-validation.json、tests.log同级。整板本次96.946s，受GPU推理和银浆配准耗时影响，未与旧运行做受控端到端速度比较。
- 限制/后续：2/1为可调初值，需要按实际线宽/反光范围标注校准，修改需重新初始化。实测银浆SIFT仍存在既有可靠性回退，不属于本次改动。
