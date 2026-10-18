
## 联合测高按 wire / thin 分别剔除反光边缘
- 完成：HeightConfig 新增 wire_measurement_erode_px / thin_measurement_erode_px，非负整数、旧配置缺项为0；当前配置初值2/1。配准后对独立引线测量mask在包围框内原地椭圆腐蚀，再选原始深度P95代表点。thin只作用中间段；空/无有效内区返回无效NG，不回退边缘。
- 效率：复用已映射uint8缓冲，缓存核；measure_mask_height对uint8按bbox枚举，保留原行序/并列值选择。完整语义掩码、平台/芯片面、焊脚、2D弯曲和独立3D接口不变；Mask3DLocal返回实际收缩后测量区域，C#字段未变。
- 文件：python/algorithm/height_measurement.py、detect_board.py、config/3D/config.json、README.md、process/OUTPUT_PARAMETERS.md、tests/test_wire_height_erosion.py。
- 验证：144项Python测试通过（新增10项），含类别独立、反光点剔除、映射后像素单位、图像边界、零开销关闭、内区耗尽、平台/芯片/焊脚不变与uint8采样等价。真实10片+整板3D请求成功，52个有效测点（wire42/thin10）坐标范围、原始深度、参考面差及阈值一致性通过。
- 性能：上一轮52个公开测量区域多边形重建+真实深度回放，9次中位：旧测高核心127.817ms，新关闭15.911ms，新开启2/1为23.014ms；关闭结果逐字段完全一致。该对照不包含配准、推理、公开几何生成，不能当作整板加速结论。
- 实测产物：visualization/output/wire-height-erosion-20261018/live-report/report.html；benchmark.json、live-validation.json、tests.log同级。整板本次96.946s，受GPU推理和银浆配准耗时影响，未与旧运行做受控端到端速度比较。
- 限制/后续：2/1为可调初值，需要按实际线宽/反光范围标注校准，修改需重新初始化。实测银浆SIFT仍存在既有可靠性回退，不属于本次改动。

## 银浆溢出边界切换为检测 ROI 四边像素内缩
- 用户确认优先px并要求注意耗时。当前配置 boundary.mode=roi_inset、inset_unit=px、insets左376/上146/右34/下128；初值由原标定模板的检测ROI与原四边形各边中点距离取整得到（模板圆4379,1360,r569，crop2059,95,3847,2591）。
- 实现：直接复用当前片crop_box生成矩形，像素单位不依赖圆半径；可选radius_ratio。有限非负四边参数必填，内缩后宽高须正且crop/shape匹配；无效uncertain/is_ok=null，无模板回退。右下连续边界使用x2/y2，0内缩覆盖全ROI。
- 性能：构造阶段跳过模板IO/hash/解码/SIFT/FLANN/RANSAC，也不为边界预先转灰度；Boundary.roi_inset用轴对齐矩形精确有符号欧氏距离（含角点）取代通用多边形距离，容差和连通域语义保持。
- 返回：C#调用/字段不变，Boundary.Mode及Registration.Method新增值roi_inset；Status=ok/IsValid=true表示生成成功，非模板配准；Points仍原图坐标。旧polygon模板模式保留。可视化注明ROI内缩边界、检查卡银浆边界生成。
- 验证：153项算法+26项报告+8项前端测试通过；新模式无模板/转灰路径有断言；多密度/容差/面积下与相同矩形的通用polygon距离/缺陷掩码一致。真实10片及整板3D完成，检测19.147s（启动另计），10片边界坐标/联合结果一致、日志无模板阶段/回退。
- 基准：10个历史实拍银浆公开轮廓重建，同一矩形几何、6次中位，通用polygon溢出核心1409.952ms -> roi_inset 602.805ms；单片构造中位0.08465ms。该基准不包含模板/GPU/3D，整板耗时变化不能全归因于此核优化。
- 文件：silver_inspection.py、detect_board.py、config/2D/config.json及README、process/OUTPUT_PARAMETERS.md、visualization/viewer.js及html/README、tests/test_silver_roi_inset.py、visualization/tests/test_bending_viewer.cjs。
- 实测报告：visualization/output/silver-roi-inset-20261018/live-report/report.html；benchmark.json/live-validation.json/tests.log同级。四边值需按现场样本调整；矩形不补偿旋转/透视。无新增C#字段，硬编码polygon/SIFT字符串的上位机需识别roi_inset。
