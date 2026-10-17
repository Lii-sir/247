
## 引线弯曲检测改为 ROI 四交点评估
- 已完成：共享 measure_wire_bending 使用 ROI 周界两个接触区间的四端点构造四边形；覆盖率为其中实例像素数 / ROI 内实例像素数。ROI 为半开区间。两条 2D 入口及联合高度回退一致；汇总复用清理后的 ROI 分配。
- 文件：python/algorithm/2D/defect_geometry.py、wire_roi_presence.py、detect_2D.py、detect_board.py；tests/test_wire_bending.py；config/2D/README.md、process/OUTPUT_PARAMETERS.md；src/Bridge/BridgeDtos.cs（现有 src 忽略规则下的本地源码）。
- 决策：保留 rectangle/mask_ratio/rectangle_ratio_threshold 字段及 0.75 初始阈值；增加 status、roi_id、分子分母像素数；交点不足、多端口、退化均无效 NG。不得回退旧旋转矩形；清理失败用原掩码标识故障 ROI。
- 验证：134 Python 测试 + 26 报告测试通过；Bridge Debug/Release 均零警告零错误。实跑10张2D及整板3D完成；72个2D实例与52个联合测量一致性校验通过。报告 visualization/output/wire-bending-validation-20261017/report.html。
- 限制/复盘：新阈值需样本校准；实跑银浆并发配准记录5次内存不足回退，与引线逻辑独立。临时目录改用 workspace 下路径解决 Windows sandbox 清理权限。未改用户已有配置值和根目录 Bridge.dll。

## 引线弯曲结果可视化修复
- 原因：canvas 原先只按类别画掩码，无弯曲四边形/结论；测量表只读联合测高实例，遗漏部分即时2D测量。
- 完成：合并全部2D引线和对应3D高度；图上按OK绿色、NG红色、ROI外灰色绘制四边形、编号、覆盖率；缺线ROI标红；表格含原因及汇总，点击编号放大定位。
- 文件：visualization/viewer.js、viewer.html、viewer.css、README.md、tests/test_bending_viewer.cjs。
- 验证：7项JS DOM/Canvas双替身交互测试、26项Python报告测试通过；已重新生成实测报告，逐片核对72条引线（43 OK、9 NG、20 ROI外）全部显示。报告路径沿用 visualization/output/wire-bending-validation-20261017/report.html。
- 限制：本机HTTP预览连接超时；浏览器策略拒绝file协议，未绕过；未完成浏览器截图验证，以实际脚本绘制指令和DOM交互测试验证。算法及已有检测JSON未修改。
