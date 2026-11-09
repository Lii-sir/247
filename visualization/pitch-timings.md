# Python Pitch矫正耗时字段

新增计时输出在现有响应的timings中，板级通常位于timings_board。矫正关闭时不产生投影分项。不改变投影公式、输出像素或参数。

## 先看实际耗时

|字段|含义|
|---|---|
|smartray_pitch_ms|整个Pitch函数的墙钟耗时：实际等待多久|
|pitch_prepare_ms|参数检查、三角函数、任务池准备与条带划分|
|pitch_projection_wall_ms|提交全部投影任务直到全部返回的墙钟耗时，包含调度、等待及任务计时汇总|
|pitch_executor_close_ms|本次调用自建线程池的关闭耗时；使用常驻池时不产生|
|pitch_output_allocate_ms|分配最终整板输出数组，包含同步强度图输出（若启用）|
|pitch_output_merge_ms|将各条带复制到最终整板，汇总点数|

上述各边界不含所有元数据、计时及日志开销，因此不要求严格加和等于smartray_pitch_ms。

## 再看累计计算落在哪里

以下work字段是所有条带对应子步骤时间之和，不是实际整板延迟：

|字段|具体操作|
|---|---|
|pitch_projection_work_ms|原有字段：全部条带任务的累计工作时间|
|pitch_strip_view_work_ms|取源数据条带视图和尺寸|
|pitch_valid_points_work_ms|判断0<depth<65535，nonzero提取有效点行列索引|
|pitch_z_decode_work_ms|读取有效点深度，转float64并按零点、Z步距解码高度|
|pitch_transform_work_ms|计算扫描方向偏移与矫正高度：tan/cos或sin/cos公式|
|pitch_round_coordinates_work_ms|复制坐标、计算并取整目标行列及深度编码|
|pitch_filter_indices_work_ms|判断坐标/深度越界，筛选目标索引与uint16编码|
|pitch_strip_allocate_work_ms|为当前条带分配零值深度输出；页面提交不等于所有物理页写入均在此发生|
|pitch_scatter_max_work_ms|np.maximum.at散射写入及碰撞取最大深度；首次写页成本可能也落在这里|
|pitch_intensity_work_ms|可选同步强度图：挑选胜出源像素、生成对应强度图；正式3D-only路径接近0|
|pitch_output_count_work_ms|统计输出非零点数，组织条带返回值|
|pitch_projection_other_work_ms|任务总时间减上述分项，包括局部临时数组释放、函数/计时边界等未归属开销|

累计子步骤加other约等于projection_work。不要把projection_work和其分项再相加，也不要把它们与projection_wall相加。线程竞争、内存等待和被系统换出的时间仍包含在这些墙钟测量内，不能仅凭某项偏大就断定对应算术运算慢。

## 判断是否有慢条带

- pitch_projection_max_strip_ms：最慢条带函数体时间，不含线程开始执行前的等待、函数返回后的局部数组释放。
- pitch_projection_start_delay_max_ms：从调用任务调度前到各条带开始执行的最大延迟。它包含提交顺序与排队/线程启动影响，不是精确的单任务队列计时；它与projection_wall有重叠。

开启现有TIMING日志（BRIDGE_TIMING_LOG=1，且调用方提供日志接收器）后，新增event=pitch_projection_details。strips列表包含每个条带的范围、分项timings、wall_ms、start_delay_ms、输入有效点数、投影有效点数和输出有效点数。该列表只在日志中输出，不放进检测结果几何字段。

下一次分析时提供smartray_pitch_ms、全部pitch_*字段，最好附对应pitch_projection_details日志。由分项与最慢条带对照，判断瓶颈是有效点扫描、浮点数组运算、碰撞写入、任务启动还是最终汇总。

此计时针对现有Python流程；独立SmartRay.PitchCorrection.dll不会输出这些字段。未做实机测试，计时开销和实际占比需在目标机器确认。
