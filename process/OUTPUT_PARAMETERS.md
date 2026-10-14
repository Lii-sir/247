# 上位机接口返回参数说明

更新日期：2026-10-14。依据当前 `PythonBridge.cs`、`BridgeDtos.cs`（代码版本 `9d9ab70`）。

**本文只说明上位机调用 Bridge 后取得的返回对象和参数。** 不包含测试脚本输出目录、
CSV、日志、时间线或可视化专用字段。JSON 名称与 C# 属性名称对照列出；
使用与当前源码匹配的 DLL。

**哪些返回字段可用于上位机绘图、标签和结果卡片，直接查看第 14 节。**

## 1. 调用接口与返回类型

| 上位机接口                                                                                                       | 返回类型                 | 主要内容                                             |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------- |
| `bridge.Init()`                                                                                                | `BridgeBaseResponse`   | 通信初始化状态                                       |
| `bridge.InitAlgorithm()` / `InitAlgorithm(stage)`                                                            | `BridgeBaseResponse`   | 算法初始化状态、模型加载耗时                         |
| `bridge.TwoD.DetectPieceFile(...)` / `DetectPieceSharedMemory(...)` / `DetectPieceSharedMemoryLayout(...)` | `DetectResponse`       | 指定板号、片号的 2D 结果及缓存状态                   |
| 上述单片接口的`...Async(...)` 版本                                                                             | `Task<DetectResponse>` | `await` 后得到同样的返回对象                       |
| `bridge.ThreeD.DetectBoardFile(...)` / `DetectBoardSharedMemory(...)`                                        | `DetectResponse`       | 整板 3D 与同板 2D 联合结果，主要读取`ResultsBoard` |
| 上述整板 3D 接口的`...Async(...)` 版本                                                                         | `Task<DetectResponse>` | `await` 后得到同样的返回对象                       |
| `bridge.Board.DetectStreamingFilesAsync(...)`                                                                  | `Task<DetectResponse>` | 提交十片 2D 与一张整板 3D，返回最终联合整板结果      |
| `bridge.Board.DetectFiles(...)` / `DetectSharedMemory(...)`                                                  | `DetectResponse`       | 批量板级结果；不保证执行全部流式 2D 检查             |
| `bridge.TwoD.Detect2DFile(...)` / `Detect2DSharedMemory(...)`                                                | `DetectResponse`       | 独立 2D 结果，主要读取`Results2D`                  |
| `bridge.ThreeD.Detect3DFile(...)` / `Detect3DSharedMemory(...)`                                              | `DetectResponse`       | 独立 3D 结果，主要读取`Results3D`                  |
| `bridge.ReleaseBoard(boardId)`                                                                                 | `DetectResponse`       | 指定板缓存释放状态                                   |
| `bridge.Release()`                                                                                             | `BridgeBaseResponse`   | 服务及模型资源释放状态                               |
| `bridge.ReloadDefectConfig(...)` / `ReloadDefectConfigText(...)`                                             | `BridgeBaseResponse`   | 检测配置更新状态                                     |

初始化失败、参数校验失败、通信中断、请求超时等可能直接抛出 C# 异常，
不一定返回 `Ok=false` 对象。检测服务返回的业务/执行失败则可能通过返回对象描述。

## 2. 通用返回字段：`BridgeBaseResponse`

`DetectResponse` 继承这些字段。

| JSON 字段               | C# 属性               | 类型                          | 说明                                                                       |
| ----------------------- | --------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| `ok`                  | `Ok`                | `bool`                      | 请求是否执行成功；不是产品合格结论                                         |
| `request_id`          | `RequestId`         | `string?`                   | 请求唯一标识，用于关联响应；不是板号                                       |
| `cmd`                 | `Cmd`               | `string?`                   | 执行命令，例如`init_algorithm`、`detect_2d_piece`、`detect_3d_board` |
| `stage`               | `Stage`             | `string?`                   | 接口阶段：`2d/3d/board`；有些通用操作不提供                              |
| `message`             | `Message`           | `string?`                   | 人类可读的状态说明                                                         |
| `error`               | `Error`             | `string?`                   | 错误码或错误原因；正常时通常为空                                           |
| `timings`             | `Timings`           | `Dictionary<string,double>` | 服务端与 Bridge 填入的阶段统计；详见第 11 节                               |
| `model_load_timings`  | `ModelLoadTimings`  | `Dictionary<string,double>` | 初始化时模型加载耗时，单位毫秒，例如`YOLO/chip_short/chip_long`          |
| `csharp_roundtrip_ms` | `CsharpRoundtripMs` | `double`                    | C# 请求发送至对应原始响应返回的往返耗时，毫秒                              |
| `csharp_total_ms`     | `CsharpTotalMs`     | `double`                    | 检测接口内部计时范围的总耗时，毫秒；初始化等操作不一定填充                 |

### 2.1 状态读取规则

- **先检查 `Ok`，再检查对应结果对象中的 `IsOk`。**
- `Ok=true`、`IsOk=false`：请求正常，产品判定 NG。
- `bool? IsOk=null`：未提供确定结论，不能解释为 OK。
- 结果对象为 `null`：该接口未返回这一类结果或执行失败。
- DTO 的普通 `bool/int/double` 在 JSON 缺字段时可能保留 `false/0` 默认值，
  不能仅凭默认值推断算法实际测到了什么；应结合接口类型、`Status` 和对象是否存在。

## 3. 检测返回字段：`DetectResponse`

| JSON 字段                         | C# 属性                       | 类型                | 说明                                                                 |
| --------------------------------- | ----------------------------- | ------------------- | -------------------------------------------------------------------- |
| `elapsed_ms`                    | `ElapsedMs`                 | `double`          | 服务端汇总耗时，毫秒；接口不同，计时边界不同，不等于 C# 完整调用时间 |
| `board_id`                      | `BoardId`                   | `string?`         | 流式检测板号，2D 与 3D 通过它关联                                    |
| `batch_id`                      | `BatchId`                   | `string?`         | 批量板号；部分联合响应也返回该兼容字段                               |
| `piece_id`                      | `PieceId`                   | `int?`            | 单片 2D 的片号；整板响应通常不提供                                   |
| `results_2d`                    | `Results2D`                 | `Results2D?`      | 单片/独立 2D 结果                                                    |
| `results_board`                 | `ResultsBoard`              | `ResultsBoard?`   | 联合或批量整板结果                                                   |
| `results_3d`                    | `Results3D`                 | `Results3D?`      | 独立 3D 检测结果；与`ResultsBoard` 不同                            |
| `cache_state`                   | `CacheState`                | `CacheState?`     | 流式 2D 的板数据缓存状态                                             |
| `released`                      | `Released`                  | `ReleaseSummary?` | 本次板缓存释放情况                                                   |
| `missing_piece_ids`             | `MissingPieceIds`           | `List<int>`       | 等待输入超时等情况下的缺失片号；部分 C# 汇总失败路径会填入失败片号   |
| `csharp_image_read_ms`          | `CsharpImageReadMs`         | `double`          | 文件接口记录的读图/输入准备时间，毫秒；外部共享内存接口通常为 0      |
| `csharp_shared_memory_write_ms` | `CsharpSharedMemoryWriteMs` | `double`          | Bridge 创建/写入共享内存的时间，毫秒；外部已写好共享内存时为 0       |

### 3.1 根据调用方式选择结果对象

| 调用场景                | 应读取                           |
| ----------------------- | -------------------------------- |
| 逐片 2D                 | `response.Results2D`           |
| 一块板的 2D+3D 联合检测 | `response.ResultsBoard.Pieces` |
| 独立 3D                 | `response.Results3D.Pieces`    |

`response.GetRectangles2D()` 只读取 `Results2D.Rectangles`。
`response.GetPieces3D()` 只读取独立的 `Results3D.Pieces`，**不会返回联合整板的片列表**。

`DetectStreamingFilesAsync` 成功时返回的是最终 3D 整板响应，不会自动把十条单片
`DetectResponse` 嵌套到其中。需要逐片即时结果时，分别调用单片 2D 接口取得返回对象。

## 4. 单片 2D：`Results2D`

| JSON 字段                | C# 属性                      | 说明                                                       |
| ------------------------ | ---------------------------- | ---------------------------------------------------------- |
| `is_ok`                | `IsOk`                     | 单片 2D 综合结论                                           |
| `circle`               | `Circle`                   | 圆定位结果：`X/Y/Radius`，单位像素，使用完整 2D 原图坐标 |
| `crop_box`             | `CropBox`                  | 检测 ROI，格式`[x1,y1,x2,y2]`，使用原图坐标              |
| `wire_rois_2d`         | `WireRois2D`               | 引线 ROI 列表，坐标相对于检测 crop                         |
| `rectangles`           | `Rectangles`               | 弯曲评估矩形列表：`Points/MaskRatio/IsOk`                |
| `instances`            | `Instances`                | 分割实例列表                                               |
| `wire_bending_2d`      | `WireBending2D`            | 引线 ROI 和弯曲综合诊断                                    |
| `silver_continuity_2d` | `SilverContinuity2D`       | 银浆连通诊断                                               |
| `silver_overflow_2d`   | **`SilverOverflow`** | 银浆溢出诊断，注意该 C# 属性没有`2D` 后缀                |
| `chip_rotation_2d`     | `ChipRotation2D`           | 芯片旋转诊断                                               |
| `chip_anomaly_2d`      | `ChipAnomaly2D`            | 芯片外观诊断                                               |

### 4.1 分割实例：`InstanceResult2D`

| JSON 字段               | C# 属性               | 说明                                                        |
| ----------------------- | --------------------- | ----------------------------------------------------------- |
| `instance_id`         | `InstanceId`        | 当前片内实例编号，不是跨片/跨批次跟踪 ID                    |
| `class_id/class_name` | `ClassId/ClassName` | 模型类别编号/名称，常见名称为`chip/wire/thin/bond/silver` |
| `confidence`          | `Confidence`        | 模型检测置信度，不是产品通过概率                            |
| `mask_2d`             | `Mask2D`            | 分割掩码几何，使用完整 2D 原图坐标                          |
| `rectangle`           | `Rectangle`         | 弯曲评估矩形顶点，不适用时可能为空                          |
| `rectangle_ratio`     | `RectangleRatio`    | 对应矩形掩码覆盖率；不适用实例可能保留默认值                |
| `bending_2d`          | `Bending2D`         | 当前实例的弯曲结果                                          |
| `chip_rotation_2d`    | `ChipRotation2D`    | 当前芯片实例旋转结果                                        |
| `chip_anomaly_2d`     | `ChipAnomaly2D`     | 当前芯片实例外观结果                                        |
| `is_ok`               | `IsOk`              | 当前实例综合结论                                            |

### 4.2 共用掩码几何：`MaskGeometry`

| JSON 字段        | C# 属性         | 类型/单位       | 说明                                           |
| ---------------- | --------------- | --------------- | ---------------------------------------------- |
| `area_px`      | `AreaPx`      | `int`，像素数 | 实际掩码面积，不是矩形面积                     |
| `bounding_box` | `BoundingBox` | `List<int>`   | `[x1,y1,x2,y2]`；右/下端点为排他边界         |
| `contours`     | `Contours`    | 多组点列表      | 每组为一个外轮廓；可能经过简化，不是逐像素掩码 |

坐标系由字段位置确定：`Mask2D` 是 2D 原图坐标，`Mask3DLocal` 是本片局部深度坐标。

### 4.3 引线弯曲：`WireBending2DResult` / `WireBendingResult`

`WireBending2D` 提供 `Enabled/IsOk/RoiChecks/Measurements`。

| 对象/属性                      | 说明                                                               |
| ------------------------------ | ------------------------------------------------------------------ |
| `RoiChecks[].RoiId`          | 配置 ROI 编号                                                      |
| `RoiChecks[].Box`            | 引线 ROI 在完整 2D 原图中的矩形                                    |
| `RoiChecks[].MaskPixelCount` | 分配给该 ROI 的引线/细线掩码像素数                                 |
| `RoiChecks[].InstanceIds`    | 参与检查的实例 ID                                                  |
| `RoiChecks[].BendingChecks`  | 对应实例的覆盖率和有效性/通过状态                                  |
| `RoiChecks[].Status`         | 常见为`ok/wire_missing/bending_unavailable/bending_out_of_range` |
| `Measurements[]`             | `RegionId/InstanceId/ClassName/Mask2D/Bending2D` 等逐实例测量    |
| `Bending2D.IsValid`          | 是否成功得到有效弯曲测量                                           |
| `Bending2D.IsOk`             | 是否满足弯曲条件                                                   |
| `Bending2D.MaskRatio`        | 弯曲矩形的掩码覆盖率                                               |
| `Bending2D.Rectangle`        | 原图矩形顶点                                                       |

## 5. 银浆诊断：`Diagnostic2D`

适用于 `Results2D.SilverOverflow`、`Results2D.SilverContinuity2D`，
以及联合片结果中的 `SilverOverflow2D/SilverContinuity2D`。

| JSON 字段        | C# 属性          | 类型                        | 说明                                                   |
| ---------------- | ---------------- | --------------------------- | ------------------------------------------------------ |
| `enabled`      | `Enabled`      | `bool`                    | 是否启用；**仅在服务实际返回该字段时有明确意义** |
| `status`       | `Status`       | `string`                  | 诊断状态                                               |
| `is_ok`        | `IsOk`         | `bool?`                   | 通过/不通过/无法形成确定结论                           |
| `message`      | `Message`      | `string?`                 | 失败、未配置或不确定原因                               |
| `boundary`     | `Boundary`     | `BoundaryResult2D?`       | 实际允许边界                                           |
| `registration` | `Registration` | `BoundaryRegistration2D?` | 模板边界配准/回退情况                                  |
| `measurement`  | `Measurement`  | **`JsonElement?`**  | 动态检测明细，使用 JSON 属性读取，不是强类型测量 DTO   |

流式银浆响应可能没有 `enabled`，此时 C# 的 `Enabled` 保持默认 `false`。
**不要仅凭该默认值认为银浆未运行，应结合 `Status`、`IsOk` 和测量对象判断。**

银浆溢出 `Status`：`ok/overflow/no_silver/not_configured/uncertain/disabled`。
银浆连通 `Status`：`ok/disconnected/uncertain/disabled`。
`no_silver/not_configured/uncertain/disabled` 不应作为检测通过处理。

### 5.1 银浆允许边界：`BoundaryResult2D`

| JSON 字段             | C# 属性              | 说明                                                                     |
| --------------------- | -------------------- | ------------------------------------------------------------------------ |
| `mode`              | `Mode`             | 当前主要使用`polygon`                                                  |
| `points`            | `Points`           | 实际允许边界顶点，不是 SIFT 特征点                                       |
| `coordinate_system` | `CoordinateSystem` | 正常公开结果为`image`，表示目标原图坐标；内部局部坐标可标记为 `crop` |
| `is_within_crop`    | `IsWithinCrop`     | `bool?`；是否全部位于检测 ROI                                          |

**边界允许超出 ROI。** `IsWithinCrop=false` 不表示产品 NG，也不阻止检测。
算法保留完整多边形，只评价 ROI 内实际分割到的银浆；不能保证 ROI 外未检测区域没有缺陷。

### 5.2 配准和回退：`BoundaryRegistration2D`

| JSON 字段                                       | C# 属性                                      | 说明                                                                                               |
| ----------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `method`                                      | `Method`                                   | `sift_ransac`：匹配；`template_points_fallback`：直接使用模板点；兼容模式可为 `fixed_points` |
| `status`                                      | `Status`                                   | 常见为`ok/fallback/failed/not_configured`；与缺陷诊断状态不同                                    |
| `is_valid`                                    | `IsValid`                                  | 是否为有效配准，回退时为`false`                                                                  |
| `fallback_used`                               | `FallbackUsed`                             | 是否使用模板原始点回退                                                                             |
| `matches`                                     | `Matches`                                  | `int?`；筛选及去重后的候选匹配数                                                                 |
| `inliers`                                     | `Inliers`                                  | `int?`；RANSAC 内点数                                                                            |
| `inlier_ratio`                                | `InlierRatio`                              | `double?`；内点数/候选匹配数，不是概率置信度                                                     |
| `reprojection_error_px`                       | `ReprojectionErrorPx`                      | `double?`；内点平均重投影误差，目标原图像素单位；回退时通常缺失                                  |
| `template_name/template_size/template_sha256` | `TemplateName/TemplateSize/TemplateSha256` | 模板名称、尺寸`[width,height]` 和标定哈希                                                        |
| `homography_template_to_image`                | `HomographyTemplateToImage`                | 模板原图到目标原图的 3×3 矩阵，不是到检测 crop 的矩阵                                             |
| `message`                                     | `Message`                                  | 匹配失败或回退原因                                                                                 |

回退时 `FallbackUsed=true`、`Status="fallback"`，直接使用模板原图点，
不缩放、不平移。单位矩阵仅表示直接使用这些坐标，不代表 SIFT 成功。
回退后仍可能检测 OK 或 NG，应读取 `Diagnostic2D.IsOk`，不能直接用配准有效性替代银浆缺陷结论。

### 5.3 `Measurement` 的动态 JSON 字段

银浆溢出：

| JSON 字段                   | 类型/单位    | 说明                                                   |
| --------------------------- | ------------ | ------------------------------------------------------ |
| `silver_area_px`          | 整数，像素数 | 参与检测的银浆面积                                     |
| `outside_area_px`         | 整数，像素数 | 边界外银浆面积                                         |
| `candidate_area_px`       | 整数，像素数 | 超过距离容差的候选面积                                 |
| `defect_area_px`          | 整数，像素数 | 最小区域面积筛选后的实际缺陷面积                       |
| `defect_ratio`            | 小数         | 缺陷面积/银浆面积                                      |
| `max_outside_distance_px` | 小数，像素   | 相对边界的最大外侧距离                                 |
| `regions`                 | 数组         | 越界区域的`area_px/box_xyxy/max_outside_distance_px` |
| `defect_contours`         | 多组点列表   | 越界缺陷轮廓，正常公开结果为原图坐标                   |

银浆连通：`chip_area_px/ring_area_px/silver_area_px/occluded_area_px/valid_area_px`
为对应像素统计；`covered_sector_count/valid_sector_count/ignored_sector_count/missing_sector_count`
为扇区统计；`coverage_ratio` 为覆盖扇区/有效扇区；`center_xy/sectors/chips`
提供中心、扇区和逐芯片明细。这些字段直接从 `Measurement` 的 JSON 读取。

## 6. 芯片旋转与外观

### 6.1 `ChipAlignment2DResult` / `ChipAlignmentEntry2D`

`ChipRotation2D`：`Enabled/IsOk/Status/MaxAngleDeg/Chips`。

| 芯片条目属性             | 说明                                      |
| ------------------------ | ----------------------------------------- |
| `InstanceId/ClassName` | 对应实例                                  |
| `CenterXy/Rectangle`   | 原图中心和矩形顶点                        |
| `WidthPx/HeightPx`     | 旋转矩形长、短边像素长度                  |
| `AngleDeg`             | 芯片主方向角，度                          |
| `AxisDeviationDeg`     | 相对最近图像轴的有符号偏角，度            |
| `MaxAngleDeg`          | 实际允许偏角阈值                          |
| `MaskFillRatio`        | 掩码与旋转矩形的面积比                    |
| `IsOk/Status`          | 当前芯片旋转结论，例如`aligned/rotated` |

旋转判定主要比较 `abs(AxisDeviationDeg)` 与 `MaxAngleDeg`，不是直接比较 `AngleDeg`。

### 6.2 `ChipAnomaly2DResult` / `ChipAnomalyEntry2D`

`ChipAnomaly2D`：`Enabled/IsOk/Status/RatioDefinition/AspectRatioModels/Chips`。

| 芯片条目属性（JSON）                                      | 说明                                                                   |
| --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `InstanceId/ClassName`（`instance_id/class_name`）    | 对应实例                                                               |
| `Model/Category`（`model/category`）                  | 使用的外观模型及类别标签                                               |
| `AspectRatio`（`aspect_ratio`）                       | 长边/短边比例                                                          |
| `ModelAspectRatioRange`（`model_aspect_ratio_range`） | 对应模型的比例区间                                                     |
| `RectificationAngleDeg/Rectangle`                       | 矫正角度和原图芯片矩形                                                 |
| `RectifiedSize/ModelInputSize`                          | 矫正后图片、模型输入的`[width,height]`                               |
| `Score`（`score`）                                    | 当前评分规则最终用于判断的异常分数                                     |
| `ScoreTopK/ScoreMax`（`score_topk/score_max`）        | 辅助异常分数，不能随意替代最终`Score`                                |
| `Threshold`（`threshold`）                            | 当前芯片实际生效阈值，各模型独立                                       |
| `Prediction/IsOk`（`prediction/is_ok`）               | `Score > Threshold` 为 NG，否则为 OK                                 |
| `DecisionRule`（`decision_rule`）                     | 当前判定规则文本                                                       |
| `ActiveScales`（`active_scales`）                     | 评分尺度信息，某些模式下为空                                           |
| `AnomalyBoxes`（`anomaly_boxes`）                     | 异常框列表：`CropXyxy` 是矫正图位置，`ImagePolygon` 是原图异常区域 |

当前强类型 DTO 没有 `ScoreMode/ScorePoolKernel/ThresholdSource` 属性，即使 Python
原始 JSON 包含对应字段，也不会自动变成 C# 可访问属性。

## 7. 联合整板返回：`ResultsBoard`

| JSON 字段           | C# 属性            | 说明                                         |
| ------------------- | ------------------ | -------------------------------------------- |
| `is_ok`           | `IsOk`           | 整板产品综合结论                             |
| `registration_ok` | `RegistrationOk` | 各片联合配准`IsValid` 是否全部为真         |
| `pieces`          | `Pieces`         | `List<BoardPieceResult>`；当前场景应为十片 |

### 7.1 单片联合结果：`BoardPieceResult`

| JSON 字段                         | C# 属性                        | 说明                                                            |
| --------------------------------- | ------------------------------ | --------------------------------------------------------------- |
| `piece_id`                      | `PieceId`                    | 片号，与 2D 请求对应                                            |
| `timings`                       | `Timings`                    | 当前片联合 3D 耗时字典，数值单位毫秒                            |
| `image_2d_size`                 | `Image2DSize`                | 原始 2D 尺寸`[width,height]`                                  |
| `crop_2d`                       | `Crop2D`                     | 2D 检测 crop 的原图矩形                                         |
| `wire_rois_2d`                  | `WireRois2D`                 | 相对于检测 crop 的引线 ROI                                      |
| `wire_roi_checks`               | `WireRoiChecks`              | 引线 ROI 存在性/弯曲诊断                                        |
| `depth_start_row/depth_end_row` | `DepthStartRow/DepthEndRow`  | 本片在整板深度图的行范围，结束行不包含                          |
| `height_point_coordinate_space` | `HeightPointCoordinateSpace` | 当前正常联合结果为`board_3d`                                  |
| `registration`                  | `Registration`               | `RegistrationResult`，当前片联合配准                          |
| `is_ok`                         | `IsOk`                       | 当前片的联合产品结论                                            |
| `reference_planes`              | `ReferencePlanes`            | 平台/芯片参考面                                                 |
| `wire_measurements`             | `WireMeasurements`           | 引线弯曲和高度结果                                              |
| `solder_measurements`           | `SolderMeasurements`         | 焊点及细线端部高度结果                                          |
| `chip_anomaly_2d`               | `ChipAnomaly2D`              | 缓存的芯片外观诊断                                              |
| `silver_overflow_2d`            | **`SilverOverflow2D`** | 缓存的银浆溢出诊断；与`Results2D.SilverOverflow` 的属性名不同 |
| `silver_continuity_2d`          | `SilverContinuity2D`         | 缓存的银浆连通诊断                                              |
| `masks`                         | `Masks`                      | `List<AlignedMaskResult>`，对齐后的实例输出                   |

### 7.2 联合配准：`RegistrationResult`

| JSON 字段           | C# 属性          | 说明                                         |
| ------------------- | ---------------- | -------------------------------------------- |
| `scale`           | `Scale`        | 尺度参数                                     |
| `tx/ty`           | `Tx/Ty`        | 到本片局部深度坐标的平移参数，像素           |
| `score`           | `Score`        | 配准匹配评分，不是产品缺陷分数               |
| `is_valid`        | `IsValid`      | 当前配准是否被算法判为有效                   |
| `matrix_2d_to_3d` | `Matrix2DTo3D` | 返回的 2×3 映射参数，目标为本片局部深度坐标 |

当前 DTO 没有 3×3 `homography_2d_to_3d`、仿射/透视详细诊断属性。
`Matrix2DTo3D` 不应被当作已包含完整投影分母的 3×3 透视矩阵。
这些额外 Python 字段默认反序列化时被忽略；本文不把它们列为现有上位机属性。

## 8. 引线、焊点和高度

### 8.1 `WireMeasurementResult`

`RegionId/InstanceId/ClassName` 标识引线；`Mask2D` 为原图几何；
`Mask3DLocal` 为局部深度几何；`Bending2D` 为弯曲结果；`Height3D` 为高度测量。

### 8.2 `SolderMeasurementResult`

| JSON 字段                 | C# 属性                 | 说明                                    |
| ------------------------- | ----------------------- | --------------------------------------- |
| `measurement_id`        | `MeasurementId`       | 测量编号                                |
| `source_instance_id`    | `SourceInstanceId`    | 来源分割实例 ID                         |
| `class_name`            | `ClassName`           | 来源类别                                |
| `endpoint`              | `Endpoint`            | `bond`：焊点；`start/end`：细线端部 |
| `location`              | `Location`            | 参考位置类别，例如平台或芯片            |
| `reference_instance_id` | `ReferenceInstanceId` | 关联参考芯片编号，可能为空              |
| `chip_inside`           | `ChipInside`          | 是否位于关联芯片内                      |
| `reference_plane`       | `ReferencePlane`      | 实际用于测高的参考面                    |
| `mask_2d/mask_3d_local` | `Mask2D/Mask3DLocal`  | 原图/局部深度的测量区域                 |
| `height_3d`             | `Height3D`            | 高度结果                                |

### 8.3 `HeightMeasurementResult`

| JSON 字段                  | C# 属性                  | 类型/单位         | 说明                                                       |
| -------------------------- | ------------------------ | ----------------- | ---------------------------------------------------------- |
| `reference`              | `Reference`            | `string`        | 使用的参考面标签                                           |
| `is_valid`               | `IsValid`              | `bool`          | 是否有有效深度点和可用参考面                               |
| `is_ok`                  | `IsOk`                 | `bool?`         | 高度差是否满足对应阈值；无确定判定时可能为空               |
| `height_mm`              | `HeightMm`             | `double?`，毫米 | 按灰度基准换算的测量点高度，不是相对参考面的高度差         |
| `height_diff_mm`         | `HeightDiffMm`         | `double?`，毫米 | 测量点相对于同位置参考面的高度差，通常是业务判断主要读取值 |
| `point`                  | `Point`                | `ImagePoint?`   | 测量像素点`X/Y`，不是毫米位置；无效时可能为空            |
| `point_coordinate_space` | `PointCoordinateSpace` | `string?`       | 当前正常联合结果为`board_3d`                             |

代表点按配置的高度百分位选择，不必等于单个最高像素。
业务应优先检查 `IsValid` 和 `IsOk`，再使用高度数值；不要把 `null` 当 0。

**联合测高点已使用整板深度原图坐标，不需要再次加 `DepthStartRow`。**
在本片局部预览中使用 `Point.Y - DepthStartRow`。
旧 DLL/旧结果可能缺少坐标标签，须确认来源，不能自动重复偏移。

### 8.4 参考面：`ReferencePlanesResult` / `PlaneResult`

- `ReferencePlanes.WirePlatform`：平台参考面，可能为空。
- `ReferencePlanes.Chips[]`：`InstanceId/Plane`，各芯片参考面。
- `Plane.Name`：参考面标签。
- `Plane.Coefficients`：`[a,b,c]`，按局部坐标拟合 `depth_gray = a*x + b*y + c`；
  系数对应深度灰度，不是直接的毫米值。
- `Plane.ValidPoints`：保留的拟合采样点数。

参考面和 `Mask3DLocal` 保持本片局部坐标，不会因测高点转成整板坐标而平移。
当前 C# 类型没有 Python 的 `platform_roi` 采样诊断属性。

### 8.5 对齐实例：`AlignedMaskResult`

`InstanceId/ClassId/ClassName/Confidence`：分割实例标识；
`Mask2D/Mask3DLocal`：原图和局部深度轮廓；
`Bending2D/Height3D`：对应弯曲/高度结果；
`BondLocation/ReferenceInstanceId/ChipInside`：焊点参考关联；
`Plane3D`：芯片参考面；`ChipAnomaly2D`：芯片外观诊断。
并非每一种类别都提供所有测量对象，应处理空值。

## 9. 独立 3D 返回：`Results3D`

仅对应 `ThreeD.Detect3DFile/Detect3DSharedMemory` 等独立 3D 接口。
联合接口读取第 7、8 节的 `ResultsBoard`，不要混用。

| 对象/JSON 字段                   | C# 属性               | 说明                                       |
| -------------------------------- | --------------------- | ------------------------------------------ |
| `results_3d.is_ok`             | `Results3D.IsOk`    | 独立 3D 综合结论                           |
| `results_3d.pieces`            | `Results3D.Pieces`  | `List<Piece3D>`                          |
| `pieces[].piece_id`            | `PieceId`           | 片号                                       |
| `pieces[].is_ok`               | `IsOk`              | 当前片独立 3D 结论                         |
| `pieces[].substrate_height_mm` | `SubstrateHeightMm` | 基板基准高度，毫米                         |
| `pieces[].points`              | `Points`            | `List<DepthPoint3D>`，各中心线代表测量点 |
| `points[].height_mm`           | `HeightMm`          | 点的灰度换算高度，毫米                     |
| `points[].height_diff_mm`      | `HeightDiffMm`      | 点相对于本片基板基准的高度差，毫米         |
| `points[].x/y`                 | `X/Y`               | 整板深度图像素坐标，非毫米位置             |

独立 3D 的基板基准方式与联合检测的芯片/平台平面方式不同，不能默认结果完全等价。

## 10. 缓存状态和释放状态

### 10.1 `CacheState`

| JSON 字段                | C# 属性                | 说明                                                       |
| ------------------------ | ---------------------- | ---------------------------------------------------------- |
| `board_id`             | `BoardId`            | 缓存所属板号                                               |
| `expected_piece_count` | `ExpectedPieceCount` | 配置要求的片数，当前场景为 10                              |
| `received_piece_count` | `ReceivedPieceCount` | 已完成并进入缓存的 2D 片数，不是刚提交的请求数             |
| `piece_ids`            | `PieceIds`           | 已缓存片号列表                                             |
| `ready_for_3d`         | `ReadyFor3D`         | 是否已经收到要求的全部 2D 结果；不表示整板联合检测已经完成 |

并行时，2D 响应可能乱序。按 `BoardId + PieceId` 关联，不要按返回顺序关联。
3D 可以先提交并等待匹配输入，不必等 `ReadyFor3D=true` 才能提交。

### 10.2 `ReleaseSummary`

| JSON 字段             | C# 属性             | 说明                   |
| --------------------- | ------------------- | ---------------------- |
| `board_id`          | `BoardId`         | 被释放的板号           |
| `released_2d_count` | `Released2DCount` | 释放的已缓存 2D 片数   |
| `released_3d`       | `Released3D`      | 是否释放该板的 3D 缓存 |

`ReleaseBoard` 或联合结果中的 `Released` 只描述板数据缓存。
**释放板缓存不等于卸载模型；`bridge.Release()` 才用于服务/模型资源释放。**

## 11. 返回对象中的耗时参数

只说明通过返回对象可取得的参数，不包括文本日志或测试客户端额外统计。

### 11.1 C# 侧耗时

检测接口通常同时填入对象属性及 `Timings` 的同名 JSON 键。

| C# 属性 / 字典键                                                  | 说明                                                                                         |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `CsharpImageReadMs` / `csharp_image_read_ms`                  | 文件读取或输入准备；部分旧接口未计时，可能为 0；批量文件接口的这一计时范围也包含输入准备操作 |
| `CsharpSharedMemoryWriteMs` / `csharp_shared_memory_write_ms` | Bridge 自己创建/写共享内存的时间；外部共享内存接口为 0                                       |
| `CsharpRoundtripMs` / `csharp_roundtrip_ms`                   | 发送到对应原始响应返回的往返时间；不包含其后的 DTO 反序列化                                  |
| `CsharpTotalMs` / `csharp_total_ms`                           | 检测内部总经过时间，包含该计时范围内的共享内存写入、往返、反序列化等                         |

`CsharpTotalMs` 不保证包含进入检测接口的并发准入等待，也不保证包含计时起点之前的
文件读取；不同文件/共享内存接口的边界不同。不能强行认为它等于其他三项相加。
`DetectStreamingFilesAsync` 成功时这些字段主要来自最终 3D 请求，不是十条 2D 调用耗时总和。

### 11.2 服务端 `response.Timings`

| 常见字典键                                                                              | 说明                                                                                                                        |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `total_init_ms`                                                                       | 初始化方法汇总时间，毫秒；不一定包含外层首次导入模块                                                                        |
| `init_board_import_and_construct_ms/init_dispatch_ms`                                 | 流式板级初始化的模块导入及请求处理汇总                                                                                      |
| `shared_memory_read_decode_ms`                                                        | 输入读取/解码汇总                                                                                                           |
| `cpu_prepare_queue_ms/cpu_prepare_phase_ms`                                           | 2D CPU 预处理排队/执行                                                                                                      |
| `gpu_queue_ms/gpu_phase_ms`                                                           | GPU 阶段排队/执行；GPU 阶段包括掩码提取和外观检测                                                                           |
| `cuda_before_yolo_ms/yolo_predict_wall_ms/cuda_after_yolo_ms`                         | YOLO 前等待、完整调用、后等待                                                                                               |
| `yolo_reported_preprocess_ms/yolo_reported_inference_ms/yolo_reported_postprocess_ms` | 模型框架上报的子阶段时间，不一定覆盖全部调用开销                                                                            |
| `mask_extract_to_cpu_ms/efficientad_total_ms`                                         | 掩码提取传回 CPU、芯片外观检测                                                                                              |
| `cpu_postprocess_queue_ms/cpu_postprocess_phase_ms`                                   | 2D 后处理排队/执行                                                                                                          |
| `silver_total_ms`                                                                     | 银浆检查总时间；`silver_template_sift_ms/silver_target_sift_ms/silver_flann_match_ms/silver_ransac_ms` 是其主要配准子步骤 |
| `request_pipeline_ms`                                                                 | 单片流式 2D 的服务端流程经过时间，含排队与结果几何构建，不含之后的协议输出                                                  |
| `segmentation_3d_ms/depth_segmentation_ms`                                            | 整板深度分片时间                                                                                                            |
| `total_detect_ms` / `ElapsedMs`                                                     | 接口相关的兼容汇总值；联合整板包含等待 2D 与联合任务，不是纯 3D 计算                                                        |

单片联合 `response.ResultsBoard.Pieces[i].Timings` 常见键：

| 键                                                        | 说明                                    |
| --------------------------------------------------------- | --------------------------------------- |
| `matching_input_wait_ms`                                | 2D 与深度输入就绪时间差，不占用计算线程 |
| `joint_queue_ms/joint_phase_ms`                         | 联合任务 CPU 排队/执行                  |
| `coarse_registration_ms`                                | 粗配准                                  |
| `affine_registration_ms`                                | 动态仿射优化                            |
| `perspective_registration_ms`                           | 透视优化                                |
| `mask_alignment_ms`                                     | 各实例掩码对齐                          |
| `height_measurement_ms`                                 | 参考面与引线、焊点高度测量              |
| `height_fit_platform_plane_ms/height_fit_chip_plane_ms` | 平台/芯片平面拟合                       |
| `height_wire_regions_ms/height_solder_ms`               | 引线/焊点测高                           |

字典允许出现其他已执行子阶段键。`*_ms` 是毫秒，**不是字典中所有数值都必然是毫秒**：
例如独立 3D 的 `received_bytes` 是字节数量。
父阶段包含子阶段，多片并行时间互相重叠，不能将所有值直接相加。
缺少某个字典键表示该接口/路径没有提供该项，不能当作耗时为零。

## 12. C# 读取示例

以下片段位于已有初始化 `bridge` 的异步上位机方法内部。
`images2D` 为十张原图路径，`depthPath` 为整板深度图路径，`boardId` 为本批板号。

```csharp
DetectResponse response = await bridge.Board.DetectStreamingFilesAsync(
    images2D, depthPath, boardId);

if (!response.Ok)
{
    Console.WriteLine(response.Error ?? response.Message ?? "检测请求失败");
    return;
}

ResultsBoard? board = response.ResultsBoard;
if (board == null)
{
    Console.WriteLine("接口未返回联合整板结果");
    return;
}

Console.WriteLine($"板号={response.BoardId}，产品OK={board.IsOk}");
foreach (BoardPieceResult piece in board.Pieces)
{
    Console.WriteLine($"片号={piece.PieceId}，片级OK={piece.IsOk}");
    bool fallback = piece.SilverOverflow2D?.Registration?.FallbackUsed == true;
    Console.WriteLine($"银浆OK={piece.SilverOverflow2D?.IsOk}，边界回退={fallback}");

    foreach (WireMeasurementResult wire in piece.WireMeasurements)
    {
        HeightMeasurementResult? height = wire.Height3D;
        if (height?.IsValid == true)
            Console.WriteLine($"引线={wire.InstanceId}，高度差={height.HeightDiffMm} mm");
    }
}
```

读取单片 2D 银浆缺陷面积（需 `using System.Text.Json;`）：

```csharp
Diagnostic2D? silver = response.Results2D?.SilverOverflow;
if (silver?.Measurement is JsonElement measurement &&
    measurement.ValueKind == JsonValueKind.Object &&
    measurement.TryGetProperty("defect_area_px", out JsonElement area))
{
    int defectPixels = area.GetInt32();
    Console.WriteLine($"银浆越界缺陷面积={defectPixels} 像素");
}
```

此片段的 `response` 应来自单片 2D 调用；联合片的银浆对象应改为
`piece.SilverOverflow2D`。不要读取测试脚本外包的 `response/responses_2d` 结构作为 DLL 返回类型。

## 13. 当前 DTO 的边界

- 本文以 `BridgeDtos.cs` 实际已定义的属性为准，JSON 名称大小写与 C# 属性名称不同。
- Python 未映射到 DTO 的额外字段默认会被忽略，不会自动生成 C# 属性。
- 当前没有强类型属性的例子：联合配准完整透视矩阵/仿射详细诊断、平台 ROI 采样诊断、
  外观结果的 `score_mode/score_pool_kernel/threshold_source`、2D 顶层某些重复测量列表。
- 银浆 `Measurement` 保留为 `JsonElement?`，其中的动态统计字段仍可直接读取。
- 本次文档修订只说明返回值，没有增删接口、修改 DTO 或改变检测逻辑。

## 14. 哪些返回参数可以用于上位机可视化

本节只使用当前 Bridge 已返回的字段，不要求重新运行检测算法。
下表中的 `two` 指 `response.Results2D`；`piece` 指当前选中的
`response.ResultsBoard.Pieces[i]`；银浆对象在 2D 返回中是 `two.SilverOverflow`，
在联合片返回中是 `piece.SilverOverflow2D`。

### 14.1 先准备底图和结果对应关系

- **DLL 不返回原始 2D 图片、整板深度数组或图片字节。** 上位机需要保留自己采集/提交的图像作为底图。
- 用 `BoardId + PieceId` 将结果绑定到对应图像，不要按异步响应先后顺序绑定。
- 2D 叠加使用原始分辨率坐标；缩放显示时统一将底图和所有图层转换到显示坐标。
- 联合 3D 本片底图使用整板深度图的 `DepthStartRow` 到 `DepthEndRow` 行，结束行不包含，X 方向保持原图全宽。
- `piece.Image2DSize` 可用于核对 2D 底图尺寸；它不是本片 3D 深度图尺寸。

### 14.2 2D 图像可直接绘制的字段

| 要显示的图层            | 上位机返回字段                                                                                                                             | 如何显示                                                 | 坐标系/注意事项                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | --------------------------------------------------------------------- |
| 各类分割轮廓            | `two.Instances[].Mask2D.Contours`；或 `piece.Masks[].Mask2D.Contours`                                                                  | 按`ClassName` 分色，绘制闭合轮廓；可近似透明填充       | 完整 2D 原图坐标，不再加 crop 偏移                                    |
| 分割包围框              | 上述`Mask2D.BoundingBox`                                                                                                                 | 绘制`[x1,y1,x2,y2]` 矩形                               | 原图坐标；它是包围框，不是精确掩码                                    |
| 实例文字                | `InstanceId/ClassName/Confidence`                                                                                                        | 在对应轮廓旁标注类别、编号、置信度                       | 置信度不能当作产品通过率                                              |
| 圆定位结果              | `two.Circle.X/Y/Radius`                                                                                                                  | 绘制圆和圆心                                             | 原图坐标，单位像素；最终联合片 DTO 没有`Circle` 属性                |
| 检测区域                | `two.CropBox`；或 `piece.Crop2D`                                                                                                       | 绘制检测 ROI 矩形                                        | 原图坐标                                                              |
| 引线 ROI                | `two.WireRois2D`；或 `piece.WireRois2D`                                                                                                | 绘制各配置 ROI                                           | 相对于 crop；画到原图时加 crop 左上角偏移                             |
| 已公开的引线 ROI 检查框 | `two.WireBending2D.RoiChecks[].Box`；或 `piece.WireRoiChecks[].Box`                                                                    | 按`IsOk/Status` 显示 ROI 状态                          | 已是原图坐标，不再加偏移                                              |
| 引线弯曲评估矩形        | `two.Rectangles[].Points`、`two.WireBending2D.Measurements[].Bending2D.Rectangle`；或 `piece.WireMeasurements[].Bending2D.Rectangle` | 绘制矩形，旁边标注`MaskRatio/IsOk`                     | 原图坐标；无效或空矩形不绘制                                          |
| 芯片旋转矩形和中心      | `two.ChipRotation2D.Chips[].Rectangle/CenterXy`                                                                                          | 绘制矩形和中心点，标注`AxisDeviationDeg`               | 原图坐标；最终`BoardPieceResult` 没有强类型 `ChipRotation2D` 属性 |
| 芯片外观检查范围        | `two.ChipAnomaly2D.Chips[].Rectangle`；或 `piece.ChipAnomaly2D.Chips[].Rectangle`                                                      | 绘制芯片检查矩形，标注`Score/Threshold/Prediction`     | 原图坐标                                                              |
| 芯片外观缺陷区域        | 上述芯片的`AnomalyBoxes[].ImagePolygon`                                                                                                  | 绘制缺陷多边形，例如红色半透明区域                       | 原图坐标；不要把`CropXyxy` 直接画到原图                             |
| 银浆允许边界            | 银浆对象的`Boundary.Points`                                                                                                              | 连接顶点绘制完整允许多边形，例如黄色线，顶点可标 B1/B2… | 根据`CoordinateSystem` 判断；正常返回为 `image`，直接画到原图     |
| 银浆越界缺陷轮廓        | 银浆对象`Measurement` 中的 `defect_contours`                                                                                           | 绘制各越界区域，例如红色轮廓                             | 正常公开结果为原图坐标；从`JsonElement` 读取                        |
| 银浆越界缺陷框          | 银浆对象`Measurement.regions[].box_xyxy`                                                                                                 | 绘制区域框并标注`area_px/max_outside_distance_px`      | 原图坐标；不要重复加偏移                                              |
| 银浆连通检查范围        | 连通对象`Measurement.chips[].bbox_xyxy/outer_bbox_xyxy/center_xy`                                                                        | 显示芯片内框、外环检查框和中心                           | 正常公开结果为原图坐标，字段为动态 JSON                               |

`MaskGeometry.Contours` 只提供简化外轮廓，可能不保留内部孔洞。
用轮廓填充显示可以辅助观察，但**不要用填充后的面积替代算法返回的 `AreaPx` 或缺陷面积**。

#### 银浆图层的额外规则

1. 直接绘制 `Boundary.Points`；算法已做模板映射或回退，**不要再用 `HomographyTemplateToImage` 对这些点变换一次**。
2. `Boundary.CoordinateSystem="image"`：原图坐标；若遇到明确标记 `"crop"` 的结果，才加 crop 偏移。
3. `Boundary.IsWithinCrop=false`：边界仍可绘制，允许超出检测 ROI；不能把边界顶点裁到 ROI 边缘。
4. `Registration.FallbackUsed=true`：建议显示“使用模板原始点回退，未配准”的提示。
5. 回退提示与缺陷颜色分开：配准无效不等于实际越界；缺陷判断仍读取银浆对象 `IsOk/Status`。
6. `Measurement` 为空或状态为 `uncertain/not_configured/no_silver` 时，不要生成不存在的红色缺陷区域。

### 14.3 3D 深度图可直接绘制的字段

| 要显示的图层          | 上位机返回字段                                                                       | 如何显示                                                       | 坐标系/注意事项                                                    |
| --------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| 对齐后的各类轮廓      | `piece.Masks[].Mask3DLocal.Contours`                                               | 按类别绘制在本片深度图上                                       | 本片局部深度坐标；画在整板图上时给 Y 加`DepthStartRow`           |
| 对齐实例包围框        | `piece.Masks[].Mask3DLocal.BoundingBox`                                            | 绘制对象框                                                     | 本片局部坐标，转换规则同上                                         |
| 引线实际测高区域      | `piece.WireMeasurements[].Mask3DLocal`                                             | 显示参与引线测量的轮廓/包围框                                  | 本片局部坐标；不一定等于完整分割实例范围                           |
| 焊点/细线端部测高区域 | `piece.SolderMeasurements[].Mask3DLocal`                                           | 显示焊点和 start/end 端部区域                                  | 本片局部坐标                                                       |
| 联合引线测高点        | `piece.WireMeasurements[].Height3D.Point`                                          | 标点，旁边显示`HeightDiffMm` 和 `IsOk`                     | 正常为整板坐标；画在本片图上时给 Y 减`DepthStartRow`             |
| 联合焊点测高点        | `piece.SolderMeasurements[].Height3D.Point`                                        | 标点并显示高度差、参考面和端部标签                             | 正常为整板坐标，转换规则同上                                       |
| 实例关联测高点        | `piece.Masks[].Height3D.Point`                                                     | 可用于选中实例后高亮关联测量点                                 | 可能与上述测量列表重复，不必重复绘制                               |
| 独立 3D 测量点        | `response.Results3D.Pieces[].Points[].X/Y`                                         | 绘制在整板深度原图，标注`HeightMm/HeightDiffMm`              | 独立 3D 返回为整板像素坐标，不要套用联合片的局部规则               |
| 参考面说明            | `piece.ReferencePlanes.WirePlatform`、`Chips[].Plane`，或焊点 `ReferencePlane` | 显示`Name/Coefficients/ValidPoints` 文本，标明测高所用参考面 | 系数使用本片局部坐标和深度灰度单位；不是可直接绘制的采样区域多边形 |

只在 `Height3D.IsValid=true` 且 `Point` 非空时绘制有效测高点。
无效高度不要显示成“0 mm”；可以显示“无效/未测得”。

优先看 `Height3D.PointCoordinateSpace`，也可结合 `piece.HeightPointCoordinateSpace`：
明确为 `board_3d` 时按整板点转换；旧 DLL 或旧结果没有标签时先确认来源，不能盲目再加减行偏移。

**优先使用已经返回的 `Mask3DLocal`，不要再次把 `Mask2D` 通过 `Matrix2DTo3D` 重映射后当作算法最终对齐轮廓。**
当前 DTO 未提供完整联合透视矩阵，用 2×3 参数重新计算可能与算法结果不同。

### 14.4 不用于绘图，但可用于结果卡片/表格的字段

| 显示内容           | 返回字段                                                                      | 建议处理                                               |
| ------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| 板号、片号         | `response.BoardId/BatchId/PieceId`、`piece.PieceId`                       | 切换视图、关联图像及对象                               |
| 请求是否成功       | `response.Ok/Error/Message`                                                 | 与产品结果单独展示；错误信息不能当作缺陷框坐标         |
| 整板/单片结果      | `ResultsBoard.IsOk`、`piece.IsOk`、`two.IsOk`                           | 显示 OK/NG；空结果显示未提供                           |
| 引线弯曲           | `Bending2D.IsValid/IsOk/MaskRatio`                                          | 配合评估矩形显示有效性、覆盖率和结论                   |
| 芯片旋转           | `AxisDeviationDeg/MaxAngleDeg/IsOk`                                         | 显示偏角、阈值、结论                                   |
| 芯片外观           | `Score/Threshold/Prediction/Model`                                          | 显示实际模型和异常分数，不把分数当概率                 |
| 银浆越界面积和距离 | `Measurement.defect_area_px/defect_ratio/max_outside_distance_px`           | 展示缺陷数量和严重程度，注意像素单位                   |
| 银浆连通           | `Measurement.coverage_ratio/missing_sector_count`、诊断 `IsOk/Status`     | 展示覆盖率、缺失扇区数和结论                           |
| 模板边界配准质量   | `Registration.Matches/Inliers/InlierRatio/ReprojectionErrorPx/FallbackUsed` | 展示内点、误差和回退来源；回退不显示成配准成功         |
| 联合配准质量       | `piece.Registration.Score/IsValid`                                          | 显示匹配分数及有效性，不直接作为产品缺陷分数           |
| 引线/焊点高度      | `Height3D.HeightMm/HeightDiffMm/Reference/IsValid/IsOk`                     | 高度值单位毫米，业务通常重点展示高度差                 |
| 输入到齐进度       | `CacheState.ReceivedPieceCount/ExpectedPieceCount/PieceIds/ReadyFor3D`      | 显示该板已完成缓存的片数；不是整板测高进度百分比       |
| 耗时               | `ElapsedMs/CsharpRoundtripMs/CsharpTotalMs/Timings`、`piece.Timings`      | 显示时间表或柱状图；父子阶段和并发片的时间不能直接相加 |

可建议采用绿色 OK、红色 NG、灰色未提供/无效，但颜色是上位机 UI 约定，不是 DLL 返回参数。

### 14.5 必须遵守的坐标转换规则

| 数据                            | 绘制到完整原图                                       | 绘制到局部图                                         |
| ------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| 2D 原图坐标的轮廓/边界/缺陷     | 原样使用，不加 crop 偏移                             | 若底图是 crop，减去`CropBox/Crop2D` 的左上角       |
| `WireRois2D` 的 crop 局部矩形 | 加`CropBox/Crop2D` 左上角                          | 原样使用                                             |
| `Mask3DLocal`                 | `x_full=x_local`，`y_full=y_local+DepthStartRow` | 本片全宽深度切片上原样使用                           |
| 标记`board_3d` 的联合高度点   | 原样使用                                             | `x_local=x_full`，`y_local=y_full-DepthStartRow` |

示例仅说明偏移，不是生产标定：若 crop 左上角为 `(2000,100)`，
ROI 局部点 `(100,50)` 在原图上为 `(2100,150)`。
若 `DepthStartRow=1000`，局部深度点 `(75,31)` 的整板点为 `(75,1031)`；
高度点若已返回 `(75,1031)`，在本片图上显示为 `(75,31)`，不能再加 1000。

最后统一进行视图缩放和平移，例如：

```text
screen_x = image_x * zoom + pan_x
screen_y = image_y * zoom + pan_y
```

这里 `image_x/image_y` 必须先转换到当前底图坐标。
若控件还做了居中、裁剪或留白，应把对应显示偏移合并到 `pan_x/pan_y`。
不要缩放底图却忘记同步缩放图层；面积统计和高度数值不随显示缩放改变。

### 14.6 当前返回对象不能单独支持的可视化

- 原图、深度伪彩底图：需上位机保留的原始图像/深度数据，返回对象没有图像字节。
- EfficientAD 异常热力图：当前 DTO 只有分数和异常区域，没有逐像素异常响应图。
- SIFT 特征连线/匹配点：当前只返回匹配质量统计和边界矩阵，没有特征点位置及配对列表。
- 平台参考面采样 ROI 多边形：当前 `ReferencePlanesResult` 没有对应强类型采样区域属性。
- 精确原始分割像素填充：当前仅有简化外轮廓，不是完整掩码数组。
- 仅凭返回对象恢复完整点云或物理坐标 3D 曲面：还需要原始深度与相机标定/像素分辨率等信息。

只取得 `Board.DetectStreamingFilesAsync` 的最终响应时，可以显示其已携带的
2D/3D 轮廓、银浆诊断、芯片外观和测高结果；但圆定位、芯片旋转角等只在单片
`Results2D` 中映射的字段，需要上位机保留相应单片返回值，不能从当前联合片 DTO 的默认值推算。

需要这些额外图层时应明确扩展接口/DTO，或保留已经取得的单片返回结果，
不要依赖当前 DLL 实际没有返回的可视化字段。
