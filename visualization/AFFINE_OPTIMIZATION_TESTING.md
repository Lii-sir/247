# 仿射优化：由使用者执行的实图验收

跨机器环境安装、自动并发对照、完整PowerShell命令与报告判读，见 [Windows完整测试流程](AFFINE_CROSS_MACHINE_TESTING.md)。

代码已接入四项功能。原生`score_mode=full`、当前`ftol=0.01`及所有生产标定保持。仅精确缓存默认开启；标定/多尺度默认关闭，预算默认null。以下速度和精度尚未经实图验证，不承诺800ms或150次目标已达到。

## 测试输入和环境

使用同一板的同一套10片2D与对应深度，Pitch、fixed_rois、评分区、阈值、容差、线程和模型配置保持不变。记录输入文件名及板号/片号。不要拿不同配置或不同冷热状态的389次/1863ms旧例作为唯一基线。

先在配准台对同片比较叠加和分区误差；参数从高级设置“仿射参数”传入，运行后导出JSON。确认后再在正式验证台对相同10片并发运行，比较`timings.affine_registration_ms`。正式服务读取磁盘配置，要重新初始化；验证台检测到配置指纹变化会重新加载模型，需区别冷启动和稳定运行。

## A：原始基线

保留其它字段，只设置以下开关：

```json
{
  "score_mode": "full",
  "cache_enabled": false,
  "calibration": {"enabled": false, "profile": null},
  "multiscale": {"enabled": false},
  "maxfev": null,
  "total_maxfev": null
}
```

这是开关片段，不是完整替代参数对象。保留原有preview_scale、阈值、搜索边界、maxiter、xtol、ftol。

## B：仅缓存

在A基础上改`cache_enabled=true`。应检查最终损失、各区损失及矩阵完全一致，nfev仍相同；真实变换次数可下降。查看cache_hits/misses/hit_rate与compute_mean_ms，不用30%命中率作硬门槛。真实耗时以整个仿射阶段为准。

## C：仅增加标定候选

在B基础上，先用配准台人工确认一片对齐并“保存当前阶段矩阵”。把该JSON对象整体放入`calibration.profile`，然后`calibration.enabled=true`。不要只粘贴六个数或只粘贴matrix而丢失context。

只复用矩阵的线性四项。源2D尺寸、深度宽度、独立相机坐标、Pitch几何设置必须匹配，当前片粗配准中心位置和原搜索范围保留；原片高/分片行起点可不同，因为不复用旧平移。

看`affine_initialization`：source为calibration_linear才实际选中；profile_valid和reason解释拒绝/不改善原因。初始损失变低不保证最终几何更准，应比较每个区域的损失、关键边缘以及最终测高结果。测试同板和另一板，不能只验证标定片本身。

## D：两级搜索

先在B基础上单独开启（不同时开启标定，便于判断各项效果）：

```json
{
  "multiscale": {
    "enabled": true,
    "search_scale": 0.5,
    "search_maxiter": 15,
    "refine_maxiter": 10,
    "search_maxfev": null,
    "refine_maxfev": null,
    "refine_window_ratio": 0.25
  }
}
```

完整二值掩码先闭运算再降采样；源/目标实际比例、评分区、矩阵同步换算，6参数始终保持同一原坐标单位。各分辨率独立缓存。精修使用完整分辨率，最终质量只用完整分辨率损失。

完整损失退化、阈值失败、精修窗口触边或阶段异常时会尝试原范围的完整搜索。检查`affine_multiscale.fallback_used/fallback_reason/solution_source`；频繁回退可能更慢。`affine_stages`提供各级耗时、nfev、尺寸、窗口和完整评分，跨尺度损失不要互相比较。

## E：评分预算

先在B的单级模式试`maxfev=150`、`total_maxfev=150`，比较质量，不能仅因为次数≤150就宣布成功。达到上限时返回本阶段已评估的最佳有限候选副本；保留优化器原停止状态。正常收敛且不受预算截断时不强行替换原优化器输出。

多尺度可分别设置search_maxfev、refine_maxfev；null继承顶层maxfev。total_maxfev限制所有优化阶段（含回退）的目标函数调用总数，缓存命中也计数。初始候选、级间验证和最后验收仍会评分，故score_calls可能超过预算；这些工作计入实际阶段墙钟，不伪造省略。总预算耗尽会在stage.skipped和fallback_not_run明确说明，不能认为已经执行了原路径回退。

预算触顶只决定停止，不意味着合格；最终仍检查矩阵有限和完整损失阈值。粗初值本身可能已通过宽松阈值，所以还需人工检查关键位置，不能只看affine_valid。

## 要回传的记录

- 同片输入、配置快照，baseline和candidate完整JSON。
- `affine_registration_ms`（配准台对应timings中的“仿射配准”），不要与optimizer_ms混为同一口径。
- 顶层及每级nfev、实际warp次数、缓存命中；多级顶层为各级总和，不再固定score_calls=nfev+2。
- 最终损失、各区域损失、实际矩阵、关键点映射变化、配准有效率。
- 所有引线/焊脚记录数、采样掩码、高度值和阈值附近NG变化。
- 回退比例、总阶段平均及较慢片。不同片平均改善不能掩盖个别片精度或耗时退化。

800ms可作阶段目标，但必须同时满足可接受的几何和测高精度。金字塔总nfev变多也可能更快，因此将完整分辨率真实warp次数和总墙钟作为主要性能依据。

可用不运行检测的比较工具读取已保存报告：

```powershell
.\.venv\Scripts\python.exe -B .\visualization\compare_affine_results.py --baseline A\result.json --candidate B\result.json --output comparison.json
```

支持正式验证台result.json及配准台导出的registration-test.json；工具只整理现有数据，不能自动证明对齐精度或测高一致。
