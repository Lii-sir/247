# Bridge 配置目录

运行时维护两份阶段配置：

- `2D/config.json`：2D 圆定位、ROI、YOLO、引线弯曲、银浆、芯片旋转和
  EfficientAD。模型路径统一指向 `config/models/`。
- `3D/config.json`：3D 分片、初始配准、动态仿射、透视配准、焊脚/引线高度和
  独立 3D 引线高度接口。

`config/config.json` 是给 Bridge DLL 使用的轻量 manifest，只保存两份阶段配置的
相对路径，不承载算法参数。

每个功能块采用一致结构：

```json
{
  "description": "功能说明",
  "parameters": { "...": "现场需要调整的参数" }
}
```

不使用 `enabled` 开关。功能块存在即执行；可选功能不需要时，从交付配置中移除
对应功能块。`common_2d`、`common_3d` 只存放阶段公共参数。

板级接口初始化时由 C# 读取两份阶段配置并在内存中合并；Python 算法和传输服务
统一从 `python/` 加载。配置中的模型路径相对于 2D 配置文件解析，不依赖开发机
桌面绝对路径。
