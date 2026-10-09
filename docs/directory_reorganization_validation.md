# 目录整理验证（2026-10-09）

本次调整是源码组织与入口重构，不是模型算法升级。

## 通过的检查

- 整理前：174 项原有 unittest 通过。
- 整理后：174 项原有 unittest 通过；新增 7 项目录/入口测试单独通过，共 181 项。
- 新增测试覆盖根目录默认输出、旧导入的同模块别名、统一入口分发/返回码/异常处理、其他工作目录调用、数据合成命令。
- 使用合成图和随机 Teacher，完成两个 CPU DDP 进程的训练、校准、三种 score 评估、热图、单图推理、保存/加载和断点恢复。
- 推理交付包复制到项目之外，在 Python isolated mode 下运行成功。
- 同一合成权重经 run.py infer、旧 infer_efficientad.py 和旧 efficientad_ccd.py predict 输出的整图分数完全一致。
- 现有 outputs 中 CCD1、CCD2、CCD3、CCD6 的四份 model.pt 均严格恢复成功。
- 校验原文件 SHA256：pyproject.toml、uv.lock、requirements.txt、.python-version、四份 model.pt 完全不变；circle_config.json 内容完全不变，仅移动到 configs/。
- 11 个核心模块忽略导入路径和文档字符串后的 AST 对比一致，数学计算、网络结构和训练控制流未改。
- compileall 和 git diff --check 通过。

## 验证记录

- outputs/validation/directory_reorganization_tests.log
- outputs/validation/directory_structure_entry_tests.log
- outputs/validation/directory_reorganization_smoke.log

测试模型和图片仅用于验证程序；不是生产模型检测精度证明。原生产数据未移动，GUI 本次未实际启动窗口。

## 使用提示

统一命令为 `python run.py <子命令>`，配置参数改为 `--circle-config configs/circle_config.json`。
旧训练/推理入口仍可用；其他 Python 程序直接导入旧核心模块时，应更新为 ccd_efficientad 包路径。
参考仓库归入 references，原平铺交付源码与 process 缓存归档在 outputs/migration_backup，未删除。
