# 参考实现

- `efficientad_official/`：原根目录 `.efficientad_check/`。
- `efficientad_alternative/`：原根目录 `.efficientad_rximg_check/`。

这两份参考检出不是当前训练/推理依赖。目录移动时保留了各自 Git 元数据和文件内容，不修改参考实现。
业务实现请维护 `ccd_efficientad/`，不要把参考仓库的权重当成本项目 model.pt。
