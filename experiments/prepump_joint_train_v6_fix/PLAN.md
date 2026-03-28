# V6 修正版

目标：
- 修正旧 `V6` 使用了不一致 experimental 样本的问题
- 使用固定 `alpha=0.015` 重新导出的 `1400` 条 pre-pump 实验样本
- 完全按 `V3` 的训练规模重跑：`60K + 8 labels + 低噪声 + merged`
- 保存每个 epoch checkpoint
- 逐 epoch 评估固定 `alpha=0.015` 下的实验推理质量，画 `epoch vs quality` 曲线

本轮关键变量：
- 保留 `8th label = h_C5_signed`
- 不改采样
- 不改模型结构
- 不改训练方式（merged）
- 只修正 experimental 样本口径
