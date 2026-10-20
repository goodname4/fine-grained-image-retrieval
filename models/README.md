# 本地模型目录

放置预训练权重与训练checkpoint，不提交Git。使用 `python -m fgir prepare-clip-model --output models/clip-vit-base-patch32` 准备本地CLIP，收据记录来源/版本/SHA-256。C的适配权重位于训练目录 `best.pt`，同时保存 `last.pt` 和checkpoint摘要索引；加载/重建图片索引见[训练说明](../docs/training.md)。
