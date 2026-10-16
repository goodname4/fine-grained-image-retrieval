# A：数据与评测工作记录

工作分支：`codex/a-data-evaluation`。按25点规划推进，区分已实际完成的交接和仍需其他成员产出的任务，不按代码量宣称完成比例。

| 任务 | 已实际完成 | 仍需完成 |
| --- | --- | --- |
| 数据整理（5点） | 来源/用途核对、真实下载、官方导入、8000图解码、21对重复候选辅助看图、两张训练图有据排除、7998图交接包、真实统计与一键重建 | 后续新发现的近重复需记录并统一更新版本 |
| 细粒度测试（7点） | 100条真实源图初稿，四人各25条；A的25条AI辅助初标；B的25条辅助预审；离线工作台、合并冲突检查、复核门槛 | B/C/D初标；B复核A，A在B初标后复核B，C/D互审；共同池补审与正式冻结 |
| 正式评测（6点） | val/test各1000图5000查询协议；排名契约；总体/分类/逐查询指标；摘要/图库/深度检查；E0/E1/E2比较；按源图bootstrap；多种子汇总 | B/C提供真实排名和配置后，由A统一执行正式测试 |
| 错误分析（4点） | 改善/退化/Top-1状态；类别分层选例及覆盖报告；离线图片复核与原因导出 | 真排名到位后实际分析至少12例，四类及成功/失败均覆盖 |
| 文档与复核（3点） | 数据与评测章节、实际数据证据、交接手册、标注指南、排名契约、可运行示例和回归测试 | 独立复核记录及真实结果章节待补齐 |

## 当前可交接产物

- `data/prepared/flickr8k-v1/`：train5998图/29990对，val/test各1000图/5000查询。path相对data-root，交接文件和图片均有摘要。
- `data/annotations/fine-v1/annotations.work.jsonl`：25条annotated、75条draft、0条reviewed。A当前25图池不代表全库相关性。
- `artifacts/annotation-board-current/board.html`：可离线打开，支持成员筛选、预审意见、逐图标签、复核门槛、导入/导出。已在浏览器实测，导出的100条与工作文件一致。
- `configs/data/`及`configs/annotations/a-source-checks.json`：可重建的版本化元数据。图片和完整生成数据不入Git。
- `examples/evaluation/`：明确标为synthetic，仅验收接口。真实E0/E1/E2尚无。
- `data/annotations/pool-request-v1/`：已冻结当前查询文本与图库，供B/C先产出真实Top-K；共同池合并不把新增图片当作已审，详见标注指南。

## 开始使用

```bash
python -m pip install -e ".[data]"
python -m fgir prepare-a-data
python -m fgir verify-handoff --bundle data/prepared/flickr8k-v1 --data-root data
```

本机已有文件时可用`prepare-a-data --skip-download`，校验复用且不覆盖人工工作文件。B/C现在可直接开展编码、缓存和训练，无需等待细粒度独立复核。D可先用synthetic协议接指标与表格，再换真实结果。

详见[交接手册](handoff-a.md)、[数据与评测章节](reports/a-data-and-evaluation.md)、[标注指南](annotation-guide.md)、[排名契约](ranking-contract.md)。
