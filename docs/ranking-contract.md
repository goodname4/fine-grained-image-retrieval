# 冻结评测与排名契约 v1

> 2026-10-27版本说明：本文是当前E0/E1/E2的run-v1操作契约。N0—N3共享原始图像索引并使用不同文本LoRA，需要新的run-v2和retrieval receipt，设计见[实施清单](implementation-plan.md)。当前导出、服务和评分接口尚未完成该迁移；不得仅改method或伪造图像adapter摘要来导出N结果。固定检索protocol-v1的候选/查询/qrels仍可沿用，语义诊断另用独立协议。

## 固定协议

标准协议位于`data/prepared/flickr8k-v1/{val,test}/protocol.json`。它绑定同目录候选ID、查询JSONL和qrels的文件SHA-256，也记录清洗版manifest摘要、查询/图库规模、数据类型和标注范围。修改文件而不生成新协议会被拒绝。

标准协议为英文文本到图片检索，每条原始描述对应其源图片，`annotation_scope=source_pair`。这衡量已标注源图命中率，并不宣称其余图片均不相关。细粒度人工协议需通过`freeze-annotations`单独生成，不能与标准协议合并计分。

## B/C交给A的run JSON

下面是结构示意；示意字符串不是可运行结果。可运行的**人工排名**在`examples/evaluation/`，其`data_kind=synthetic`，不能改一个标签就作为模型结果。

```json
{
  "schema_version": "fgir-run-v1",
  "run_id": "E0-seed42-model-version",
  "method": "E0",
  "model_id": "实际编码器与预处理版本",
  "data_kind": "real",
  "protocol_sha256": "实际protocol.json的64位小写SHA-256",
  "weights_sha256": "本地编码器权重的64位小写SHA-256",
  "adapter_sha256": null,
  "seed": 42,
  "code_revision": "实际commit；未提交修改另记录源文件摘要",
  "config": {"device": "实际设备", "preprocessing": "实际预处理", "text_processing": "实际分词/截断规则"},
  "ranking_depth": 10,
  "rankings": {
    "实际query_id": ["按相似度从高到低的图片ID"]
  }
}
```

用`from fgir.data import file_digest`计算摘要。E0不填写适配层；E1/E2必须填写实际适配层摘要。`config`应包括数据/模型/索引版本、训练选择依据和必要参数；`code_revision`不能只记一个已有commit而隐瞒工作区变更。

`rankings`必须覆盖协议中**全部且仅有**的query_id。非空排名长度须等于声明深度，ID不得重复或越出图库。查询执行失败可显式返回`[]`并按0计分，同时在配置/日志说明失败；不能遗漏查询或把异常查询删除。深度至少覆盖需要的Recall@K（K大于图库时按图库规模），最多为图库总数。

## 评测和比较

真实排名准备好后执行：

```bash
python -m fgir evaluate-run --protocol data/prepared/flickr8k-v1/test/protocol.json --run artifacts/e0.run.json --output artifacts/evaluation/e0
python -m fgir compare-runs --protocol data/prepared/flickr8k-v1/test/protocol.json --runs artifacts/e0.run.json artifacts/e1.run.json artifacts/e2.run.json --output artifacts/evaluation/compare
```

`--runs`第一份作为基线，其余分别与其比较。每份run必须绑定同一协议，run_id不同，排名深度相同。新输出目录必须不存在；输入错误退出2，不产生半成品评测包。

输出：

- `evaluate-run`：`metrics.json`（总体、四类指标、运行元数据、输入摘要）、`per-query.jsonl`（首个相关项排名、RR、各K命中、完整供应排名、空结果标识）。
- `compare-runs`：每个run的指标、`comparison.json`、`report.md`、每个对比的`changes-N.jsonl`及`cases-N.jsonl`。
- 无某类查询时该类query_count=0、指标为null，不显示虚构零分。一条查询可进入多类，不将分类数量相加当作总查询数。

## 指标含义与统计限制

Recall@K是查询级成功率：前K项有任一已标注相关图片则记1。它不是找回全部相关图片的集合召回率。MRR为第一个相关项名次倒数的均值；只有供给完整图库排名才标作`mrr`，Top-10写`mrr@10`。

对比输出逐查询RR差、改善/退化/保持首位成功/保持非首位的计数，并保留负向变化。“保持非首位”仍可能在Top-5命中。bootstrap按源图片成对分组重采样，同图描述一起抽取；无源图分组时用query_id。固定种子、默认1000次，给出固定运行下RR均值差的95%百分位区间，不能代替多训练种子实验或宣布显著。

实际种子通过`python -m fgir summarize-seeds --protocol <同一协议> --runs <各实际种子run文件> --output <新目录>`汇总。按方法、编码器和深度分组，拒绝重复种子或不同的与种子无关的`config`。输出总体/分类均值和样本标准差，1个种子时标准差为null。seed和适配层摘要放run顶层，不在config混入各次checkpoint文件名。

## 候选池标注不能冒充全图库标注

`annotation_scope=candidate_pool`时，所有方法返回的每个图片ID都必须出现在该查询的`judged_image_ids`内；出现未审结果直接报错。先将各方法结果的并集加入共同池，人工补审，再冻结新协议并重跑全部方法。不得把未审结果自动视为不相关，也不得分别为每个模型用不同的正样本集合。

`full_gallery`必须逐查询审核全部固定候选。所有细粒度查询必须有另一位标注人的真实独立复核和说明；工具只检查记录完整性，不能替代复核行为本身。

## 错误分析

自动案例按color/count/action/relation与改善、退化、保持首位成功、保持非首位分层选取，尽量提供至少12条；可用案例不足时按实际数量输出。`analysis_status=needs_visual_review`、`error_type=null`。这只是选例，不是错误原因判断。

```bash
python -m fgir render-cases --manifest data/prepared/flickr8k-v1/manifest.jsonl --cases artifacts/evaluation/compare/cases-1.jsonl --data-root data --output artifacts/evaluation/case-review
```

打开生成的`review.html`，看查询、相关图片、两组Top-K，记录错误类型与图像证据并导出。类型包括文本歧义、颜色、数量、动作、关系、标注不全、其他和成功。不能只留下改善案例；报告类别覆盖、不存在的案例类型和真实局限。

## 无模型也能先验收接口

```bash
python -m fgir evaluate-run --protocol examples/evaluation/protocol.json --run examples/evaluation/run-e0.json --output artifacts/demo-evaluation
python -m fgir compare-runs --protocol examples/evaluation/protocol.json --runs examples/evaluation/run-e0.json examples/evaluation/run-e1.json examples/evaluation/run-e2.json --output artifacts/demo-comparison
```

这些文件是12个虚构候选、16个虚构查询的人工排序，包含改善和退化，用于B/C/D开发输出接口和验证表格。任何分数都不属于真实Flickr8k或CLIP实验。
