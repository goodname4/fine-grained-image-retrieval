# 新版A交接：语义数据、审核与独立评测

2026-10-27，research-v2。本文中的七个CLI已实现，均不需要新增模型依赖。当前交付的是可运行的协议/审核/诊断工具和真实val任务包；不是N模型成绩，也不是已冻结的人工金标。

## 1. 本机产物与真实状态

| 产物 | 内容 |
| --- | --- |
| `data/semantic/pilot-val-v1/` | 30个不同val源图、原始caption、确定性抽样记录，全部draft |
| `data/semantic/pilot-val-assisted-v1/samples.work.jsonl` | 保留全部30组；A的8组仅做原句看图辅助预审，其余22组draft |
| `artifacts/semantic-pilot-val-assisted-v1/board.html` | 含30张本地图片的离线审核页，可筛成员、改JSON、导入/导出 |
| `configs/semantic/a-pilot-source-checks-v1.json` | 8条辅助预审的caption/图片摘要与具体证据，可重建 |
| `examples/semantic-evaluation/` | 完整synthetic协议、输入和手算分数，只供接口验收 |

30图按A/B/C/D分配8/8/7/7，独立复核人对应B/A/D/C。抽样只依赖seed、图片ID和caption ID的SHA-256排序，未使用模型分数，也没有按易判样本筛选。语义覆盖率目前未知，不能把随机任务包当成已平衡诊断集。

8组辅助检查中4组supported、4组uncertain。30组均未填实体、关系或变体；0组独立人工复核、0组可用于正式诊断。`annotated=8`仅表示有原句图片预审记录，不代表该组全部语义标注完成。

## 2. 复现与继续审核

先按[旧数据交接](handoff-a.md)准备Flickr8k清洗包，再执行以下命令。所有输出必须是新目录；不能覆盖已有工作文件。已有目录时直接校验或指定新版本。

```powershell
python -m fgir prepare-semantic-pilot --output data/semantic/pilot-val-v1 --count 30 --seed 42 --code-revision c6cfa34
python -m fgir apply-semantic-source-checks --samples data/semantic/pilot-val-v1/samples.work.jsonl --checks configs/semantic/a-pilot-source-checks-v1.json --output data/semantic/pilot-val-assisted-v1
python -m fgir validate-semantics --samples data/semantic/pilot-val-assisted-v1/samples.work.jsonl --data-root data
python -m fgir render-semantics --samples data/semantic/pilot-val-assisted-v1/samples.work.jsonl --output artifacts/semantic-pilot-val-assisted-v1
```

可直接双击board.html，图片均在其相邻images目录，无服务器依赖。保存本组仅写浏览器内存，必须导出全部JSONL才持久保存；刷新前先下载。导入要求完整且不重复的同一批样本。导出后用`validate-semantics`校验，原文与划分不能在页面中改写。

审核页使用完整JSON编辑，字段定义见[语义协议](semantic-data.md)，可照完整[合成样本](../examples/semantic-evaluation/gold.jsonl)了解结构。真正的语义等价性、一次只改一个条件、对象绑定和图像真假仍需人工判断；结构校验器不代替这些判断。

## 3. A/B的数据接口

每条记录以`source_query_id`逐条联结原始pairs，`group_id=source_image_id`，`scene_group_id`来自保留图片的已知同场景连通组。校验器检查源manifest、inventory、三份pairs和近重复审核摘要。inventory须完整覆盖manifest且image_id/path/split一致；审核图片摘要及实际字节必须匹配inventory.file_sha256，不能通过替换图片并更新审核摘要改换源图。训练入口验证用`--purpose train`，它拒绝val/test、draft、uncertain和争议样本。

B补充`entities/relations/variants/units`及生成器来源。实体`head`、属性`value`、关系`predicate`均保留原文字符span。组件`text`是原句的连续子串，`foil`必须准确应用其`variant_id`关联的同一次最小编辑；不能为N1/N2分别构造不同错误。

每个变体必须有独立`id`，不得使用`original`或`sentence`等保留ID；`target_unit_id`指实体、关系，或整句保留值`sentence`。同义句用`equivalent/supported`，最小错误及错误追加用`changed/contradicted`，正确追加用`changed/supported`。仅仅caption没提及某对象不能判负。

追加变体通过`control_pair_id`配对正确/错误描述。评测器单列配对不完整、长度不等和长度匹配数量；长度按空白分词计，不冒充CLIP token数。

审核状态为draft、annotated、auto_verified、reviewed、rejected。初标注明`annotator_kind=human/assistant/automatic`；有审核时必须保存notes、image_evidence、image_sha256。自动核验标为auto_verified。正式金标要求human初标、不同的human复核人、review_notes及空争议记录；助手看图不充当这两个人。身份是可审计声明，不是账户认证。

已有辅助预审没有生成新文本，provenance仍是source-only；人工接手可保留这些来源记录，增加自己的实际初标和复核。使用本地语言模型生成文本时必须记录实际weights/prompt摘要和代码/seed，不能写占位摘要。

## 4. 冻结门槛与test接触

只有人工补齐并独立复核后，以下命令才会成功；当前30组会被拒绝。不要为了通过工具改审核状态。

```powershell
python -m fgir validate-semantics --samples data/semantic/reviewed-val-v1.jsonl --purpose eval --data-root data
python -m fgir freeze-semantics --samples data/semantic/reviewed-val-v1.jsonl --data-root data --data-kind real --test-access-notes "Validation pilot only; historical test exposure disclosed in docs/experiments.md" --output data/semantic/frozen-val-v1
```

val/test分开冻结。冻结产物绑定samples、图库、来源上下文、审核审计及各文件SHA-256；冻结前核对图片实际字节。拒绝非法span、缺失引用、未审/不确定变体、同人复核、图片证据错配与覆盖已有目录。结构检查无法发现伪写的人工身份或错误的视觉判断，需保留组织层面的实际复核。

冻结与解析F1评分共用解析结构校验：重复的规范化实体提及、属性和有向关系会在冻结前被拒绝。来源上下文绑定源inventory摘要及所用图片的原始摘要。早期未绑定inventory的语义协议需重新冻结到新目录，合成示例也遵守同一要求；不能原地改协议摘要冒充旧版本。

本次只抽val图片并看图，不新增test图像或caption级分析；校验来源时读取了全量manifest/pairs及近重复元数据。旧100条test草稿已有历史接触，继续按[实验设计](experiments.md)披露，不能称为完全未见测试集。正式test的选择、生成协议和配置冻结仍未执行。

## 5. C/D给A的分数接口

独立语义运行格式为`fgir-semantic-scores-v1`，完整样例见[run.json](../examples/semantic-evaluation/run.json)。主要字段：

| 字段 | 要求 |
| --- | --- |
| protocol_sha256 / data_kind | 绑定本次冻结protocol文件的实际字节及real/synthetic |
| run_id / method / seed / code_revision | 唯一运行ID、实际方法、训练种子、源码版本 |
| model | model_id、model_type、基础权重和tokenizer摘要；MLP额外adapter摘要；N系列额外LoRA、receipt、语义训练数据摘要 |
| predictions | 每个sample_id恰好一次，不缺失或混入额外样本 |
| scores | `original`及每个variant ID的源图匹配分数，高分表示更匹配；拒绝非有限数值或bool |
| parsing（可选） | 每条原句的预测entities/relations；若提交则必须覆盖全部样本 |
| rankings（可选） | 原句和全部paraphrase的图库ID列表；统一ranking_depth，完整列表或显式空结果 |

错误编辑不是整库检索qrels，不允许将其加入rankings；其正确相关图需另行标注。rankings只用源图作已知相关图，不声明其他图片不相关。全量原始5000查询的标准检索仍由旧协议执行，N系列run-v2的完整加载/receipt链需C/D继续实现。

本评分器要求模型摘要字段并检查基本类型/方法约束，但不加载模型验证分数来源，不认证尚未实现的run-v2制品链。不同N种子允许不同LoRA摘要；同一比较须使用同一基础权重、tokenizer、协议和配对种子，并提供一致的解析/排名能力及深度。训练结构和数据曝光的控制仍须C的运行配置支持。

## 6. 计分与比较

```powershell
python -m fgir evaluate-semantics --protocol examples/semantic-evaluation/frozen/protocol.json --run examples/semantic-evaluation/run.json --output artifacts/semantic-example-e0-v2 --bootstrap-samples 100
python -m fgir compare-semantics --protocol examples/semantic-evaluation/frozen/protocol.json --runs examples/semantic-evaluation/run.json examples/semantic-evaluation/candidate.json --output artifacts/semantic-example-comparison-v2 --bootstrap-samples 100
```

上面全为synthetic手工分数。真实运行时换成真实冻结协议及C/D的真实分数，不能改data_kind伪装。结果包含metrics.json、per-sample.jsonl、per-source.jsonl；比较包含comparison.json、cases.unreviewed.jsonl，原因初始为null，改善/退化/不变均保留。

- 配对区分：原句分数大于错误句记1，反向记0，严格相等记0.5；按现象输出。先在源图内平均，再对源图宏平均。
- 追加对照：同时输出真/假追加相对原句的分数变化、真句胜过假句的准确率、等长配对准确率及缺失/不等长数量。
- 解析：实体span、规范化实体、属性所属对象、有向关系的micro P/R/F1及源图宏F1。通过原文span和Unicode casefold/空白归一化对齐ID，不做模糊同义词匹配。无金标且无预测的分项为null。
- 同义检索：一组原句及全部同义句都命中源图才算joint recall；另报原句和变体recall。共同失败计0。绝对排名差仅使用完整图库排名，排除显式空返回并报告覆盖，避免将失败记为rank0。
- 不确定性：对源图均值bootstrap，不把同图变体当独立图片；成对方法按同种子比较，跨种子均值/样本标准差另列。单种子SD为null。某运行没有有效完整排名差时，比较披露未配对来源，不将缺失填成0。

排名差比较先按`(sample_id, variant_id)`对齐双方共同有效项，再计算逐项差和源图宏平均。`rank_gap_pairing`输出共同项、基线独有项和候选独有项的数量及具体ID；共同项为空时配对差为null。seed_summary中各运行自己的描述性均值仍保留，两次描述性均值之差不能替代配对差。修复与复验详情见[记录](reports/a-v2-fixes.md)。

未提供的解析或排名指标为null，不能当0分，也不能当已通过。外部匹配任务如SugarCrepe++与整库R@K需单列，本次未下载或运行该外部任务。

## 7. 下一步与依赖

具体执行入口已整理为[接续任务板](research-v2-tasks.md)，后续agent从根目录[AGENTS.md](../AGENTS.md)按成员角色接手。B-V2-01/02负责语言候选和train数据，H-V2-01由全体真人初标互审，C-V2-01/02负责LoRA与实验，D-V2-01/02/03负责制品、真实分数排名及服务，A-V2-04/05待依赖到位后冻结和评测。任务卡列出实际输入、交付路径、验收与解除条件；最新工具验收见[再次验收](reports/a-v2-reacceptance.md)。

B完成真实语言候选、人工初标与交叉复核后，A才能冻结val金标并对实际模型诊断。C/D需提供模型分数、原句/同义句排名及身份摘要；此后A执行正式比较、独立看图分析至少12例。当前工具和源图预审均不能替代这些外部产出，不能宣布新版A全部验收通过。
