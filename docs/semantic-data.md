# 语义数据协议 semantic-v1

更新日期：2026-10-27。数据校验、试点抽样、审核页、冻结和评分CLI已实现，详见[新版A交接](handoff-a-v2.md)。本地已有30张val源图任务及8条助手原句预审，没有独立人工金标或语言变体。本地语言模型生成器仍待B实现。旧图片清单及检索qrels继续使用原格式。

## 1. 目录、划分与角色

每行以一张源图的一条原始caption为样本，group_id等于source_image_id；scene_group_id采用保留图片已知同场景连通组的最小ID，不让同场景跨集合。已有data/semantic/pilot-val-v1和pilot-val-assisted-v1；后续train/val/test分别导出新目录。冻结语义诊断用protocol.json绑定文件，不复用旧dataset.json或qrels语义。

- train用于语言数据生成、训练及可选失败驱动采样。
- val用于约30源图试点、独立诊断和选参。
- test只在方法及生成/复核协议确定后用于固定评价；记录已有接触，不使用其错误改进生成模板或损失。

一个语义样本可以有多个变体，但图片、原句、同义句、组件、错误变体、追加细节均跟随同一源图划分。来源必须能回连A交接包的image_id与query_id，禁止重新随机划分派生文本。

源交接包必须提供被dataset.json摘要绑定的inventory.jsonl，真实和synthetic包使用同一规则。inventory须恰好覆盖manifest图片，image_id/path/split逐条一致且file_sha256有效；图片字节和审核摘要都必须匹配原始inventory。冻结上下文保留所用图片的inventory条目与源文件摘要，拒绝仅更新审核摘要的源图替换。

## 2. 记录字段

| 字段 | 类型及要求 |
| --- | --- |
| schema_version | 固定fgir-semantic-v1 |
| sample_id/group_id | 全局唯一样本ID/稳定源图分组ID |
| source_image_id/source_query_id/split | 必须与绑定manifest和pairs逐条一致 |
| scene_group_id | 由源交接包duplicate-review.json和保留图像计算，不由生成器自定 |
| text | 保留原始英文caption，非空 |
| source_visual_label | supported/uncertain；原始caption也需看图，不能默认成立 |
| entities | id、head、span、attributes；属性引用所属entity_id |
| relations | id、subject、predicate、object、span；subject/object引用存在的实体 |
| variants | 唯一id、type、phenomenon、text、target_unit_id、edit、semantic_relation、visual_label、naturalness、review；追加有control_pair_id |
| units | 唯一id、连续原文text/span、type(entity/relation)、entity_ids、foil、关联完整句错误variant_id、review |
| provenance | source_manifest_sha256、generator模型/权重/提示词摘要、seed、代码版本 |
| review | 初标人与独立复核人、状态、依据与争议记录 |

span为Python字符串的Unicode字符半开区间[start,end)，不是token位置或UTF-8字节偏移；必须可回取原文。关系证据允许覆盖完整原句，但不可据此推断句外内容。属性必须同时有属性值与原文证据。

源查询采用原pairs的`image_id#编号`，不能按文本去重或重新编号。实体和关系ID在记录内唯一，variant和unit也不许复用这些ID；`sentence`为整句目标，`original`为评分原句保留键。组件foil必须准确应用variant_id关联的那一次最小编辑。

数据校验与解析评分共用parsing_sets：即使使用不同ID，同一原文span及规范化head的实体也不能重复；规范化后的属性和有向关系三元组同样去重检查。金标错误和预测错误分别标明，防止冻结出无法评分的金标。

variant的type枚举：paraphrase、minimal_edit、true_addition、false_addition；edit包含目标字段、before/after、原文替换区间或追加位置。semantic_relation记录equivalent/changed，不将“语义变了”等同于“图片不匹配”。visual_label为supported/contradicted/uncertain/not_applicable；仅凭caption未提及不能判contradicted。

同义句必须保留所有已知对象身份、属性绑定、数量、关系方向和否定范围。禁止无依据地添加“快乐”“正在玩耍”等常识。表述是否等价与是否被图片支持分别核验。

## 3. 示例（示意、未审，不是可训练数据）

下面为简化草稿示意，使用占位来源，不能通过真实源数据绑定。完整可校验的合成记录见[gold.jsonl](../examples/semantic-evaluation/gold.jsonl)，仅供字段/算术验收。实际摘要、图片依据与人工身份必须真实。

```json
{
  "schema_version": "fgir-semantic-v1",
  "sample_id": "example-0001",
  "group_id": "image-example",
  "scene_group_id": "image-example",
  "source_image_id": "image-example",
  "source_query_id": "image-example#0",
  "split": "train",
  "text": "A man in a red shirt walks a black dog.",
  "source_visual_label": "uncertain",
  "entities": [
    {"id": "e1", "head": "man", "span": [2, 5],
     "attributes": [{"entity_id": "e1", "name": "shirt_color", "value": "red", "span": [11, 14]}]},
    {"id": "e2", "head": "dog", "span": [35, 38],
     "attributes": [{"entity_id": "e2", "name": "color", "value": "black", "span": [29, 34]}]}
  ],
  "relations": [
    {"id": "r1", "subject": "e1", "predicate": "walks", "object": "e2", "span": [21, 26]}
  ],
  "variants": [
    {"id": "v1", "type": "minimal_edit", "phenomenon": "attribute_binding", "text": "A man in a black shirt walks a red dog.",
     "target_unit_id": "e1", "semantic_relation": "changed",
     "edit": {"field": "e1,e2.color_binding", "before": "red shirt walks a black", "after": "black shirt walks a red", "span": [11, 34]},
     "visual_label": "uncertain", "review": {"status": "draft"}}
  ],
  "units": [],
  "provenance": {
    "source_manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "generator": {"name": "source-only", "weights_sha256": null, "prompt_sha256": null},
    "seed": 42, "code_revision": "documentation-example"
  },
  "review": {"status": "draft"}
}
```

正式记录不得因为给出编辑例子就声称图片真假已判断。示例交换了人物衣服与狗的颜色归属，是否contradicted需要真实图片证据。单独把red换成blue只属于属性值替换，不能据此声称完成属性绑定评测；现象标签需要人工核验。

## 4. 生成与质量门槛

B从train原始pairs生成实体和关系候选，再生成同义句和每次只修改一个条件的foil。使用本地小模型；规则只做结构校验，不代替视觉真假判断。

检查顺序：

1. source_query_id对应原句与image_id，split及manifest摘要匹配。
2. 实体/关系ID唯一、引用存在、span可回取、属性有归属。
3. 最小编辑只改声明条件；避免近义替换、对称关系的无效主客交换及模板可猜偏差。
4. 看图或由记录了图片证据的本地VLM核验：编辑后语句必须明确与源图矛盾；不明确则uncertain。
5. 抽样人工核验自动标签，逐现象记录准确率、采纳/拒绝/争议/未审数；评测样本全部独立人工复核。
6. 冻结语义文件、审核与排除记录、生成器/提示词版本；所有N组共用冻结版本。

默认训练只接受受支持正例、经核验矛盾负例；draft/uncertain/有争议样本排除，不自动改成已审。批次可没有关系foil，但整个训练版本必须有有效监督及覆盖统计。训练自动核验和独立人工复核使用不同状态，不能冒充。

校验器使用`--purpose draft/train/eval`。train模式拒绝val/test并检查监督状态，报告错误而非静默删除；B/C如需排除须导出排除清单。eval模式要求每条记录及每个variant/unit均reviewed，原句supported、同义和真追加supported、错误变体contradicted且naturalness=acceptable。仅结构正确的最小编辑不能证明只改一个语义条件，仍须实际审查。

review.status枚举draft/annotated/auto_verified/reviewed/rejected；annotated/auto_verified/reviewed须有annotator、annotator_kind、notes、image_evidence、image_sha256，rejected须注明拒绝原因。reviewed额外要求human初标、不同的human reviewer、reviewer_kind=human、review_notes和空disputes。助手预审标assistant，不能升级成独立人工金标。生成器name=source-only或human可无模型权重；本地模型生成需实际weights_sha256/prompt_sha256，另记seed和code_revision。

## 5. 诊断集合与类别兼容

旧contracts.CATEGORIES限定color/count/action/relation，新现象不能直接写成binding等类别传入旧验证器。

已单独引入phenomenon字段，首版枚举attribute_binding、relation_direction、paraphrase、false_addition。真追加作为false_addition现象的正确控制项，不作为负例计分。检索查询仍可映射到旧color/relation等类别，但显式绑定诊断任务版本，不修改历史标注枚举的含义。

诊断协议fgir-semantic-eval-v1绑定图像集合、样本JSONL、审核记录与各文件摘要，声明按源图分组和并列计分规则。语义二选一准确率、解析F1和整库R@K分别输出，不能把诊断记录直接充作旧retrieval qrels。

实际冻结文件为samples.jsonl、candidates.json、audit.json、source-context.json及protocol.json；每份文件均有摘要。独立分数格式fgir-semantic-scores-v1，计分定义、缺失策略、模型身份声明边界及配对种子要求见[交接](handoff-a-v2.md)。

原句与同义句的共同命中可复用新检索run。minimal_edit若只是对源图的错误描述，仅参加配对匹配诊断；若要评估该编辑查询的整库检索，需另标实际相关图。

## 6. 标注组织

首批约30张val源图用于试点，不从test挑有利案例。正式诊断目标100—200张test源图，由A组织均分；选100张则每人25个源图组，每组多个变体和视觉核验，工作量大于原来的25条句子。A/B互审、C/D互审；争议由第三人或全组讨论并留痕。

对正确/错误追加细节同时记录语法自然性、长度和涉及对象，避免系统只学会长句降分。若同义表达或空间关系存在视角歧义，改写或排除并说明，不为凑数量强行赋值。
