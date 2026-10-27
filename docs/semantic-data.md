# 语义数据协议 semantic-v1（待实现）

更新日期：2026-10-27。本协议定义新语言数据，尚无对应生成/校验CLI。下列目录、字段与JSON都是设计，不代表已有真实标注。旧图片清单及检索qrels继续使用原格式。

## 1. 目录、划分与角色

计划新增data/semantic/semantic-v1/{train,val,test}.jsonl及dataset.json。每行以一张源图的一条原始caption为样本，group_id稳定绑定源图；有近重复场景组时记录scene_group_id，不让同场景跨集合。

- train用于语言数据生成、训练及可选失败驱动采样。
- val用于约30源图试点、独立诊断和选参。
- test只在方法及生成/复核协议确定后用于固定评价；记录已有接触，不使用其错误改进生成模板或损失。

一个语义样本可以有多个变体，但图片、原句、同义句、组件、错误变体、追加细节均跟随同一源图划分。来源必须能回连A交接包的image_id与query_id，禁止重新随机划分派生文本。

## 2. 记录字段

| 字段 | 类型及要求 |
| --- | --- |
| schema_version | 固定fgir-semantic-v1 |
| sample_id/group_id | 全局唯一样本ID/稳定源图分组ID |
| source_image_id/source_query_id/split | 必须与绑定manifest和pairs逐条一致 |
| text | 保留原始英文caption，非空 |
| entities | id、head、span、attributes；属性引用所属entity_id |
| relations | id、subject、predicate、object、span；subject/object引用存在的实体 |
| variants | type、text、target_unit_id、edit、semantic_relation、visual_label、review |
| units | 原始组件短语、组件类型、实体引用及对应foil，保留生成与核验信息 |
| provenance | source_manifest_sha256、generator模型/权重/提示词摘要、seed、代码版本 |
| review | 初标人与独立复核人、状态、依据与争议记录 |

span为Python字符串的Unicode字符半开区间[start,end)，不是token位置或UTF-8字节偏移；必须可回取原文。关系证据允许覆盖完整原句，但不可据此推断句外内容。属性必须同时有属性值与原文证据。

variant的type枚举：paraphrase、minimal_edit、true_addition、false_addition；edit包含目标字段、before/after、原文替换区间或追加位置。semantic_relation记录equivalent/changed，不将“语义变了”等同于“图片不匹配”。visual_label为supported/contradicted/uncertain/not_applicable；仅凭caption未提及不能判contradicted。

同义句必须保留所有已知对象身份、属性绑定、数量、关系方向和否定范围。禁止无依据地添加“快乐”“正在玩耍”等常识。表述是否等价与是否被图片支持分别核验。

## 3. 示例（示意、未审，不是可训练数据）

下面为简化示意，完整生成器还需填units及provenance中的实际摘要；占位ID和摘要不得进入正式产物。

```json
{
  "schema_version": "fgir-semantic-v1",
  "sample_id": "example-0001",
  "group_id": "image-example",
  "source_image_id": "image-example",
  "source_query_id": "image-example#0",
  "split": "train",
  "text": "A man in a red shirt walks a black dog.",
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
    {"type": "minimal_edit", "text": "A man in a blue shirt walks a black dog.",
     "target_unit_id": "e1", "semantic_relation": "changed",
     "edit": {"field": "e1.shirt_color", "before": "red", "after": "blue", "span": [11, 14]},
     "visual_label": "uncertain", "review": {"status": "draft"}}
  ],
  "review": {"status": "draft"}
}
```

正式记录不得因为给出编辑例子就声称图片真假已判断。示例中minimal_edit只确定改变了衣服颜色语义，是否contradicted需要真实图片证据。

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

## 5. 诊断集合与类别兼容

旧contracts.CATEGORIES限定color/count/action/relation，新现象不能直接写成binding等类别传入旧验证器。

计划单独引入phenomenon字段，首版枚举attribute_binding、relation_direction、paraphrase、false_addition。检索查询仍可映射到旧color/relation等类别，但显式绑定诊断任务版本，不修改历史标注枚举的含义。

诊断协议fgir-semantic-eval-v1绑定图像集合、样本JSONL、审核记录与各文件摘要，声明按源图分组和并列计分规则。语义二选一准确率、解析F1和整库R@K分别输出，不能把诊断记录直接充作旧retrieval qrels。

原句与同义句的共同命中可复用新检索run。minimal_edit若只是对源图的错误描述，仅参加配对匹配诊断；若要评估该编辑查询的整库检索，需另标实际相关图。

## 6. 标注组织

首批约30张val源图用于试点，不从test挑有利案例。正式诊断目标100—200张test源图，由A组织均分；选100张则每人25个源图组，每组多个变体和视觉核验，工作量大于原来的25条句子。A/B互审、C/D互审；争议由第三人或全组讨论并留痕。

对正确/错误追加细节同时记录语法自然性、长度和涉及对象，避免系统只学会长句降分。若同义表达或空间关系存在视角歧义，改写或排除并说明，不为凑数量强行赋值。
