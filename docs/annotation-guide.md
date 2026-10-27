# 细粒度标注、复核与合并

> 2026-10-27版本说明：下文是旧100条检索相关性标注的现有操作。新组合语义诊断按源图组标注同义/最小编辑及真假追加细节，遵循[semantic-v1](semantic-data.md)，尚未实现对应页面或CLI。旧test初稿已被接触，需披露该事实；新验证试点从val构建，不据test错误调整方法。历史初标/复核数字以实际文件为准。

## 已准备的工作

`prepare-a-data`从真实test描述中按固定seed42筛选100张不同源图，每种主类别25条，轮转分给A/B/C/D各25条。额外类别可多标签。自动规则只是筛选：例如号码“six”不是六个主体，“through the air”不一定是主体之间的空间关系。

初稿位于`data/annotations/fine-v1/annotations.jsonl`。当前工作文件为同目录`annotations.work.jsonl`：A的25条已逐图完成AI辅助初标，以固定25图源图池逐条判断相关性；B的25条有逐图预审建议，其余仍为初稿。没有任何记录被冒充为独立复核完成。A的辅助初标文本、类别修订与B预审意见可从`configs/annotations/a-source-checks.json`复现。

25图池并不是1000图的完整相关性标注，也未覆盖未来模型排名的并集；正式评测前必须扩展审图范围。B应复核A，A应在B完成初标后进行独立复核；预审意见不能算已经完成B的独立复核。C/D同理互审。

## 打开工作台

现有本地页面：`artifacts/annotation-board-current/board.html`。双击可离线使用，全部缩略图嵌入HTML，不需要图像服务器或外部AI接口。其他电脑重建最新工作台：

```bash
python -m fgir render-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.work.jsonl --data-root data --output artifacts/annotation-board
```

按“分配”筛选成员，再选查询。页面显示源图和已有预审建议；全文原图路径在清单中，细节不清时查看原图，不凭模糊缩略图决定。

1. 核实文本只包含图中能够判断的条件。修正拼写，避免无依据的国籍、不可见动作或不明确计数。
2. 选择color/count/action/relation中真实相关类别。颜色指物体/衣着属性；数量指主体/物体数量；动作需定义是否含静态姿态；关系应明确主体及参照物。
3. 逐图标记“相关/不相关/未审”。自定义描述可能匹配多个图片，相关ID不止源图。未审图片不得当作负样本。“本页全部不相关”仅在实际逐图检查后使用。
4. 记录标注人、范围、池构建方法和不确定事项。保存初标后导出JSONL，不能只关页面。页面在内存保存，重启浏览器不保证恢复；先点保存，再点导出。
5. 另一位成员独立看图并填写复核人、说明及争议处理。复核后才能保存`reviewed`。名字不同是最低结构检查，不是独立复核的证明；不得代填他人的复核。

## 范围与版本

- `candidate_pool`：`judged_image_ids`列出已审池，`pool_method`记录构建方法及模型版本，`relevant_image_ids`为池内全部已知正例。未来返回的未审结果会被评测工具拒绝，必须扩大共同池后重评全部方法。
- `full_gallery`：每条查询的`judged_image_ids`必须等于1000个固定候选，才可宣称全图库复核。
- `draft → annotated → reviewed`分别表示筛选初稿、已初标、已独立复核；修改文字、类别或相关图片后应重新复核。工作台图片标记变更会降回初稿。
- 不混合两种范围生成同一正式协议；不根据测试错误反复调模型后继续把此集合称为未见测试。

校验进度：

```bash
python -m fgir validate-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.work.jsonl
```

## 多人导出的合并

所有人从**同一份基准文件**开始，把导出另存为`data/annotations/exports/A.jsonl`等，每份仍包含原来的100个query_id。不要分别覆盖公共工作文件。

```bash
python -m fgir merge-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --base data/annotations/fine-v1/annotations.work.jsonl --updates data/annotations/exports/A.jsonl data/annotations/exports/B.jsonl data/annotations/exports/C.jsonl data/annotations/exports/D.jsonl --output data/annotations/merged-v1
```

只合并相对基准发生变化的查询；同一查询有不同版本则报冲突，不悄悄采用后提交者。query_id、原始源图/源描述、分配和计划复核人不允许改变。合并结果还须通过整体校验：即使各份导出单独合法，不同查询合并后出现规范化文本重复，也会报错且不发布输出目录。输出合并文件和摘要回执；冲突由A组织核对后用一致版本重跑。多轮初标、复核要各自固定共同基准。

## 先收集各模型结果，再完成共同候选池

人工相关性尚未完成时，不应为了让模型返回Top-K而伪造一个已复核协议。先单独冻结**查询文本和图库**的排名请求，不包含qrels：

```bash
python -m fgir make-pool-request --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.work.jsonl --output data/annotations/pool-request-v1
```

本机已生成该请求，仍包含未完成初标的描述，仅供开发接入。B/C读取请求目录的`queries.jsonl`和`candidates.json`，使用实际模型输出所有query_id的有序图片ID，保存为以下结构：

```json
{
  "schema_version": "fgir-pool-run-v1",
  "request_sha256": "实际request.json摘要",
  "model_id": "实际方法和版本，例如E0-seed42-encoder版本",
  "rankings": {"实际query_id": ["有序图片ID，至少Top-10"]}
}
```

各方法必须使用同一请求；文本修订后新建请求并重跑，工具拒绝旧文本的结果。真实E0/E1/E2候选返回后：

```bash
python -m fgir expand-annotation-pool --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.work.jsonl --request data/annotations/pool-request-v1/request.json --runs artifacts/e0.pool.json artifacts/e1.pool.json artifacts/e2.pool.json --depth 10 --output data/annotations/common-pool-v1
python -m fgir render-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/common-pool-v1/annotations.jsonl --data-root data --output artifacts/common-pool-board
```

共同池是各模型Top-K、此前已审图片和相关图片的并集，记录各run摘要。新增成员保持未审，不自动作为正例或负例；旧reviewed状态会退回待复核。工作台可选“当前候选池/候选池未审”，加快补审。声明已复核时仍有候选池成员未审，工具将拒绝。

这些pool run只用于收集待标注图片，不是正式模型效果。正式评测仍需最终冻结协议及绑定它的`fgir-run-v1`真实排名。

## 冻结正式协议

所有100条通过真实独立复核且每类至少20条后：

```bash
python -m fgir freeze-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/merged-v1/annotations.jsonl --output data/annotations/fine-reviewed-v1
```

当前工作文件应被此命令拒绝，因为仍缺其他成员初标、独立复核和共同池补审。这是实际未完成项，不能通过改状态字符串绕过。标准val/test源图配对协议已经可供B/C开发，不受此项阻塞。
