# 新版A验收复查（修复前记录）

后续状态：本文记录的三项代码问题已修复，复验结果见[修复记录](a-v2-fixes.md)。下文保留修复前的结果与证据；真实语义金标仍待完成。

验收日期：2026-10-27。结论：**尚未通过正式验收**。正常流程可运行，但额外反例复现出3项代码问题；真实语言标注和独立人工金标也尚未达到P0门槛。

本次只复查、生成隔离反例和记录结论，没有修改算法实现或真实工作标注。所有反例均使用synthetic夹具，输出位于`artifacts/a-v2-review-evidence-v1/`。本报告补充此前的实施记录：测试通过并不等于完整验收通过。

## 1. 发现的问题

### A-V2-01 · P1：未用原始inventory校验源图身份

位置：`src/fgir/semantic_data.py:84`及`:267`。source_context只检查manifest、pairs和duplicate-review，没有绑定inventory.jsonl；validate_semantics只比较审核记录中的image_sha256与当前文件。

复现：为合成数据包建立并绑定原始inventory，先确认旧verify_handoff通过；随后在相同路径替换一张图片，并让新审核记录如实记录替换后的图片摘要。旧verify_handoff正确报`image file hash mismatch: val-a`，但新版validate_semantics、freeze_semantics和prepare_semantic_pilot全部通过，审计仍报告image_bytes_checked=true。

影响：在同一个source_image_id及旧manifest版本下，替换后的图片会被当作原固定数据集的一部分，破坏训练/评测来源与缓存的一致性。审核者看过当前图片不能证明它就是已冻结数据集中的图片。

修复要求：校验并绑定源inventory的文件摘要和image_id/path/split，逐张比较实际图片字节、审核image_sha256与inventory.file_sha256。对于明确的synthetic夹具，应提供相同结构的inventory或采用明确且隔离的夹具规则。添加“图片已替换且审核摘要同步更新”仍被拒绝的测试。

真实数据核查：当前30张试点图片与原始inventory的字节摘要全部一致。此发现是校验缺口，不表示真实图片已经发生替换。

### A-V2-02 · P2：排名差的配对比较没有对齐具体同义句

位置：`src/fgir/semantic_evaluation.py`的compare_semantics，先从per-source读取均值，再直接相减的循环。

复现：同一源图有para、para2两个同义句。基线只有para返回完整结果，排名差为0，para2显式返回空；候选只有para2返回完整结果，排名差为2，para显式返回空。两次运行没有共同可计算排名差的同义句，但比较输出`source_macro_deltas[paraphrase/absolute_rank_gap_full_rankings].source_macro_mean=2.0`，且unpaired_metric_sources为空。

影响：把不同查询的均值相减并称为配对变化，会把返回失败引起的样本构成变化混入模型效果；现有“未配对源图”检查发现不了同一源图内部的缺失错位。

修复要求：先按`(sample_id, variant_id)`对齐双方都有完整排名的有效对，计算同一对的差，再做源图宏平均及bootstrap。共同有效对为空时输出null，并披露双方缺失的具体ID与数量。两次运行各自的描述性均值仍可保留，但需与真正的配对差分开。

### A-V2-03 · P2：冻结校验与解析评分器不一致

位置：`src/fgir/semantic_data.py:166`和`src/fgir/semantic_evaluation.py:121`。

复现：在合法合成金标中复制同一实体提及，使用不同entity ID，但保留相同head/span并更新其属性所属ID。validate_semantics、freeze_semantics及load_semantic_protocol均允许；提交正常预测进行evaluate_semantics时，评分器在读取金标阶段报`duplicate predicted entity mention`。问题在金标，却被描述为预测错误。

影响：审核后冻结的协议可能无法用于解析F1评分，必须返工数据版本；属性重复和关系重复也有同类校验口径风险。

修复要求：让冻结和评分共用解析结构验证规则，至少提前拒绝重复的规范化实体提及、属性和有向关系；区分金标错误与预测错误。添加“冻结成功的金标必能进入解析评分”的针对性测试。

## 2. 已实际通过的检查

| 检查 | 结果 |
| --- | --- |
| 全量单元测试重新运行 | 124项，123通过、1项因Windows符号链接权限跳过 |
| 30源图试点重新生成 | 与原始任务包逐对象一致 |
| 真实工作文件来源/划分检查 | 30个不同val源图，通过 |
| 实际图片对照原始inventory | 30/30摘要一致 |
| 当前草稿作为正式金标输入 | 正确拒绝：not independently reviewed |
| 现有审核进度核对 | 8组助手原句预审、22组draft，与文档一致 |

测试日志：`artifacts/a-v2-review-tests.log`。上述通过项不覆盖第1节的3个反例。

## 3. 尚未完成的内容验收

当前30组的entities、relations、variants及units尚未补齐；独立人工复核为0，正式可用金标为0。现有8组annotated只代表助手对原句做过图片预审。

因此，即使先修复3项代码问题，P0仍需完成真实实体/关系、语义变体和实际人工初标互审。正式test、N系列真实模型分数、标准run-v2以及至少12个真实模型错误案例也仍待完成。不能将此阶段判断成“新版A全部工作完成”。

## 4. 证据和复现

机器可读结论：`artifacts/a-v2-review-evidence-v1/review.json`。

三个独立证据目录：

- `source-image-binding/`：原inventory、替换后的合成图片、被接受的冻结协议。
- `unpaired-paraphrase-gaps/`：两条同义句、两次运行和错误的配对差输出。
- `unevaluable-frozen-gold/`：含重复实体的已冻结协议和评分输入。

选择一个尚不存在的新输出目录执行：

```powershell
python artifacts/a-v2-review-repros.py --output artifacts/a-v2-review-evidence-v2
```

复现脚本使用现有synthetic测试夹具，仅修改新证据目录内的文件；真实试点只进行读取与摘要核对。

修复后的验收顺序：先运行这3个反例对应的回归测试，确认异常输入在正确边界被拒绝、配对统计正确；再跑原套件和真实任务包复现；最后按实际人工复核状态判断数据阶段是否通过。
