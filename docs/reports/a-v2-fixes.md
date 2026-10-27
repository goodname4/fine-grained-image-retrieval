# 新版A三项问题修复与复验

日期：2026-10-27。结论：**A-V2-01、A-V2-02、A-V2-03三项代码问题均已修复并通过复验**。数据阶段仍待真实语义标注和独立人工复核，不据此宣布新版A全部完成。

修复前的现象及原始证据保留在[验收复查历史](a-v2-acceptance-review.md)。本轮没有更改真实工作标注的内容或审核身份。

## 修复结果

| 问题 | 修复 | 反例复验 |
| --- | --- | --- |
| A-V2-01：源图身份缺口 | source_context强制绑定inventory文件摘要，检查完整覆盖及image_id/path/split/file_sha256；实际图片和审核摘要均须匹配原inventory。冻结上下文保存所用源图的inventory条目 | 同路径替换图片并同步更新所有审核摘要，validate、freeze、pilot均拒绝；不产生无效冻结/试点输出 |
| A-V2-02：排名差未按句配对 | 按(sample_id,variant_id)提取双方共同有效的完整排名差，先计算同一对的差，再按源图平均和bootstrap | 两次运行只命中不同同义句时，配对差为null、共同对数为0，双方排除ID可追溯；部分共同有效时只用交集 |
| A-V2-03：金标冻结/评分口径不一致 | validate_record和parsing_scores共用parsing_sets，统一实体提及、属性、关系的规范化与去重规则，错误区分gold parsing与prediction parsing | 不同ID但相同span/head的重复金标在冻结前被拒绝；重复属性、重复有向关系同样拒绝；合法金标仍能评分 |

源inventory缺失、摘要不匹配、漏图、重复图、路径或划分错配、非法图片摘要均会报错。此要求同样适用于synthetic数据，已更新合成夹具和示例；不会默认绕过来源检查。旧语义协议若没有绑定inventory，需从实际源包重新冻结到新目录，不原地补写摘要。旧E系列检索协议与真实工作文件内容不受影响。

排名差的`seed_summary`仍保留每次运行自己的描述性均值。两次运行各自均值之差不再冒充配对差。`source_macro_deltas`中的该指标基于共同有效项；`rank_gap_pairing`提供共同对、基线独有对和候选独有对的数量，以及各自sample/variant/source ID。显式空返回仍参与同义共同命中失败计分。

## 测试与证据

新增9项测试，语义测试共36项，通过；全套共133项，132项通过、1项因Windows符号链接权限跳过。

新增覆盖包括：图片和审核摘要一起替换、inventory绑定和完整性、冻结来源上下文、重复金标、金标/预测错误区分、同义句无共同有效项、部分共同有效项，以及多个源图的宏平均和bootstrap。多源图例子中一个源图有2个配对差(0,+2)，另一个源图有1个配对差(-2)，结果按源图宏平均为-0.5，避免按3条句子算成0。

运行记录：

- `artifacts/a-v2-fix-semantic-tests.log`：36项语义测试。
- `artifacts/a-v2-fix-tests.log`：133项完整测试。
- `artifacts/a-v2-fix-acceptance-v1/acceptance.json`：三个问题逐项复验、真实数据审计与实现文件摘要。
- `artifacts/a-v2-fix-acceptance-v1/source-image-binding/`：替换图像反例和被拒绝结果。
- `artifacts/a-v2-fix-acceptance-v1/paraphrase-pairing/comparison/`：无共同同义句时的null配对差及排除明细。
- `artifacts/a-v2-fix-acceptance-v1/gold-validation/`：重复金标在冻结前被拒绝，合法金标仍能正确评分。

可复验的仓库测试命令：

```powershell
python -m unittest discover -s tests -p test_semantics.py -v
python -m unittest discover -s tests -v
```

修复前`artifacts/a-v2-review-repros.py`用于记录旧缺陷，保留其原始断言和历史输出；当前代码应当阻止其中的异常输入。修复后的验收以新增回归测试和本轮证据为准。

## 真实数据与剩余工作

真实val任务仍为30张源图，A的8组仍仅是助手原句预审，其余22组draft；实体、关系、变体及独立人工金标尚未补齐。

使用当前实现，在临时新目录重建原始试点并重新应用版本化source-checks，两个工作文件的SHA-256均与现有文件一致：

- 原始任务包：`89cd73664d5022a87b8ad24154740d57eae19affbe180a094ea14654742bad31`。
- 助手预审工作文件：`33d44c2bd2c8e142e768bd6544fa0d664a83b1bd4b9ddd7b6dd59cedfbc49beb`。

当前30张图片已通过原始inventory的字节核对。源图重建通过不表示语义金标已完成；人工复核、C/D真实模型分数、正式test和真实错误分析仍按[新版A交接](../handoff-a-v2.md)继续。
