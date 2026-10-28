# B-V2-01：语言解析与候选生成（30 组 val 工作包）

2026-10-28，B / agent。本报告记录生成器的真实权重、提示词、seed、源码版本与全部产物；候选全部为 draft，视觉真假、语义等价与对象绑定留待 H-V2-01 真人复核，本报告不认证任何人工金标。

## 1. 实际输入与本机重建

本机原先没有 `data/semantic/`。按[新版A交接](handoff-a-v2.md)第2节重建（不覆盖任何已有目录，均为新目录）：

```powershell
python -m fgir prepare-semantic-pilot --output data/semantic/pilot-val-v1 --count 30 --seed 42 --code-revision c6cfa34
python -m fgir apply-semantic-source-checks --samples data/semantic/pilot-val-v1/samples.work.jsonl --checks configs/semantic/a-pilot-source-checks-v1.json --output data/semantic/pilot-val-assisted-v1
```

- 30 组 val 源图、分配 A/B/C/D=8/8/7/7、独立复核 B/A/D/C 与既有分工一致；8 条助手原句预审（4 supported / 4 uncertain）原样保留，annotator_kind=assistant 未改动。
- 源包 `data/prepared/flickr8k-v1` 的 manifest 摘要 `a7f7d25d…dc0d`，图片字节逐张校验通过。

## 2. 生成器与真实权重

| 项 | 值 |
| --- | --- |
| 本地模型 | Qwen/Qwen2.5-0.5B-Instruct（经 hf-mirror 下载，快照 `7ae557604adf67be50417f59c2c2f167def9a775`，单文件 `model.safetensors`） |
| weights_sha256 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`（对实际权重文件流式计算） |
| prompt_sha256 | `b28c65ee6e46518e60798be21f2b45280ea996da5fa2c9533da0f44ffcc79be0`（`configs/semantic/b-generation-prompts-v1.json` 实际字节；副本在产物 `prompts.json`） |
| seed / code_revision | 42 / `22efaf5-dirty`（HEAD 22efaf5 + 未提交源码） |
| 环境 | torch 2.12.1+cpu、transformers 5.20.1、CPU，greedy 解码（确定性） |
| provenance.generator.name | `local-qwen2.5-0.5b-instruct-templates-v1` |

生成器不调用外部推理服务；权重缺失时（`--no-download`）报错并给出获取步骤，不伪造摘要（测试覆盖）。

## 3. 管线设计（`src/fgir/semantic_generation.py`）

1. **LM 解析提案**：少样本提示要求输出 `ENTITY / ATTRIBUTE / RELATION` 行、多词短语下划线连接、谓词逐字照抄。所有短语必须**唯一地锚定**为原句的连续子串（词边界、大小写回落到原文字符）；锚定失败即拒绝并记原因。
2. **确定性谓词回退**：模型谓词非逐字时，在已锚定主语与宾语之间的原文区间内按规则恢复真实动词短语（有限动词/助动词+ing），恢复不出即拒绝；模型把多条关系合并成一行时，抢救其中可锚定的名词短语为实体。
3. **词典补充**：颜色/尺寸词表（color/size）按词边界扫描，按三条规则解析归属实体；与 LM 属性按值去重。
4. **模板变体**（每次只改一个条件，全部由声明 edit 精确重建）：
   - `attribute_binding`：交换两个不同属性值，edit/unit span 覆盖两处值；
   - `relation_direction`：保守 SVO 检查 + 单词有限动词门槛（排除助动词、-ing/-ed），交换主语/宾语名词短语；
   - `paraphrase`：LM 改写提案，排除空串、原样复制（忽略标点前后空格差异）与长度异常；
   - `true/false_addition`：词表模板成对追加（`control_pair_id="addition-pair"`）。
5. 每条记录保留 `assigned_to`、`independent_reviewer`、原 review（含 8 条 assistant 预审）与 `source_visual_label`；新增 `generation_notes` 说明逐项拒绝原因；LM 原始输出存 `raw-outputs.jsonl` 备查。

## 4. 产物与统计

- `data/semantic/pilot-val-candidates-v1/samples.work.jsonl`：完整 30 组，`samples_sha256=5058e091f5f55a44d4ba0c960c818d9b6237f4b48f9e8f82b380a62b7a36b0be`
- 同目录 `generation.json`（配置/来源记录）、`rejections.jsonl`（逐项拒绝与适用性说明）、`raw-outputs.jsonl`、`prompts.json`、`receipt.json`
- 离线审核页：`artifacts/semantic-pilot-val-candidates-v1/board.html`（30 张本地图，可筛选、编辑、导入/导出；导出须先下载）

| 统计 | 数量 |
| --- | --- |
| 实体 / 关系 | 100 / 19 |
| 变体 | 99 = 6 绑定 + 4 方向 + 29 改写 + 60 追加（30 组真/假配对） |
| units | 10（每个最小编辑恰一个关联 foil） |
| 逐项拒绝 | 78（parse 27、binding 24、direction 26、paraphrase 1），原因均保留在 rejections.jsonl |

抽样质量（结构校验已通过，语义待人工）：

- 绑定：`The white poodle dog is coming out of a blue tent .` → `The blue poodle dog is coming out of a white tent .`
- 方向：`A dog chases a toy in the grass with its owner in the background .` → `A toy in the grass with its owner in the background chases a dog .`
- 已知弱点：`a/an` 一致（`An large man with a old belly`）、换位后大小写（`Young blossoms`）等由真人复核时修正；不存在的编辑不强制构造（如仅含系动词/无两个属性的记录只留拒绝原因）。

## 5. 验收与测试

- `python -m fgir validate-semantics --samples data/semantic/pilot-val-candidates-v1/samples.work.jsonl --purpose draft --data-root data`：通过（30/30，图片字节逐一核对）。
- 新增 `tests/test_generation.py` 10 项：完整管线产出合法 draft 记录、每个 foil 仅对应声明编辑、非法 span/错引用/垃圾 LM 输出逐项拒绝并留原因、来源错配整体拒绝且不落盘、词典补充、assistant 预审与分配字段保持、同 seed 确定性、追加配对恒在、缺权重报下一步且不伪造、权重摘要必须真实 64 位 hex。
- 全套 `python -m unittest discover -s tests -v`：143 项全部通过。附带修复本机 `runs.py` 的 `_git` 用默认 GBK 解码 git 输出导致 `code_revision` 崩溃的既有缺陷（改为 UTF-8 + errors=replace），影响 training/runs 旧测试，与 B-V2-01 功能无关。

## 6. 复现命令

```powershell
python -m fgir generate-semantics --samples data/semantic/pilot-val-assisted-v1/samples.work.jsonl --output data/semantic/pilot-val-candidates-v1 --seed 42 --code-revision 22efaf5-dirty
python -m fgir validate-semantics --samples data/semantic/pilot-val-candidates-v1/samples.work.jsonl --purpose draft --data-root data
python -m fgir render-semantics --samples data/semantic/pilot-val-candidates-v1/samples.work.jsonl --output artifacts/semantic-pilot-val-candidates-v1
```

## 7. 下一步与依赖

- **H-V2-01**：全体真人按 `assigned_to`/`independent_reviewer` 在审核页逐组复核候选（改写同义性、对象绑定、视觉真假、追加真假），导出完整 JSONL 到新版本目录；助手/模型来源不得升级为 human。
- **B-V2-02（train 独立候选）**：生成器对任意源绑定记录通用，但当前 `prepare-semantic-pilot` 只抽 val。train 工作包需要独立抽样/导出接口（计划新增 `prepare-semantic-train` 或对生成器加 `--split train` 抽样参数），从 `data/prepared/flickr8k-v1/train/pairs.jsonl` 独立构建、绝不复制 val 任务；核验（真人抽检）与 `semantic_dataset_sha256` 绑定留待该任务完成，保持 BLOCKED。
- 正式 val 冻结（A-V2-04）仍需 H-V2-01 完成；本产物是候选输入，不是已冻结金标。
