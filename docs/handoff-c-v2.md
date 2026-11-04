# C 接续交接与全组输入清单

更新：2026-11-04，C / Codex assistant。接手先拉取最新 `main`，再读[任务板](research-v2-tasks.md)。C/D实现已合并；C可导LoRA、受控损失、编排与D验证接口均已具备。[本次真实核验报告](reports/c-v2-followup-20261104.md)记录已做工作和摘要；正式N0—N3及消融尚未运行，C-V2-02保持BLOCKED。

## 1. 当前实际收到什么

收到的 `fgir-human-operation-20261030-v1.zip` 含train106组/334变体/39组件，val33组/102变体/13组件，全部是assistant工作稿，0组真人金标。它是待审核任务包。原30组val加3组补充、原100组train加6组补充，不能沿用旧任务板的8/8/7/7分配。

本机166项文件摘要、139张图片、582个本地链接、源图划分与draft校验通过。C原始两步开发权重经合并后的D接口导出5000×1000完整val排名，严格重载及离线查询通过；173项测试172过、1项环境跳过。这些证据支持开发联调，不产生正式效果结论。

## 2. 先由 A 组织四位真人完成审核

负责人将A/B/C/D映射到四位不同真人，填写操作包的 `forms/member-mapping.md`；实际姓名留在另传审核文件，禁止假名绕过初标/复核分离。[完整人工操作手册](human-operation-manual-v2.md)已随本次文档交接纳入Git。

| 真人分工 | 初标val / train | 初标总数 | 复核谁 | 复核val / train | 复核总数 | 复核页面“成员” |
| --- | --- | --- | --- | --- | --- | --- |
| A | 9 / 27 | 36 | B | 9 / 27 | 36 | B |
| B | 9 / 27 | 36 | A | 9 / 27 | 36 | A |
| C | 8 / 26 | 34 | D | 7 / 26 | 33 | D |
| D | 7 / 26 | 33 | C | 8 / 26 | 34 | C |

执行与交回顺序：

1. 完整解压操作包，打开 `index.html`，成员从自己的任务册看图，逐组完成原句、解析、每个保留变体、组件及依据检查。需要修改、排除或不能判断的项真实留痕，不能照抄助手结论。
2. 每人每轮交val、train各一份“导出全部JSONL”，再附 `forms/submission-template.md` 的实际填写版本。说明真实姓名/分工、阶段、实际日期、输入基线、已审核ID、未审ID、修正/排除/争议及输出文件名。仅截图、CSV、口头“已完成”不够；导出33/106行也不表示本人审完全部。
3. A汇齐四位初标文件，交agent合并为val/train各一份统一初标基线；保留来源与冲突，不静默覆盖。四位真人导入对应划分的完整统一基线后再交叉复核，不能自行叠加导入不同人的零散文件。
4. 复核仍按第2项交两份完整JSONL及说明。存在争议时由agent整理清单，第三位真人看图裁决；裁决人不能是该组初标/复核人。解析修改和排除分歧也需列出，不能只看页面争议数。
5. 实际负责人和成员填写 `forms/research-decisions.md`，确认原30组及补充组用途、最终有效规模、每种现象的有效源图覆盖、排除/未决项、是否补样及结论范围。助手建议的val25/train88不能代替真人决策。

每人每轮的文件名建议使用“阶段_分工_姓名_划分_实际日期_修订号.jsonl”，保留历史备份；agent可以合并/修正结构，不能代签真人记录。train此次按手册同样安排交叉复核。正式test不在这一轮。

## 3. B 必须发布的训练输入（B-V2-02 → C）

交付一个完整、不可变的新版本训练目录，默认约定 `data/semantic/train-verified-v1/`；已存在时升版，并明确通知C更新参数，不覆盖旧文件。

- **`samples.jsonl`**：独立train源图的核验后记录，含来源、原文锚定解析、保留变体、唯一正确关联的组件及实际审核证据；保留样本必须通过train校验。assistant/draft/uncertain/未决争议不能直接进入优化器；淘汰项在独立清单留痕。
- **审核和质量材料**：工作稿版本、真实人工初标/复核来源、修正/拒绝/排除/未审清单、各现象有效源图/编辑/组件/同义数、组件是否等于整句及适用性。N1/N2使用同一组最小错误编辑；组件必须属于同一次编辑。
- **来源和文件摘要**：原始采样与生成配置、权重/prompt/seed/代码版本、源manifest摘要、交付目录所有成员的相对路径/大小/SHA-256。`semantic_dataset_sha256`必须等于最终 `samples.jsonl` 的实际字节摘要。
- **若包含automatic核验**：另交真实人工抽检JSON，schema=`fgir-semantic-training-audit-v1`；必须有 `samples_sha256`、`reviewer_kind="human"`、真实 `reviewer`、实际抽检 `notes`、非空且确属该训练集的 `sampled_sample_ids`、`unresolved_disputes=[]`。C通过 `--training-audit` 传入该文件。全部为真实human核验时使用各记录证据，不需凭空补automatic抽检。

验收先执行：

```powershell
python -m fgir validate-semantics --bundle data/prepared/flickr8k-v1 --samples data/semantic/train-verified-v1/samples.jsonl --purpose train --data-root data
```

还须通过C的人工/automatic来源审核及token过滤，单独结构命令通过不代表可以训练。图片/原句/变体始终跟随源图划分，禁止从val复制到train。现有106组是待审输入，不承诺最终保留106组。

## 4. A 必须发布的验证输入（A-V2-04 → C、D）

在真人初标、另一位真人复核、必要裁决及覆盖决策完成后，冻结新版本目录，默认 `data/semantic/frozen-val-v1/`，提交完整绑定文件而非孤立protocol：

| 必须文件或材料 | 接收要求 |
| --- | --- |
| `protocol.json` | schema=`fgir-semantic-eval-v1`，real、split=val、independently_reviewed=true，真实样本数/图库数及摘要绑定，披露历史test接触 |
| `samples.jsonl` | 保留原句、解析、变体和组件的真实human初标、不同human复核、图片依据及历史来源；无未决争议，不适用项独立排除 |
| `candidates.json` | 固定完整val图库，必须与标准val检索协议的图库一致；本数据版本1000图 |
| `audit.json`、`source-context.json` | 协议实际绑定的结构/来源/划分/图片摘要记录，路径可解析且成员SHA-256匹配 |
| 人工/排除/覆盖与决策记录 | 已审核集合、原30组及3组补充用途、各现象有效源图数、淘汰理由、裁决、是否补样/缩小结论、实际负责人/日期 |
| 交付manifest | 所有另传成员相对路径、字节大小、SHA-256，冻结协议文件自身的SHA-256及对应源数据版本 |

冻结前执行 `validate-semantics --purpose eval --data-root data`，冻结后通过 `load_semantic_protocol` 的成员摘要与记录检查。只有旧标准检索 `data/prepared/flickr8k-v1/val/protocol.json` 无法提供语义选模金标；val工作稿也不能直接替代。

## 5. C 收齐之后执行什么，D 同时准备什么

C本机已有基础CLIP/tokenizer、源数据bundle、raw train缓存及raw val索引。D的 `fgir.text_export:validate_text_lora_checkpoint` 已实现；换机还须另传/重建实际资源。回调的可选路径配置为bundle下 `retrieval-v2.json`，参考 `configs/retrieval-v2.example.json`，相对索引路径按bundle目录解析。

先在隔离环境进行正式门槛预检：

```powershell
.\.venv-semantic\Scripts\python.exe -m fgir run-text-experiments --model-dir models/clip-vit-base-patch32 --image-features data/features/clip-vit-base-patch32/train.images.npz --samples data/semantic/train-verified-v1/samples.jsonl --val-protocol data/semantic/frozen-val-v1/protocol.json --output artifacts/c-v2/formal-paired-v1 --check-only
```

该命令已实测，当前返回BLOCKED，缺B训练文件与A冻结val。automatic核验数据必须额外加 `--training-audit` 指向实际抽检文件。READY以后复核token过滤后的训练覆盖、同一数据曝光及固定预算/选择规则；确认真实数据、配置和研究范围达到门槛后，用同一参数移除 `--check-only`，显式 `--seeds 42 43 44` 开始正式运行。输出必须为新目录，禁止覆盖历史目录。

C交付N0、N1、N2、N3、N2-no-relation各3种子，共15次真实实验，每次交best/last、配置、模型/数据/代码/环境摘要、训练/梯度/冻结/重载/丢样本审计、全部val候选/选择/失败记录和成本。所有方法共用数据、结构、种子、预算和文本曝光，仅规定目标项改变。若只完成部分，明确方法/种子与失败，不把任务标全完成。

当前待冻结配置见 `configs/training/n0.json` 至 `n3.json` 及 `n2-no-relation.json`：文本末两层q/v LoRA、rank4/alpha8/dropout0、10轮固定预算、微批8/累积4。累积不扩大对比负例池。N0按val R@1、MRR选择；其余按适用绑定/方向诊断宏平均，要求R@1比同种子N0下降不超过0.01，再按R@1/MRR/较早轮次破平；无合格候选报告失败。所有方法保持共同全句正例池，N1/N2共用编辑、N2/N3保持相同文本曝光。实际冻结后变更数据、配置或规则须重新版本化并重跑受影响比较，不根据test选模。

D在每个候选保存后即可调用固定val回调导出receipt/run-v2、完整原句检索及语义分数，返回实际checkpoint与协议摘要、标准val R@1/MRR、绑定/方向诊断供C选模，无需等待15次全部完成。D收到正式N2/N3后完成N2→E0→N3→N2恢复、真实Top-K及预热性能验收。开发权重与正式权重严格区分。

D逐方法/种子的正式交付应包含checkpoint/receipt、`run.json`、`semantic-scores.json`（`fgir-semantic-scores-v1`）、`retrieval-evaluation.json`、`semantic-evaluation/`及实际命令/环境/成员摘要。语义scores覆盖每个sample的original与全部variant ID；原句/同义句提供统一深度完整图库排名或明确空返回，错误编辑不当整库qrels。标准5000原句查询的R@1/MRR与语义子集诊断分别报告。解析指标如需报告，由B交真实解析预测，必须绑定冻结样本并完整覆盖；D/A不得复制金标冒充预测，未提供时按缺测/null披露。

正式结果交给A执行evaluate/compare、覆盖/缺测与配对统计、至少12例真实看图分析。正式test仍须在配置/选模规则冻结后另行建立独立人工金标并冻结，披露历史test接触；本轮不把val改名成test。

## 6. Git 与另传文件的边界

本次推送说明、任务状态、人工手册及小型核验证据JSON。Git不会自动下载下列忽略制品，发送方须另传完整原包或约定重建方式；不提交数据集图片、模型、特征和154MB排名文件。

| 谁负责另传 | 文件/目录 | 当前本机情况与摘要 |
| --- | --- | --- |
| A/操作包保管人 | `fgir-human-operation-20261030-v1.zip` | 18,999,859字节；SHA-256=`2d790753c84a81be5b40578c56505ebf110f8612fa4353114182d3233438f031`；包内index/成员册/图片/审核页须一起保留 |
| C | `artifacts/c-v2/c-v2-development-handoff-v1.zip` | 140,666字节；SHA-256=`5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`；原始开发包本机核对通过，可另传D；不当正式训练输入 |
| B | `data/semantic/train-verified-v1/` 或明确的新版本 | 待发布，不能编造包大小/摘要 |
| A | `data/semantic/frozen-val-v1/` 或明确的新版本 | 待冻结，须完整传协议绑定成员及审核/覆盖材料 |
| C（正式训练后） | 每个方法/种子的完整训练与验证产物包 | 尚未生成；另附实际相对路径/大小/摘要，逐项交给D/A |

收到另传文件后先核对摘要、版本、成员齐全，再跑加载与预检。相同文件名、截图或口头完成不替代交付。若操作包构建脚本/审核工具仅在保管人未提交的工作区，保管人还应记录源码版本或补交工具代码；目前原包已可离线使用，不凭空提供不存在的下载链接或构建命令。

可直接发群：

> C已完成新版训练代码、原始开发包核对和D接口真实导出，正式15次实验等待核验数据。请A先组织四位真人完成初标→agent统一基线→交叉复核→必要第三人裁决，并填写覆盖/研究决策；每人每轮交val/train两份完整JSONL及提交说明。B发布verified train及审核/排除/来源/质量/摘要（automatic还需真实人工抽检）；A发布完整frozen-val协议、绑定文件及审核/覆盖/排除/摘要。两项就绪后C开正式配对实验并逐项交D，D同步导出选模结果并验收正式N2/N3切换。操作包和模型/数据另传，Git中的说明不表示这些制品已收到。
