# research-v2 接续任务板

更新：2026-10-27。供成员和接手 agent 直接执行。根目录 [AGENTS.md](../AGENTS.md)负责引导；本页是剩余任务状态的统一入口。详细设计见[实施清单](implementation-plan.md)、[实验设计](experiments.md)、[架构](architecture.md)和[语义协议](semantic-data.md)。后续完成者须更新本页，不能只在聊天中说完成。

## 验收基线与任务状态

[本轮复验](reports/a-v2-reacceptance.md)：A-V2-01/02/03均通过，36项语义测试通过；全套133项中132通过、1项环境跳过。A工具可接手使用，P0数据阶段及N实验未完成。

真实输入是 `data/semantic/pilot-val-assisted-v1/samples.work.jsonl`：30张val图，A/B/C/D分配8/8/7/7，8组仅助手原句预审，全部缺少实体/变体/组件，0组独立人工金标。审核页在 `artifacts/semantic-pilot-val-assisted-v1/board.html`。这些是本机忽略文件，其他机器按[新版A交接](handoff-a-v2.md)及[源数据交接](handoff-a.md)准备；不要重新下载或覆盖本机已有数据。

状态：READY＝可立即开展；IN_PROGRESS＝已开始且尚未验收；BLOCKED＝依赖产物缺失；WAITING_HUMAN＝需真人执行；DONE＝有验收证据。BLOCKED任务中的独立准备工作仍可先做，依赖解除后更新状态。

| ID | 提供方 / 执行方式 | 当前状态 | 后继接收方 | 依赖与当前缺口 |
| --- | --- | --- | --- | --- |
| A-V2-01/02/03 | A / agent | DONE | 全组 | [三项修复](reports/a-v2-fixes.md)及本轮复验 |
| B-V2-01 | B / agent | DONE | A、全体标注人 | 已交付：`data/semantic/pilot-val-candidates-v1/`（30组完整候选，samples_sha256=5058e091…，draft校验+图片字节通过）、`artifacts/semantic-pilot-val-candidates-v1/board.html`、`src/fgir/semantic_generation.py`+CLI `generate-semantics`、`tests/test_generation.py`（10项）、[报告](reports/b-v2-generation.md)。生成器=本地 Qwen2.5-0.5B-Instruct（weights fdf756fa…，prompts b28c65ee…，seed 42，code 22efaf5-dirty）；统计：实体100/关系19/变体99（绑定6+方向4+改写29+追加60）/units 10/拒绝78项带原因。全套143测试通过（含修复本机 runs.py GBK 解码既有缺陷）。真人复核仍待 H-V2-01 |
| H-V2-01 | A组织，A/B/C/D真人 | WAITING_HUMAN | A | 全体真实初标与交叉复核缺失；候选已由 B-V2-01 提供（`data/semantic/pilot-val-candidates-v1/samples.work.jsonl` + 审核页），可先检查原句图片 |
| B-V2-02 | B / agent＋真实核验 | READY | C、A | 依赖B-V2-01已解除；独立train候选及经核验版本尚无。生成器对任意源绑定记录通用，但 train 工作包需新增独立抽样/导出接口（prepare-semantic-pilot 只抽 val），见 b-v2-generation.md 第7节 |
| C-V2-01 | C / agent | READY | D、C后续任务 | 可导LoRA、N0小批次、checkpoint未实现；本地模型/环境先实查 |
| C-V2-02 | C / agent | BLOCKED | A、D | 正式实验依赖C-V2-01、B-V2-02、A-V2-04及D-V2-01/02；损失和配置代码可先做 |
| D-V2-01 | D / agent，C提供checkpoint接口 | READY | C、A | receipt/run-v2与严格加载未实现；可先契约和夹具测试，真实重载等C-V2-01 |
| D-V2-02 | D / agent，C提供真实权重 | BLOCKED | A | 依赖D-V2-01、C-V2-01、A-V2-04；真实分数/排名导出缺失，导出器代码可先做 |
| D-V2-03 | D / agent | READY | 全组 | 注册表/缓存隔离可先做；真实N切换和性能验收依赖D-V2-01/02及C权重 |
| A-V2-04 | A / agent＋人工数据门槛 | BLOCKED | C、D | 依赖B-V2-01、H-V2-01；val独立金标及覆盖审计缺失 |
| A-V2-05 | A / agent＋真实看图归因 | BLOCKED | 全组 | 依赖A-V2-04、C-V2-02、D-V2-02；正式test另须冻结配置和真实test人工复核 |

READY只表示能开始，不表示可跳过后续真人或真实模型验收。C-V2-02先由D导出N0/试点checkpoint的val结果用于选择，再导出全部正式候选的结果；不要求先完成所有N实验才能做D-V2-02，避免循环等待。

## B-V2-01：语言解析与候选生成

- 先读：语义协议、实施清单P0、新版A交接第3节、`src/fgir/semantic_data.py`及`examples/semantic-evaluation/gold.jsonl`（仅结构示例）。
- 立即做：实现计划中的 `src/fgir/semantic_generation.py`及独立CLI，原文锚定实体/属性归属/有向关系；生成同义、单条件错误和真假追加候选，构建与同一最小编辑相连的units。优先先打通少量val记录，再覆盖现有30组；保留source/sample ID和分配字段。
- 约束：检查本地生成权重可用性，记录实际权重、prompt、seed和源码摘要。缺权重时先完成适配接口、结构测试和审核导出，明确缺失模型及下一步，不能伪造摘要或用外部API代跑。agent生成/辅助判断保留真实来源，不认证human gold。
- 交付：新版本 `data/semantic/pilot-val-candidates-v1/`（若已存在则升版），包含完整 `samples.work.jsonl`、生成配置/来源记录、逐项拒绝与适用性说明；新离线审核页；版本化代码、测试及 `docs/reports/b-v2-generation.md`。
- 验收：草稿结构与图片字节校验通过；原文span可恢复、每个foil仅对应关联的同一次编辑；同义信息保持、对象绑定与真假仍交给人审；不能因为原caption没提到就判负。不存在的编辑保留排除原因，不强造变体。至少测试非法span、错引用、来源/划分错配和来源摘要缺失。
- 解锁：H-V2-01完整语义复核、B-V2-02训练候选。

## H-V2-01：全体真人初标与独立复核

此项不能由agent声明替人完成。A组织进度；B提供语言候选；各成员按记录字段筛选，既有分工不重新随机分配。

| 初标人 | 初标组数 | 独立复核人 |
| --- | --- | --- |
| A | 8 | B |
| B | 8 | A |
| C | 7 | D |
| D | 7 | C |

1. 按 `assigned_to`筛初标任务，按 `independent_reviewer`筛复核任务。实际审核应逐图核对原句、实体、属性归属、关系、同义信息、错误条件及真假追加，不能只批量改status。
2. 保存真实初标/复核身份、图像依据、图片摘要、复核意见与争议；争议由第三人裁决并留痕。助手预审来源保留，不能把assistant字段改成human声称做过复核。
3. 导出完整JSONL到新版本目录，记录输入摘要、修改的sample ID及剩余未审项；页面保存只在内存，必须导出。agent可合并与结构校验，有冲突时保留双方内容和冲突清单，不能静默覆盖。
4. 无法支持的现象不强造金标：草稿可保留uncertain/not_applicable，冻结输入须剔除不适用变体并在独立排除清单中保留原因。不能把not_applicable直接塞入正式eval输入绕过当前校验器。

完成证据：约30个源图组的真实初标及独立复核、争议处理、各现象覆盖/排除数量。若有组无法成为有效正例，由A记录原因并按公开规则补充val组、重新分配复核，不能挑易例凑数。交给A-V2-04。

## B-V2-02：独立train语义数据

- 立即可准备：确定train采样/导出接口及来源记录；复用B-V2-01生成器，从 `data/prepared/flickr8k-v1/train/pairs.jsonl`独立构建，绝不复制val任务进入train。
- 交付：版本化 `data/semantic/train-verified-v1/`，包括工作候选、核验后samples、拒绝/未审清单、各现象质量统计与独立人工抽检、实际生成配置和摘要。与C/D约定 `semantic_dataset_sha256`指向哪个不可变文件及其绑定成员，写进报告；不凭空定一个摘要。
- 验收：对正式训练samples运行 `validate-semantics --purpose train --data-root data`；人工或有真实图像依据的自动核验，自动核验有真实人工抽检。draft/uncertain/争议项排除；N1/N2编辑一致，所有N全句正例池一致，N2/N3同义曝光一致。
- 注意：当前 `prepare-semantic-pilot`只抽val，`freeze-semantics`只冻结eval协议。B须实现/记录train版本导出，不能拿这两个入口伪造train冻结。候选未核验时保留BLOCKED并交代真人需要做的抽检。

## C-V2-01：可导LoRA与N0最小闭环

- 先读：架构第3/4节、实施清单P1/P3；检查当前本地基础CLIP、tokenizer、设备及环境。现有 `encoding.py`推理路径会detach，不能直接用作训练前向。
- 立即做：新增计划中的 `text_lora.py`、`text_training_data.py`、`text_training.py`；白名单注入文本q_proj/v_proj，独立可导Tensor路径，保留旧MLP。建立隔离的可选依赖和实测版本记录。
- 交付：实际注册并试跑的 `train-text-lora`入口、N0配置、真实train小batch至少两步的日志、参数/梯度/冻结审计、best/last checkpoint与安全重载测试；`docs/reports/c-v2-training.md`记录环境、真实命令和输出摘要。模型协议与D-V2-01对接。
- 验收：LoRA进入优化器且至少两步后更新；图像塔、原文本权重、投影不变，图片向量不变；重载前后分数在记录容差内一致；有限数值/未知结构/缺参数校验；原始模型文件不覆盖。
- 独立边界：可先用真实train原句验证梯度和N0小batch；缺少共同语义正例池时仅称开发试跑，不当作正式可比较N0成绩。记录截断、丢样本、显存与NaN；不能拿旧MLP耗时充作LoRA耗时。

## C-V2-02：受控损失、N0—N3及消融

- 可先做：`semantic_losses.py`、多正例mask、空监督处理、配置模板及编排器；检查梯度方向、掩码和同义塌缩风险。计划中的 `run_semantic_experiments.py`与 `configs/training/n0.json`至`n3.json`仅在可运行后提供实测命令。
- 开跑门槛：C-V2-01通过，B-V2-02交付核验train，A-V2-04提供val金标，D-V2-01/02能验证/导出真实checkpoint和val分数。模型、tokenizer、LoRA结构、训练预算和val选择规则先冻结。
- 交付：配对种子目标42/43/44的N0—N3、N2去掉关系监督的消融；所有val尝试与选择记录、失败、best/last、训练数据/配置/源码摘要及环境成本。若实际只跑部分，明确缺哪些方法/种子，任务保持未完成。
- 验收：N1/N2共用编辑，仅监督粒度不同；N2/N3文本曝光相同，仅一致性项不同；微批次梯度累积不谎称扩大负例池；无test参与选择。checkpoint及逐项分数交给D/A。

## D-V2-01：模型制品、receipt和run-v2

- 先读：实施清单P3、架构第5/6节、旧 `runs.py`/`evaluation.py`/`adapters.py`和[排名契约](ranking-contract.md)。
- 立即做：与C的 `fgir-text-lora-v1`接口对齐；实现receipt、文本特征与图像索引绑定、`fgir-run-v2`版本分发和严格加载；先用明确synthetic的夹具测试，不依赖全部N训练完成。
- 交付：契约/导出加载代码、正反例夹具与测试，`docs/reports/d-v2-integration.md`写出字段、实际命令、路径、摘要及待C提供的文件；新协议文档与旧v1兼容说明。
- 验收：缺失/错配基础模型、tokenizer、LoRA、结构、索引、receipt被拒绝；合法不同N checkpoint共享E0原始图像索引；不同种子允许不同LoRA摘要；旧E0/E1/E2回归通过。最后用C-V2-01真实checkpoint完成重载和导出闭环才标DONE。

## D-V2-02：给A的真实分数与完整排名

- 可先做：新增计划中的 `text_export.py`和导出入口；真实模型计算原句/变体匹配分数；原句和同义句检索，错误编辑不当整库qrels。
- 交付：每个实际method/seed一个 `fgir-semantic-scores-v1`文件（字段见[新版A交接第5节](handoff-a-v2.md)）；对应标准检索run-v2、receipt、模型/协议摘要、实际命令、环境与日志。逐样本scores覆盖original及全部variant ID；完整图库排名或显式空返回，统一深度；解析预测由B提供，不用金标拷贝冒充预测。
- 验收：用实际冻结val协议运行A的evaluate/compare；排名、模型与receipt身份能追到真实制品；缺失解析/排名明确为缺测，不能填0。多方法比较保持相同协议、基础权重、tokenizer、配对种子和能力覆盖。
- 阶段顺序：先交N0/试点val结果给C选择，随后按C-V2-02的实际训练逐个补齐N1/N2/N3。正式test导出等A-V2-05的test门槛解除。手工synthetic接口测试不能标成真实交付。

## D-V2-03：服务、界面与性能

- 立即做：`service.py`按raw_clip/feature_mlp/text_lora分发；注册表驱动 `app.py`显示实际可用模型；缓存键绑定base/LoRA摘要，独立文本模型实例，不能向E0共享encoder注入LoRA。
- 交付：真实可用配置、缺制品诊断、身份完整且不泄漏个人绝对图片路径的JSON导出，更新演示文档和D报告。
- 验收：C/D制品齐备后实测N2→E0→N3→N2输出恢复、真实Top-K图、准备权重后断网查询；预热后p50/p95，冷启动/编码/排序/端到端与GPU同步、设备/K/图库/样本数齐全。不存在的N模型不展示为可成功入口；未完成真实切换验收前不能标DONE。

## A-V2-04：冻结真实val金标

- 输入：B-V2-01候选与H-V2-01真实复核结果。核查完整30组或有据补充的源图覆盖、各现象有效数、排除原因、争议及人工身份；结构校验不会替你判断标注是否充分。
- 执行：将已审完整集合保存为新版本，运行 `validate-semantics --purpose eval --data-root data`；按新版A交接第4节冻结到全新的val目录，记录协议及各成员SHA-256、test接触说明、覆盖审计。
- 交付：真实val协议、samples、审核/排除报告及摘要给C/D；train与val/test各自管理。P0只有数据及人工门槛一并达到才标通过。

## A-V2-05：真实评测、正式test与至少12例分析

- 先做：收到D-V2-02分数后运行val evaluate/compare，核查null及覆盖、排名差共同有效项和源图宏平均，给C保留可追溯反馈。
- 正式test门槛：C的配置、checkpoint选择规则冻结；A/B按既定规则构建独立test语义组，A组织真人初标/不同人复核并另行冻结；披露既有test接触。30组val不得换标签冒充test。正式100—200源图规模若缩减，明确实际范围和限制。
- 交付：固定标准检索与语义诊断结果、逐样本/源图指标、配对源图bootstrap与多种子统计；至少12个真实图像案例，覆盖绑定/方向/同义/错误追加和改善/失败/退化。无某种结果时如实报告缺口，不杜撰案例。
- 验收：原因字段来自真实看图归因，保留审核来源；外部任务与整库R@K单列，synthetic不进入真实效果报告。更新A进度、结果章节及任务板。

## 可以直接给接手agent的指令

- B：`执行新版B：先读AGENTS.md和docs/research-v2-tasks.md，落实B-V2-01并准备B-V2-02；完成代码、可运行候选导出、测试和交接，真实人工复核保持待办。`
- C：`执行新版C：按任务板从C-V2-01实现可导LoRA和真实N0小batch，接着完成C-V2-02可独立部分；依赖缺失要列出具体文件并继续可做工作。`
- D：`执行新版D：按任务板开展D-V2-01和D-V2-03，准备D-V2-02导出；保留旧v1/E流程，真实权重到位后完成制品、排名与切换验收。`
- A：`继续新版A：核实H-V2-01和B/C/D交付，依次执行A-V2-04、A-V2-05；不得把未审草稿或合成结果当真实金标和成绩。`

本轮仅建立交接入口与任务卡，没有向其他会话发送消息，也没有代替B/C/D完成以上实现。之后每项更新请记录：状态、执行者、实际交付路径/摘要、测试结果、仍缺输入及下游解除条件。
