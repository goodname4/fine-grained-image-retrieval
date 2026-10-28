# 从现有代码迁移到新路线：实施与验收清单

2026-10-28 D更新：P3制品/导出/验证回调与P5服务注册表代码已交付，tiny-CLIP合成验收通过。真实C权重、图片/索引与独立val未在D工作区到位，D任务仍BLOCKED；详见[D报告](reports/d-v2-integration.md)。下文“待开发”描述保留迁移设计背景，当前状态以任务板为准。

2026-10-28 C更新：P1可导LoRA/真实N0开发小batch已验收，P2损失/配对配置/编排代码及P3的C checkpoint接口已交付；正式N实验仍BLOCKED。实际命令、证据和缺口见[C报告](reports/c-v2-training.md)与[C/D契约](text-lora-contract.md)。以下计划条目保留设计背景，当前状态以任务板为准。

版本research-v2，2026-10-27。本文针对当前仓库实际接口。A已实现P0的数据/审核/冻结工具和P4语义诊断工具，详情见[新版A交接](handoff-a-v2.md)。P0的约30源图独立金标尚未验收；B生成、C训练、D的run-v2与服务仍为待开发设计。

## 1. 已确认的迁移阻点

| 当前代码 | 已有行为 | 对新路线的影响 |
| --- | --- | --- |
| encoding.py：ClipEncoder.encode_texts/_project | inference_mode、detach、NumPy输出 | 训练时直接复用会切断LoRA梯度 |
| training_data.py：CachedSplit | 只加载detached图文特征 | 无原文/token流，不能更新文本Transformer |
| training.py / adapters.py | E1/E2缓存训练、DualAdapter、fgir-adapter-v1 | 不能拿旧checkpoint加载器读LoRA |
| runs.py | METHODS限制E0/E1/E2；要求图文adapter摘要一致 | 不能直接导出共享E0图片索引的N文本特征 |
| evaluation.py | run-v1；仅对E0/E1/E2部分约束，不能认证N的LoRA | 需要新版本验证及比较约束，不能只改method |
| service.py | 固定模型名、基础目录共享encoder、旧adapter加载 | 注入LoRA可能污染E0或其他N模型 |
| contracts.py / annotations.py | 旧四类检索标注与qrels | 新语义任务应独立协议，避免破坏旧数据 |
| app.py / configs/demo.json | 固定E0/E1/E2多选与路径 | 需按注册表展示，仅列实际准备好的模型 |
| pyproject.toml | training依赖不包含transformers/PEFT | 新训练需独立可选依赖，不能假定旧环境够用 |

E0/E1/E2的脚本、配置、日志、权重与排名保留原含义。旧C训练8秒级耗时仅适用于缓存MLP，不适用于新路线。

## 2. 建议开发顺序与按文件修改

按角色开工时先读[接续任务板](research-v2-tasks.md)：本页定义技术实施范围，任务板记录实际状态、交付、人工依赖与解除条件，避免将下面的计划入口误当已实现命令。

### P0：语义数据与冻结（A/B）

实际进度：semantic_data.py、semantic_review.py/semantic-board.html及相关CLI已实现，真实val任务包30组已生成，8组仅有助手原句预审；未生成解析/变体或冒充独立人工金标。semantic_generation.py仍由B待实现。本节验收尚未通过。

新增semantic_data.py，负责读取/校验fgir-semantic-v1、源图划分、原文锚定、实体关系引用、编辑位置及审核状态。新增semantic_generation.py隔离本地生成模型，生成器输出候选，不自行认证“人工复核”。新增小型合成测试夹具，不把合成文本作为真实图片效果。

扩展现有标注页面或新增语义审核页面，保留原100条检索标注不覆盖。已有contracts.py只在必要处增加独立入口，不改变旧四类枚举。

验收：约30张val源图通过独立复核；同图变体不跨集合；非法span/引用、把未审当已审、无图片依据的负例被拒绝。训练候选从train另行生成，val试点不混入优化器。

### P1：文本LoRA与可导训练（C）

新增text_lora.py，注入文本attention的q_proj/v_proj，显式白名单与零初始化；新增text_training_data.py，按image_id联结原始图片向量与原文、组件及变体token。新增text_training.py负责训练，不把新功能塞入旧train_adapter。

encoding.py保留推理函数，新训练使用独立可导torch路径。若抽取共享tokenizer帮助函数，确保不会隐式包入inference_mode或转NumPy。图像塔、文本原权重和投影冻结；LoRA更新后原始图片向量逐项不变。

pyproject.toml计划新增semantic-training可选依赖，至少包含模型训练所需torch/numpy/transformers；若采用PEFT则显式加入并固定实测版本。可手写最小LoRA以减少依赖，但二者二选一，不能文档写PEFT而checkpoint实际为另一结构。

待实现CLI名称：train-text-lora；当前不能运行。先只实现N0和一个真实mini-batch，再接全部数据。

验收：LoRA参数进入优化器、有效梯度可达，至少两步后有更新；非LoRA参数和图像输出不变；运行日志记录文本截断/丢样本、参数量、峰值显存、NaN与无效批次；基础模型文件不被原地覆盖。

### P2：损失、对照与选择（C/A）

新增semantic_losses.py，实现完整句成对损失、组件成对损失及同义一致性；复用或适配现有多正例mask，保留同图多描述和已知假负样本规则。正确组件之间可能相同，不自动互作负例。

计划新增configs/training/n0.json至n3.json，初始超参写入[实验设计](experiments.md)，当前不创建可误执行配置。实验间的结构/数据摘要一致，N2/N3使用同样同义文本；损失按有效样本归一化。

新增run_semantic_experiments.py编排配对种子，验证不读test。Checkpoint选择及失败定义依实验文档冻结，不以test结果反向改规则。

验收：N0/N1/N2/N3仅有约定差异；批次缺少某类编辑能正确mask；全空监督显式报出；标量损失、梯度方向、多正例处理及同义塌缩风险有针对性验证；保留所有val记录，非只保存最优曲线。

### P3：制品与导出（C/D/A）

定义新的checkpoint schema fgir-text-lora-v1，保存基础权重及tokenizer摘要、冻结策略、target_modules、rank/alpha/dropout、LoRA state_dict、实验/种子、语义数据及训练配置摘要。保存best与last；加载使用安全的tensor/basic-type路径并检查有限数值，拒绝未知结构/缺失参数，不传给DualAdapter。

新增text_export.py及计划CLI export-text-lora，离线重新编码协议中全部查询，生成带text_lora_sha256的文本特征；原始图片索引可共享，receipt绑定索引与文本LoRA。

计划引入fgir-run-v2，继续引用固定retrieval protocol-v1的query/qrels/candidate摘要。新增字段至少包括model_type、base_weights_sha256、image_encoder_sha256、text_lora_sha256、image_index_sha256、retrieval_receipt_sha256、semantic_dataset_sha256；image预处理和tokenizer分别记录。新权重字段语义明确，不把text LoRA伪装成图像adapter摘要。v1仍由旧路径读写。

runs.py、evaluation.py、服务和导出一起增加版本分发。v2验证N0—N3属于text_lora类型且LoRA字段完整；E0不带任何适配，E1/E2保留旧约束。比较N方法时要求相同原始CLIP、图像索引、tokenizer、协议、维度与LoRA结构。按方法汇总多个种子时允许不同LoRA checkpoint，不能要求不同种子的文件摘要相同。

验收：保存重载分数在记录容差内一致；缺失/错误LoRA、基础模型或索引摘要被拒绝；换错N文本特征被拒绝；同图片索引下合法不同N模型可比较；旧v1回归仍通过。

### P4：NLP诊断（A/B）

实际进度：semantic_evaluation.py、evaluate-semantics与compare-semantics已实现，合成CLI及针对性测试通过。支持配对区分、解析F1、同义共同命中、追加长度对照、源图统计及配对种子比较。真实评分需先有独立金标和C/D分数；标准retrieval run-v2仍待C/D接入。

已新增semantic_evaluation.py，读取fgir-semantic-eval-v1及fgir-semantic-scores-v1逐项分数，输出绑定/关系/错误追加的准确率、解析F1、同义共同命中及逐源图分组结果。实际CLI为evaluate-semantics，字段见[交接](handoff-a-v2.md)。

诊断参考标签的实体ID只在组内有效；做F1时先用源图金标映射和规范化规则对齐，不把生成顺序造成的ID不同当关系错误。并列分数记0.5、缺失查询报错，源图bootstrap不把变体当独立图。

验收：正确/错误追加和长度对照均有输出；同义共同失败不会被计成成功；不能把错误编辑对源图不匹配误算成“查询无相关图片”；外部匹配任务与整库检索单列。

### P5：查询服务与界面（D）

service.py改为model_type分发（raw_clip、feature_mlp、text_lora），默认兼容旧configs/demo.json。新增统一注册表后，app.py从已配置且制品可校验的模型读取选项，不固定硬编码N名称就宣布可运行。

缓存键至少绑定基础模型/LoRA摘要；不得向E0共享实例反复注入或merge不同LoRA。首版可独立加载文本实例、顺序比较并按需释放，记录实际显存/耗时。

界面显示原句与预先审核的同义/最小编辑查询对应的真实图片结果；解析仅能标成训练/诊断辅助，不能把离线解析冒充在线推理过程。JSON导出包含模型与receipt身份，继续去掉个人图片绝对路径。

验收：N2→E0→N3→N2切换输出恢复；新旧方法都可返回真实图；未准备的N模型不提供可点击成功入口；缺失制品有定位信息；权重准备后断网查询；在线耗时不混入离线语言生成。

## 3. 必须覆盖的正确性检查

语义数据与指标部分已有针对性测试；训练与制品部分仍待各模块实现后验证：

- 语义数据：源图泄漏、span偏移、实体引用、假负例/未审过滤、同义信息遗漏。
- 训练：LoRA零初始化、梯度两步可达、冻结参数不变、组件mask、多正例、截断导致关键编辑丢失。
- 制品：重载一致、结构/基础权重/索引/LoRA错配拒绝、旧v1可读。
- 服务：模型切换隔离、缓存键、真实图库ID与协议一致。
- 指标：并列、缺失组、共同失败、源图分组、长度对照及多种子汇总。

micro-batch梯度累积只累计梯度，不自动扩大对比负样本池；报告中显式记录实际对比分母大小。CLIP分词截断若删除了关键编辑，应在编码前过滤或缩写并重审，不让等价token串冒充语义负例。

## 4. 当前可运行入口与未来入口

当前仍可使用README、training.md和demo.md中的E0/E1/E2命令。七个新版A命令已注册并实测，见[交接](handoff-a-v2.md)。train-text-lora和export-text-lora尚未注册。

迁移后再提供独立的新训练环境、版本锁、配置样例和经过实际执行的完整命令；不直接升级现有已验证E1/E2环境。现有功能不能因N代码导入而额外要求未安装的大模型依赖，CLI新增训练依赖采用延迟导入。

## 5. 初始计划文档变更范围（历史记录）

- 重写README、项目计划、开题正文/提纲、架构、实验和分工。
- 新增语义数据协议及本实施清单。
- 旧训练、数据、排名、演示和交接文档增加版本边界提示，保留实际操作与历史结果。
- 研究依据文档改为已采纳的设计依据，并链接正式计划。
- 本次不修改Python算法、模型权重、依赖或真实数据，不创建虚构N结果；实施进度以后由真实代码/日志更新。
