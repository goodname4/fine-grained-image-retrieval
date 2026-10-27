# 面向组合语义理解的自然语言图像检索

**实体关系监督与语义一致性轻量训练 · Natural Language Processing Coursework**

## 当前路线与实现状态

2026-10-27起采用research-v2：参考2026年CS-CLIP组件监督研究，通过实体/属性绑定/有向关系解析、最小语义变体与同义改写，在CLIP文本编码器内部进行LoRA训练，研究关键语义敏感性与同义表达稳定性。

**A的语义协议、审核页、冻结和诊断评分工具已实现；30张真实val源图已分配，8组完成原句助手预审，尚无独立人工金标。N0—N3尚未实现或训练。** 历史E0/E1/E2代码与实验记录保留。实际状态和运行步骤见[新版A交接](docs/handoff-a-v2.md)及[实施记录](docs/reports/a-semantic-evaluation.md)。

从这里开始：

- [Agent接手入口](AGENTS.md)与[接续任务板](docs/research-v2-tasks.md)：按A/B/C/D直接开工，明确依赖、真人待办及验收标准；[最新复验](docs/reports/a-v2-reacceptance.md)。
- [新版项目计划](docs/project-plan.md)：研究问题、范围、阶段及验收。
- [代码迁移与修改清单](docs/implementation-plan.md)：当前阻点、按文件修改、依赖和正确性检查。
- [新版开题报告](docs/proposal-report.md)与[汇报提纲](docs/proposal.md)。
- [语义数据协议](docs/semantic-data.md)、[方法架构](docs/architecture.md)、[实验设计](docs/experiments.md)。
- [四人分工](docs/roadmap.md)、[近期研究依据与选择](docs/research-redesign.md)。

## 1. 研究什么

让用户用英文描述搜索本地图片，检验模型能否：

1. 区分“红衣男子牵黑狗”与属性归属错误的描述。
2. 区分施事/受事和非对称关系方向。
3. 对保持原意的措辞/句式变化保持检索稳定。
4. 惩罚追加的错误细节，同时避免只学会偏好短句。

语言工作贯穿语义解析、数据构建、文本Transformer训练与NLP诊断。首版重点为属性绑定和关系方向；数量、否定、中文、主动补查及大模型重排序为可选扩展。

核心问题为：同一文本训练结构下，组件监督是否优于完整句负例监督；显式同义一致性约束能否兼顾稳定性、语义区分和总体检索。方法增量与效果待实验验证，LoRA、文本侧训练及语义解析本身不作为原创声明。

## 2. 技术路线与实验

训练：固定图片划分 → 本地离线语义解析/变体生成 → 图片依据核验 → 冻结图像分支、文本LoRA可导训练 → val选择 → 固定test。

查询：英文文本 → 选定文本LoRA编码 → 归一化 → 与原始图片向量精确余弦检索 → 真实图片Top-K。

图片塔/投影冻结时可共享原始图像索引，文本LoRA与索引配对由新版receipt校验；旧整句缓存不能更新文本编码器。新增制品与服务接口尚待实现。

| 方法 | 状态 | 内容 |
| --- | --- | --- |
| E0 | 已有 | 原始CLIP |
| E1/E2 | 已有训练记录 | 图文残差MLP；E2加入困难图片负样本 |
| N0 | 待实现 | 文本LoRA＋完整句对齐 |
| N1 | 待实现 | N0＋完整句最小语义负例 |
| N2 | 待实现 | 同一批编辑的组件监督 |
| N3 | 待实现 | N2＋显式同义一致性 |

所有N组共用全句正例池；N2/N3的数据曝光相同，只改变一致性项，完整控制条件见[实验设计](docs/experiments.md)。

## 3. 数据、资源与边界

Flickr8k清洗版train5998图/29990描述，val/test各1000图/5000描述。所有组件及变体跟随源图划分，相关性不完整与近重复筛查限制继续披露。旧100条细粒度草稿不等同于新语义诊断，独立复核状态见实际记录。

先用约30张val源图做语义试点，train训练数据独立生成；正式诊断目标100—200张test源图，方法与协议先冻结。使用本地开源预训练权重，不从头训练大模型，不调用外部AI推理/翻译/嵌入/重排序接口；依赖、数据与权重准备可以联网。语言生成工具离线使用，在线检索不依赖它。

已有记录使用RTX4060 Laptop 8GB；新LoRA训练的batch、显存和耗时先小规模实测，不能沿用旧缓存MLP训练成本。CS-CLIP原论文设置不同，本项目是受限资源的方法研究，不冒称完整复现。

## 4. 当前可以运行的旧流程

以下标准库示例仅验收向量检索和指标，示例ID不具有真实图片语义：

```powershell
python -m pip install -e .
python -m fgir search --index examples/index.json --query-vector examples/query_vector.json --top-k 3
python -m fgir evaluate --rankings examples/rankings.json --qrels examples/qrels.json
python -m unittest discover -s tests -v
```

真实旧流程：

- [A数据交接](docs/handoff-a.md)：prepare-a-data、固定划分与摘要校验。
- [B编码检索](docs/encoding-retrieval.md)：本地CLIP、图片/文本缓存、E0。
- [C旧训练](docs/training.md)：E1/E2缓存MLP训练、导出及重载。
- [D演示](docs/demo.md)：install.cmd、start-demo.cmd和Streamlit接入。
- [C真实记录](docs/reports/c-training-and-ablation.md)：单种子val结果及局限。
- [旧排名契约](docs/ranking-contract.md)：当前run-v1与统一计分。

双击preview.cmd可查看静态离线交互预览，包含示意图片与固定排名，**不是模型效果**。真实演示需另行准备本地图片、权重、索引及配置；源码安装ZIP/wheel可由python scripts/build-demo-package.py构建。

新版A已可执行prepare-semantic-pilot、apply-semantic-source-checks、validate-semantics、render-semantics、freeze-semantics、evaluate-semantics、compare-semantics，详见[交接命令](docs/handoff-a-v2.md)。当前草稿会被正式冻结门槛拒绝；[合成示例](examples/semantic-evaluation/README.md)可以先验收接口。

train-text-lora、export-text-lora及N系列标准run-v2仍是[实施提案](docs/implementation-plan.md)，尚不可执行。

## 5. 评价与交付

同时报告标准检索R@1/5/10与完整MRR、属性绑定/关系区分、错误追加及长度对照、同义共同命中与排名差、解析质量、参数/显存/耗时。语义匹配准确率与整库检索分开，不以示例或预期提升充当结果。

A负责协议与独立评测，B负责语言解析/变体，C负责文本LoRA/消融，D负责新模型服务/界面；各25点，见[分工](docs/roadmap.md)。先完成语义试点和可导小batch，再做N0—N3受控实验、固定测试和软件集成。

## 6. 仓库结构与历史文档

```text
src/fgir/    当前数据、编码、E1/E2训练、检索和评测代码
docs/        新路线计划、语义协议、迁移清单及历史操作记录
configs/     当前数据/标注/E1/E2/演示配置；N配置待实现
tests/       当前功能正确性检查；新功能测试随实现补充
examples/    人工构造示例，不能作为真实模型结论
data/        本地数据和特征，不随Git分发
models/      本地预训练权重说明
artifacts/   真实运行产物及摘要，按忽略规则保存
```

[旧数据规范](docs/data.md)、[旧标注操作](docs/annotation-guide.md)、[开发说明](CONTRIBUTING.md)继续供现有功能使用；新语义数据以semantic-v1为准。原有报告及历史数字不追溯改写。第三方模型/数据使用条款分别适用，仓库不再分发图片和完整权重；代码暂不另加开源许可证。
