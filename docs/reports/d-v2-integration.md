# D research-v2：制品、验证回调与服务接入

2026-10-28，D / Codex，会话 01a15788-8365-77a1-813b-01177a384cd7。基于 C 分支提交 `c377cec` 实施，保留旧 E0/E1/E2 流程。

2026-10-29交接：代码与开发联调版本可接收，提供方、缺失路径、解除条件和资源包交付方式统一见[新版D交接清单](../handoff-d-v2.md)。正式D-V2-02/03仍按下述依赖保持BLOCKED。

## 当前交付与边界

D-V2-01/02/03 的可独立代码已经实现。后续已在本项目内准备基础 CLIP、真实 Flickr8k 图片、原始 val 索引，并使用 C 发布的训练接口与配置另存两步开发重建。重建路径为 `artifacts/c-v2/n0-smoke-seed42-d-rebuild-v1`，不冒充 C 原始开发包或其 SHA-256。真实联调验收记录见下文。

此前 tiny-CLIP 与 synthetic 样本验证了加载、编码、排名、评分、导出和服务切换接口；这些历史测试单列保留。正式实验仍依赖 B 的核验 train 样本、A 的独立 val 金标及 C 后续训练；两步开发重建也不能解除这些数据门槛。

## 实现与制品链

| 文件 | 交付 |
| --- | --- |
| `src/fgir/text_artifacts.py` | `fgir-retrieval-receipt-v2` 写入/加载、raw E0 索引核对、run-v2 校验、配对比较控制字段 |
| `src/fgir/text_export.py` | 严格加载 C 的模型、重新编码全部标准查询、完整排名、语义逐项分数、C 的验证回调 |
| `src/fgir/text_commands.py` | 注册并实测 `export-text-lora` 和 `validate-text-lora`；两条CLI及回调均通过本地 synthetic 联调 |
| `src/fgir/evaluation.py` / `runs.py` | v1/v2 分发，旧 E 系列不变；N 系列不得伪装为 run-v1；LoRA 文本缓存经 receipt 搜索 |
| `src/fgir/service.py` / `app.py` | raw_clip、feature_mlp、text_lora 注册表；独立 LoRA 实例、摘要缓存键、可用模型检查与缺失诊断 |
| `src/fgir/benchmark.py` | 注册模型 ID、实际设备、明确 warmup、各阶段 p50/p95、图库/K/样本数与 synthetic 标记 |
| `configs/demo-v2.json` | val 图库的 E0 与 N0-development 配置；不存在或加载失败的模型只显示诊断 |
| `configs/retrieval-v2.example.json` | C 的回调读取 raw val 索引的位置与编码设备/批量配置模板 |

receipt 顶层绑定：model_type、experiment_id、method、seed、mode、model_id、base_weights_sha256、tokenizer_sha256、image_encoder_sha256、projection_dimension、preprocessing、text_processing、structure、training_config、semantic_dataset_sha256、text_lora_sha256、image_index_sha256、text_features_sha256，以及五个带字节摘要的相对路径（checkpoint、图片索引及 sidecar、文本特征及 sidecar）。

导出目录内包含 `checkpoint.pt`、`index.npz`、`index.npz.meta.json`、`texts.npz`、`texts.npz.meta.json`、`receipt.json`、`run.json` 和 `export.json`。checkpoint 和索引是输入的逐字节快照，保证原子发布后与目录搬迁后仍可校验；不复制基础 CLIP，不向共享原始索引写 LoRA 字段。多个 N 的图片索引摘要应相同，但每份自足导出多占一份索引空间。

加载 receipt 时核对实际文件摘要、checkpoint 身份/结构/LoRA 参数、raw 索引基础模型/预处理/维度、文本缓存身份及有限非零向量。指定 model_dir 时额外执行 C 的完整重载（真实基础权重/tokenizer/准确目标层和 dtype）；离线 run 评分不重新加载基础 CLIP，但会校验制品字节链并用绑定特征重新排序，拒绝排名与特征不一致。摘要可追溯，不等于认证外部训练与人工身份。

`fgir-run-v2` 继续绑定 retrieval protocol-v1；记录完整 LoRA/基础模型/索引/receipt 身份。比较固定基础权重、tokenizer、图像塔、图片索引、维度、结构及训练样本摘要，允许不同种子的 LoRA 文件摘要不同。N2 去关系监督仍为 method=N2，使用独立 experiment_id=`N2-no-relation`；服务和标准多种子汇总按 experiment_id 区分。

v2 标准评分需要 NumPy/PyTorch 读取实际制品链（可安装 `.[semantic-training]`）；旧 v1 和标准库 CLI 不引入该依赖。A 的语义 `compare-semantics` 仍按 method/seed 分组，不能把普通 N2 与 N2-no-relation 当不同 method 塞入同一份语义比较；这项消融需单独报告或由 A 后续增加 experiment_id 分组，不改 checkpoint 的实际 method。

开发权重保留 mode=`development_smoke`、semantic_dataset_sha256=null，仅可输出非正式加载/检索制品。标准 N 评分、语义评分和 C 的选模回调均拒绝把它当正式结果。正式导出原句与所有变体的源图余弦分数；只有原句/同义句输出完整图库排名，错误编辑不作为整库 qrels。没有 B 的解析预测时完全省略 parsing，A 将其视为缺测。

## C 的验证回调

函数名与 C 默认配置一致：`fgir.text_export:validate_text_lora_checkpoint`。仅接收已保存 checkpoint，并独立加载模型，不接触训练 encoder。要求 semantic val 协议、标准 val 协议与源 manifest 一致；标准查询必须完整等于 `val/pairs.jsonl`，不能拿语义子集的 R@1 代替全量标准查询。

如有自定义索引路径，在交接包内新建 `data/prepared/flickr8k-v1/retrieval-v2.json`（可复制模板）。其中 image_index 相对交接包根目录解析；device 默认 cpu，batch_size 默认64。无此文件时兼容旧实验 raw E0 val 索引约定路径。缺索引将明确报错，调用方仍应在正式训练前核实这项输入。

回调严格返回四项 metrics：标准验证集 recall@1、完整排名 mrr，及语义 attribute_binding/relation_direction 的 source_macro_mean；没有适用诊断时为 null。另返回实际 checkpoint/protocol/base/tokenizer 摘要及 receipt/run/评分文件路径和摘要，保存 `validation.json`。C 的 `validation-history.json` 继续保留每轮结果。

## 运行方式

独立环境使用 C 已核验的 transformers 4.57.6，保留旧环境。实际本地资源均存放在项目内；新机器可按以下方式准备环境：

```powershell
python -m venv .venv-d-v2 --system-site-packages
.\.venv-d-v2\Scripts\python.exe -m pip install -e ".[demo,semantic-training]" "transformers==4.57.6"
```

开发联调只生成非正式制品，不传 `--semantic-protocol`：

```powershell
.\.venv-d-v2\Scripts\python.exe scripts/prepare-d-v2-local.py --rebuild-development --device auto
.\start-v2.cmd
```

收到正式候选与冻结 val 后，C 可直接调用回调，或用新增 CLI 独立调用：

```powershell
.\.venv-d-v2\Scripts\python.exe -m fgir validate-text-lora --checkpoint artifacts/c-v2/formal-paired-v1/N0-seed42/candidate-epoch000.pt --model-dir models/clip-vit-base-patch32 --bundle data/prepared/flickr8k-v1 --val-protocol data/semantic/frozen-val-v1/protocol.json --output artifacts/d-v2/n0-seed42-val-epoch0
```

准备脚本复用已有校验通过的缓存，调用 C 的两步训练后导出真实 val 查询，不下载数据或基础模型。运行前需按 `prepare-a-data` 与 `prepare-clip-model` 接口准备它们；不能覆盖冻结目录。正式 test 导出仍等 A 的独立协议和冻结选模门槛，本次只联调 val。

可用模型上实测性能：

```powershell
.\.venv-d-v2\Scripts\python.exe -m fgir.benchmark --config configs/demo-v2.json --device auto --models E0 N0-development --queries examples/demo-queries.json --repeats 5 --output artifacts/d-v2/performance-real-v1.json
```

报告不包含浏览器渲染/图片解码。initial_call 按记录顺序执行，raw/MLP 可能复用基础 encoder，不宣称每个模型都在独立进程完全冷启动。GPU 前后显式同步，文本结果回传 CPU 后排序；JSON 包含设备、图库、有效 K、采样数与预热数。不能把 synthetic 或两步开发权重的耗时称为正式 N2/N3 性能。

安装包已更新为0.2.0，demo依赖与C相同限制为transformers<5；打包器只收录当前版本wheel，避免旧wheel被安装脚本误选。

## 本机验证及剩余交接

隔离环境全套173项：172通过，1项Windows符号链接权限跳过；D新增9项专项通过。

专项测试命令：`python -m unittest discover -s tests -p test_text_export.py -v`。测试实际覆盖导出/搬迁后重载、完整排名及语义分数、四项回调、开发门槛、同索引不同种子比较、错配/缺失制品、N2→E0→N3→N2 恢复、注册表不可用项和 synthetic 性能标记。CI 已增加 D 专项。

发现并修复 live processor 的 tuple 配置与 JSON sidecar 的 list 不相等问题：`ClipEncoder._describe` 统一 JSON 表示，不放宽实际预处理字段检查。C 的旧预检测试原先断言 D 模块不存在，现更新为断言 B/A 输入仍缺、D 模块可被解析。

当前真实实验预检保存在 `artifacts/d-v2-preflight-updated.json`，Git 副本为 `d-v2/preflight.json`；目前仅缺 B 核验 train 与 A 独立 val 输入。摘要与环境/测试证据见本目录 `d-v2/`。以下输入缺失时相关任务保持 BLOCKED：

- C：开发 ZIP `c-v2-development-handoff-v1.zip`，报告 SHA-256=`5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`；best=`2d72425de399f210c7a1888a8da377c3c41db63a50800e3809c581d694bc3d7b`。收到后先核验包和 checkpoint 摘要。
- 本地基础 CLIP/tokenizer、数据交接包、raw val 图片索引与实际图片已补齐。D 开发重建另存版本，原包的逐字节核对仍等待 C；没有补写语义训练摘要。
- B：核验 train samples；A：独立复核后的 frozen val protocol。齐备后才能为正式候选生成选模分数和完整排名。
- C：后续实际 N2/N3 制品。齐备后再记录真实模型切换、离线查询和 p50/p95。当前 tiny-CLIP 的隔离验证不将 D-V2-03 标为 DONE。

H-V2-01 仍为真人待办，本次没有替成员签署 human 标注或复核。A 原语义评分器仍仅认证它自己的协议/分数结构；本次 D 的标准 run-v2 制品链校验不能替代独立图像判断。

## 项目内真实开发联调补验

本机使用 Python 3.12、torch 2.9.1+cu128、transformers 4.57.6、RTX 4060 Laptop。
基础模型位于 `models/clip-vit-base-patch32`，权重摘要与 C 一致为 `d5853196…`。
数据 bundle 的 clean manifest 摘要为 `a7f7d25d…`，图库1000张 val 图，未运行正式 test。
开发重建 best 摘要为 `de71a68db5c50d2a7452b2036c7c13ad3c3423f1035cd5445449eb7002e57246`，
raw val 索引为 `6065a067a0706c24f5919d102220fd6064ca41a60757267fa765c718d9dcb0b9`。
`artifacts/d-v2/local-assets.json` 保存完整来源与冻结审计。

两步各8张真实train图，16,384个LoRA参数；原参数摘要不变、图片输出差0。
独立CPU重载文本最大差8.94e-8，匹配分数最大差4.47e-8，低于原定1e-5。
完整日志与摘要见 `artifacts/c-v2/n0-smoke-seed42-d-rebuild-v1/`，Git精简副本为
`d-v2/local-rebuild-summary.json`。语义数据摘要保持null；没有语义人工审核或正式成绩。

首次GPU缓存实测与冻结图片塔的差为0.0012336，被原有门槛拒绝，训练未开始。
该缓存已移至 `artifacts/d-v2/rejected-cache-fast-v1/`；拒绝证据为 `d-v2/rejected-cache.json`。
关闭TF32并启用确定性算法后另存 `data/features/clip-vit-base-patch32/train.d-v2-deterministic-v1.images.npz`，
严格审计通过。没有覆盖原缓存或放宽容差。

真实开发导出位于 `artifacts/d-v2/n0-development/`：5000条标准val查询，各含1000张图的完整排名，
receipt/run-v2已绑定真实模型、索引与文本缓存。只作开发检索，不计算或报告正式R@1。
启动 `start-v2.cmd` 可在页面中对比E0与N0-development；`scripts/verify-d-v2-local.py`
执行离线环境查询、来回切换恢复、图片存在性、便携JSON和Streamlit实际按钮验收。
这些检查已通过，页面实际渲染10张结果图；证据为 `d-v2/real-development-verification.json`。
4条查询各重复5次，每个模型预热4次后采集20次：E0端到端p50/p95为17.01/22.61毫秒，
N0-development为38.60/46.28毫秒，均在cuda:0上测量，图库1000、Top-K=10。
这是本机开发查询延迟，不包含浏览器绘制与图片解码，也不是正式N系列性能结论。
真实N2→E0→N3→N2仍需C提供相应正式权重，D-V2-03不据此标DONE。

源码安装包为 `dist/fgir-d-demo-windows.zip`；资源齐备包通过
`python scripts/build-demo-package.py --include-v2-assets` 构建为 `dist/fgir-d-v2-ready-windows.zip`。
后者包含本地CLIP、val图片及完整开发导出；首次依赖安装需要联网，之后可离线查询。
大型文件留在本项目和资源ZIP内，不加入Git。C原ZIP的摘要核对仍待原文件，不能拿重建代替。
