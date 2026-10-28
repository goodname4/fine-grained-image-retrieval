# D research-v2：制品、验证回调与服务接入

2026-10-28，D / Codex，会话 01a15788-8365-77a1-813b-01177a384cd7。基于 C 分支提交 `c377cec` 实施，保留旧 E0/E1/E2 流程。

## 当前交付与边界

D-V2-01/02/03 的可独立代码已经实现。**真实制品验收仍 BLOCKED**：本工作区没有 C 的开发权重、基础 CLIP、真实图片/缓存/索引，也没有独立复核的 val 协议。没有运行真实 N0—N3，未记录或虚构真实检索效果、GPU 延迟和人工复核。

本次联调使用本地随机 tiny-CLIP 和明确 synthetic 的样本，实际执行加载、编码、排名、评分、导出和服务切换。它们验证接口，不构成 C 真实开发权重的重载验收。正式实验还依赖 B 的核验 train 样本、A 的独立 val 金标及 C 后续训练；不能因为 callback 可以 import 就宣布这些数据门槛通过。

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

以下安装和真实数据命令供收到交接包后使用；本机尚不能执行真实权重导出。独立环境可使用 C 已核验的 transformers 4.57.6，避免改旧环境：

```powershell
python -m venv .venv-d-v2 --system-site-packages
.\.venv-d-v2\Scripts\python.exe -m pip install -e ".[demo,semantic-training]" "transformers==4.57.6"
```

开发联调只生成非正式制品，不传 `--semantic-protocol`：

```powershell
.\.venv-d-v2\Scripts\python.exe -m fgir export-text-lora --checkpoint artifacts/c-v2/n0-smoke-seed42-v2/best.pt --model-dir models/clip-vit-base-patch32 --index artifacts/c-experiments/seed42/retrieval/E0/val/index.npz --protocol data/prepared/flickr8k-v1/val/protocol.json --experiment-id N0-development --output artifacts/d-v2/n0-development
.\.venv-d-v2\Scripts\python.exe -m streamlit run app.py -- --config configs/demo-v2.json
```

收到正式候选与冻结 val 后，C 可直接调用回调，或用新增 CLI 独立调用：

```powershell
.\.venv-d-v2\Scripts\python.exe -m fgir validate-text-lora --checkpoint artifacts/c-v2/formal-paired-v1/N0-seed42/candidate-epoch000.pt --model-dir models/clip-vit-base-patch32 --bundle data/prepared/flickr8k-v1 --val-protocol data/semantic/frozen-val-v1/protocol.json --output artifacts/d-v2/n0-seed42-val-epoch0
```

checkpoint 参数应换为实际存在的候选路径；不覆盖已有导出目录。正式 test 导出仍等 A 的独立协议和冻结选模门槛，本次只联调 val。

可用模型上实测性能：

```powershell
.\.venv-d-v2\Scripts\python.exe -m fgir.benchmark --config configs/demo-v2.json --models E0 N0-development --queries examples/demo-queries.json --repeats 5 --output artifacts/d-v2/performance-real-v1.json
```

报告不包含浏览器渲染/图片解码。initial_call 按记录顺序执行，raw/MLP 可能复用基础 encoder，不宣称每个模型都在独立进程完全冷启动。GPU 前后显式同步，文本结果回传 CPU 后排序；JSON 包含设备、图库、有效 K、采样数与预热数。不能把 synthetic 或两步开发权重的耗时称为正式 N2/N3 性能。

安装包已更新为0.2.0，demo依赖与C相同限制为transformers<5；打包器只收录当前版本wheel，避免旧wheel被安装脚本误选。

## 本机验证及剩余交接

隔离环境全套173项：172通过，1项Windows符号链接权限跳过；D新增9项专项通过。

专项测试命令：`python -m unittest discover -s tests -p test_text_export.py -v`。测试实际覆盖导出/搬迁后重载、完整排名及语义分数、四项回调、开发门槛、同索引不同种子比较、错配/缺失制品、N2→E0→N3→N2 恢复、注册表不可用项和 synthetic 性能标记。CI 已增加 D 专项。

发现并修复 live processor 的 tuple 配置与 JSON sidecar 的 list 不相等问题：`ClipEncoder._describe` 统一 JSON 表示，不放宽实际预处理字段检查。C 的旧预检测试原先断言 D 模块不存在，现更新为断言 B/A 输入仍缺、D 模块可被解析。

当前真实实验预检保存在 `artifacts/d-v2-preflight.json`；摘要与环境/测试证据见本目录 `d-v2/`。以下输入缺失时任务保持 BLOCKED：

- C：开发 ZIP `c-v2-development-handoff-v1.zip`，报告 SHA-256=`5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`；best=`2d72425de399f210c7a1888a8da377c3c41db63a50800e3809c581d694bc3d7b`。收到后先核验包和 checkpoint 摘要。
- B/C：本地基础 CLIP/tokenizer、数据交接包、raw val 图片索引与实际图片。先以 C 的开发包完成真实加载、导出及开发检索，不能补假训练数据摘要。
- B：核验 train samples；A：独立复核后的 frozen val protocol。齐备后才能为正式候选生成选模分数和完整排名。
- C：后续实际 N2/N3 制品。齐备后再记录真实模型切换、离线查询和 p50/p95。当前 tiny-CLIP 的隔离验证不将 D-V2-03 标为 DONE。

H-V2-01 仍为真人待办，本次没有替成员签署 human 标注或复核。A 原语义评分器仍仅认证它自己的协议/分数结构；本次 D 的标准 run-v2 制品链校验不能替代独立图像判断。
