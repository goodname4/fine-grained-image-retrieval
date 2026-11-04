# C 接续检查与缺失交付：2026-11-04

执行者：C / Codex assistant。同步到 `origin/main` 的 `ffbc9b0e16ae85e52433bf20007a9dbd75fb95cb`，C 的 `c377cec` 与 D 的实现均已合并。此次完成输入核验、C 原始开发制品与合并后 D 接口的真实导出联调；正式 C-V2-02 仍 BLOCKED。未代填真人记录、未启动正式训练、未操作正式 test。用户后续已要求将本报告、核验证据及完整交接需求提交推送；统一接手入口为[C交接清单](../handoff-c-v2.md)，人工流程见[操作手册](../human-operation-manual-v2.md)。实际数据和模型仍按清单另传。

## 收到的文件与实际状态

人工操作包：`fgir-human-operation-20261030-v1.zip`，SHA-256=`2d790753c84a81be5b40578c56505ebf110f8612fa4353114182d3233438f031`。随附 `human-operation-manual-v2.md` 与包内 `MANUAL.md` 字节一致。手册是人工操作说明，不能视作已经执行的审核、研究决策或训练授权记录。

完整解压副本：`artifacts/c-v2/followup-20261104-v1/human-operation-package/`，入口 `index.html`，C 的任务册 `members/C.html`。保留原ZIP、原始基线及历史证据，不将其复制成 `train-verified` 或 `frozen-val`。

实际核验：ZIP CRC、manifest 绑定的166个文件摘要、582个本地HTML链接、139张图的权威来源字节摘要均通过；两份工作稿通过 `validate_semantics(..., purpose="draft", data_root="data")`，train/val源图交集为空。

| 划分 | 源图组 | 变体 | 组件 | 原句 assistant 核验 | 真人初标 | 独立真人金标 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 106 | 334 | 39 | 106 | 0 | 0 |
| val | 33 | 102 | 13 | 33 | 0 | 0 |

这是比仓库旧30组试点更新的工作包，仍是草稿。实际train发布校验拒绝 `source positive needs image support`，val评测校验拒绝 `not independently reviewed`。结构通过、`status=annotated` 或助手建议筛选数量都不等于真人已核验。

## 已直接完成的 C 工作

1. 严格核对本机 C 原始开发包 SHA-256=`5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`，CRC及包内每个文件与原运行目录逐字节一致。现成文件：`artifacts/c-v2/c-v2-development-handoff-v1.zip`（140,666字节），可以另传D；此处的核对不表示D机器已经收到文件。
2. 用原 C `n0-smoke-seed42-v2/best.pt`，经合并后的 `export-text-lora` 实现导出至 `artifacts/c-v2/followup-20261104-v1/n0-original-development-export/`。不是D重建权重，也未重新训练。
3. 导出包含严格绑定的checkpoint、raw图像索引、文本特征、receipt和run-v2。5000条原始val查询，每条1000张图库图片，所有排名完整且无重复。receipt严格重载及run开发模式校验通过；未开启开发许可时正确拒绝 `development_smoke is not a formal N-series result`。
4. 原 C checkpoint 的CPU离线查询通过，Top-5图片均存在。无外部AI推理服务调用，`semantic_dataset_sha256=null`保留原义，不报告正式N0指标。
5. 隔离环境 `.venv-semantic` 全套173项测试：172通过、1项Windows符号链接权限跳过，46.698秒。日志在 `artifacts/c-v2/followup-20261104-v1/tests.log`。

实际导出参数对应的注册CLI：

```powershell
.\.venv-semantic\Scripts\python.exe -m fgir export-text-lora --checkpoint artifacts/c-v2/n0-smoke-seed42-v2/best.pt --model-dir models/clip-vit-base-patch32 --index artifacts/c-experiments/seed42/retrieval/E0/val/index.npz --protocol data/prepared/flickr8k-v1/val/protocol.json --output artifacts/c-v2/followup-20261104-v1/n0-original-development-export --device cuda --batch-size 64 --experiment-id N0-original-C-development
```

本次实际通过同一 `export_text_lora` Python函数运行，显式设置torch线程4，HF/Transformers离线。输出目录已存在，重跑须另取新版本，不能覆盖。

| 文件 | SHA-256 |
| --- | --- |
| 原 C best/checkpoint.pt | `2d72425de399f210c7a1888a8da377c3c41db63a50800e3809c581d694bc3d7b` |
| 新导出 receipt.json | `2a0509f8d0cf67faeaf8ab83d9b3ca5648b29f6482e75168a483af751d26af6d` |
| 新导出 run.json | `44d51a4f9cca7f61222f57b488c42c9a4c33e45205ea0cd88b7e77ba71097569` |

## 现在需要谁准备什么

| 提供方 | 必须交回的内容 | 与 C 的关系 |
| --- | --- | --- |
| A组织；A/B/C/D四位真人 | 实际成员对应表；各自真实初标后导出的完整val、train JSONL及提交说明；agent合并初标基线后，再交叉复核并导出；有争议另交合规第三人裁决 | 当前包没有这些已填写交付，助手不能代签。每人每轮的导出含该划分全部记录，不代表此人审核全部。 |
| B / B-V2-02 | `data/semantic/train-verified-v1/samples.jsonl`，连同版本、来源、审核/拒绝/排除及各现象质量记录、实际文件摘要。若使用automatic核验，还要真实人工抽检记录，符合 `fgir-semantic-training-audit-v1`，训练时用 `--training-audit` 传入 | 106组工作稿可作待审核输入，不能直接开正式训练。|
| A / A-V2-04 | `data/semantic/frozen-val-v1/protocol.json`及协议实际绑定的全部samples、图库、源数据成员；真人初标/不同真人复核、排除/争议、覆盖审计、文件摘要及负责人实际研究决策 | 单独一个protocol文件、旧标准检索val协议或33组工作稿均不够。 |
| C / C-V2-02（上游到位后） | 42/43/44配对种子的N0/N1/N2/N3及N2去关系消融，共15次正式运行；best/last、配置、训练来源、全部val尝试/选择及失败记录、环境和摘要 | 训练/损失/编排代码已具备；此次未产生任何正式权重，不需要让别人代造C结果。 |
| D（收到正式C权重后） | 正式receipt/run-v2/语义分数与N2→E0→N3→N2切换验收 | 默认验证回调已能导入，已从最新C预检缺失项中解除；后续仍需真实冻结数据和正式权重。 |

新手册中，C本人初标val8＋train26＝34组，复核D的val7＋train26＝33组。复核阶段页面成员选D，必须等待agent返回统一初标基线后进行。A/B各初标36、复核36；D初标33、复核34。旧任务板8/8/7/7分配属于原30组试点，不适用于这个包。

负责人研究决策需根据真人最终有效源图覆盖决定是否补样或缩小结论。手册的val25/train88等数字只是助手建议，不当作已冻结规模；不能为了开训强行认可模糊负例。

## 最新正式预检与交接话语

已实际调用 `run_text_experiments(..., check_only=True)`；状态BLOCKED，只有B的verified-train与A的frozen-val路径缺失。基础权重、tokenizer、源数据、raw train缓存、D验证回调均已存在。没有创建正式训练输出目录。输入到位后还需运行真实数据校验，路径存在本身不构成人工质量认证。

机器可读证据：本报告旁 `c-v2-followup-20261104/{input-audit,formal-preflight,original-c-d-export-audit,test-summary}.json`；完整本地制品在 `artifacts/c-v2/followup-20261104-v1/`，被Git忽略，需另传或重建。

可直接转述：

> C已同步最新main（ffbc9b0），C/D代码都已合并。C原始开发包摘要核对通过，并用原权重完成D接口真实5000×1000排名导出；全套173项测试172过、1项环境跳过。操作包106组train、33组val仍全部是assistant工作稿，0组真人金标。目前请A组织四位真人按手册初标、统一基线复核及必要裁决；B据此交付verified train及审核/摘要，A交付冻结val协议和完整绑定文件。两项到位后C继续15次正式训练并逐个交D。现有两步开发权重不能当作正式N0，N2/N3尚未训练。
