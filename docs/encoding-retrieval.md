# 编码与检索方法（B）

> 2026-10-27版本说明：本文保留当前CLIP推理、原始特征和E0/E1/E2的操作。新路线仍可复用原始图片向量，但文本LoRA训练必须使用可导前向；当前encode_texts会关闭梯度并detach，不能用于更新文本参数。N模型的文本导出/索引兼容设计见[架构](architecture.md)及[实施清单](implementation-plan.md)。

本文是 B 模块的方法章节与模型准备说明，覆盖本地编码、特征缓存、精确检索、E0 排名导出与困难负样本挖掘。命令均为 `python -m fgir <command>` 形式（本机 `fgir` 不在 PATH）。

## 1. 模型准备说明

- 模型：`openai/clip-vit-base-patch32`（CLIP ViT-B/32，图文投影维度 512）。E0 为零样本基线，权重冻结，不参与训练。
- 权重获取：`python -m fgir prepare-clip-model --source openai/clip-vit-base-patch32 --output models/clip-vit-base-patch32`。该命令一次性下载权重并写入显式本地路径，产出 `models/clip-vit-base-patch32/receipt.json`，记录来源、transformers/torch 版本、逐权重文件 SHA-256 与合计 `weights_sha256`。
- 权重放 `models/`，Git 忽略（见 `.gitignore`），不随仓库再分发；其他机器用上述命令重建。权重就位后全部编码、建库、检索**断网可执行**（`local_files_only=True`）。
- `weights_sha256` 定义：单权重文件即该文件的 SHA-256；分片权重为「文件名:文件摘要」清单的 SHA-256。所有 E0 run 的 `weights_sha256` 与收据一致。
- 依赖为可选组：`python -m pip install -e ".[model]"`（numpy、torch、transformers、Pillow）。核心 `fgir` 工具保持零依赖，CI 只装 `.[data]` 时相关测试自动跳过。

## 2. 本地编码

实现：`src/fgir/encoding.py`。

- 图片编码：`fgir.handoff.load_split` 取出图片行，路径经 `fgir.contracts.image_path(root, value)` 相对 `data-root` 解析（防目录逃逸），Pillow 解码转 RGB。预处理为模型官方 `CLIPImageProcessor`（最短边缩放、中心裁剪、官方均值方差），不做额外改动。
- 文本编码：按 `train/val/test 的 pairs.jsonl` 逐条取**原始描述**，不截断、不翻译；分词/截断由官方 `CLIPTokenizer` 完成（含 77 token 上限），规则记入 run 的 `config.text_processing`。
- 批处理与设备：`--batch-size`（图片默认 32，文本默认 64）；`--device cpu|cuda|auto`，auto 优先 CUDA、无则 CPU。推理用 `torch.inference_mode`。
- 输出为 CLIP 投影空间的**原始特征**（未归一化）；归一化推迟到建库时（见 §3）。缓存位置与命名：`data/features/<model>/<split>.images.npz`、`<split>.texts.npz`，每份附 `<文件>.meta.json`（model_id、weights_sha256、设备、预处理描述、维度、计数与文件摘要）。相似度只用于排序，不是概率。

命令：

```bash
python -m fgir encode-features --bundle data/prepared/flickr8k-v1 --data-root data \
  --model-dir models/clip-vit-base-patch32 --output data/features/clip-vit-base-patch32 --device cpu
```

## 3. 特征缓存与索引

实现：`src/fgir/features.py`。

- 特征文件加载时校验 `features_sha256`，模型版本/维度不一致直接拒绝（文本与图片特征必须同权重）。
- 索引：对行做 L2 归一化后余弦点积精确检索（`FeatureIndex`）。并列分数按 image_id 字典序稳定打破，与 `fgir.retrieval.ExactIndex` 语义一致、确定可复现。
- 重建索引不重跑编码：`build-index --features <raw.npz> --output <index.npz>` 直接读缓存；C 的适配层走 Python API `build_index(path, out, adapter=f)`（`f: 原始特征矩阵 -> 适配特征矩阵`，适配后再归一化），适配特征=f(原始特征)，绝不触碰编码器。

```bash
python -m fgir build-index --features data/features/clip-vit-base-patch32/test.images.npz \
  --output data/features/clip-vit-base-patch32/test.index.npz
python -m fgir search-index --index data/features/clip-vit-base-patch32/test.index.npz \
  --query-features data/features/clip-vit-base-patch32/test.texts.npz --output artifacts/search-val.json --top-k 10
```

## 4. E0 基线与排名导出

实现：`src/fgir/runs.py`。E0 = 预训练 CLIP 冻结零样本检索，无适配层。

- 每条协议查询经缓存文本特征与图库索引余弦排序；查询 ID 沿用 `pairs.jsonl` 的 `image_id#编号`，不重编号、不按文本合并。
- run JSON 为 `fgir-run-v1`：`run_id / method / model_id / data_kind=real / protocol_sha256 / weights_sha256 / adapter_sha256(E0 为 null) / seed / code_revision / config / ranking_depth / rankings`。`code_revision` 取实际 commit；工作区有未提交修改时记为 `<commit>-dirty` 并在 `config.uncommitted_source_sha256` 列出各改动文件摘要。
- 排名深度：**默认全图库 1000 项**，可评真 `mrr`（Top-10 只能记 `mrr@10`）。非空排名长度必须等于声明深度，ID 不得越出图库；`rankings` 覆盖协议全部且仅有的 query_id。
- 先 val 跑通管线（`evaluate-run` 本地自检），test 正式排名交 A 统一评测。

```bash
python -m fgir export-run --protocol data/prepared/flickr8k-v1/val/protocol.json \
  --index data/features/clip-vit-base-patch32/val.index.npz \
  --query-features data/features/clip-vit-base-patch32/val.texts.npz \
  --output artifacts/runs/e0-val.run.json --method E0 --seed 42
python -m fgir evaluate-run --protocol data/prepared/flickr8k-v1/val/protocol.json \
  --run artifacts/runs/e0-val.run.json --output artifacts/evaluation/e0-val
```

## 5. 困难负样本挖掘

实现：`src/fgir/negatives.py`。只用 train（5998 图 / 29990 查询），val/test 绝不进入候选。

对每条训练查询（`image_id#N`）按文本→图片余弦取 Top-M（默认 50），过滤后取前 K 条（默认 5）为困难负样本。硬性排除：

1. 源图（永远是正样本，进 `positive_image_ids`）；
2. **完全相同描述文本**的其他训练图（可能也是正样本，整条排除，记 `excluded_image_ids`）；
3. `duplicate-review.json` 判定同场景/相同的帧（union-find 同组，即 `negative_exclusion_group_id` 同组成员）——不能当负样本，也不冒充正样本。

输出 `artifacts/negatives/train-negatives.jsonl`，逐行含 `query_id / positive_image_ids / negative_image_ids / mining_model / mining_parameters / excluded_image_ids / reviewed`，通过校验：

```bash
python -m fgir validate-negatives --manifest data/prepared/flickr8k-v1/manifest.jsonl \
  --negatives artifacts/negatives/train-negatives.jsonl
```

**假负样本检查说明**：相关性未穷尽，结构过滤（同文本、同场景组、源图）只能消除*已知*风险，无法证明剩余负样本都是真负样本。挖掘同时产出：

- `train-negatives.review-sample.jsonl`：固定种子抽样的人工复核清单，含图片 ID、查询文本与 `false_negative_image_ids / decision` 待填槽位；
- `train-negatives.report.json`：挖掘参数、逐类排除计数与风险说明。

人工看图抽检后把发现的假负样本写回主文件并置 `reviewed=true`（含 reviewer 与理由）；未抽检前 `reviewed=false`，`validate-negatives` 输出 `reviewed_query_fraction` 如实反映复核进度。C 训练时对每条查询取少量困难负样本与普通批内负样本混合，混合权重在验证集选择。

## 6. 已知限制

- 单模型（CLIP ViT-B/32）单种子；多种子与不同预训练模型对比见 roadmap 加分项。
- 全部在 CPU 完成（本机无 CUDA），编码耗时较长但结果确定（推理无随机性）。
- 标准协议 `annotation_scope=source_pair`：每条描述只以其源图为已标注相关，指标衡量源图命中率，不宣称其余图片不相关。细粒度人工协议另行冻结。
