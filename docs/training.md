# E1/E2历史基线：轻量适配与训练（C）

> 2026-10-28版本说明：本文保留缓存MLP历史训练与复现命令。新版可导文本LoRA已实现，真实N0开发试跑及N0—N3代码状态见[C新版报告](reports/c-v2-training.md)，正式实验仍待审核数据和D接口。不能用本文的detached文本缓存训练LoRA，也不能把旧MLP checkpoint加载为LoRA。

已完成的真实训练、验证结果、成本、产物摘要与交接清单见[训练与消融报告](reports/c-training-and-ablation.md)。

## 方法与数据边界

训练只读取 A 的 train、val 和 B 的同名原始缓存。`CachedSplit` 验证图片/描述清单摘要、完整 ID 覆盖、描述与源图对应、split/kind、维度、非零有限数值、图文权重与预处理版本；拒绝已经适配的输入。优化器只包含两个残差 MLP，编码器不加载到训练器，缓存张量 `requires_grad=False`。线上可用 `freeze_encoder(encoder)` 冻结编码器。

原始投影特征先归一化，以避免模长改变固定 alpha 的含义：

```text
u = normalize(x)
g(x) = normalize(u + alpha * W2(GELU(W1(u))))
```

图像、文本分别使用独立 MLP，W2 权重/偏置初始化为零，初始输出与 E0 一致。第一步 W2 获得梯度，后续 W1 也获得梯度。alpha 和温度固定。512 维、瓶颈 128 时可训练参数共 **263,424** 个，每分支 `(512×128+128)+(128×512+512)=131,712`。

基础损失为对称多正样本 InfoNCE：图片→文本、文本→图片分别计算 `logsumexp(允许项)-logsumexp(正项)` 再平均。同 image_id 的描述均为正样本。默认按 A 的 `training_pairs_for_epoch` 每图每 epoch 一条描述并轮转，最后不足 batch 的样本保留；损失仍支持重复图片和多正样本。

同场景不同图、相同描述的其他图片从分母忽略，不擅自标成正样本。B 明确排除项、额外已知正样本和已标记假负样本也不充当批内负样本。

E2 增加单向辅助项：

```text
L_hard = mean_valid(max(0, margin + cos(text, negative_image) - cos(text, source_image)))
L_total = L_symmetric + hard_weight * L_hard
```

困难图片经过当前图像适配层，图文 MLP 均接收梯度；不将没有对应描述的困难图片塞进对称矩阵。每条查询按 epoch 轮转取最多 `hard_per_query` 个候选，不足时 mask 补齐位置。验证负样本只来自 train、覆盖全部训练查询、无重复、无源图/同场景/相同描述图片。`negative_audit` 记录复核比例与已移除假负样本；未复核候选仍有假负样本风险。

按验证集 Recall@1 选模型，平分比较完整图库 MRR，再平分保留较早 epoch。epoch 0 恒等适配参与选择；训练退化时可保留基线行为。测试特征只在训练结束后的导出阶段读取，正式测试由 A 统一评分。

## 环境与命令

Python 3.11；缓存训练安装 `.[training]`（NumPy、PyTorch），重建特征安装 B 的 `.[model]`（另含 transformers、Pillow）。按硬件选择 CPU/CUDA PyTorch。权重准备后编码、训练、检索可断网运行。Windows PowerShell 从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[model]"
.\.venv\Scripts\python.exe -m fgir prepare-a-data
.\.venv\Scripts\python.exe -m fgir prepare-clip-model --output models/clip-vit-base-patch32
.\.venv\Scripts\python.exe -m fgir encode-features --model-dir models/clip-vit-base-patch32 --output data/features/clip-vit-base-patch32 --device auto
.\.venv\Scripts\python.exe -m fgir mine-negatives --text-features data/features/clip-vit-base-patch32/train.texts.npz --image-features data/features/clip-vit-base-patch32/train.images.npz --output artifacts/negatives/train-negatives.jsonl
.\.venv\Scripts\python.exe -m fgir run-c-experiments --features data/features/clip-vit-base-patch32 --negatives artifacts/negatives/train-negatives.jsonl --output artifacts/c-experiments/seed42
```

已有 A/B 本地交接物时直接执行最后一条。图片、权重和完整产物不随 Git 分发。输出必须使用新目录；失败保留日志，重跑改用新的目录。本版不提供中断续训。

单独训练：

```powershell
python -m fgir train-adapter --features data/features/clip-vit-base-patch32 --config configs/training/e1.json --output artifacts/c-training/e1-seed42
python -m fgir train-adapter --features data/features/clip-vit-base-patch32 --config configs/training/e2.json --negatives artifacts/negatives/train-negatives.jsonl --output artifacts/c-training/e2-seed42
```

配置严格检查未知键，路径从命令行传入。配对实验配置只在 method、hard_weight 不同，E1/E2 独立从零初始化训练，不从 E1 checkpoint 续训 E2。只运行一个种子时不声称统计显著。

## 产物与 B/D 接入

每个训练目录：`config.json`、`provenance.json`、`environment.json`、`history.jsonl`、`summary.json`、`checkpoints.json`、`best.pt`、`last.pt`。记录 commit/未提交源码摘要、数据/特征摘要、种子、软硬件、训练时长、峰值显存、缓存空间和参数量。last用于保留最后实际优化状态，不替代验证选择的best。

配对实验另输出 E0/E1/E2 验证结果、未评分的标准测试完整排名、图片索引/文本特征、`ablation.json`、`report.md`、`handoff.json`。验证结果参与模型选择，不能充当独立测试效果。

离线导出：

```powershell
python -m fgir export-adapter --checkpoint artifacts/c-experiments/seed42/E1/best.pt --image-features data/features/clip-vit-base-patch32/test.images.npz --text-features data/features/clip-vit-base-patch32/test.texts.npz --output artifacts/c-export/e1-test
```

`index.npz`、`texts.npz`、`receipt.json` 绑定同一 checkpoint SHA-256。适配后必须重建图片索引；`export-run`/`search-index` 拒绝图文 checkpoint 不一致的组合，排名导出还检查声明摘要与实际特征相符。

D 在线查询 API：

```python
from fgir.adapters import adapt_array, freeze_encoder, load_adapter
from fgir.encoding import load_encoder
from fgir.features import FeatureIndex

encoder = freeze_encoder(load_encoder("models/clip-vit-base-patch32", device="cpu"))
adapter, receipt = load_adapter("artifacts/c-experiments/seed42/E1/best.pt", expected_model=encoder.metadata())
index = FeatureIndex.load("artifacts/c-experiments/seed42/retrieval/E1/test/index.npz")
if index.meta.get("adapter_sha256") != receipt["adapter_sha256"]:
    raise ValueError("index and checkpoint differ")
query = adapt_array(adapter.text, encoder.encode_texts(["two dogs playing on grass"]))
hits = index.search(query[0], top_k=10)
```

## 检查与标注

`python -m unittest discover -s tests -v` 包含零初始化、梯度、参数量、对称交叉熵等价、多描述、同场景/重复描述忽略、辅助损失 mask、种子复现、泄漏拒绝、导出/评测兼容与配对消融。数值测试明确标为 synthetic，不代表真实检索效果。CI 增加 CPU 训练测试。

C 的25条逐图AI辅助初标和D的25条预审建议见 `configs/annotations/c-source-checks.json`。在 A 原工作文件上应用：

```powershell
python -m fgir apply-source-checks --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.work.jsonl --checks configs/annotations/c-source-checks.json --output data/annotations/fine-v1/annotations.c-assisted.jsonl
python -m fgir render-annotations --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.c-assisted.jsonl --data-root data --output artifacts/c-annotation-board
```

审图限于记录的25张 C 源图池，不代表完整图库相关性标注。D 尚未独立初标，预审不能冒充复核；C 初标亦需 D 实际独立复核。共同候选池补审和正式细粒度评测由 A 组织。

修订文本后使用同一个新请求重跑三个模型候选，不能混用旧文本排名：

```powershell
python -m fgir make-pool-request --manifest data/prepared/flickr8k-v1/manifest.jsonl --annotations data/annotations/fine-v1/annotations.c-assisted.jsonl --output data/annotations/c-pool-request-v1
python scripts/export_annotation_pool.py --request data/annotations/c-pool-request-v1/request.json --model-dir models/clip-vit-base-patch32 --experiments artifacts/c-experiments/seed42 --output artifacts/c-experiments/seed42/fine-pool --device cuda
```

本机已完成E0/E1/E2 Top-10实际导出和共同池扩展，待审工作台为 `artifacts/c-common-pool-board/board.html`。新增候选保持未审，不能直接拿该池作为已复核测试集。
