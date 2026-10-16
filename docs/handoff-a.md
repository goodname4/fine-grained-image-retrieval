# A → B/C/D：可立即使用的交接

## 当前可用范围

本地已准备真实Flickr8k清洗版`data/prepared/flickr8k-v1/`，以及`data/annotations/fine-v1/`。图片、完整清单、生成页面和评测输出被Git忽略；其他电脑用以下命令重建，不依赖本机路径：

```bash
python -m pip install -e ".[data]"
python -m fgir prepare-a-data
python -m fgir verify-handoff --bundle data/prepared/flickr8k-v1 --data-root data
```

首次下载约1.04 GiB的图片ZIP及2.2 MiB的描述ZIP，实际还需解压空间。公开镜像固定为[jbrownlee的Flickr8k release](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)，原始用途说明见[作者项目页](https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html)。仅用于非商业课程/研究，图片版权和Flickr条款仍适用，不随代码再发布图片。

下载器验证版本化SHA-256；准备器读取原始官方划分，应用明确的排除和近重复复核记录，再写入新目录。已有交接包会校验复用，已有`annotations.work.jsonl`不会覆盖。已有本地文件时可用`prepare-a-data --skip-download`；该选项仍会检查输入清单和最终图片摘要。

| 划分 | 图片 | 图文对 | 用途 |
| --- | ---: | ---: | --- |
| train | 5,998 | 29,990 | 编码、训练、困难负样本挖掘 |
| val | 1,000 | 5,000 | 选模型、超参数和早停 |
| test | 1,000 | 5,000 | A统一执行正式测试 |

清洗版清单SHA-256：`a7f7d25d5017fdb62c150885da7bc3de3a8aea1dd2e40fc06d1da3ea6296dc0d`。原始官方清单保存在`data/manifest.official.jsonl`，不要用它替换清洗版。清洗改变了训练规模，结果须注明`flickr8k-v1 cleaned`，不能当作未经处理的6000训练图片协议。

## 交接包文件

```text
data/prepared/flickr8k-v1/
  dataset.json                  所有交接文件摘要、规模与规则
  manifest.jsonl                全部7998张图片，path相对data-root
  inventory.jsonl               原文件/解码像素摘要、尺寸、dHash
  audit.json                    清洗后图片检查结果
  exclusions.json               两条排除及原因
  duplicate-review.json          原始21对候选的看图记录
  duplicate-screen.json          保留图片的近重复候选
  source-receipt.json            描述与原始划分摘要
  train/images.jsonl            每图一行，包含5条captions
  train/pairs.jsonl             每描述一行，query_id=image_id#编号
  val/images.jsonl, pairs.jsonl
  val/candidates.json, queries.jsonl, qrels.json, protocol.json
  test/images.jsonl, pairs.jsonl
  test/candidates.json, queries.jsonl, qrels.json, protocol.json
```

新版本数据、标注或评测结果写新目录；工具拒绝覆盖已有交接包。`verify-handoff`先校验包内文件，再校验全部图片字节，避免同路径下图片已被替换。图片路径仅相对调用者提供的`data-root`。

## B：编码、建库和E0

1. 用`train/images.jsonl`缓存训练图片，用`train/pairs.jsonl`缓存全部训练文本；为val/test分别缓存图库。图片ID保持字符串，图片路径通过`data-root`解析。
2. val/test候选库各1000张，查询各5000条。按`candidates.json`中的ID建立索引；查询ID采用已有`query_id`，不得按文本合并重复查询或重编号。
3. 原始描述保持原文，不统一截断或翻译。模型自己的分词和预处理写入run配置。相似度用于排序，不作为概率。
4. 先运行val确认管线，把test正式排名按[排名契约](ranking-contract.md)交给A；保存完整Top-10，若要完整MRR则保存1000项完整排名。无需等待100条细粒度复核即可开发标准协议。
5. 困难负样本只用train。挖掘文件完成后运行：

```bash
python -m fgir validate-negatives --manifest data/prepared/flickr8k-v1/manifest.jsonl --negatives artifacts/train-negatives.jsonl
```

相关性未穷尽，结构检查无法消除假负样本；保留抽检记录和误标估计。B的25条标注已分配，A的预审建议可在工作台看到；A的25条初标需B实际独立复核。

细粒度尚未复核时，可先读取`data/annotations/pool-request-v1/`的仅文本/图库请求，输出各方法真实Top-K供A组织共同池补审；该请求不是正式评测。重建命令及`fgir-pool-run-v1`格式见[标注指南](annotation-guide.md)。

## C：训练输入与同图多描述

优先从每图一行的清单抽取描述。已有标准库辅助API可避免同一epoch把同图五条描述当作五张独立图片：

```python
from fgir.handoff import load_split, training_pairs_for_epoch
from fgir.contracts import image_path

train_images = load_split("data/prepared/flickr8k-v1", "train")
epoch_pairs = training_pairs_for_epoch("data/prepared/flickr8k-v1", epoch=0, seed=42)
local_path = image_path("data", epoch_pairs[0]["path"])
```

`training_pairs_for_epoch`每个epoch每图恰好一条描述，连续5个epoch轮转全部5条，顺序可复现，返回稳定query_id。它只提供数据，不实现训练器或损失。采用全部描述训练时，C需自行实现多正样本mask。

记录中的`negative_exclusion_group_id`把已识别的同场景训练帧分在同组。不要在同batch把同组成员互当负样本，也不要据此假定任意跨帧描述都是正样本。保留的同场景图片为`3426966595_c8c4e1e872`与`3426933951_2302a941d8`，均在train。

用val选checkpoint，test不能用于早停。E1/E2排名必须记录适配层checkpoint摘要，与B使用的图像索引和文本适配层一致，再交A统一评测。

## D：展示与性能

复用包内ID/path映射和B的真实索引。总体/分类指标在`evaluate-run`输出的`metrics.json`，逐查询数据在`per-query.jsonl`；比较结果在`compare-runs`输出的`comparison.json`和`report.md`。`data_kind=synthetic`的结果必须标成工具示例，不能展示为模型结果。

计时由D实测，A提供聚合命令。每行记录`stage`、`device`、`gallery_size`、`top_k`、`batch_size`、`cache_state`、`warmup`、`cold_start`、`milliseconds`，GPU/MPS还需`synchronized: true`：

```bash
python -m fgir summarize-latency --measurements artifacts/latency.jsonl
```

阶段为`index_build/text_encoding/vector_sort/end_to_end/cold_start`。预热样本不参与聚合。输出`groups`保留原始`stage`，增加布尔字段`cold_start`作为分组维度；不同阶段、冷启动状态、设备、K、规模、批大小和缓存状态分别统计。读取稳定查询延迟时须筛选`stage=end_to_end`且`cold_start=false`，首次查询各阶段耗时则分别读取对应阶段的`cold_start=true`组。旧式`stage=cold_start`仍作为独立阶段，输出的`cold_start`恒为true。`excluded_from_steady_state`记录预热和冷启动样本数，冷启动样本仍保留在各自组中。分位数用排序后`(n-1)*p`线性插值，输出样本数、p50/p95与范围。

## 尚需外部产出的工作

- B/C的本地模型编码与E0/E1/E2真实排名、环境与配置；当前没有模型效果结论。
- B/C/D各自初标、交叉复核及所有查询共同候选池补审。A已完成25条AI辅助初标和B的25条预审，但不冒充另一位成员完成独立复核。
- 真排名到位后，A执行正式比较并逐图完成至少12个成功/失败案例。目前自动选例和复核工具已具备，案例原因不会预先编造。
