# C：轻量适配、模型训练与消融报告

## 实际结果

在真实 `flickr8k-v1 cleaned` 上完成 E1/E2 配对训练。下表是 **验证集** 结果：固定 1000 张候选图片、5000 条原始描述、源图配对相关性、完整 1000 项排名。验证集用于早停和模型选择，不是正式测试效果。

| 方法 | R@1 | R@5 | R@10 | 完整MRR | 最佳epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0：冻结CLIP | 54.82% | 81.74% | 89.08% | 0.667706 | — |
| E1：图文残差MLP | 56.08% | 83.28% | 90.42% | 0.679540 | 8 |
| E2：E1＋困难负样本辅助损失 | 56.28% | 83.38% | 90.34% | 0.680443 | 8 |

E1 相对 E0 的 R@1 增加 1.26 个百分点；E2 相对 E0 增加 1.46 个百分点。E2 比 E1 的 R@1 高 0.20 个百分点，即本次验证中额外命中 10 条查询；R@10 则低 0.08 个百分点。不能据此断言困难负样本全面改善检索，也不能将单种子、经过模型选择的验证差异解释为统计显著的泛化提升。

两套实验均实际执行 13 个 epoch，第 8 个 epoch 最佳，之后连续 5 个 epoch 未改善而早停。训练损失继续下降而验证 R@1 波动，说明继续优化训练目标不保证提高检索指标。正式 test 排名已导出但本次没有评分，由 A 统一评测；100条细粒度查询尚未完成独立复核，暂不填写分类效果。

## 模型、损失和配置

编码器为本地 `openai/clip-vit-base-patch32`，官方图像预处理与 tokenizer。图文原始特征均为512维，先归一化，再分别经过 `normalize(u + 0.2 * W2(GELU(W1(u))))`。瓶颈维128，末层零初始化，两个分支共263,424个可训练参数。编码器没有进入训练器和优化器，输入为缓存的 detached 特征。

每个epoch每张训练图恰好抽取一条描述，按 A 的固定seed规则轮转；同图多描述不互当负样本。同场景和相同描述的可疑跨图配对从对比分母忽略，不擅自标为正样本。基础损失为多正样本对称InfoNCE。E2另对文本、源图和困难图片计算单向margin辅助损失，困难图片也经过当前图像MLP。

E1/E2均使用seed42、batch128、AdamW、学习率0.001、weight decay 0.0001、温度0.07、alpha0.2、最大20epoch、patience5、梯度裁剪1.0；只改变method和hard_weight（0→0.2）。E2每查询每epoch轮转取最多2个困难候选，margin0.2；与E1相同地独立从零初始化开始。配置见 `configs/training/e1.json`、`e2.json`。

训练集为5998图/29990描述，验证与测试各1000图/5000描述。B的挖掘输入仅含train，对29990条描述各存5个候选，共149950对。结构过滤排除源图29990次、相同描述候选481次、同场景候选10次。

**本次候选的人工复核比例为0。** 已输出100条抽检任务，但未虚构人工判断。E2结果应表述为“结构过滤、尚未人工抽检的困难负样本实验”，仍存在假负样本风险；后续复核形成新负样本文件后，应以新输出目录重跑配对实验并单独报告。

## 成本与复现环境

环境：Windows、Python3.11.4、PyTorch2.11.0+cu128、CUDA12.8、NumPy2.4.3、transformers4.57.6、Pillow12.2.0，RTX4060 Laptop GPU（8GB），CPU线程4，启用确定性算法。

| 方法 | 纯训练秒 | 训练＋逐epoch验证/保存秒 | PyTorch峰值显存 | 可训练参数 |
| --- | ---: | ---: | ---: | ---: |
| E1 | 5.454 | 8.058 | 77.91 MiB | 263424 |
| E2 | 7.236 | 9.769 | 78.77 MiB | 263424 |

上述成本仅指 **已缓存特征后的适配训练**，不含图片/权重下载、CLIP特征提取、困难负样本挖掘、最终排名导出，也不代表完整系统延迟。train＋val四份原始特征文件共89,800,360字节。显存统计为PyTorch训练张量峰值，不含桌面进程及编码器推理显存。

模型权重摘要为 `d5853196afb60ca6373692d9af911aacd8fa821af6d4af54720d4eb9c8f5da0a`；清洗清单摘要为 `a7f7d25d5017fdb62c150885da7bc3de3a8aea1dd2e40fc06d1da3ea6296dc0d`。运行以远程commit `679746b` 为基线，工作区包含C实现，故产物准确记录 `679746bf68f75948f2ff5526f15a5dbe98148b63-dirty` 并逐文件保存源码摘要。已核实训练时所有Python源码摘要与当前实现一致。

完整命令、公式与加载接口见[训练说明](../training.md)。适配层训练之后，必须同时切换文本MLP与对应的重建图片索引。大型图片、缓存、checkpoint和排名保存在Git忽略的本地目录；以下小型真实记录随代码保留：

- [机器可读结果](c-training-results.json)：与本地ablation.json一致，包含指标、成本、checkpoint和test排名摘要。
- [E1日志](c-training/e1-history.jsonl)、[E2日志](c-training/e2-history.jsonl)：实际epoch0–13逐条记录，包含退化epoch。
- [运行环境](c-training/environment.json)、[编码器收据](c-training/encoder-receipt.json)。

原始 `artifacts/c-experiments/seed42/ablation.json` SHA-256：`98758a79bc648b33f44d890a2d25477eda3a9b8c24f754260bf92a89b061f36a`。

## A/D交接与验证

本地根目录：`artifacts/c-experiments/seed42/`。

| 接收方 | 文件 | 状态 |
| --- | --- | --- |
| B/D | `E1/best.pt`、`E2/best.pt`；各目录的config、summary、checkpoints | 可加载；保留last.pt供检查最终优化状态 |
| D | `retrieval/E1/test/index.npz`、`retrieval/E2/test/index.npz`和各自texts.npz | 已适配并绑定checkpoint摘要 |
| A | `runs/e0-test.run.json`、`runs/e1-test.run.json`、`runs/e2-test.run.json` | 各5000查询×1000完整排序；契约校验通过，尚未正式评分 |
| A | `fine-pool/e0.pool.json`、`e1.pool.json`、`e2.pool.json` | 对同一100条文本实际编码检索得到Top-10，供共同池补审，不是正式评测 |
| A/D | `data/annotations/c-common-pool-v1/annotations.jsonl`；`artifacts/c-common-pool-board/board.html` | 已合并三个模型候选；新增图片保持未审 |

E1 checkpoint摘要：`dfe3cdda60e9d327c30a53d0bc906c457d36550fb18fcfdd82b3debac54e084e`。

E2 checkpoint摘要：`5a1cd614b1d5f32a1508f1551b452beac7f503b4e9805a2c7640f65709f28c93`。

A可统一运行正式标准协议比较：

```powershell
python -m fgir compare-runs --protocol data/prepared/flickr8k-v1/test/protocol.json --runs artifacts/c-experiments/seed42/runs/e0-test.run.json artifacts/c-experiments/seed42/runs/e1-test.run.json artifacts/c-experiments/seed42/runs/e2-test.run.json --output artifacts/c-test-evaluation
```

91项测试中90项通过，1项因Windows符号链接权限跳过；`git diff --check`通过。额外核对三个真实test run的协议摘要、模型/适配层摘要、查询全覆盖及每条1000个候选无重复，未计算test效果。实际离线CLIP编码＋适配加载＋100条细粒度Top-10导出及共同池扩展已运行成功。

C的25条逐图AI辅助初标及D的25条预审建议见 `configs/annotations/c-source-checks.json`。预审不能冒充D尚未进行的独立初标/复核，C初标也需D独立复核。当前真实交付为代码、训练、验证消融、权重/排名接口、方法章节和辅助标注；团队独立互审、困难负样本人工抽检、正式测试/分类结论继续按roadmap责任执行。
