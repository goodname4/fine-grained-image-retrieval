# C research-v2：文本LoRA与受控损失交付

2026-10-28，C / Codex，会话01a143a6-a329-7fb1-8a09-4856feba4274。同步基线342ba1e，依AGENTS与任务板开展C-V2-01及C-V2-02可独立部分。

## 1. 状态与明确边界

**C-V2-01完成：** 已实现文本编码器内可导LoRA，真实train原句两步开发闭环，保存best/last并通过冻结、梯度、图片不变及安全重载审计。

**C-V2-02代码部分已实现，正式实验仍BLOCKED：** 全句对齐、完整句/组件成对softplus、同义分数一致性、多正例与源图宏平均、N0—N3及去关系消融配置、42/43/44配对种子编排、val回调与选择门槛均有代码和测试；尚未运行正式N0—N3、多种子或新路线test。测试夹具中运行各方法不构成真实效果。

本机可复用旧CLIP、数据与原始图像缓存，但没有 `data/semantic/train-verified-v1/samples.jsonl`、`data/semantic/frozen-val-v1/protocol.json`，也没有D的真实验证导出模块。仓库中的B产物说明是val候选，不能替代train核验或独立val金标。实际缺口见[preflight](c-v2/formal-preflight.json)。本次未创建人名、审核、分数来绕过这些门槛。

## 2. 新增实现

| 模块 | 实际能力 |
| --- | --- |
| text_lora.py | 白名单文本q_proj/v_proj原生LoRA；可导Tensor输出；冻结审计；fgir-text-lora-v1安全保存/重载 |
| text_training_data.py | 原句与原始图片特征联结；train审核门槛；同图多正例；token限长与同token编辑过滤；同源图/同场景排除 |
| semantic_losses.py | 源图宏平均全局多正例对齐、全句/组件成对softplus、原句/同义句的图片匹配分数差平方；空监督与关系mask |
| text_training.py | 独立优化器、累积与实际负例池日志、所有val尝试、N0参照检索门槛、参数/图片/重载验收 |
| run_semantic_experiments.py | 输入preflight、严格配对配置、目标种子42/43/44、N0—N3及N2去关系监督编排 |
| text_commands.py / CLI | train-text-lora、inspect-text-lora、run-text-experiments注册，基础CLI无torch依赖 |

LoRA默认文本末两层（10/11）的q_proj/v_proj，rank4、alpha8、dropout0。每层每投影参数`2×512×4`，四个投影合计16,384。A随机初始化、B零初始化，增量为`alpha/rank * B(A(x))`。图像塔、原文本塔、两个投影和logit_scale全部冻结。没有使用PEFT，不修改基础模型文件，不污染E0共享实例。

N0/N1/N2/N3使用相同原句/经核验同义正例与同一组编辑。训练会前向同一文本池，只改变目标组合：N0=L_global，N1加sentence项，N2加unit项，N3在N2上加para项；去关系只切换unit监督mask，不删关系文本曝光。同义项只约束同图匹配分数，不直接强迫所有文本向量相同。

若原句超77个CLIP token，丢弃该记录并记录原因；编辑/关联组件超限或token相同则成对排除，防止N1/N2接收不同编辑。已知同场景/相同描述的跨图配对不充当全局负例，通用组件不自动互作负例。

## 3. 独立环境与真实命令

另建 `.venv-semantic`，没有升级原 `.venv`。实际使用Python3.11.4、torch2.11.0+cu128、transformers4.57.6、NumPy2.4.3、Pillow12.2.0、RTX4060 Laptop。配置及环境记录随本报告保存；依赖组为semantic-training，基础CLI仍可在未装torch的Python运行。

准备环境示例（按机器选CPU/CUDA，权重/依赖就位后无需网络）：

```powershell
python -m venv .venv-semantic --system-site-packages
.\.venv-semantic\Scripts\python.exe -m pip install -e ".[semantic-training]"
```

实际运行命令：

```powershell
.\.venv-semantic\Scripts\python.exe -m fgir train-text-lora --model-dir models/clip-vit-base-patch32 --image-features data/features/clip-vit-base-patch32/train.images.npz --config configs/training/n0-smoke.json --output artifacts/c-v2/n0-smoke-seed42-v2
.\.venv-semantic\Scripts\python.exe -m fgir inspect-text-lora --checkpoint artifacts/c-v2/n0-smoke-seed42-v2/best.pt --model-dir models/clip-vit-base-patch32 --device cpu
.\.venv-semantic\Scripts\python.exe -m fgir run-text-experiments --model-dir models/clip-vit-base-patch32 --image-features data/features/clip-vit-base-patch32/train.images.npz --samples data/semantic/train-verified-v1/samples.jsonl --val-protocol data/semantic/frozen-val-v1/protocol.json --output artifacts/c-v2/formal-paired-v1 --check-only
```

首次试跑v1保留原样，v2为最终严格schema产物。输出目录拒绝覆盖，其他机器改用新版本目录。上述第三条真实输出BLOCKED及三个提供方缺口，没有开始正式实验或创建假权重。

## 4. 真实N0开发验收记录

本次只选16张真实train图，每图保留原5条caption供轮换，共80条原句记录；实际两次优化各8张源图/8条文本，合计16次文本前向、248个非padding token。无val/test优化或选模，无语义金标，无检索效果声明。两个batch损失为0.7264和0.5603，批次不同，不能当作收敛/效果提升结论。

| 检查 | 实际结果 |
| --- | --- |
| LoRA参数/梯度 | 16,384；8个A/B张量均在两步内观察到非零梯度并更新 |
| 原参数 | 前后完整冻结参数SHA-256相同 |
| 图像输出 | 同两张train图训练前后最大绝对差0 |
| 图片缓存核对 | 与实际冻结图片塔归一化输出最大差2.98e-7 |
| 独立CPU重载 | 文本特征最大差1.3411e-7，匹配分数最大差4.4703e-8，均小于1e-5 |
| 基础模型文件 | 权重摘要训练前后均d5853196…，未覆盖 |
| 截断/跳批/NaN | 静默截断0，丢弃0，跳批0，NaN0 |
| 纯训练/训练后验收段总计 | 0.6013秒 / 4.5554秒；只含两次小batch，不是完整训练耗时 |
| PyTorch峰值显存 | 693,329,920字节，约661.2MiB；不含其他桌面进程 |

计时不含模型/缓存载入及环境准备，也不含正式语义数据生成/核验、验证导出和完整实验。旧MLP成本没有挪作本次LoRA成本。重复开发试跑两次的LoRA参数变化逐项一致，checkpoint文件摘要因schema/来源记录变化不要求相同。

本次best仅表示最后一个通过开发检查的状态，选择规则写明latest-development-step-no-validation，不是验证集最佳模型。

小型原始记录已准确复制至Git：

- [summary](c-v2/n0-smoke-summary.json)、[两步日志](c-v2/n0-smoke-history.jsonl)、[实际完整配置](c-v2/n0-smoke-resolved-config.json)。
- [冻结审计](c-v2/freeze-audit.json)、[CPU重载审计](c-v2/reload-audit.json)、[环境](c-v2/environment.json)。

summary实际SHA-256：`43152009504d4edb269fdc0288c3f96705378579f29d61097647624fbcfe8cf1`。训练源码按基线342ba1e-dirty和逐文件摘要记录；Python源码与开发运行记录核对。

## 5. C-V2-02配置与选择

`configs/training/n0.json`至`n3.json`仅method不同；n2-no-relation只关闭include_relations。默认micro-batch8、累积4、学习率1e-4、tau0.07、句/组件权重0.2、同义权重0.1、固定10epoch共同预算。**累积4次不把InfoNCE负例池扩成32图**，分母只包含当前微批次及其正例文本。

为满足计划中相同更新/验证预算，本模板固定预算运行，不跨方法分别早停；保留epoch0和每个epoch的全部验证尝试，较早模型可仍被选中。计划建议的patience3不在本次配对模板启用。正式开跑前仍须按数据质量冻结最终配置，不根据test反馈改规则。

N0按val R@1、MRR选模型；其余按绑定/方向源图宏平均、N0检索保护阈值0.01、R@1、MRR及较早epoch选择。D回调未交付前不填虚构分数；没有合格模型就保留失败。D接口、自动核验训练的人工抽检字段、精确samples摘要定义见[制品契约](../text-lora-contract.md)。

## 6. 测试与下游交接

新增21项专项测试覆盖零初始化、至少两步梯度、冻结不变、严格重载、缺/错/非有限参数、错误基础模型/tokenizer/目标层拒绝、源图宏平均、关系mask、同义不强迫向量相同、N1/N2/N3前向曝光一致、审核门槛、拒绝test选模与真实输入缺失。测试使用本地随机tiny-CLIP与明确synthetic图片/标签/回调，不下载模型，不代表真实训练效果。CI新增独立semantic-training任务。

全套164项：163通过，1项Windows符号链接权限跳过；标准库Python未安装torch，基础search仍通过。旧E0/E1/E2、A语义工具和D旧服务回归通过。

给D的开发包已生成：`artifacts/c-v2/c-v2-development-handoff-v1.zip`（140,666字节），含best/last、完整配置、日志、冻结及重载审计。SHA-256：`5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`。Git不上传权重/ZIP，需另传此包，或用以上命令新目录重建。best摘要`2d72425de399f210c7a1888a8da377c3c41db63a50800e3809c581d694bc3d7b`；last摘要`449538c4c8b58059cea625f75e9e47125ae5e085021b263b12e73b05db99f74c`。

解除正式实验缺口：B-V2-02提供核验train样本及自动核验时的真人抽检；A-V2-04提供独立复核val协议；D-V2-01/02按契约实现真实checkpoint验证/receipt及val分数。全体H-V2-01真人标注互审仍待完成，本会话没有替任何成员签署human记录。上述依赖满足后，用run-text-experiments执行42/43/44配对实验及关系消融，再交A统一正式test。
