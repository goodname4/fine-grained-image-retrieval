# 技术架构：文本LoRA与组合语义监督

版本research-v2，2026-10-27。本文为新路线设计，N0—N3代码尚未实现；旧MLP训练的准确行为见[training.md](training.md)。

## 1. 两条隔离的执行路径

| 路径 | 图像侧 | 文本侧 | 制品 |
| --- | --- | --- | --- |
| E0 | 原始CLIP | 原始CLIP | 原始图片/文本特征 |
| E1/E2，已实现 | 冻结CLIP后接图像MLP | 冻结CLIP后接文本MLP | fgir-adapter-v1与适配图片索引 |
| N0—N3，拟实现 | 冻结原始CLIP图像塔及投影 | 冻结原始文本塔/投影，只训练注意力LoRA | 新的text-lora-v1、原始图片索引与派生文本特征 |

N模型不在E1/E2索引上叠加LoRA，也不把LoRA checkpoint传给旧load_adapter。模型类型与实验编号分离：N0—N3都是text_lora类型，其损失配置不同。

## 2. 离线语言流水线

训练原句 → 原文锚定的实体/属性/有向关系 → 同义改写与单条件错误变体 → 规则过滤 → 图片依据复核 → 版本冻结。

语义ID、原文字符区间、属性所属对象及关系subject/object必须可追溯；生成器不得补写图片常识。训练、验证、测试按源图与既有近重复排除记录隔离。小型本地生成模型离线工作；推理不依赖该生成器。输入结构见[语义数据协议](semantic-data.md)。

## 3. 可导文本前向

现有ClipEncoder.encode_texts使用torch.inference_mode，_project会detach并转NumPy，因此只能复用于推理/导出。新增训练前向返回torch.Tensor：

1. 使用官方tokenizer得到input_ids与attention_mask。
2. 调用CLIP原生get_text_features的可导路径并正确处理已验证的版本返回类型。
3. 保持输出张量、设备与计算图，按行归一化后计算损失。
4. 优化器仅接收白名单内的LoRA参数；图像塔、原文本权重、投影及温度均冻结。

训练前核对模块名：目标为text_model.encoder.layers中注意力的q_proj、v_proj，先试末两层、rank4，alpha8、dropout0；rank8或更多层只用验证集决定。以上为待测起点，不保证在8GB下的具体速度或效果。LoRA的B矩阵零初始化时首步部分参数可能无梯度；检查至少两步的有效变化，不能要求首步所有LoRA参数都有非零梯度。

图像缓存是在训练外以inference_mode生成并写盘，训练器读取普通无梯度Tensor；不得误用在线inference tensor参与需要保存输入的反向计算。旧文本缓存仅用于历史基线，不能替代N训练原文。

## 4. 损失的计划定义

设v为冻结图片的归一化特征，f_theta为带文本LoRA的归一化编码，s(v,t)=v·f_theta(t)。

- L_global：图文对齐。每源图每轮选一条原句及共用同义正例，所有N实验使用相同数据与采样；同图多描述为正例，已知跨图歧义项屏蔽。
- L_sentence：对经图片核验的完整句正确/错误编辑对，采用softplus((s(v,t_wrong)-s(v,t_right))/tau)。
- L_unit：对绑定属性的实体短语或有向关系短语及其错误变体，使用同样的成对softplus目标；每种现象按有效样本均值聚合。通用组件不直接互作批内负例，避免“a person”类短语产生大量假负例。
- L_para：对同一图片下原句与经核验同义句的分数差平方取均值；不是强制所有文本向量相同。全句与区分项保留以避免塌缩。

N0=L_global；N1=L_global+lambda_s*L_sentence；N2=L_global+lambda_u*L_unit；N3=N2+lambda_p*L_para。N1/N2是两种监督粒度的对照，不叠加不同数量编辑。该成对实现借鉴CS-CLIP思想，不冒称原论文完整损失复现。

tau固定并记录；损失按源图及有效编辑归一化，空编辑批次掩码跳过并记录比例；拒绝把全空批次伪装成零损失成功。

## 5. 索引和模型身份

图像塔不变时，N0—N3可共享E0原始图片向量及其校验过的索引。不要给共享图像索引写入特定文本LoRA摘要。

计划采用独立retrieval receipt描述一次图文配对：

- 基础CLIP权重、tokenizer、图像预处理及投影维度；
- image_index_sha256与image_encoder_sha256；
- text_lora_sha256、LoRA目标层/rank/alpha/结构版本及method；
- 数据、语义数据、checkpoint选择配置与源码版本摘要。

缓存的N文本特征必须绑定对应text_lora_sha256；排名文件绑定receipt。允许多个receipt引用同一原始图片索引；加载时必须核对receipt中的文本LoRA与实际文本特征一致。旧adapter_sha256仅用于旧MLP路径，禁止伪造索引适配摘要来绕过现有检查。

## 6. 服务与评测边界

新增模型注册表，包含experiment_id、model_type、display_name、base_model与receipt。服务按model_type分发；E0/E1/E2旧配置兼容。

旧service以基础模型目录共享encoder；若直接向共享实例注入不同LoRA，将污染E0或其他N模型。N加载缓存键至少包含基础权重与LoRA摘要；首版可独立文本实例并按需释放，验收N2→E0→N3→N2结果恢复一致。

FeatureIndex精确余弦与并列ID排序继续复用。新run/schema及诊断协议版本在编码、服务、导出、评分端一起实现，不能只给METHODS加四个名称。细节见[迁移计划](implementation-plan.md)。
