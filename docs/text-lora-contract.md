# C → D：文本 LoRA 制品与验证回调

2026-10-28 D 接入更新：`fgir.text_export:validate_text_lora_checkpoint`、receipt/run-v2、导出CLI及服务分发已实现；tiny-CLIP联调通过。真实C开发包与独立val金标尚未在D工作区到位，验收边界及命令见[D报告](reports/d-v2-integration.md)。下文原C交付时的“D尚未实现”为历史状态。

2026-10-28。C-V2-01 的可导模型/安全重载已实现。本页定义 C 实际导出的接口，不替代 D 尚待实现的 retrieval receipt、run-v2、服务与真实验证导出。

## 1. 加载实际权重

本机可用开发产物：`artifacts/c-v2/n0-smoke-seed42-v2/best.pt`，由真实 Flickr8k train 原句完成两次优化更新。它是 `development_smoke`，不是正式可比较 N0，`semantic_dataset_sha256=null` 是真实缺口，不能补造摘要。权重73,635字节，不包含基础CLIP。

```python
import torch
from fgir.text_lora import load_text_lora

encoder, checkpoint = load_text_lora(
    "artifacts/c-v2/n0-smoke-seed42-v2/best.pt",
    model_dir="models/clip-vit-base-patch32",
    device="cpu",
)
with torch.no_grad():
    query_vectors = encoder(["two dogs playing on grass"]).cpu().numpy()
print(checkpoint["text_lora_sha256"])
print(encoder.metadata)  # dict，不是旧 ClipEncoder.metadata() 方法
```

`load_text_lora`每次加载独立基础模型，不修改外部E0实例。它验证权重/tokenizer/图像分支摘要、预处理、投影维度、准确目标层、rank/alpha/dropout、参数名字/形状/dtype及有限数值。禁止交给 `load_adapter`，禁止先套旧图文MLP再套LoRA。

## 2. checkpoint结构

schema=`fgir-text-lora-v1`，原生PyTorch实现，**不属于PEFT格式**。顶层字段严格检查：

| 字段 | 含义 |
| --- | --- |
| schema_version / model_type | 固定schema及text_lora |
| method / seed | 实际N0/N1/N2/N3及整数种子 |
| freeze_policy | 视觉塔、原文本权重、投影、logit_scale全部冻结，仅白名单LoRA可训练 |
| structure | version、last_layers、rank、alpha、dropout、target_modules、完整target_paths |
| model | model_id、base_weights_sha256、tokenizer_sha256、image_encoder_sha256、projection_dimension、preprocessing、text_processing |
| state_dict | 只有准确白名单的lora_A/lora_B Tensor，不保存基础CLIP |
| semantic_dataset_sha256 | **核验训练samples文件实际字节**的SHA-256；开发原句试跑为null |
| training_config / provenance | 完整配置、源数据/特征/代码/审核/参考模型摘要、模式和数据类别 |
| selection | epoch、实际optimizer_steps、开发或验证选择依据 |

LoRA公式：`base(x) + (alpha/rank) * B(A(dropout(x)))`。默认末两层text attention的q_proj/v_proj、rank4、alpha8、dropout0；A为Kaiming初始化、B为零，因此初始等同基础CLIP。ViT-B/32下共16,384个可训练参数。

摘要定义：

- base_weights_sha256复用B的 `weights_digest`，本机为 `d5853196afb60ca6373692d9af911aacd8fa821af6d4af54720d4eb9c8f5da0a`。
- tokenizer_sha256由 `tokenizer_digest(model_dir)` 对现存tokenizer.json/tokenizer_config.json/special_tokens_map.json/vocab.json/merges.txt/added_tokens.json逐文件摘要清单计算；本机 `1071290e73855e0f1efeaae5b3f8b70847ceeb0289d05f1d9b82c8f1be13c595`。不只比较tokenizer类名。
- image_encoder_sha256由 `frozen_state_digest(model, image_only=True)` 对vision_model与visual_projection原参数的有序名称、dtype、shape、字节计算；本机 `225900851fd74e6e3dcc49c314e70974c53cb8c1632292e479d87d6be35e4e20`。
- text_lora_sha256是整个checkpoint文件的字节摘要；不可用相同method名替代。

## 3. 原始图像索引与服务边界

图像塔不变，多个N文本checkpoint可共享校验过的E0原始图片索引。D的receipt分别绑定原始image_index_sha256及该次text_lora_sha256；不要给共享图片索引本身写入某个LoRA摘要。仍需核对基础模型、预处理、图片ID/划分与维度。

N文本缓存和运行文件必须记录其实际LoRA摘要。D服务缓存键至少含基础权重和LoRA摘要，不能复用被注入过LoRA的E0 encoder。`N2-no-relation`制品method仍为N2，差异体现在training_config.include_relations=false；注册时用单独experiment_id，不能覆盖普通N2。

开发权重可以验收加载、模型隔离及真实编码。不得将其改名、补假semantic_dataset_sha256，再输出成正式N0成绩。

## 4. D给C的验证回调

默认配置引用 `fgir.text_export:validate_text_lora_checkpoint`，D已按此名称实现。C采用可配置 `module:function` 接口，正式比较前统一冻结。函数接收以下关键字参数：

```python
def validate_text_lora_checkpoint(*, checkpoint_path, model_dir, bundle, val_protocol, output):
    # D实现：严格加载真实checkpoint、生成receipt/run-v2和语义分数，
    # 再调用A的固定协议评分。这里只定义接口，不提供假分数。
    ...
```

返回dict必须有以下字段，额外保留实际receipt/run/评测文件路径和摘要便于追溯：

| 字段 | 值 |
| --- | --- |
| text_lora_sha256 | 本次candidate checkpoint实际文件摘要 |
| semantic_protocol_sha256 | 传入的val_protocol实际摘要 |
| base_weights_sha256 / tokenizer_sha256 | 与实际checkpoint及本地模型相符 |
| metrics.recall@1 / metrics.mrr | 同一固定val标准检索协议的完整排名指标，0–1 |
| metrics.attribute_binding | A语义metrics中accuracy/attribute_binding.source_macro_mean，无适用金标则null |
| metrics.relation_direction | A语义metrics中accuracy/relation_direction.source_macro_mean，无适用金标则null |

metrics对象只含上述四个键。语义诊断的源图组准确率不能冒充标准5000查询的R@1。回调不得读取test或改动训练encoder；C为它传入已保存的candidate路径，D独立加载。每次结果完整保存在validation-history.json，不只留最佳轮次。

N0以val R@1、MRR选择；N1—N3按适用绑定/方向诊断宏平均选择，要求R@1不低于同种子N0超过0.01，之后按R@1、MRR、较早epoch破平。无合格checkpoint报告failed_retrieval_guard，不为test换规则。

## 5. A/B训练输入交接

正式训练 `--samples` 使用B-V2-02核验后的不可变JSONL，C把**该文件的实际SHA-256**记为semantic_dataset_sha256，所有N方法/种子共用。其他成员文件由源数据协议及provenance分别绑定，不随便取目录名或编造摘要。

C会执行A的train结构/图片字节校验；assistant预审、draft、uncertain、争议、token超限或token相同的错误编辑不能作为默认训练负例。N1/N2只用同一组通过核验且有唯一关联unit的minimal_edit；真假追加保留作诊断，不混进该对照。组件跨度与句长需B/A审计，整个组件等同全句时，两种目标可能相同，不能据此主张粒度收益。

若监督来自automatic核验，还需 `--training-audit` 指向真实人工抽检记录。schema为fgir-semantic-training-audit-v1，字段为samples_sha256、reviewer_kind=human、reviewer、notes、sampled_sample_ids（非空实际样本ID）、unresolved_disputes=[]。这是可审计声明，工具不能代替真人或认证身份。全部人类核验的训练记录可直接使用各记录内的审核证据。
