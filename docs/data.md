# 数据准备与标注规范

## 数据选择

已在本地采用Flickr8k，不随Git附带图片。来源、非商业研究/教学用途、ZIP及清单摘要已核实记录；清洗版train5998图、val/test各1000图，见[数据报告](reports/a-data-and-evaluation.md)。其他电脑执行`python -m fgir prepare-a-data`重建，B/C/D入口见[交接手册](handoff-a.md)。

若选中文，需要采用中文图文编码器与中文图文配对数据。禁止把英文文本输入中文专用模型后直接宣布支持中文，也不使用外部AI翻译接口生成数据。

## 已支持的原始清单格式

每行一个JSON对象，UTF-8编码：

```json
{"image_id":"img_0001","path":"images/img_0001.jpg","split":"train","captions":["a dog running on grass","a dog playing outside"]}
```

要求image_id全局唯一、path相对数据根目录、split为train/val/test、captions为非空文本列表。禁止把个人绝对路径写入可复现配置。

优先保留官方划分；如自行划分，需要保存随机种子与完整清单，不能在比较不同实验时重新划分。先按图片或近重复图片组划分，再展开描述。

## 本地Flickr8k导入

当前导入器支持`Flickr8k.token.txt`的`文件名#描述编号<TAB>描述`格式，以及每行一个图片文件名的三份划分列表。不自动下载数据、不生成随机划分、不验证远程来源或代替使用条款审核；先确认来源与使用条件，再填写命令中的记录项。其他镜像的CSV描述格式需要先转换，不能直接传入此命令。

以上指低层`prepare-flickr8k`。推荐一键`prepare-a-data`，使用已核实的固定镜像、ZIP摘要和版本化复核/排除记录，校验复用产物，不覆盖人工工作文件。

本地布局示例（真实文件不提交Git）：

```text
data/
  images/                         原始图片
  raw/Flickr8k.token.txt           描述文件
  raw/Flickr_8k.trainImages.txt    训练图片列表
  raw/Flickr_8k.devImages.txt      验证图片列表
  raw/Flickr_8k.testImages.txt     测试图片列表
```

以下为命令模板，`--source-url`和`--terms-note`需替换为实际核实的信息：

```bash
python -m fgir prepare-flickr8k --tokens data/raw/Flickr8k.token.txt --train-list data/raw/Flickr_8k.trainImages.txt --val-list data/raw/Flickr_8k.devImages.txt --test-list data/raw/Flickr_8k.testImages.txt --image-dir images --source-url "实际下载来源URL" --terms-note "已核实的使用条款和限制" --output data/manifest.jsonl
```

每张图片仅生成一行，`image_id`为文件名去掉扩展名，描述按编号排序。保留列表内顺序；重复图片、跨集合复用、缺失描述、空划分和ID冲突均报错。未出现在划分中的描述图片不自动加入任何集合，数量记录在回执中，需核对是否符合所用版本。导入器保留已有划分，但不能认证输入列表确实来自官方，需人工核对来源。

同时生成`data/manifest.jsonl.receipt.json`，记录描述文件、三份划分、输出清单的SHA-256及来源和条款说明。已有清单或回执会被拒绝覆盖，避免无意改变固定划分。导入只转换标注，之后必须执行图片检查。

## 数据检查命令与边界

```bash
python -m pip install -e ".[data]"
python -m fgir validate-data --manifest data/manifest.jsonl --data-root data --check-images --output artifacts/data-audit.json
```

检查器逐行汇总异常，包括非法JSON、ID重复、非法划分、空描述、不安全路径、缺失或不可读文件、同路径重复、SHA-256内容重复及其跨集合泄漏。`--check-images`使用Pillow验证文件并实际解码，检测损坏或截断；不加此参数时仅进行文件读取与摘要检查，报告明确标注未执行解码。

路径必须使用相对数据根目录的`/`形式，拒绝绝对路径、盘符、`..`和指向根目录外的符号链接。输出报告包含逐集合图片/描述数、清单摘要与逐行问题；统计是清单记录数，存在异常时不能当作已清洗数据规模。通过检查退出0，发现问题退出1，命令或文件读取错误退出2。`--output`可省略，此时仅打印JSON；报告目录需已存在。

低层`validate-data`以完整文件字节查重复，报告`near_duplicates_checked: false`。完整交接流程另用`screen-duplicates`检查EXIF校正RGB像素摘要与dHash，再由看图记录区分误报和同场景。`build-handoff`要求绑定原清单的复核记录，拒绝未审候选、解码像素重复及保留的同场景跨集合图。筛查仍可能漏检；不自动重划分，只应用明确排除理由。

```bash
python -m fgir screen-duplicates --manifest data/manifest.official.jsonl --data-root data --threshold 8 --output artifacts/new-duplicate-screen
```

21对辅助看图记录及两张训练图排除位于`configs/data/`。原始8000图清单保留，清洗版7998图在`data/prepared/flickr8k-v1/manifest.jsonl`；同图多描述读取及同场景屏蔽组见交接手册。

### 完整数据整理仍需覆盖的事项

- ID重复、缺失图片、图片不可解码、空描述。
- 相同图片跨划分、内容重复或近重复图片跨划分。
- 同图多描述在批采样中的处理。
- 训练、验证、测试图片数与描述数统计。
- 划分清单、数据摘要及处理日志保留。

## 细粒度测试子集

目标100条，类别包含color、count、action、relation，可多标签，每类目标不少于20条。四人各初标25条；A与B互审，C与D互审，每人另复核25条。A统一规范和汇总，至少两名成员独立检查每条描述与相关图片，不一致项讨论后记录。

现有100条真实源图初稿，25条AI辅助初标、75条draft、0条独立复核。范围、候选池、复核人与合并遵循[标注指南](annotation-guide.md)。下面仅列核心字段，正式记录还需这些审核字段，不能只改状态字符串声称完成。

```json
{"query_id":"fine_001","text":"a cat under a table","categories":["relation"],"relevant_image_ids":["img_0021"],"annotation_status":"reviewed"}
```

自定义查询可能匹配多个图库图片。对固定小图库尽可能完整标注；若只审核候选池，需要记录候选池构建和未审项处理，结果单独报告为候选池评测，不能声称是完整图库Recall。

细粒度测试描述不进入训练或超参数选择。可以另建独立验证子集，不能根据测试错误反复调整参数后继续报告同一测试集为未见数据。

## 困难负样本记录

```json
{"query_id":"train_001","positive_image_ids":["img_0001"],"negative_image_ids":["img_0042"],"mining_model":"baseline-version","reviewed":true}
```

仅从训练图片选择候选。相似图片可能也是正样本，需排除已知正样本并抽样复核；数量、抽样范围、假负样本比例与剩余风险写入报告。

## 当前可运行示例格式

当前检索CLI读取JSON列表，每项为`{"image_id": "...", "embedding": [1.0, 0.0]}`；查询向量为JSON数字列表。评测CLI读取两个JSON对象：query_id到有序检索ID列表，以及query_id到相关ID列表。示例见examples目录。

人工向量仅用于代码检查，不具有真实图像语义，不能用于论文或报告的模型性能结论。
