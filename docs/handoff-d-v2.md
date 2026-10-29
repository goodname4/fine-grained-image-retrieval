# 新版 D 交接与缺失输入

更新：2026-10-29，D / Codex。代码分支为 `feat/d-retrieval-app`，实现提交为 `ddc0faa`、`2f9e483`；尚未合入 `main`。本页为接收方入口，详细实现和验收见 [D 报告](reports/d-v2-integration.md)及[任务板](research-v2-tasks.md)。

## 已交付

- 接入 `fgir.text_lora.load_text_lora`，完成 receipt-v2、run-v2、严格文件身份校验、文本特征和完整排名导出。
- C 验证回调 `fgir.text_export:validate_text_lora_checkpoint` 及 export/validate CLI 已注册并测试；旧 E0/E1/E2 流程保留。
- 注册表驱动页面、独立模型缓存、性能测量、Windows 启动及打包脚本已交付。
- D-V2-01 已完成：真实两步开发重建、CPU 重载与冻结审计、5000 条 val 查询各1000图排名、E0/开发 LoRA 切换及页面10张结果图检查通过。
- D-V2-02/03 的代码可交接，正式评分与真实 N2/N3 切换验收仍 BLOCKED；未宣称完成正式 N0—N3 实验。

历史全套173项测试中172通过、1项Windows符号链接环境跳过；D专项9项在CUDA环境再次通过。[真实开发验证](reports/d-v2/real-development-verification.json)与[重建摘要](reports/d-v2/local-rebuild-summary.json)已入Git；不需要为本次文档补交重复运行全部模型测试。

## 缺少的输入及接收条件

| 提供方 | 缺少内容 | 接收条件及影响 |
| --- | --- | --- |
| B / B-V2-02 | `data/semantic/train-verified-v1/samples.jsonl` 及审核、版本和摘要记录 | 应是核验后的独立train样本；自动核验还需真实人工抽检记录。现有val draft候选不能替代。缺少时C不能运行正式共同训练池，D不能得到正式候选权重。 |
| A / A-V2-04 | `data/semantic/frozen-val-v1/protocol.json` 及其绑定samples、图库、审核/排除记录 | 真实初标和另一位真人独立复核后冻结，附成员摘要和覆盖审计。只有protocol单文件或原标准检索val协议不够。缺少时D无法验收正式语义评分与C选模回调。 |
| C / C-V2-02 | 使用上述核验数据训练的正式N0—N3 checkpoint，尤其N2/N3，以及配置、seed、训练来源、全部val选择记录和文件摘要 | 应能经 `load_text_lora` 严格重载，真实绑定训练数据及冻结val身份。D按实际交付逐个导出；N2/N3齐备后验收N2→E0→N3→N2恢复及正式模型延迟。 |
| C（原始制品核对） | `artifacts/c-v2/c-v2-development-handoff-v1.zip` | 原包SHA-256为 `5506d1a382dd1de9eaa6fc6cb3d8520e519f0fb22d2f2d230aba95306301cf79`。它只用于C原始开发产物逐字节核对，不是正式权重；不是当前开发演示运行的阻塞项。 |

当前[预检记录](reports/d-v2/preflight.json)明确缺B的train samples和A的frozen-val协议；路径存在检查不是人工质量认证。C正式checkpoint须在这些输入及正式训练齐备后交付。H-V2-01仍由真人完成，不代签、不以助手判断替代独立复核。

## 代码、文件与安装包的交接方式

接收代码：

```powershell
git fetch origin
git switch --track origin/feat/d-retrieval-app
```

已有同名本地分支时切换后用 `git pull --ff-only`，保留本地修改。拉取代码不会下载Git忽略的图片、权重、索引及大型排名文件。

本项目内的真实开发资源：

- 基础CLIP：`models/clip-vit-base-patch32/`；数据：`data/prepared/flickr8k-v1/`、`data/images/`。
- 原始val索引：`artifacts/d-v2/e0-val/index.npz`。
- 开发导出：`artifacts/d-v2/n0-development/`，含checkpoint、索引、文本特征、receipt及run。
- 独立重建来源：`artifacts/c-v2/n0-smoke-seed42-d-rebuild-v1/`，明确不是C原包；仅两步训练，`semantic_dataset_sha256=null`。

资源安装包：`dist/fgir-d-v2-ready-windows.zip`，580,962,124字节；SHA-256：`e96da9303bb0768d2f3fb4c3d84b3c4d047cba2539a194c3339d4cec50501f06`。本地独立解压、CRC校验及禁止网络连接后的CPU查询均通过。该ZIP需要另传，未上传Git或GitHub Release。

解压资源包后先运行 `install.cmd`（Python 3.11+，首次联网安装依赖），再运行 `start-v2.cmd`。本机已有环境可直接运行 `start-v2.cmd`。页面配置为 `configs/demo-v2.json`，可展示E0和N0-development；开发权重不能作为正式N0成绩。

收到正式输入后先核对版本与摘要，再调用已有验证/导出接口，保存到新版本目录，更新D-V2-02/03及任务板。正式test仍需独立冻结与选模门槛解除。
