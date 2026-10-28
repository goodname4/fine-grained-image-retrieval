# D：本地检索演示与接入

research-v2（2026-10-28）：文本LoRA接入见[D新版报告](reports/d-v2-integration.md)，
使用 `configs/demo-v2.json` 和独立 `.[demo,semantic-training]` 环境。下文旧E0/E1/E2接口仍保留。
界面按注册表检查真实制品，缺失模型只显示诊断，不作为可运行选项。

> 2026-10-27版本说明：以下安装、配置及界面对应当前E0/E1/E2。新计划的N0—N3尚未接入，接入时需同步扩展制品校验和模型缓存隔离，见[实施清单](implementation-plan.md)。不能只在下拉选项添加N名称或给旧配置换标签就声称支持新方法。

## 安装与启动

在仓库根目录安装演示依赖：

Windows 安装包解压后，双击 `preview.cmd` 直接打开离线界面预览，无需安装 Python。
真实查询应用依次双击 `install.cmd`、`start-demo.cmd`。安装要求 Python 3.11+
和联网下载依赖，会在项目内创建独立 `.venv`；数据与权重需按下文另行准备。
安装包不包含 Python、依赖、图片或模型权重，不是完整离线推理发行包。

也可手动执行：

```powershell
python -m pip install -e ".[demo]"
streamlit run app.py -- --config configs/demo.json --device cpu
```

打包命令：`python scripts/build-demo-package.py`。输出位于 `dist/`，包括 Windows
源码安装 ZIP、SHA-256 校验文件和 Python 库 wheel。ZIP 包含启动脚本、静态预览、
配置、文档、源码及 wheel；wheel 本身仅提供 `fgir` Python 库和命令行工具。

`configs/demo.json` 集中声明数据、编码器、E0 索引以及 E1/E2 的 `best.pt` 和匹配索引。图片、权重、特征和索引不提交 Git；收到这些大文件后按相同相对路径放置，或复制配置到本机路径。首次检索会加载本地 CLIP，之后由 Streamlit 缓存。

界面支持自然语言、Top-K、E0/E1/E2 并排结果、图片、ID、相似度、分阶段耗时、版本摘要和 JSON 导出。相似度只用于同一模型内排序。

## 一致性保证

`fgir.service.RetrievalService` 在加载每个模型时检查：编码器权重与预处理摘要、索引 split/kind、图库 ID 集合、维度和索引元数据。E1/E2 还检查 `index.meta.adapter_sha256 == best.pt` 摘要以及 E1/E2 method 字段。任何不一致都会拒绝查询，因此不能把 E0 图片索引与适配文本层混用。切换模型时会同时切换该模型的 checkpoint 和图片索引。

## 离线查询与性能测量

服务也可被 Python 调用：

```python
from fgir.service import RetrievalService
service = RetrievalService(".", "configs/demo.json", device="cpu")
result = service.search("two dogs playing on grass", model="E2", top_k=10)
comparison = service.compare("a cat under a table", ("E0", "E1", "E2"), top_k=5)
```

测量使用真实本地查询，首次加载和 warm p50/p95 分开记录：

```powershell
python -m fgir.benchmark --config configs/demo.json --queries examples/demo-queries.json --output artifacts/d-performance.json --repeats 5
```

该报告不包含浏览器绘制和图片解码时间；只有在真实权重、特征、索引和图片齐全后运行，不能用 synthetic 示例替代。

## 交接验收

1. E0、E1、E2 配置均能加载，且三者 `image_id` 集合与 `data/prepared/flickr8k-v1/test/images.jsonl` 一致。
2. 故意交换 E1/E2 checkpoint 或索引时，服务报 `index/checkpoint mismatch` 并拒绝查询。
3. 重启 Streamlit 后仍能离线查询；无图片时保留排名和 ID，并明确提示图片缺失。
4. 用真实查询执行 benchmark，保存环境、权重摘要、索引摘要和 p50/p95；结果交 A 做统一评测。
