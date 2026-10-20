# D：本地检索演示与接入

## 安装与启动

在仓库根目录安装演示依赖：

```powershell
python -m pip install -e ".[demo]"
streamlit run app.py -- --config configs/demo.json --device cpu
```

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
