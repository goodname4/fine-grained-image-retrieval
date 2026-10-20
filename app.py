"""Streamlit demo for the local retrieval service.

Run with ``streamlit run app.py -- --config configs/demo.json``.
"""
import argparse
import json
from pathlib import Path

import streamlit as st

from fgir.service import RetrievalService


@st.cache_resource(show_spinner=False)
def get_service(root, config, device):
    return RetrievalService(root, config, device=device)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/demo.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="FGIR 检索演示", layout="wide")
    st.title("Fine-Grained Image Retrieval")
    st.caption("本地编码、适配与精确余弦检索；不同模型分数仅用于各自排序。")
    try:
        config_path = Path(args.root, args.config).resolve()
        service = get_service(str(Path(args.root).resolve()), str(config_path), args.device)
    except (OSError, ValueError) as exc:
        st.error(f"配置加载失败：{exc}")
        return
    if st.sidebar.button("重新加载配置和模型"):
        get_service.clear()
        st.session_state.pop("results", None)
        st.rerun()
    st.sidebar.caption("更换本地文件后请重新加载。首次检索包含模型加载耗时。")
    query = st.text_input("自然语言查询", "two dogs playing on grass")
    models = st.multiselect("模型", ["E0", "E1", "E2"], ["E0", "E1", "E2"])
    top_k = st.slider("Top-K", 1, 20, 5)
    if st.button("检索") and models:
        st.session_state.pop("results", None)
        try:
            with st.spinner("正在本地检索…"):
                st.session_state["results"] = service.compare(query, models, top_k)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(f"无法完成检索：{exc}。请核对 docs/demo.md 中的文件清单与重建步骤。")
    if "results" in st.session_state:
            results = st.session_state["results"]
            st.caption(f"本次结果：{next(iter(results.values()))['query']}")
            from fgir.service import export_results
            st.download_button("导出本次结果 JSON", export_results(results),
                               "retrieval-results.json", "application/json")
            columns = st.columns(len(results))
            for column, model in zip(columns, results):
                result = results[model]
                with column:
                    st.subheader(model)
                    st.caption(f"{result['elapsed_ms']:.1f} ms")
                    with st.expander("分阶段耗时与版本"):
                        st.json({"timing_ms": result["timing_ms"], "provenance": result["provenance"]})
                    for rank, hit in enumerate(result["hits"], 1):
                        st.write(f"{rank}. `{hit['image_id']}`  {hit['score']:.4f}")
                        if hit.get("path") and Path(hit["path"]).is_file():
                            try:
                                st.image(hit["path"], use_container_width=True)
                            except (OSError, ValueError) as exc:
                                st.warning(f"图片无法显示：{exc}")
                        else:
                            st.warning("本地图片缺失，请补齐数据目录。")


if __name__ == "__main__":
    main()
