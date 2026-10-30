"""Measure actual local queries, with cold loads separate from warm percentiles."""

import argparse
import json
import platform
import sys
from pathlib import Path
from importlib.metadata import version

from .service import RetrievalService


def benchmark(service, queries, models, repeats=5, top_k=10):
    import numpy as np
    import torch
    from .files import read_json
    if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("queries must be a non-empty list of non-empty strings")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not models or len(set(models)) != len(models):
        raise ValueError("at least one distinct registered model is required")
    dataset = read_json(service._path(service.config["bundle"]) / "dataset.json")
    report = {"schema_version": "fgir-query-benchmark-v1", "synthetic": dataset.get("data_kind", "real") == "synthetic",
              "data_kind": dataset.get("data_kind", "real"),
              "python": sys.version, "platform": platform.platform(), "device_requested": service.device,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "packages": {name: version(name) for name in ("numpy", "torch", "transformers")},
              "scope": "local service only; excludes browser rendering and image decoding",
              "queries": queries, "repeats": repeats, "top_k": top_k, "models": {}}
    for model in models:
        cold = service.search(queries[0], model, top_k)
        for query in queries:
            service.search(query, model, top_k)
        samples = [service.search(query, model, top_k) for _ in range(repeats) for query in queries]
        report["models"][model] = {
            "initial_call_ms": cold["timing_ms"], "provenance": cold["provenance"],
            "warmup_count": len(queries), "requested_top_k": top_k,
            "effective_top_k": len(cold["hits"]),
            "sample_count": len(samples), "samples_ms": [s["timing_ms"] for s in samples],
            "warm_ms": {stage: {"p50": float(np.percentile([s["timing_ms"][stage] for s in samples], 50)),
                                 "p95": float(np.percentile([s["timing_ms"][stage] for s in samples], 95))}
                        for stage in ("encoding", "ranking", "end_to_end")}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/demo.json")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--queries", required=True, help="JSON list of real query strings")
    parser.add_argument("--models", nargs="+", help="experiment IDs from the configured registry")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists():
            raise ValueError("output exists; select a new performance report path")
        queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        if not isinstance(queries, list):
            raise ValueError("queries JSON must contain a list")
        service = RetrievalService(args.root, Path(args.root, args.config), args.device)
        result = benchmark(service, queries, args.models or list(service.available_models(validate=False)[0]), args.repeats, args.top_k)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(output)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
