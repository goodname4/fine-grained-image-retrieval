"""Run the paired C ablation and hand formal test rankings to A without scoring."""

from pathlib import Path

from .adaptation import export_adapter
from .data import file_digest
from .evaluation import save_evaluation
from .features import build_index
from .files import read_json, write_json
from .runs import export_run


def run_c_experiments(bundle, features, negatives, output, config_root="configs/training"):
    from .training import train_adapter, training_config
    bundle, features, output, config_root = map(Path, (bundle, features, output, config_root))
    if output.exists():
        raise ValueError(f"experiment output already exists: {output}")
    configs = {m: training_config(config_root / f"{m.lower()}.json") for m in ("E1", "E2")}
    differences = {key for key in configs["E1"] if configs["E1"][key] != configs["E2"][key]}
    if differences != {"method", "hard_weight"} or configs["E1"]["method"] != "E1" or configs["E2"]["method"] != "E2":
        raise ValueError("paired ablation configs must differ only in method and hard_weight")
    # Training never reads test features. Test export happens after both selections finish.
    output.mkdir(parents=True)
    summaries = {}
    for method in ("E1", "E2"):
        summaries[method] = train_adapter(bundle, features, config_root / f"{method.lower()}.json",
                                          output / method, negatives if method == "E2" else None)
    validation, handoff = {}, []
    for method in ("E0", "E1", "E2"):
        for split in ("val", "test"):
            target = output / "retrieval" / method / split
            adapter_sha256 = None
            if method == "E0":
                target.mkdir(parents=True)
                index = target / "index.npz"
                build_index(features / f"{split}.images.npz", index)
                texts = features / f"{split}.texts.npz"
            else:
                export_adapter(output / method / "best.pt", features / f"{split}.images.npz",
                               features / f"{split}.texts.npz", target)
                index, texts = target / "index.npz", target / "texts.npz"
                adapter_sha256 = summaries[method]["adapter_sha256"]
            run = output / "runs" / f"{method.lower()}-{split}.run.json"
            export_run(bundle / split / "protocol.json", index, texts, run, method=method,
                       seed=configs["E1"]["seed"], adapter_sha256=adapter_sha256,
                       config_extra={"training": configs.get(method),
                                     "selection_split": "val" if method != "E0" else None})
            if split == "val":
                result = save_evaluation(bundle / "val/protocol.json", run, output / "validation" / method)
                validation[method] = result["metrics"]
            else:
                handoff.append({"method": method, "run": str(run.relative_to(output).as_posix()),
                                "sha256": file_digest(run), "adapter_sha256": adapter_sha256,
                                "formal_evaluation_owner": "A", "scored_here": False})
    result = {"schema_version": "fgir-c-ablation-v1", "validation": validation,
              "training": {m: {key: value for key, value in summary.items()
                                if key not in ("model", "provenance", "environment")}
                           for m, summary in summaries.items()},
              "test_handoff": handoff,
              "interpretation": "One paired seed; validation is model-selection data, not held-out test evidence. A owns formal test evaluation."}
    write_json(output / "ablation.json", result)
    write_json(output / "handoff.json", {"test_rankings": handoff, "models": {
        m: {"checkpoint": f"{m}/best.pt", "sha256": s["adapter_sha256"]} for m, s in summaries.items()}})
    lines = ["# C：真实验证集消融记录", "", result["interpretation"], "",
             "| 方法 | 验证R@1 | 验证R@5 | 验证R@10 | 验证MRR | 最佳epoch | 训练秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in validation.items():
        summary = summaries.get(method, {})
        lines.append(f"| {method} | {metrics['recall@1']:.4f} | {metrics['recall@5']:.4f} | "
                     f"{metrics['recall@10']:.4f} | {metrics['mrr']:.4f} | {summary.get('best_epoch', '—')} | "
                     f"{summary.get('training_seconds', 0):.2f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
