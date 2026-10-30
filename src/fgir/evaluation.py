"""Frozen protocol evaluation, paired comparison, and honest error-analysis exports."""

import math
from pathlib import Path
import random
import statistics

from .contracts import CATEGORIES, image_path, text, validate_queries
from .data import file_digest
from .files import new_directory, read_json, read_jsonl, write_json, write_jsonl
from .metrics import _validate_ids, evaluate


def load_protocol(path):
    path = Path(path)
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "fgir-protocol-v1":
        raise ValueError("unsupported evaluation protocol")
    if protocol.get("task") != "text_to_image" or protocol.get("split") not in ("val", "test"):
        raise ValueError("evaluation protocol must describe val/test text-to-image retrieval")
    if protocol.get("data_kind") not in ("real", "synthetic"):
        raise ValueError("protocol must declare real or synthetic data")
    if protocol.get("annotation_scope") not in ("source_pair", "full_gallery", "candidate_pool"):
        raise ValueError("unknown annotation scope")
    manifest_digest = protocol.get("manifest_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64 or any(c not in "0123456789abcdef" for c in manifest_digest):
        raise ValueError("protocol must bind a valid manifest SHA-256")
    files = protocol.get("files")
    if not isinstance(files, dict) or set(files) != {"candidates.json", "queries.jsonl", "qrels.json"}:
        raise ValueError("protocol must bind candidates, queries and qrels")
    bound = {}
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid protocol file binding")
        location = image_path(path.parent, entry.get("path"))
        if file_digest(location) != entry.get("sha256"):
            raise ValueError(f"protocol file hash mismatch: {name}")
        bound[name] = location
    candidates = read_json(bound["candidates.json"])
    _validate_ids(candidates, "candidate gallery", require_nonempty=True)
    queries = read_jsonl(bound["queries.jsonl"])
    fine = protocol["annotation_scope"] != "source_pair"
    validate_queries(queries, candidates, require_review=fine, fine_grained=fine)
    if fine and protocol.get("independently_reviewed") is not True:
        raise ValueError("fine-grained protocol must declare independent review")
    if fine and any(row["annotation_scope"] != protocol["annotation_scope"] for row in queries):
        raise ValueError("do not mix full-gallery and pooled annotation scopes")
    qrels = read_json(bound["qrels.json"])
    if qrels != {row["query_id"]: row["relevant_image_ids"] for row in queries}:
        raise ValueError("qrels disagree with the bound queries")
    if protocol.get("candidate_count") != len(candidates) or protocol.get("query_count") != len(queries):
        raise ValueError("protocol counts disagree with bound files")
    ks = protocol.get("recall_ks")
    if not isinstance(ks, list) or not ks:
        raise ValueError("recall_ks must be a unique, non-empty list")
    if any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("recall_ks must contain positive integers")
    if len(ks) != len(set(ks)):
        raise ValueError("recall_ks must not repeat values")
    return protocol, candidates, queries, qrels


def evaluate_run(protocol_path, run_path):
    protocol, candidates, queries, qrels = load_protocol(protocol_path)
    run = read_json(run_path)
    if isinstance(run, dict) and run.get("schema_version") == "fgir-run-v2":
        from .text_artifacts import validate_run_v2
        run = validate_run_v2(run_path, protocol_path)
        run = {**run, "weights_sha256": run["base_weights_sha256"], "adapter_sha256": None}
    elif not isinstance(run, dict) or run.get("schema_version") != "fgir-run-v1":
        raise ValueError("unsupported ranking run")
    if run.get("method") in ("N0", "N1", "N2", "N3") and run.get("schema_version") != "fgir-run-v2":
        raise ValueError("N-series retrieval requires run-v2 and a bound receipt")
    if run.get("protocol_sha256") != file_digest(protocol_path):
        raise ValueError("ranking run belongs to a different protocol")
    if run.get("data_kind") != protocol["data_kind"]:
        raise ValueError("ranking run data kind differs from protocol")
    for key in ("run_id", "method", "model_id", "code_revision"):
        text(run.get(key), key)
    if type(run.get("seed")) is not int or not isinstance(run.get("config"), dict):
        raise ValueError("run must record integer seed and configuration object")
    for key in ("weights_sha256", "adapter_sha256"):
        value = run.get(key)
        if key == "adapter_sha256" and value is None:
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if run["method"] in ("E1", "E2") and run.get("adapter_sha256") is None:
        raise ValueError("adapted E1/E2 runs must identify their adapter checkpoint")
    if run["method"] == "E0" and run.get("adapter_sha256") is not None:
        raise ValueError("unadapted E0 must not declare an adapter checkpoint")
    depth = run.get("ranking_depth")
    if type(depth) is not int or not 1 <= depth <= len(candidates):
        raise ValueError("ranking_depth must be between 1 and candidate_count")
    if depth < min(max(protocol["recall_ks"]), len(candidates)):
        raise ValueError("ranking depth is insufficient for the requested Recall@K")
    rankings = run.get("rankings")
    metrics = evaluate(rankings, qrels, protocol["recall_ks"])
    gallery = set(candidates)
    for query_id, ordered in rankings.items():
        if ordered and len(ordered) != depth:
            raise ValueError(f"non-empty ranking must have the declared depth: {query_id}")
        if not set(ordered) <= gallery:
            raise ValueError(f"ranking contains images outside the fixed gallery: {query_id}")
    if protocol["annotation_scope"] == "candidate_pool":
        for row in queries:
            if not set(rankings[row["query_id"]]) <= set(row["judged_image_ids"]):
                raise ValueError(f"pooled evaluation has unjudged results; expand the common pool and re-evaluate every method: {row['query_id']}")
    per_query = []
    for row in queries:
        ordered = rankings[row["query_id"]]
        positives = set(row["relevant_image_ids"])
        first = next((rank for rank, image_id in enumerate(ordered, 1) if image_id in positives), None)
        per_query.append({"query_id": row["query_id"], "text": row["text"], "categories": row.get("categories", []),
                          "source_image_id": row.get("source_image_id", row["relevant_image_ids"][0] if protocol["annotation_scope"] == "source_pair" else None),
                          "relevant_image_ids": row["relevant_image_ids"], "ranking": ordered,
                          "first_relevant_rank": first, "reciprocal_rank": 0.0 if first is None else 1 / first,
                          **{f"recall@{k}": int(first is not None and first <= k) for k in protocol["recall_ks"]},
                          "empty_result": not ordered})
    label = "mrr" if depth == len(candidates) else f"mrr@{depth}"
    metrics[label] = metrics.pop("mrr_over_supplied_rankings")
    metrics["declared_ranking_depth"] = depth
    metrics["empty_result_count"] = sum(row["empty_result"] for row in per_query)
    by_category = {}
    for category in CATEGORIES:
        selected = [row for row in per_query if category in row["categories"]]
        by_category[category] = {"query_count": len(selected), **{
            key: sum(row["reciprocal_rank" if key == label else key] for row in selected) / len(selected) if selected else None
            for key in [*[f"recall@{k}" for k in protocol["recall_ks"]], label]}}
    return {"schema_version": "fgir-evaluation-v1", "protocol_sha256": file_digest(protocol_path),
            "run_sha256": file_digest(run_path), "data_kind": protocol["data_kind"],
            "annotation_scope": protocol["annotation_scope"], "interpretation": protocol.get("interpretation"),
            "run": {key: value for key, value in run.items() if key != "rankings"},
            "metrics": metrics, "by_category": by_category, "per_query": per_query}


def save_evaluation(protocol_path, run_path, output):
    result = evaluate_run(protocol_path, run_path)
    with new_directory(output) as target:
        write_jsonl(target / "per-query.jsonl", result["per_query"])
        write_json(target / "metrics.json", {key: value for key, value in result.items() if key != "per_query"})
    return {key: value for key, value in result.items() if key != "per_query"}


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_bootstrap(deltas, samples=1000, seed=42, groups=None):
    if type(samples) is not int or samples < 0:
        raise ValueError("bootstrap sample count must be a non-negative integer")
    if not deltas or not samples:
        return None
    generator = random.Random(seed)
    size = len(deltas)
    if groups is not None:
        if len(groups) != size:
            raise ValueError("bootstrap group count must match query count")
        clusters = {}
        for group, delta in zip(groups, deltas):
            clusters.setdefault(group, []).append(delta)
        clusters = list(clusters.values())
        means = []
        for _ in range(samples):
            selected = [clusters[generator.randrange(len(clusters))] for _ in clusters]
            means.append(sum(sum(values) for values in selected) / sum(len(values) for values in selected))
    else:
        clusters = None
        means = [sum(deltas[generator.randrange(size)] for _ in range(size)) / size for _ in range(samples)]
    return {"samples": samples, "seed": seed, "query_count": size,
            "resampling_unit": "source_image_cluster" if groups is not None else "query",
            "cluster_count": len(clusters) if clusters is not None else size,
            "mean_delta": sum(deltas) / size, "percentile_95": [_percentile(means, .025), _percentile(means, .975)],
            "limitation": "Paired resampling for this fixed run; source-image clusters keep related captions together when available. Not variability across training seeds or a claim of statistical significance."}


def compare_runs(protocol_path, run_paths, output, bootstrap_samples=1000, seed=42):
    if len(run_paths) < 2:
        raise ValueError("comparison requires a baseline and at least one other run")
    results = [evaluate_run(protocol_path, path) for path in run_paths]
    from .text_artifacts import check_v2_comparison
    check_v2_comparison([r["run"] for r in results])
    ids = [result["run"]["run_id"] for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("comparison run IDs must be unique")
    if len({r["metrics"]["declared_ranking_depth"] for r in results}) != 1:
        raise ValueError("comparison requires equal ranking depths (including MRR truncation)")
    if {r["run"]["method"] for r in results} <= {"E0", "E1", "E2"}:
        if len({(r["run"]["weights_sha256"], r["run"]["model_id"]) for r in results}) != 1:
            raise ValueError("E0/E1/E2 comparisons require the same frozen encoder weights and model/preprocessing identifier")
    baseline = results[0]
    comparisons = []
    for current in results[1:]:
        changes = []
        for before, after in zip(baseline["per_query"], current["per_query"]):
            delta = after["reciprocal_rank"] - before["reciprocal_rank"]
            changes.append({"query_id": before["query_id"], "text": before["text"],
                            "categories": before["categories"], "relevant_image_ids": before["relevant_image_ids"],
                            "baseline_ranking": before["ranking"], "candidate_ranking": after["ranking"],
                            "baseline_first_rank": before["first_relevant_rank"], "candidate_first_rank": after["first_relevant_rank"],
                            "reciprocal_rank_delta": delta,
                            "outcome": "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged_top1_success" if before["first_relevant_rank"] == 1 else "unchanged_top1_failure",
                            "analysis_status": "needs_visual_review", "error_type": None, "analysis_notes": ""})
        comparisons.append({"baseline": ids[0], "candidate": current["run"]["run_id"],
                            "metric_deltas": {key: current["metrics"][key] - value for key, value in baseline["metrics"].items()
                                              if key.startswith("recall@") or key.startswith("mrr")},
                            "paired_reciprocal_rank_bootstrap": paired_bootstrap([row["reciprocal_rank_delta"] for row in changes], bootstrap_samples, seed,
                                [row["source_image_id"] or row["query_id"] for row in baseline["per_query"]]),
                            "outcome_counts": {outcome: sum(row["outcome"] == outcome for row in changes)
                                               for outcome in ("improved", "regressed", "unchanged_top1_success", "unchanged_top1_failure")},
                            "changes": changes})
    with new_directory(output) as target:
        for i, result in enumerate(results):
            write_json(target / f"run-{i}.metrics.json", {key: value for key, value in result.items() if key != "per_query"})
        write_json(target / "comparison.json", {
            "protocol_sha256": baseline["protocol_sha256"], "data_kind": baseline["data_kind"],
            "annotation_scope": baseline["annotation_scope"],
            "comparisons": [{k: v for k, v in row.items() if k != "changes"} for row in comparisons]})
        table = ["# 检索实验比较", "", f"数据类型：{baseline['data_kind']}；标注范围：{baseline['annotation_scope']}。", "",
                 "同一冻结协议、同一候选库与相同排名深度。分类计数允许多标签重叠。", ""]
        keys = [key for key in baseline["metrics"] if key.startswith("recall@") or key.startswith("mrr")]
        table.extend(["| Run | " + " | ".join(keys) + " |", "| --- | " + " | ".join("---" for _ in keys) + " |"])
        for result in results:
            table.append("| " + result["run"]["run_id"].replace("|", "\\|") + " | " + " | ".join(f"{result['metrics'][key]:.6f}" for key in keys) + " |")
        table.extend(["", "## 分类结果", "", "| Run | 类别 | 查询数 | " + " | ".join(keys) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in keys) + " |"])
        for result in results:
            for category, values in result["by_category"].items():
                table.append("| " + result["run"]["run_id"].replace("|", "\\|") + f" | {category} | {values['query_count']} | " + " | ".join("—" if values[key] is None else f"{values[key]:.6f}" for key in keys) + " |")
        for index, comparison in enumerate(comparisons, 1):
            changes = comparison["changes"]
            write_jsonl(target / f"changes-{index}.jsonl", changes)
            selected, used = [], set()
            for category in CATEGORIES:
                for outcome in ("improved", "regressed", "unchanged_top1_success", "unchanged_top1_failure"):
                    options = [row for row in changes if category in row["categories"] and row["outcome"] == outcome and row["query_id"] not in used]
                    if options:
                        row = sorted(options, key=lambda r: (-abs(r["reciprocal_rank_delta"]), r["query_id"]))[0]
                        selected.append(row)
                        used.add(row["query_id"])
            for row in sorted(changes, key=lambda r: (-abs(r["reciprocal_rank_delta"]), r["query_id"])):
                if len(selected) >= 12:
                    break
                if row["query_id"] not in used:
                    selected.append(row)
                    used.add(row["query_id"])
            write_jsonl(target / f"cases-{index}.jsonl", selected)
            write_json(target / f"case-coverage-{index}.json", {"selected_count": len(selected), "minimum_12_met": len(selected) >= 12,
                "category_counts": {category: sum(category in row["categories"] for row in selected) for category in CATEGORIES},
                "visually_reviewed_count": 0, "interpretation": "Automatically selected candidates; human visual analysis remains outstanding."})
            table.extend(["", f"## {comparison['candidate']} 与 {comparison['baseline']}", "",
                          str(comparison["outcome_counts"]), "",
                          "案例按类别和改善/退化/保持成功/保持失败自动选取。错误原因待看图复核；无对应案例时不编造。",
                          "同一次训练的查询bootstrap不能代替多训练种子实验。"])
        (target / "report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return {"run_count": len(results), "comparison_count": len(comparisons), "output": str(output)}


def summarize_seeds(protocol_path, run_paths, output):
    """Aggregate actual runs by method; sample std is undefined for one seed."""
    if not run_paths:
        raise ValueError("at least one run is required")
    results = [evaluate_run(protocol_path, path) for path in run_paths]
    from .text_artifacts import check_v2_comparison
    check_v2_comparison([r["run"] for r in results])
    groups, ids = {}, set()
    for result in results:
        run = result["run"]
        if run["run_id"] in ids:
            raise ValueError("run IDs must be unique")
        ids.add(run["run_id"])
        key = (run.get("experiment_id", run["method"]), run["model_id"], run["weights_sha256"], result["metrics"]["declared_ranking_depth"])
        groups.setdefault(key, []).append(result)
    summary = []
    for key, selected in sorted(groups.items()):
        seeds = [result["run"]["seed"] for result in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError("same method/model/depth has repeated seeds; do not count repeats as independent runs")
        if any(result["run"]["config"] != selected[0]["run"]["config"] for result in selected):
            raise ValueError("seed aggregation requires identical seed-invariant configuration; record seed/checkpoint in top-level run fields")
        keys = [name for name in selected[0]["metrics"] if name.startswith(("recall@", "mrr"))]
        def aggregate(values):
            return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}
        summary.append({"method": key[0], "model_id": key[1], "weights_sha256": key[2], "ranking_depth": key[3],
                        "seed_count": len(seeds), "seeds": seeds, "run_ids": [r["run"]["run_id"] for r in selected],
                        "metrics": {name: aggregate([r["metrics"][name] for r in selected]) for name in keys},
                        "by_category": {category: {"query_count": selected[0]["by_category"][category]["query_count"],
                            **{name: aggregate([r["by_category"][category][name] for r in selected]) if selected[0]["by_category"][category]["query_count"] else None for name in keys}}
                            for category in CATEGORIES}})
    result = {"protocol_sha256": file_digest(protocol_path), "data_kind": results[0]["data_kind"], "groups": summary,
              "limitation": "Sample standard deviation across supplied distinct seeds. One seed has undefined std; no significance test is implied. Keep hyperparameters and selection rules fixed across seeds."}
    with new_directory(output) as target:
        write_json(target / "seeds.json", result)
    return result
