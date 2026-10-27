"""Independent semantic diagnostics; no model execution or retrieval qrels reuse."""

from collections import defaultdict
import math
from pathlib import Path
import statistics
import tempfile

from .contracts import image_path, text
from .data import file_digest
from .evaluation import paired_bootstrap
from .files import new_directory, read_json, read_jsonl, write_json, write_jsonl
from .semantic_data import digest, objects, parsing_sets, validate_record


RANK_GAP_METRIC = "paraphrase/absolute_rank_gap_full_rankings"


def load_semantic_protocol(path):
    path = Path(path)
    protocol = read_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != "fgir-semantic-eval-v1":
        raise ValueError("unsupported semantic evaluation protocol")
    if protocol.get("data_kind") not in ("real", "synthetic") or protocol.get("split") not in ("val", "test"):
        raise ValueError("semantic protocol needs data_kind and val/test split")
    if (protocol.get("tie_credit") != .5 or protocol.get("aggregation") != "source_image_macro"
            or protocol.get("normalization") != "unicode-casefold-whitespace-v1"
            or protocol.get("independently_reviewed") is not True):
        raise ValueError("unsupported semantic scoring or review policy")
    digest(protocol.get("source_manifest_sha256"), "protocol source manifest")
    text(protocol.get("test_access_notes"), "test access disclosure")
    names = {"samples.jsonl", "candidates.json", "audit.json", "source-context.json"}
    files = protocol.get("files")
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError("semantic protocol must bind samples, gallery, audit and source context")
    bound = {}
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid semantic file binding")
        bound[name] = image_path(path.parent, entry.get("path"))
        if file_digest(bound[name]) != entry.get("sha256"):
            raise ValueError(f"semantic protocol file hash mismatch: {name}")
    rows = read_jsonl(bound["samples.jsonl"])
    context = read_json(bound["source-context.json"])
    if context.get("manifest_sha256") != protocol["source_manifest_sha256"] or context.get("data_kind") != protocol["data_kind"]:
        raise ValueError("source context manifest mismatch")
    candidates = read_json(bound["candidates.json"])
    if (not isinstance(candidates, list) or not candidates or any(not isinstance(x, str) or not x for x in candidates)
            or len(candidates) != len(set(candidates))):
        raise ValueError("invalid semantic candidate gallery")
    ids = set()
    for row in rows:
        validate_record(row, context, "eval")
        if row["sample_id"] in ids or row["split"] != protocol["split"] or row["source_image_id"] not in candidates:
            raise ValueError("duplicate sample or source outside protocol split/gallery")
        ids.add(row["sample_id"])
    if (protocol.get("sample_count") != len(rows) or protocol.get("candidate_count") != len(candidates)
            or protocol.get("source_image_count") != len({r["source_image_id"] for r in rows})):
        raise ValueError("semantic protocol counts disagree")
    ks = protocol.get("recall_ks")
    if not isinstance(ks, list) or not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("invalid semantic recall_ks")
    return protocol, rows, candidates


def _prf(tp, predicted, gold):
    return {"true_positive": tp, "predicted": predicted, "gold": gold,
            "precision": tp / predicted if predicted else (0.0 if gold else None),
            "recall": tp / gold if gold else (0.0 if predicted else None),
            "f1": 2 * tp / (predicted + gold) if predicted + gold else None}


def parsing_scores(gold, prediction):
    expected = parsing_sets(gold, gold["text"], "gold parsing")
    actual = parsing_sets(prediction, gold["text"], "prediction parsing")
    return {name: _prf(len(values & actual[name]), len(actual[name]), len(values)) for name, values in expected.items()}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("semantic scores must be finite numbers, not booleans")
    return value


def _credit(delta):
    return 1.0 if delta > 0 else (0.0 if delta < 0 else .5)


def _summary(items, samples, seed):
    """First average inside each source image, then resample image means."""
    groups = defaultdict(list)
    for group, value in items:
        groups[group].append(value)
    means = [sum(values) / len(values) for values in groups.values()]
    bootstrap = paired_bootstrap(means, samples, seed, groups=list(groups)) if means else None
    if bootstrap is not None:
        bootstrap["source_macro_mean"] = bootstrap.pop("mean_delta")
        bootstrap["source_image_count"] = bootstrap.pop("query_count")
    return {"item_count": len(items), "source_image_count": len(groups),
            "source_macro_mean": sum(means) / len(means) if means else None,
            "bootstrap": bootstrap}


def evaluate_semantics(protocol_path, run_path, output, bootstrap_samples=1000, seed=42):
    if type(bootstrap_samples) is not int or bootstrap_samples < 0 or type(seed) is not int:
        raise ValueError("bootstrap_samples must be nonnegative and seed an integer")
    protocol, gold, candidates = load_semantic_protocol(protocol_path)
    run = read_json(run_path)
    if not isinstance(run, dict) or run.get("schema_version") != "fgir-semantic-scores-v1":
        raise ValueError("unsupported semantic score run")
    if run.get("protocol_sha256") != file_digest(protocol_path) or run.get("data_kind") != protocol["data_kind"]:
        raise ValueError("semantic run protocol/data kind mismatch")
    for field in ("run_id", "method", "code_revision"):
        text(run.get(field), field)
    if type(run.get("seed")) is not int:
        raise ValueError("semantic run must record an integer training seed")
    model = run.get("model")
    if not isinstance(model, dict) or model.get("model_type") not in ("raw_clip", "feature_mlp", "text_lora"):
        raise ValueError("declare semantic scoring model identity")
    text(model.get("model_id"), "model_id")
    for field in ("base_weights_sha256", "tokenizer_sha256"):
        digest(model.get(field), field)
    if ((run["method"] == "E0" and model["model_type"] != "raw_clip")
            or (run["method"] in ("E1", "E2") and model["model_type"] != "feature_mlp")):
        raise ValueError("E method/model_type mismatch")
    if model["model_type"] == "text_lora":
        if run["method"] not in ("N0", "N1", "N2", "N3"):
            raise ValueError("text LoRA run must identify N0/N1/N2/N3")
        for field in ("text_lora_sha256", "retrieval_receipt_sha256", "semantic_dataset_sha256"):
            digest(model.get(field), field)
        if model.get("adapter_sha256") is not None:
            raise ValueError("text LoRA must not declare a feature MLP adapter")
    elif run["method"] in ("N0", "N1", "N2", "N3"):
        raise ValueError("N methods require text_lora identity")
    elif model["model_type"] == "feature_mlp":
        digest(model.get("adapter_sha256"), "adapter_sha256")
        if model.get("text_lora_sha256") is not None:
            raise ValueError("feature MLP must not declare text LoRA")
    if model["model_type"] == "raw_clip" and (model.get("text_lora_sha256") is not None or model.get("adapter_sha256") is not None):
        raise ValueError("raw CLIP must not declare adapted weights")
    predictions = objects(run.get("predictions"), "predictions")
    by_id = {}
    for row in predictions:
        sid = text(row.get("sample_id"), "prediction sample_id")
        if sid in by_id:
            raise ValueError("duplicate semantic prediction")
        by_id[sid] = row
    if set(by_id) != {r["sample_id"] for r in gold}:
        raise ValueError("semantic predictions must cover exactly all source samples")
    has_parsing = ["parsing" in row for row in predictions]
    has_rankings = ["rankings" in row for row in predictions]
    if any(has_parsing) and not all(has_parsing):
        raise ValueError("parsing predictions must cover every sample or be omitted entirely")
    if any(has_rankings) and not all(has_rankings):
        raise ValueError("rankings must cover every sample or be omitted entirely")
    depth = run.get("ranking_depth")
    if any(has_rankings) and (type(depth) is not int or not min(max(protocol["recall_ks"]), len(candidates)) <= depth <= len(candidates)):
        raise ValueError("semantic ranking depth insufficient or invalid")
    metrics, per_sample = defaultdict(list), []
    parse_totals = defaultdict(lambda: [0, 0, 0])
    for row in gold:
        prediction = by_id[row["sample_id"]]
        variants = {v["id"]: v for v in row["variants"]}
        scores = prediction.get("scores")
        if not isinstance(scores, dict) or set(scores) != {"original", *variants}:
            raise ValueError("scores must cover original and exactly all variants")
        for value in scores.values():
            _number(value)
        source = row["source_image_id"]
        detail = {"sample_id": row["sample_id"], "source_image_id": source, "group_id": row["group_id"],
                  "scene_group_id": row["scene_group_id"], "comparisons": [], "addition_controls": [],
                  "rankings": None, "parsing": None}
        addition_pairs = defaultdict(lambda: defaultdict(list))
        for vid, variant in variants.items():
            delta = scores["original"] - scores[vid]
            entry = {"variant_id": vid, "type": variant["type"], "phenomenon": variant["phenomenon"],
                     "original_score": scores["original"], "variant_score": scores[vid],
                     "original_minus_variant": delta, "credit": None,
                     "original_word_count": len(row["text"].split()), "variant_word_count": len(variant["text"].split())}
            if variant["type"] in ("minimal_edit", "false_addition"):
                entry["credit"] = _credit(delta)
                metrics[f"accuracy/{variant['phenomenon']}"].append((source, entry["credit"]))
            if variant["type"].endswith("addition"):
                metrics[f"score_delta/{variant['type']}_minus_original"].append((source, -delta))
                addition_pairs[variant["control_pair_id"]][variant["type"]].append(vid)
            detail["comparisons"].append(entry)
        for pair_id, pair in addition_pairs.items():
            if len(pair["true_addition"]) != 1 or len(pair["false_addition"]) != 1:
                detail["addition_controls"].append({"control_pair_id": pair_id, "status": "missing_or_ambiguous_pair"})
                continue
            true_id, false_id = pair["true_addition"][0], pair["false_addition"][0]
            matched = len(variants[true_id]["text"].split()) == len(variants[false_id]["text"].split())
            delta = scores[true_id] - scores[false_id]
            detail["addition_controls"].append({"control_pair_id": pair_id, "true_id": true_id, "false_id": false_id,
                                              "status": "length_matched" if matched else "length_mismatch",
                                              "true_minus_false": delta, "credit": _credit(delta)})
            metrics["accuracy/addition_true_vs_false"].append((source, _credit(delta)))
            if matched:
                metrics["accuracy/addition_length_matched"].append((source, _credit(delta)))
        if all(has_rankings):
            equivalent = {"original", *[v["id"] for v in variants.values() if v["type"] == "paraphrase"]}
            rankings = prediction["rankings"]
            if not isinstance(rankings, dict) or set(rankings) != equivalent:
                raise ValueError("rankings must contain original and exactly the equivalent variants; foils have no retrieval qrels")
            first = {}
            for qid, ordered in rankings.items():
                if (not isinstance(ordered, list) or any(not isinstance(x, str) for x in ordered)
                        or len(set(ordered)) != len(ordered) or not set(ordered) <= set(candidates)
                        or (ordered and len(ordered) != depth)):
                    raise ValueError("invalid, duplicate, unknown or inconsistent-depth semantic ranking")
                first[qid] = ordered.index(source) + 1 if source in ordered else None
            detail["rankings"] = {"first_source_rank": first, "empty_result_count": sum(not v for v in rankings.values())}
            para_ids = equivalent - {"original"}
            rank_gaps = {}
            if para_ids:
                for k in protocol["recall_ks"]:
                    metrics[f"paraphrase/joint_recall@{k}"].append((source, float(all(rank is not None and rank <= k for rank in first.values()))))
                    metrics[f"paraphrase/original_recall@{k}"].append((source, float(first["original"] is not None and first["original"] <= k)))
                    metrics[f"paraphrase/variant_recall@{k}"].extend((source, float(first[vid] is not None and first[vid] <= k)) for vid in para_ids)
                if depth == len(candidates):
                    for vid in para_ids:
                        # Explicit empty returns count as failed retrieval, never rank zero.
                        if first["original"] is not None and first[vid] is not None:
                            rank_gaps[vid] = abs(first[vid] - first["original"])
                            metrics[RANK_GAP_METRIC].append((source, rank_gaps[vid]))
            detail["rankings"]["absolute_rank_gaps_full_rankings"] = rank_gaps
            detail["rankings"]["rank_gap_eligible_pairs"] = len(rank_gaps)
        if all(has_parsing):
            detail["parsing"] = parsing_scores(row, prediction["parsing"])
            for name, values in detail["parsing"].items():
                totals = parse_totals[name]
                for i, key in enumerate(("true_positive", "predicted", "gold")):
                    totals[i] += values[key]
                if values["f1"] is not None:
                    metrics[f"parsing_source_macro/{name}_f1"].append((source, values["f1"]))
        per_sample.append(detail)
    required = ["accuracy/attribute_binding", "accuracy/relation_direction", "accuracy/false_addition",
                "score_delta/true_addition_minus_original", "score_delta/false_addition_minus_original",
                "accuracy/addition_true_vs_false", "accuracy/addition_length_matched",
                *[f"paraphrase/{kind}_recall@{k}" for kind in ("joint", "original", "variant") for k in protocol["recall_ks"]],
                "paraphrase/absolute_rank_gap_full_rankings"]
    summary = {name: _summary(metrics[name], bootstrap_samples, seed) for name in sorted(set(metrics) | set(required))}
    groups = defaultdict(list)
    for entry in per_sample:
        groups[entry["source_image_id"]].append(entry["sample_id"])
    result = {"schema_version": "fgir-semantic-evaluation-v1", "protocol_sha256": file_digest(protocol_path),
              "run_sha256": file_digest(run_path), "data_kind": protocol["data_kind"], "split": protocol["split"],
              "run": {k: v for k, v in run.items() if k != "predictions"}, "sample_count": len(gold), "source_image_count": len(groups),
              "metrics": summary, "parsing_micro": {name: _prf(*counts) for name, counts in parse_totals.items()} if all(has_parsing) else None,
              "coverage": {"parsing_supplied": all(has_parsing), "rankings_supplied": all(has_rankings),
                           "addition_control_status": {status: sum(c["status"] == status for r in per_sample for c in r["addition_controls"])
                                                       for status in ("length_matched", "length_mismatch", "missing_or_ambiguous_pair")}},
              "limitations": ["Source-pair relevance only; semantic matching accuracy is not whole-gallery Recall@K.",
                              "Null metrics mean absent supervision/predictions, not zero accuracy.",
                              "Image macro bootstrap measures fixed-run source uncertainty, not training seed variability.",
                              "Rank gap excludes empty returns and requires complete gallery rankings; joint Recall includes failures.",
                              "Model hashes identify declared artifacts; scoring does not load weights or certify the pending run-v2 receipt chain."]}
    with new_directory(output) as target:
        write_json(target / "metrics.json", result)
        write_jsonl(target / "per-sample.jsonl", per_sample)
        write_jsonl(target / "per-source.jsonl", [{"source_image_id": source, "sample_ids": ids,
                   "metrics": {name: sum(v for s, v in values if s == source) / sum(s == source for s, _ in values)
                               for name, values in metrics.items() if any(s == source for s, _ in values)}} for source, ids in groups.items()])
    return result


def _paired_rank_gaps(baseline, candidate):
    """Pair eligible (sample, variant) IDs before averaging inside a source."""
    def eligible(details):
        return {(sid, vid): (row["source_image_id"], gap)
                for sid, row in details.items() if row["rankings"] is not None
                for vid, gap in row["rankings"]["absolute_rank_gaps_full_rankings"].items()}

    left, right = eligible(baseline), eligible(candidate)
    shared = sorted(left.keys() & right.keys())
    left_only, right_only = sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())
    values = []
    for key in shared:
        if left[key][0] != right[key][0]:
            raise ValueError("paired rank-gap source image mismatch")
        values.append((left[key][0], right[key][1] - left[key][1]))

    def ids(keys, pairs):
        return [{"sample_id": sid, "variant_id": vid, "source_image_id": pairs[(sid, vid)][0]} for sid, vid in keys]

    return values, {"shared_pair_count": len(shared), "baseline_only_pair_count": len(left_only),
                    "candidate_only_pair_count": len(right_only), "shared_pairs": ids(shared, left),
                    "baseline_only_pairs": ids(left_only, left), "candidate_only_pairs": ids(right_only, right)}


def compare_semantics(protocol_path, run_paths, output, bootstrap_samples=1000, seed=42):
    """Paired comparisons use the same seed; seed SD is reported separately."""
    if len(run_paths) < 2:
        raise ValueError("semantic comparison needs a baseline and another method")
    results, sources, details = [], [], []
    with tempfile.TemporaryDirectory(prefix="fgir-semantic-compare-") as temporary:
        for i, path in enumerate(run_paths):
            target = Path(temporary) / str(i)
            results.append(evaluate_semantics(protocol_path, path, target, bootstrap_samples, seed))
            sources.append({r["source_image_id"]: r["metrics"] for r in read_jsonl(target / "per-source.jsonl")})
            details.append({r["sample_id"]: r for r in read_jsonl(target / "per-sample.jsonl")})
    runs = [r["run"] for r in results]
    recall_ks = read_json(protocol_path)["recall_ks"]
    case_k = 5 if 5 in recall_ks else min(recall_ks)
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("comparison run IDs must be unique")
    keys = [(r["method"], r["seed"]) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate method/seed in semantic comparison")
    for field in ("base_weights_sha256", "tokenizer_sha256"):
        if len({r["model"][field] for r in runs}) != 1:
            raise ValueError(f"comparison model mismatch: {field}")
    methods = defaultdict(dict)
    for i, run in enumerate(runs):
        methods[run["method"]][run["seed"]] = i
    baseline = runs[0]["method"]
    if len(methods) < 2 or any(set(by_seed) != set(methods[baseline]) for by_seed in methods.values()):
        raise ValueError("methods must have the same paired seeds and include a different method")
    comparisons, cases = [], []
    for method, by_seed in methods.items():
        if method == baseline:
            continue
        for training_seed, index in sorted(by_seed.items()):
            base = methods[baseline][training_seed]
            if (results[base]["coverage"]["parsing_supplied"] != results[index]["coverage"]["parsing_supplied"]
                    or results[base]["coverage"]["rankings_supplied"] != results[index]["coverage"]["rankings_supplied"]
                    or runs[base].get("ranking_depth") != runs[index].get("ranking_depth")):
                raise ValueError("comparison prediction coverage or ranking depth differs")
            if set(sources[base]) != set(sources[index]):
                raise ValueError("comparison source-image groups disagree")
            deltas = defaultdict(list)
            unpaired = defaultdict(lambda: {"baseline_only": [], "candidate_only": []})
            for source, metrics in sources[base].items():
                other = sources[index][source]
                for name in metrics.keys() - other.keys():
                    unpaired[name]["baseline_only"].append(source)
                for name in other.keys() - metrics.keys():
                    unpaired[name]["candidate_only"].append(source)
                for name in metrics.keys() & other.keys():
                    if name != RANK_GAP_METRIC:
                        deltas[name].append((source, other[name] - metrics[name]))
            # A shared source alone is insufficient when different paraphrases
            # failed. Keep descriptive run means separate from paired changes.
            deltas[RANK_GAP_METRIC], rank_gap_pairing = _paired_rank_gaps(details[base], details[index])
            for side in ("baseline_only", "candidate_only"):
                excluded = rank_gap_pairing[f"{side}_pairs"]
                if excluded:
                    unpaired[RANK_GAP_METRIC][side] = sorted({item["source_image_id"] for item in excluded})
            comparisons.append({"baseline_run_id": runs[base]["run_id"], "candidate_run_id": runs[index]["run_id"],
                                "training_seed": training_seed,
                                "rank_gap_pairing": rank_gap_pairing,
                                "unpaired_metric_sources": dict(unpaired),
                                "source_macro_deltas": {name: _summary(values, bootstrap_samples, seed) for name, values in deltas.items()}})
            for sid, before in details[base].items():
                after = details[index][sid]
                by_variant = {v["variant_id"]: v for v in after["comparisons"]}
                for item in before["comparisons"]:
                    candidate = by_variant[item["variant_id"]]
                    if item["credit"] is None:
                        continue
                    change = candidate["credit"] - item["credit"]
                    cases.append({"sample_id": sid, "variant_id": item["variant_id"], "source_image_id": before["source_image_id"],
                                  "phenomenon": item["phenomenon"], "baseline_run_id": runs[base]["run_id"],
                                  "candidate_run_id": runs[index]["run_id"], "training_seed": training_seed,
                                  "outcome": "improved" if change > 0 else ("regressed" if change < 0 else "unchanged"),
                                  "baseline": item, "candidate": candidate, "review_status": "unreviewed", "cause": None})
                if before["rankings"] is not None:
                    left, right = before["rankings"]["first_source_rank"], after["rankings"]["first_source_rank"]
                    for vid in sorted(set(left) - {"original"}):
                        credits = [float(all(values[qid] is not None and values[qid] <= case_k for qid in ("original", vid))) for values in (left, right)]
                        delta = credits[1] - credits[0]
                        cases.append({"sample_id": sid, "variant_id": vid, "source_image_id": before["source_image_id"],
                                      "phenomenon": "paraphrase", "baseline_run_id": runs[base]["run_id"], "candidate_run_id": runs[index]["run_id"],
                                      "training_seed": training_seed, "case_metric": f"pair_joint_recall@{case_k}",
                                      "outcome": "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged"),
                                      "baseline": {"credit": credits[0], "first_source_rank": left},
                                      "candidate": {"credit": credits[1], "first_source_rank": right},
                                      "review_status": "unreviewed", "cause": None})
    seed_summary = {}
    for method, by_seed in methods.items():
        all_names = set.intersection(*[set(results[i]["metrics"]) for i in by_seed.values()])
        seed_summary[method] = {}
        for name in sorted(all_names):
            values = [results[i]["metrics"][name]["source_macro_mean"] for _, i in sorted(by_seed.items())]
            if any(value is None for value in values):
                seed_summary[method][name] = {"mean": None, "sample_sd": None, "reason": "missing metric coverage"}
                continue
            seed_summary[method][name] = {"seeds": sorted(by_seed), "values": values,
                                         "mean": statistics.mean(values),
                                         "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
    result = {"schema_version": "fgir-semantic-comparison-v1", "data_kind": results[0]["data_kind"],
              "protocol_sha256": file_digest(protocol_path), "baseline_method": baseline,
              "runs": [{"run_id": run["run_id"], "run_sha256": result["run_sha256"]} for run, result in zip(runs, results)],
              "comparisons": comparisons, "seed_summary": seed_summary,
              "case_count": len(cases), "limitation": "Causes require image review. Source-image bootstrap and training-seed SD measure different uncertainty. Lower score/rank gaps are better; accuracy and recall are higher-is-better."}
    with new_directory(output) as target:
        write_json(target / "comparison.json", result)
        if cases:
            write_jsonl(target / "cases.unreviewed.jsonl", cases)
        else:
            (target / "cases.unreviewed.jsonl").write_text("", encoding="utf-8")
    return result
