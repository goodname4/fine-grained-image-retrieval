"""Training-only hard negative mining with explicit false-negative guards."""

from pathlib import Path

from .data import file_digest
from .files import read_json, read_jsonl, write_json, write_jsonl

MINING_SCHEMA = "fgir-negatives-v1"


def exclusion_groups(bundle):
    """Union near-identical/same-scene train frames from the recorded review."""
    bundle = Path(bundle)
    dataset = read_json(bundle / "dataset.json")
    review_path = bundle / "duplicate-review.json"
    if file_digest(review_path) != dataset["files"].get("duplicate-review.json"):
        raise ValueError("duplicate-review hash mismatch")
    review = read_json(review_path)
    parent = {}

    def find(image_id):
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    for pair in review["pairs"]:
        left, right = pair["left"], pair["right"]
        if pair["decision"] in ("same_scene", "identical"):
            for name in (left, right):
                parent.setdefault(name, name)
            a, b = find(left), find(right)
            parent[max(a, b)] = min(a, b)
    return {name: find(name) for name in parent}


def mine_negatives(bundle, text_features_path, image_features_path, output,
                   top_m=50, per_query=5, seed=42, review_sample=100):
    """Mine hard negatives per train query from the train gallery only.

    Excludes the source image, identical-caption images (possible positives) and
    same-scene exclusion-group frames (unverifiable positives). Identical-caption
    and group images are recorded but claimed as neither positive nor negative.
    """
    import numpy
    from .encoding import load_features
    from .features import normalize_rows

    if type(top_m) is not int or top_m < 1 or type(per_query) is not int or per_query < 1:
        raise ValueError("top_m and per_query must be positive integers")
    if per_query > top_m:
        raise ValueError("per_query must not exceed top_m")
    if type(review_sample) is not int or review_sample < 0:
        raise ValueError("review_sample must be a non-negative integer")
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    pairs = read_jsonl(Path(bundle) / "train" / "pairs.jsonl")
    query_ids, _, text_meta = load_features(text_features_path)
    gallery_ids, gallery_raw, image_meta = load_features(image_features_path)
    if text_meta.get("weights_sha256") != image_meta.get("weights_sha256"):
        raise ValueError("text and image features come from different weights")
    if sorted(query_ids) != sorted(row["query_id"] for row in pairs):
        raise ValueError("text features do not cover the training pairs exactly")
    if sorted(gallery_ids) != sorted(row["image_id"] for row in read_jsonl(Path(bundle) / "train" / "images.jsonl")):
        raise ValueError("image features do not cover the training images exactly")
    groups = exclusion_groups(bundle)
    id_position = {image_id: index for index, image_id in enumerate(gallery_ids)}
    groups = {name: group for name, group in groups.items() if name in id_position}
    caption_index = {}
    for row in pairs:
        caption_index.setdefault(" ".join(row["text"].lower().split()), set()).add(row["image_id"])
    gallery = normalize_rows(gallery_raw.astype(numpy.float32))
    query_by_id = {row["query_id"]: row for row in pairs}
    id_rank = numpy.empty(len(gallery_ids), dtype=numpy.int64)
    id_rank[numpy.argsort(numpy.array(gallery_ids, dtype="S"), kind="stable")] = numpy.arange(len(gallery_ids))
    text_vectors = normalize_rows(load_features(text_features_path)[1].astype(numpy.float32))
    records = []
    exclusion_counts = {"source": 0, "identical_caption": 0, "same_scene_group": 0}
    for start in range(0, len(query_ids), 2048):
        chunk = text_vectors[start:start + 2048]
        scores = chunk @ gallery.T
        for row_scores, query_id in zip(scores, query_ids[start:start + 2048]):
            pair = query_by_id[query_id]
            source = pair["image_id"]
            group = groups.get(source)
            excluded = {source}
            reasons = {source: "source"}
            normalized = " ".join(pair["text"].lower().split())
            for other in caption_index.get(normalized, ()):
                if other != source:
                    excluded.add(other)
                    reasons[other] = "identical_caption"
            if group is not None:
                for other, other_group in groups.items():
                    if other_group == group and other != source:
                        excluded.add(other)
                        reasons[other] = "same_scene_group"
            masked = row_scores.copy()
            for image_id in excluded:
                masked[id_position[image_id]] = -numpy.inf
            order = numpy.lexsort((id_rank, -masked))
            candidates = [gallery_ids[index] for index in order[:top_m] if numpy.isfinite(masked[index])]
            negatives = candidates[:per_query]
            if not negatives:
                raise ValueError(f"no eligible negative candidates for {query_id}")
            for reason in set(reasons.values()):
                exclusion_counts[reason] += len([name for name in excluded if reasons[name] == reason])
            records.append({
                "query_id": query_id, "text": pair["text"],
                "positive_image_ids": [source], "negative_image_ids": negatives,
                "negative_scores": [float(row_scores[id_position[name]]) for name in negatives],
                "excluded_image_ids": reasons,
                "mining_model": f"{image_meta.get('model_id')} (weights {image_meta.get('weights_sha256', '')[:12]})",
                "mining_parameters": {"top_m": top_m, "per_query": per_query, "seed": seed,
                                      "similarity": "cosine", "gallery": "train only"},
                "reviewed": False,
            })
    write_jsonl(output, records)
    sample_path = output.with_suffix(".review-sample.jsonl")
    generator = numpy.random.default_rng(seed)
    indices = sorted(generator.choice(len(records), size=min(review_sample, len(records)), replace=False).tolist())
    write_jsonl(sample_path, [{
        "query_id": records[index]["query_id"], "text": records[index]["text"],
        "positive_image_ids": records[index]["positive_image_ids"],
        "negative_image_ids": records[index]["negative_image_ids"],
        "review_instruction": "看图判断每条负样本是否真的不相关；把假负样本写入 false_negative_image_ids 并注明理由",
        "false_negative_image_ids": [], "decision": None, "reviewer": None,
    } for index in indices])
    report = {
        "schema_version": MINING_SCHEMA, "negatives": str(output),
        "bundle": str(bundle), "query_count": len(records),
        "negative_count": sum(len(row["negative_image_ids"]) for row in records),
        "mining_model": image_meta.get("model_id"), "weights_sha256": image_meta.get("weights_sha256"),
        "parameters": {"top_m": top_m, "per_query": per_query, "seed": seed,
                       "review_sample": review_sample, "similarity": "cosine"},
        "exclusions": exclusion_counts,
        "false_negative_controls": [
            "只用train图库挖掘，绝不触及val/test",
            "源图永远是正样本且不进入负样本",
            "与查询文本完全相同的其他训练图视为可能正样本，整条排除",
            "duplicate-review判定的同场景/相同帧（negative_exclusion_group_id同组）整条排除",
            "相关性未穷尽：相似图片仍可能是未标注正样本，结构过滤无法消除假负样本",
        ],
        "review_sample_file": str(sample_path),
        "reviewed_query_count": 0,
        "limitation": "结构检查不能证明未标注相似图是真负样本；抽样复核记录须由人工看图填写后合并回主文件。",
    }
    report_path = output.with_suffix(".report.json")
    write_json(report_path, report)
    return {**report, "report": str(report_path)}
