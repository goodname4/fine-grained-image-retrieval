"""Query-level Recall@K (any relevant hit) and reciprocal rank."""


def _validate_ids(values, label, require_nonempty=False):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if require_nonempty and not values:
        raise ValueError(f"{label} must contain at least one relevant item")
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{label} IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate IDs")


def evaluate(rankings, qrels, ks=(1, 5, 10)):
    """Require the exact query set; explicit empty rankings count as failures.

    MRR is calculated over the supplied ranking lists. If lists are truncated,
    report the returned mrr_over_supplied_rankings as MRR@L in experiment tables.
    """
    if not isinstance(rankings, dict) or not isinstance(qrels, dict) or not qrels:
        raise ValueError("rankings and non-empty qrels must be objects")
    if any(not isinstance(q, str) or not q.strip() for q in qrels):
        raise ValueError("query IDs must be non-empty strings")
    if set(rankings) != set(qrels):
        raise ValueError("rankings and qrels must contain exactly the same queries")
    ks = tuple(ks)
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    ks = tuple(sorted(set(ks)))
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    depths = []
    for query_id, relevant_ids in qrels.items():
        _validate_ids(relevant_ids, f"qrels[{query_id}]", require_nonempty=True)
        ordered = rankings[query_id]
        _validate_ids(ordered, f"rankings[{query_id}]")
        depths.append(len(ordered))
        relevant = set(relevant_ids)
        first_rank = next((i for i, image_id in enumerate(ordered, 1) if image_id in relevant), None)
        if first_rank is not None:
            rr_total += 1.0 / first_rank
            for k in ks:
                hits[k] += int(first_rank <= k)
    count = len(qrels)
    return {
        "query_count": count,
        "recall_definition": "fraction_of_queries_with_any_relevant_hit",
        **{f"recall@{k}": hits[k] / count for k in ks},
        "mrr_over_supplied_rankings": rr_total / count,
        "ranking_depth_min": min(depths),
        "ranking_depth_max": max(depths),
    }
