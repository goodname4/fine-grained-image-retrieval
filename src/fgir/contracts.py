"""Versioned image, query and annotation contracts; independent of the encoder."""

from pathlib import Path

from .data import relative_path, SPLITS
from .files import read_jsonl
from .metrics import _validate_ids


CATEGORIES = ("color", "count", "action", "relation")


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def image_path(root, value):
    root = Path(root).resolve()
    path = (root / str(relative_path(value))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes data root")
    return path


def load_manifest(path):
    rows = read_jsonl(path)
    ids, paths = set(), set()
    for row in rows:
        image_id = text(row.get("image_id"), "image_id")
        if image_id in ids:
            raise ValueError(f"duplicate image_id: {image_id}")
        ids.add(image_id)
        rel = str(relative_path(row.get("path")))
        if rel in paths:
            raise ValueError(f"duplicate image path: {rel}")
        paths.add(rel)
        if row.get("split") not in SPLITS:
            raise ValueError(f"invalid split for {image_id}")
        captions = row.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ValueError(f"empty captions for {image_id}")
        for caption in captions:
            text(caption, f"caption for {image_id}")
    return rows


def validate_queries(rows, candidate_ids, require_review=False, fine_grained=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError("queries must be a non-empty list")
    candidates = set(candidate_ids)
    seen, seen_text = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("query record must be an object")
        query_id = text(row.get("query_id"), "query_id")
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        text(row.get("text"), f"text for {query_id}")
        if row.get("source_image_id") is not None:
            source = text(row["source_image_id"], f"source image for {query_id}")
            if source not in candidates:
                raise ValueError(f"source image outside candidate gallery: {query_id}")
        if "assigned_to" in row and row["assigned_to"] not in ("A", "B", "C", "D"):
            raise ValueError(f"invalid member assignment: {query_id}")
        relevant = row.get("relevant_image_ids")
        _validate_ids(relevant, f"relevant images for {query_id}", require_nonempty=True)
        if not set(relevant) <= candidates:
            raise ValueError(f"relevant image outside candidate gallery: {query_id}")
        categories = row.get("categories", [])
        _validate_ids(categories, f"categories for {query_id}")
        if not set(categories) <= set(CATEGORIES) or (fine_grained and not categories):
            raise ValueError(f"invalid or empty fine-grained categories: {query_id}")
        if fine_grained:
            normalized = " ".join(row["text"].lower().split())
            if normalized in seen_text:
                raise ValueError(f"duplicate fine-grained query text; merge relevance or revise the query: {query_id}")
            seen_text.add(normalized)
            status = row.get("annotation_status")
            if status not in ("draft", "annotated", "reviewed"):
                raise ValueError(f"invalid annotation status: {query_id}")
            scope = row.get("annotation_scope")
            if scope not in ("full_gallery", "candidate_pool"):
                raise ValueError(f"invalid annotation scope: {query_id}")
            judged = row.get("judged_image_ids", [])
            _validate_ids(judged, f"judged images for {query_id}")
            if not set(judged) <= candidates:
                raise ValueError(f"judged image outside candidate gallery: {query_id}")
            pool = row.get("pool_candidate_image_ids", judged)
            _validate_ids(pool, f"pool candidates for {query_id}")
            if not set(pool) <= candidates:
                raise ValueError(f"pool candidate outside gallery: {query_id}")
            if status != "draft":
                text(row.get("annotator"), f"annotator for {query_id}")
                if not set(relevant) <= set(judged):
                    raise ValueError(f"relevant images must have been judged: {query_id}")
            if status == "reviewed":
                reviewer = text(row.get("reviewer"), f"reviewer for {query_id}")
                if reviewer == row.get("annotator"):
                    raise ValueError(f"independent reviewer required: {query_id}")
                text(row.get("review_notes"), f"review notes for {query_id}")
                if scope == "full_gallery" and set(judged) != candidates:
                    raise ValueError(f"full-gallery review must judge every candidate: {query_id}")
                if scope == "candidate_pool":
                    text(row.get("pool_method"), f"pool method for {query_id}")
                    if not set(pool) <= set(judged):
                        raise ValueError(f"candidate pool has unjudged members: {query_id}")
        if require_review and row.get("annotation_status") != "reviewed":
            raise ValueError(f"query has not been independently reviewed: {query_id}")
    return rows
