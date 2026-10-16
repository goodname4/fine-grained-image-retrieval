"""Balanced real-caption annotation drafts, an offline board and review gates."""

import base64
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from .contracts import CATEGORIES, image_path, load_manifest, validate_queries
from .data import file_digest
from .files import new_directory, read_json, read_jsonl, write_json, write_jsonl
from .metrics import _validate_ids


PATTERNS = {
    "color": r"\b(red|blue|green|yellow|pink|purple|orange|black|white|brown|gray|grey)\b",
    "count": r"\b(two|three|four|five|six|seven|eight|nine|ten|2|3|4|5|6|7|8|9|10)\b",
    "action": r"\b(running|jumping|swimming|riding|walking|climbing|sitting|standing|playing|catching|throwing)\b",
    "relation": r"\b(under|above|below|behind|between|beside|next to|in front of|on top of|across|through|over)\b",
}


def suggested_categories(caption):
    return [category for category, pattern in PATTERNS.items() if re.search(pattern, caption, re.IGNORECASE)]


def thumbnail(path, size=176):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((size, size))
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=65)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def render_board(rows, images, data_root, path):
    gallery = [{"image_id": row["image_id"], "thumbnail": thumbnail(image_path(data_root, row["path"]))} for row in images]
    payload = json.dumps({"queries": rows, "gallery": gallery}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("annotation-board.html").read_text(encoding="utf-8")
    Path(path).write_text(template.replace("__FGIR_DATA__", payload), encoding="utf-8")


def draft_annotations(manifest, data_root, output, seed=42, per_category=25):
    if type(per_category) is not int or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    options = []
    for image in images:
        for index, caption in enumerate(image["captions"]):
            categories = suggested_categories(caption)
            key = hashlib.sha256(f"{seed}:{image['image_id']}:{index}".encode()).hexdigest()
            options.append((key, image, index, caption, categories))
    options.sort(key=lambda item: item[0])
    chosen, used_images, used_text = [], set(), set()
    # Select the scarce relationship/count captions first; no category claims are final.
    for category in ("relation", "count", "color", "action"):
        found = 0
        for _, image, index, caption, categories in options:
            normalized = " ".join(caption.lower().split())
            if category not in categories or image["image_id"] in used_images or normalized in used_text:
                continue
            chosen.append((image, index, caption, categories, category))
            used_images.add(image["image_id"])
            used_text.add(normalized)
            found += 1
            if found == per_category:
                break
        if found < per_category:
            raise ValueError(f"not enough distinct real source images for {category}: {found}/{per_category}")
    rows = []
    for i, (image, index, caption, categories, primary) in enumerate(chosen):
        assigned = "ABCD"[i % 4]
        rows.append({"query_id": f"fine_{i + 1:03d}", "text": caption, "categories": categories,
                     "primary_category": primary, "relevant_image_ids": [image["image_id"]],
                     "source_query_id": f"{image['image_id']}#{index}", "source_image_id": image["image_id"],
                     "annotation_status": "draft", "annotation_scope": "candidate_pool", "judged_image_ids": [],
                     "assigned_to": assigned, "planned_reviewer": {"A": "B", "B": "A", "C": "D", "D": "C"}[assigned],
                     "annotator": None, "reviewer": None, "review_notes": "", "pool_method": "",
                     "annotation_notes": "Automatically selected source caption and paired image; categories and all relevance judgments require visual checking."})
    validate_queries(rows, [row["image_id"] for row in images], fine_grained=True)
    with new_directory(output) as target:
        write_jsonl(target / "annotations.jsonl", rows)
        summary = annotation_summary(rows)
        summary.update({"manifest_sha256": file_digest(manifest), "seed": seed, "per_primary_category": per_category,
                        "status": "draft_not_formal_evaluation", "selection": "deterministic lexical screening of test source captions, distinct source images"})
        write_json(target / "allocation.json", summary)
        render_board(rows, images, data_root, target / "board.html")
    return summary


def annotation_summary(rows):
    return {"query_count": len(rows),
            "status_counts": dict(Counter(row.get("annotation_status") for row in rows)),
            "assignment_counts": dict(Counter(row.get("assigned_to") for row in rows)),
            "category_counts": {category: sum(category in row.get("categories", []) for row in rows) for category in CATEGORIES},
            "scope_counts": dict(Counter(row.get("annotation_scope") for row in rows))}


def validate_annotations(manifest, annotations, require_review=False):
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    rows = read_jsonl(annotations)
    validate_queries(rows, [row["image_id"] for row in images], require_review, fine_grained=True)
    return annotation_summary(rows)


def render_annotations(manifest, annotations, data_root, output):
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    rows = read_jsonl(annotations)
    validate_queries(rows, [row["image_id"] for row in images], fine_grained=True)
    with new_directory(output) as target:
        render_board(rows, images, data_root, target / "board.html")
        write_jsonl(target / "annotations.jsonl", rows)
    return annotation_summary(rows)


def apply_source_checks(manifest, annotations, checks, output):
    """Apply versioned assisted initial annotations without inventing peer review."""
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    rows = read_jsonl(annotations)
    changes = read_json(checks)
    if changes.get("manifest_sha256") != file_digest(manifest):
        raise ValueError("source checks belong to another manifest")
    by_id = {row["query_id"]: row for row in rows}
    seen = set()
    for check in changes["checks"]:
        query_id = check["query_id"]
        if query_id not in by_id or query_id in seen:
            raise ValueError("unknown or repeated source-check query")
        seen.add(query_id)
        row = by_id[query_id]
        if row["source_image_id"] != check["source_image_id"] or row["annotation_status"] != "draft":
            raise ValueError("source checks must match the exact draft source image")
        if check["kind"] == "initial_annotation":
            row.update({"text": check["text"], "categories": check["categories"],
                        "annotation_status": "annotated", "annotator": changes["actor"], "reviewer": None,
                        "relevant_image_ids": check["relevant_image_ids"], "judged_image_ids": changes["judged_pool_image_ids"],
                        "pool_method": changes["pool_method"], "annotation_notes": check["notes"],
                        "assistance_disclosure": "AI-assisted visual initial annotation; human independent review is outstanding."})
        elif check["kind"] == "pre_review":
            row["pre_review"] = {"actor": changes["actor"], "notes": check["notes"],
                                 "status": "suggestions_only_awaiting_member_initial_annotation"}
        else:
            raise ValueError("unknown source-check kind")
    validate_queries(rows, [row["image_id"] for row in images], fine_grained=True)
    output = Path(output)
    if output.exists() or output.resolve() == Path(annotations).resolve():
        raise ValueError("source-check output must be a new file")
    write_jsonl(output, rows)
    return annotation_summary(rows)


def merge_annotations(manifest, base, updates, output):
    """Three-way merge member exports; conflicting changes require resolution."""
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    candidates = [row["image_id"] for row in images]
    rows = read_jsonl(base)
    validate_queries(rows, candidates, fine_grained=True)
    original = {row["query_id"]: row for row in rows}
    chosen, sources = {}, []
    for path in updates:
        changed_rows = read_jsonl(path)
        validate_queries(changed_rows, candidates, fine_grained=True)
        if {row["query_id"] for row in changed_rows} != set(original):
            raise ValueError("member exports must contain exactly the base query set")
        count = 0
        for row in changed_rows:
            query_id = row["query_id"]
            for key in ("assigned_to", "planned_reviewer", "source_image_id", "source_query_id"):
                if row.get(key) != original[query_id].get(key):
                    raise ValueError(f"member export changed immutable provenance: {query_id}/{key}")
            if row != original[query_id]:
                if query_id in chosen and chosen[query_id] != row:
                    raise ValueError(f"conflicting member edits; resolve explicitly: {query_id}")
                chosen[query_id] = row
                count += 1
        sources.append({"filename": Path(path).name, "sha256": file_digest(path), "changed_query_count": count})
    merged = [chosen.get(row["query_id"], row) for row in rows]
    with new_directory(output) as target:
        write_jsonl(target / "annotations.jsonl", merged)
        write_json(target / "merge-report.json", {"base_sha256": file_digest(base), "sources": sources,
                                                   "changed_query_count": len(chosen), **annotation_summary(merged)})
    return annotation_summary(merged)


def make_pool_request(manifest, annotations, output):
    """Freeze query texts/gallery for B/C to rank before relevance review is final."""
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    rows = read_jsonl(annotations)
    candidates = [row["image_id"] for row in images]
    validate_queries(rows, candidates, fine_grained=True)
    with new_directory(output) as target:
        write_json(target / "candidates.json", candidates)
        write_jsonl(target / "queries.jsonl", [{"query_id": row["query_id"], "text": row["text"]} for row in rows])
        request = {"schema_version": "fgir-pool-request-v1", "purpose": "annotation_pool_not_formal_evaluation",
                   "manifest_sha256": file_digest(manifest), "annotations_sha256": file_digest(annotations),
                   "files": {name: {"path": name, "sha256": file_digest(target / name)} for name in ("candidates.json", "queries.jsonl")}}
        write_json(target / "request.json", request)
    return request


def expand_annotation_pool(manifest, annotations, request_path, runs, output, depth=10):
    """Union actual method results, keeping every newly added image unjudged."""
    if type(depth) is not int or depth < 1 or not runs:
        raise ValueError("pool depth must be positive and at least one method run is required")
    request = read_json(request_path)
    if request.get("schema_version") != "fgir-pool-request-v1" or request.get("manifest_sha256") != file_digest(manifest):
        raise ValueError("pool request belongs to another manifest or schema")
    bound = {}
    for name in ("candidates.json", "queries.jsonl"):
        entry = request["files"][name]
        path = image_path(Path(request_path).parent, entry["path"])
        if file_digest(path) != entry["sha256"]:
            raise ValueError("pool request file hash mismatch")
        bound[name] = path
    candidates = read_json(bound["candidates.json"])
    actual_candidates = [row["image_id"] for row in load_manifest(manifest) if row["split"] == "test"]
    if candidates != actual_candidates:
        raise ValueError("pool request gallery differs from the fixed test gallery")
    rows = read_jsonl(annotations)
    validate_queries(rows, candidates, fine_grained=True)
    texts = {row["query_id"]: row["text"] for row in read_jsonl(bound["queries.jsonl"])}
    if texts != {row["query_id"]: row["text"] for row in rows}:
        raise ValueError("query texts changed since pool request; rerun all methods on a new request")
    gallery = set(candidates)
    pools = {row["query_id"]: set(row.get("pool_candidate_image_ids", [])) | set(row["judged_image_ids"]) | set(row["relevant_image_ids"]) for row in rows}
    models, sources = set(), []
    for path in runs:
        run = read_json(path)
        if run.get("schema_version") != "fgir-pool-run-v1" or run.get("request_sha256") != file_digest(request_path):
            raise ValueError("pool run belongs to another request")
        model = run.get("model_id")
        if not isinstance(model, str) or not model.strip() or model in models:
            raise ValueError("pool model IDs must be distinct non-empty identifiers")
        models.add(model)
        rankings = run.get("rankings")
        if not isinstance(rankings, dict) or set(rankings) != set(texts):
            raise ValueError("pool rankings must contain exactly the requested queries")
        for query_id, ordered in rankings.items():
            _validate_ids(ordered, f"pool rankings[{query_id}]")
            if len(ordered) < min(depth, len(candidates)) or not set(ordered) <= gallery:
                raise ValueError("pool rankings are too short or contain unknown images")
            pools[query_id].update(ordered[:depth])
        sources.append({"model_id": model, "run_sha256": file_digest(path)})
    for row in rows:
        row["pool_candidate_image_ids"] = [image_id for image_id in candidates if image_id in pools[row["query_id"]]]
        row["pool_method"] = f"Union of Top-{depth} from {sorted(models)} plus previous judged/relevant images; request SHA-256 {file_digest(request_path)}"
        if row["annotation_status"] == "reviewed":
            row["annotation_status"] = "annotated"
            row["reviewer"] = None
            row["review_notes"] = "Pool expanded; new independent review is required."
    with new_directory(output) as target:
        write_jsonl(target / "annotations.jsonl", rows)
        write_json(target / "pool-receipt.json", {"request_sha256": file_digest(request_path), "depth": depth, "sources": sources,
            "new_images_are_unjudged": True, "pool_sizes": {row["query_id"]: len(row["pool_candidate_image_ids"]) for row in rows}})
    return annotation_summary(rows)


def freeze_annotations(manifest, annotations, output, expected_count=100, min_per_category=20):
    if type(expected_count) is not int or expected_count < 1 or type(min_per_category) is not int or min_per_category < 1:
        raise ValueError("expected query count and per-category minimum must be positive integers")
    images = [row for row in load_manifest(manifest) if row["split"] == "test"]
    candidates = [row["image_id"] for row in images]
    rows = read_jsonl(annotations)
    validate_queries(rows, candidates, require_review=True, fine_grained=True)
    summary = annotation_summary(rows)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} independently reviewed queries, got {len(rows)}")
    if any(count < min_per_category for count in summary["category_counts"].values()):
        raise ValueError("fine-grained category coverage is below the required minimum")
    scopes = {row["annotation_scope"] for row in rows}
    if len(scopes) != 1:
        raise ValueError("freeze full-gallery and candidate-pool annotations separately")
    with new_directory(output) as target:
        write_json(target / "candidates.json", candidates)
        write_jsonl(target / "queries.jsonl", rows)
        write_json(target / "qrels.json", {row["query_id"]: row["relevant_image_ids"] for row in rows})
        write_json(target / "protocol.json", {
            "schema_version": "fgir-protocol-v1", "task": "text_to_image", "split": "test", "data_kind": "real",
            "manifest_sha256": file_digest(manifest), "annotation_scope": next(iter(scopes)),
            "independently_reviewed": True, "candidate_count": len(candidates), "query_count": len(rows),
            "recall_ks": [1, 5, 10], "files": {name: {"path": name, "sha256": file_digest(target / name)}
                for name in ("candidates.json", "queries.jsonl", "qrels.json")},
            "interpretation": "Full-gallery relevance" if next(iter(scopes)) == "full_gallery" else "Judged candidate-pool success only; unjudged returned images cause rejection, not automatic negatives."})
        write_json(target / "coverage.json", summary)
    return summary
