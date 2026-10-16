"""Descriptive data statistics; lexical category counts are never human labels."""

from collections import Counter, defaultdict
from pathlib import Path

from .annotations import suggested_categories
from .contracts import load_manifest
from .data import file_digest, SPLITS
from .evaluation import _percentile
from .files import new_directory, read_json, read_jsonl, write_json


def dataset_statistics(bundle, output):
    bundle = Path(bundle)
    dataset = read_json(bundle / "dataset.json")
    for name in ("manifest.jsonl", "inventory.jsonl"):
        if file_digest(bundle / name) != dataset["files"].get(name):
            raise ValueError("statistics input hash mismatch")
    rows = load_manifest(bundle / "manifest.jsonl")
    inventory = read_jsonl(bundle / "inventory.jsonl")
    descriptions = defaultdict(list)
    for row in rows:
        for i, caption in enumerate(row["captions"]):
            descriptions[" ".join(caption.lower().split())].append({"image_id": row["image_id"], "split": row["split"], "query_id": f"{row['image_id']}#{i}"})
    duplicates = [{"normalized_caption": caption, "occurrences": occurrences}
                  for caption, occurrences in descriptions.items() if len(occurrences) > 1]
    cross_split = [item for item in duplicates if len({row["split"] for row in item["occurrences"]}) > 1]
    splits = {}
    for split in SPLITS:
        images = [row for row in rows if row["split"] == split]
        captions = [caption for row in images for caption in row["captions"]]
        lengths = [len(caption.split()) for caption in captions]
        splits[split] = {"image_count": len(images), "caption_count": len(captions),
                         "captions_per_image_distribution": dict(sorted(Counter(len(row["captions"]) for row in images).items())),
                         "whitespace_token_length": {"min": min(lengths), "p50": _percentile(lengths, .5), "p95": _percentile(lengths, .95), "max": max(lengths)},
                         "lexical_category_caption_counts": {category: sum(category in suggested_categories(caption) for caption in captions)
                                                             for category in ("color", "count", "action", "relation")}}
    summary = {"manifest_sha256": dataset["manifest_sha256"], "splits": splits,
               "total_image_count": len(rows), "total_caption_count": sum(len(row["captions"]) for row in rows),
               "repeated_normalized_caption_count": len(duplicates), "cross_split_repeated_caption_count": len(cross_split),
               "width_range": [min(row["width"] for row in inventory), max(row["width"] for row in inventory)],
               "height_range": [min(row["height"] for row in inventory), max(row["height"] for row in inventory)],
               "interpretation": "Repeated generic captions do not by themselves prove image leakage; they highlight possible unlabelled relevant images. Lexical categories are screening counts, not verified fine-grained labels."}
    with new_directory(output) as target:
        write_json(target / "statistics.json", summary)
        write_json(target / "repeated-captions.json", duplicates)
    return summary
