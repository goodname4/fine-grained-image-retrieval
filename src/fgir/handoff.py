"""Small standard-library reader APIs for B/C; no encoder or trainer included."""

from pathlib import Path
import random

from .data import file_digest, SPLITS
from .files import read_json, read_jsonl


def load_split(bundle, split):
    if split not in SPLITS:
        raise ValueError("split must be train, val or test")
    bundle = Path(bundle)
    dataset = read_json(bundle / "dataset.json")
    if dataset.get("schema_version") != "fgir-dataset-v1":
        raise ValueError("unsupported dataset bundle")
    name = f"{split}/images.jsonl"
    if file_digest(bundle / name) != dataset["files"].get(name):
        raise ValueError("split file hash mismatch")
    rows = read_jsonl(bundle / name)
    if any(row["split"] != split for row in rows):
        raise ValueError("split file contains records from another split")
    return rows


def training_pairs_for_epoch(bundle, epoch, seed=42):
    """Return one caption per unique train image, rotating all captions over epochs.

    C should avoid treating members of one negative_exclusion_group_id as batch
    negatives. This flag does not assert every cross-frame caption is a positive.
    """
    if type(epoch) is not int or epoch < 0 or type(seed) is not int:
        raise ValueError("epoch must be non-negative and seed must be an integer")
    rows = load_split(bundle, "train")
    bundle = Path(bundle)
    dataset = read_json(bundle / "dataset.json")
    review_path = bundle / "duplicate-review.json"
    if file_digest(review_path) != dataset["files"].get("duplicate-review.json"):
        raise ValueError("duplicate-review hash mismatch")
    review = read_json(review_path)
    parent = {row["image_id"]: row["image_id"] for row in rows}

    def find(image_id):
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    for pair in review["pairs"]:
        left, right = pair["left"], pair["right"]
        if pair["decision"] in ("same_scene", "identical") and left in parent and right in parent:
            a, b = find(left), find(right)
            parent[max(a, b)] = min(a, b)
    result = []
    for row in rows:
        offset = random.Random(f"{seed}:{row['image_id']}").randrange(len(row["captions"]))
        index = (offset + epoch) % len(row["captions"])
        result.append({"query_id": f"{row['image_id']}#{index}", "image_id": row["image_id"],
                       "path": row["path"], "text": row["captions"][index], "split": "train",
                       "negative_exclusion_group_id": find(row["image_id"])})
    random.Random(f"{seed}:{epoch}:order").shuffle(result)
    return result
