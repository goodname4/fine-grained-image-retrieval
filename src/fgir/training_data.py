"""Validate immutable A bundles and B caches before any optimizer step."""

from pathlib import Path

import numpy as np
import torch

from .adapters import check_model_metadata
from .data import file_digest
from .encoding import load_features
from .files import read_json, read_jsonl
from .handoff import load_split
from .negatives import exclusion_groups


class CachedSplit:
    def __init__(self, bundle, features, split):
        if split not in ("train", "val"):
            raise ValueError("training loader only permits train/val")
        bundle, features = Path(bundle), Path(features)
        self.rows = load_split(bundle, split)
        dataset = read_json(bundle / "dataset.json")
        name = f"{split}/pairs.jsonl"
        if file_digest(bundle / name) != dataset["files"].get(name):
            raise ValueError("training pair file hash mismatch")
        self.pairs = read_jsonl(bundle / name)
        expected = {f"{row['image_id']}#{i}": (row["image_id"], caption)
                    for row in self.rows for i, caption in enumerate(row["captions"])}
        if len(self.pairs) != len(expected) or {p["query_id"] for p in self.pairs} != set(expected):
            raise ValueError("pairs do not cover the split captions exactly")
        for pair in self.pairs:
            if pair.get("split") != split or (pair["image_id"], pair["text"]) != expected[pair["query_id"]]:
                raise ValueError("pair text/source/split differs from image manifest")
        self.image_ids, images, self.meta = load_features(features / f"{split}.images.npz")
        self.query_ids, texts, text_meta = load_features(features / f"{split}.texts.npz")
        check_model_metadata(self.meta, text_meta)
        for meta, kind in ((self.meta, "image"), (text_meta, "text")):
            if meta.get("split") != split or meta.get("kind") != kind or meta.get("adapter_sha256") is not None:
                raise ValueError("training requires raw features from the correct split/kind")
        if len(set(self.image_ids)) != len(self.image_ids) or set(self.image_ids) != {r["image_id"] for r in self.rows}:
            raise ValueError("image features do not cover the split exactly")
        if len(set(self.query_ids)) != len(self.query_ids) or set(self.query_ids) != set(expected):
            raise ValueError("text features do not cover the split exactly")
        if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
            raise ValueError("feature dimensions disagree")
        if self.meta.get("dimension") != images.shape[1] or text_meta.get("dimension") != texts.shape[1]:
            raise ValueError("feature metadata dimension mismatch")
        for matrix in (images, texts):
            if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
                raise ValueError("features must be finite and nonzero")
        self.images = torch.from_numpy(images).detach()
        self.texts = torch.from_numpy(texts).detach()
        self.image_position = {value: i for i, value in enumerate(self.image_ids)}
        self.query_position = {value: i for i, value in enumerate(self.query_ids)}
        self.pair_by_id = {p["query_id"]: p for p in self.pairs}
        self.groups = exclusion_groups(bundle) if split == "train" else {}
        caption_index = {}
        for pair in self.pairs:
            caption_index.setdefault(" ".join(pair["text"].lower().split()), set()).add(pair["image_id"])
        self.excluded = {p["query_id"]: caption_index[" ".join(p["text"].lower().split())] - {p["image_id"]}
                         for p in self.pairs}
        self.dimension = images.shape[1]


def load_training_negatives(path, split):
    records = read_jsonl(path)
    if len(records) != len(split.query_ids) or {r["query_id"] for r in records} != set(split.query_ids):
        raise ValueError("negatives must cover every training query exactly once")
    by_query, reviewed, removed, remaining = {}, 0, 0, 0
    gallery = set(split.image_ids)
    for row in records:
        query_id = row["query_id"]
        source = split.pair_by_id[query_id]["image_id"]
        positive, negative = row["positive_image_ids"], row["negative_image_ids"]
        if (not isinstance(positive, list) or source not in positive or not set(positive) <= gallery
                or not isinstance(negative, list) or not set(negative) <= gallery
                or len(negative) != len(set(negative))):
            raise ValueError("negative record contains invalid or non-training images")
        false_negatives = set(row.get("false_negative_image_ids", []))
        excluded = set(row.get("excluded_image_ids", {})) - {source}
        excluded |= split.excluded[query_id]
        if set(negative) & set(positive) or set(negative) & excluded:
            raise ValueError("negative record includes a positive or excluded image")
        if any(split.groups.get(n, n) == split.groups.get(source, source) for n in negative):
            raise ValueError("negative record includes a same-scene image")
        if row.get("reviewed") is True:
            if not row.get("reviewer"):
                raise ValueError("reviewed negative needs an identified reviewer")
            reviewed += 1
        kept = [n for n in negative if n not in false_negatives]
        removed += len(negative) - len(kept)
        remaining += len(kept)
        by_query[query_id] = kept
        # Additional explicitly known positives must also be masked from InfoNCE.
        split.excluded[query_id] |= (set(positive) | false_negatives | excluded) - {source}
    return by_query, {"queries": len(records), "reviewed_queries": reviewed,
                      "reviewed_fraction": reviewed / len(records), "removed_false_negatives": removed,
                      "eligible_negative_pairs": remaining,
                      "limitation": "unreviewed mined candidates may still be false negatives"}
