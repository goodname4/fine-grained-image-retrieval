"""Original-text / verified-semantic cohorts, never cached text embeddings."""

import copy
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_digest
from .encoding import load_features
from .files import read_json, read_jsonl
from .handoff import load_split, training_pairs_for_epoch
from .negatives import exclusion_groups
from .semantic_data import validate_semantics


class ImageFeaturePool:
    def __init__(self, bundle, image_features, model_identity):
        self.bundle = Path(bundle)
        rows = load_split(bundle, "train")
        self.rows = {row["image_id"]: row for row in rows}
        ids, matrix, meta = load_features(image_features)
        expected = {"model_id": model_identity["model_id"], "weights_sha256": model_identity["base_weights_sha256"],
                    "preprocessing": model_identity["preprocessing"], "kind": "image", "split": "train"}
        if any(meta.get(k) != value for k, value in expected.items()) or meta.get("adapter_sha256") is not None:
            raise ValueError("LoRA training requires matching raw frozen train image features")
        if len(ids) != len(set(ids)) or set(ids) != set(self.rows):
            raise ValueError("raw image features do not exactly cover the training split")
        if (matrix.ndim != 2 or matrix.shape[1] != model_identity["projection_dimension"]
                or not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=-1) == 0).any()):
            raise ValueError("invalid raw image feature dimensions/values")
        self.ids, self.position = ids, {image_id: i for i, image_id in enumerate(ids)}
        self.features = F.normalize(torch.from_numpy(matrix).detach(), dim=-1)
        self.groups = exclusion_groups(bundle)
        self.dataset = read_json(self.bundle / "dataset.json")
        self.data_kind = self.dataset.get("data_kind", "real")
        self.caption_sources = defaultdict(set)
        for image_id, row in self.rows.items():
            for text in row["captions"]:
                self.caption_sources[" ".join(text.casefold().split())].add(image_id)
        self.feature_sha256 = file_digest(image_features)

    def images(self, ids, device="cpu"):
        return self.features[[self.position[i] for i in ids]].to(device)


def verify_training_audit(path, samples_path, rows):
    """Automatic labels require a version-bound, honestly recorded human spot-check."""
    reviews = []
    for row in rows:
        reviews.append(row["review"])
        reviews.extend(item["review"] for item in row["variants"] + row["units"])
    if any(r.get("annotator_kind") not in ("human", "automatic") for r in reviews):
        raise ValueError("assistant annotations are not verified default training supervision")
    if not any(r.get("annotator_kind") == "automatic" for r in reviews):
        return None
    if path is None:
        raise ValueError("automatic verification requires --training-audit with real human spot-check evidence")
    audit = read_json(path)
    if (audit.get("schema_version") != "fgir-semantic-training-audit-v1"
            or audit.get("samples_sha256") != file_digest(samples_path)
            or audit.get("reviewer_kind") != "human" or not audit.get("reviewer") or not audit.get("notes")
            or audit.get("unresolved_disputes") != []):
        raise ValueError("invalid or unbound human training spot-check audit")
    checked = audit.get("sampled_sample_ids")
    if (not isinstance(checked, list) or not checked or len(checked) != len(set(checked))
            or not set(checked) <= {row["sample_id"] for row in rows}):
        raise ValueError("human spot-check must identify actual training sample IDs")
    return {"sha256": file_digest(path), "checked_samples": len(checked), "total_samples": len(rows)}


class TextTrainingCorpus:
    def __init__(self, pool, encoder, samples=None, data_root="data", training_audit=None, max_images=None):
        self.pool, self.encoder = pool, encoder
        self.semantic_dataset_sha256 = file_digest(samples) if samples else None
        self.rejections, self.audit, self.records = [], None, []
        if samples is None:
            # Development-only originals. No invented semantic review status.
            rows = [{"sample_id": f"source:{image_id}#{j}", "source_image_id": image_id,
                     "text": text, "variants": [], "units": []}
                    for image_id, row in pool.rows.items() for j, text in enumerate(row["captions"])]
        else:
            self.audit = validate_semantics(samples, pool.bundle, "train", data_root)
            if self.audit["data_kind"] != pool.data_kind:
                raise ValueError("semantic and image data kinds differ")
            rows = read_jsonl(samples)
            self.human_audit = verify_training_audit(training_audit, samples, rows)
        if max_images is not None:
            if type(max_images) is not int or max_images < 2:
                raise ValueError("development max_images must be at least two")
            chosen = set(sorted({r["source_image_id"] for r in rows},
                                key=lambda i: hashlib.sha256(f"development:{i}".encode()).hexdigest())[:max_images])
            rows = [row for row in rows if row["source_image_id"] in chosen]
        for row in rows:
            if row["source_image_id"] not in pool.rows:
                raise ValueError("semantic corpus includes a non-training image")
            prepared = self._prepare(row)
            if prepared is not None:
                self.records.append(prepared)
        self.by_image = defaultdict(list)
        for row in self.records:
            self.by_image[row["source_image_id"]].append(row)
        if max_images is not None:
            chosen = sorted(self.by_image, key=lambda i: hashlib.sha256(f"development:{i}".encode()).hexdigest())[:max_images]
            self.by_image = {i: self.by_image[i] for i in chosen}
            self.records = [r for i in chosen for r in self.by_image[i]]
        if len(self.by_image) < 2:
            raise ValueError("fewer than two training images remain after token/review checks")
        for row in self.records:
            for phrase in [row["text"]] + [v["text"] for v in row["variants"] if v["type"] == "paraphrase"]:
                pool.caption_sources[" ".join(phrase.casefold().split())].add(row["source_image_id"])
        self.stats = {"source_images": len(self.by_image), "source_records": len(self.records),
                      "variant_types": dict(Counter(v["type"] for r in self.records for v in r["variants"])),
                      "unit_phenomena": dict(Counter(u["phenomenon"] for r in self.records for u in r["units"])),
                      "rejected_items": len(self.rejections), "silent_truncations": 0,
                      "cohort_sha256": hashlib.sha256("\n".join(sorted(r["sample_id"] for r in self.records)).encode()).hexdigest()}

    def _tokens(self, text):
        ids = self.encoder.token_ids([text])[0]
        return tuple(ids) if len(ids) <= self.encoder.max_length else None

    def _reject(self, row, item, reason):
        self.rejections.append({"sample_id": row["sample_id"], "item_id": item, "reason": reason})

    def _prepare(self, source):
        row = copy.deepcopy(source)
        original = self._tokens(row["text"])
        if original is None:
            self._reject(row, "original", "original_over_token_limit")
            return None
        variants, units = [], []
        for variant in row["variants"]:
            if variant["type"] in ("true_addition", "false_addition"):
                self._reject(row, variant["id"], "diagnostic_addition_not_in_controlled_minimal_edit_training")
                continue
            tokens = self._tokens(variant["text"])
            if tokens is None or tokens == original:
                self._reject(row, variant["id"], "variant_over_token_limit_or_token_identical")
                continue
            if variant["type"] == "minimal_edit":
                linked = [u for u in row["units"] if u["variant_id"] == variant["id"]]
                if len(linked) != 1:
                    self._reject(row, variant["id"], "requires_exactly_one_component_for_paired_N1_N2")
                    continue
                unit = linked[0]
                right, wrong = self._tokens(unit["text"]), self._tokens(unit["foil"])
                if right is None or wrong is None or right == wrong:
                    self._reject(row, variant["id"], "linked_component_over_token_limit_or_token_identical")
                    continue
                units.append({**unit, "phenomenon": variant["phenomenon"]})
            variants.append(variant)
        row["variants"], row["units"] = variants, units
        return row

    def epoch(self, epoch, seed):
        rows = []
        for image_id, choices in sorted(self.by_image.items()):
            offset = random.Random(f"{seed}:{image_id}").randrange(len(choices))
            rows.append(choices[(offset + epoch) % len(choices)])
        random.Random(f"text-lora:{seed}:{epoch}").shuffle(rows)
        return rows

    def batch(self, records, device):
        texts, text_positions = [], {}

        def position(text):
            if text not in text_positions:
                text_positions[text] = len(texts)
                texts.append(text)
            return text_positions[text]

        global_positions, global_sources, global_exclusions = [], [], []
        sentence, units, paraphrases = [], [], []
        ids = [row["source_image_id"] for row in records]
        for row in records:
            source = row["source_image_id"]
            original = position(row["text"])
            positives = [(original, row["text"])]
            for variant in row["variants"]:
                other = position(variant["text"])
                if variant["type"] == "paraphrase":
                    positives.append((other, variant["text"]))
                    paraphrases.append((source, original, other))
                elif variant["type"] == "minimal_edit":
                    sentence.append((source, original, other))
            for unit in row["units"]:
                units.append((source, position(unit["text"]), position(unit["foil"]), unit["phenomenon"]))
            for index, text in positives:
                global_positions.append(index)
                global_sources.append(source)
                global_exclusions.append(self.pool.caption_sources[" ".join(text.casefold().split())] - {source})
        return {"texts": texts, "image_ids": ids, "images": self.pool.images(ids, device),
                "global_positions": global_positions, "global_sources": global_sources,
                "global_exclusions": global_exclusions, "sentence_pairs": sentence,
                "unit_pairs": units, "paraphrase_pairs": paraphrases}
