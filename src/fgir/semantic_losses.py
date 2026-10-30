"""Differentiable, source-macro semantic objectives for research-v2."""

import math

import torch
from torch.nn import functional as F

from .losses import pair_masks


def global_alignment_loss(images, texts, image_ids, text_image_ids, groups=None,
                          excluded_by_text=None, temperature=0.07):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    positive, allowed = pair_masks(image_ids, text_image_ids, groups, excluded_by_text, images.device)
    if not positive.any(0).all() or not positive.any(1).all():
        raise ValueError("global alignment has an unpaired source")
    if not (allowed & ~positive).any():
        raise ValueError("global alignment has no informative negative pairs")
    logits = F.normalize(images, dim=-1) @ F.normalize(texts, dim=-1).T / temperature
    denominator, numerator = logits.masked_fill(~allowed, -torch.inf), logits.masked_fill(~positive, -torch.inf)
    image_loss = torch.logsumexp(denominator, 1) - torch.logsumexp(numerator, 1)
    text_loss = torch.logsumexp(denominator, 0) - torch.logsumexp(numerator, 0)
    return (source_macro_mean(image_loss, image_ids) + source_macro_mean(text_loss, text_image_ids)) / 2


def _check_vectors(images, right, wrong):
    if images.ndim != 2 or images.shape != right.shape or right.shape != wrong.shape:
        raise ValueError("paired semantic embeddings must share a 2D shape")
    if any(not torch.isfinite(x).all() for x in (images, right, wrong)):
        raise ValueError("non-finite semantic embeddings")


def source_macro_mean(values, source_ids, valid=None):
    """First average inside each source image, then average source images."""
    if values.ndim != 1 or len(values) != len(source_ids):
        raise ValueError("source grouping and loss values disagree")
    if valid is None:
        valid = torch.ones(len(values), dtype=torch.bool, device=values.device)
    if valid.shape != values.shape or valid.dtype != torch.bool:
        raise ValueError("invalid supervision mask")
    means = []
    for source in sorted(set(source_ids)):
        mask = torch.tensor([s == source for s in source_ids], dtype=torch.bool, device=values.device) & valid
        if mask.any():
            means.append(values[mask].mean())
    return torch.stack(means).mean() if means else values.sum() * 0


def sentence_pair_loss(images, right, wrong, source_ids, temperature=0.07, valid=None):
    _check_vectors(images, right, wrong)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    images, right, wrong = (F.normalize(x, dim=-1) for x in (images, right, wrong))
    delta = ((images * wrong).sum(-1) - (images * right).sum(-1)) / temperature
    return source_macro_mean(F.softplus(delta), source_ids, valid)


def component_pair_loss(images, right, wrong, source_ids, phenomena, temperature=0.07,
                        include_relations=True, valid=None):
    """Equal means across supported phenomena, then source means within each."""
    _check_vectors(images, right, wrong)
    if len(phenomena) != len(source_ids):
        raise ValueError("component phenomena and sources disagree")
    if valid is None:
        valid = torch.ones(len(source_ids), dtype=torch.bool, device=images.device)
    parts = []
    for phenomenon in ("attribute_binding", "relation_direction"):
        if phenomenon == "relation_direction" and not include_relations:
            continue
        mask = torch.tensor([p == phenomenon for p in phenomena], dtype=torch.bool, device=images.device) & valid
        if mask.any():
            parts.append(sentence_pair_loss(images, right, wrong, source_ids, temperature, mask))
    return torch.stack(parts).mean() if parts else (right.sum() + wrong.sum()) * 0


def paraphrase_consistency_loss(images, original, paraphrase, source_ids, valid=None):
    _check_vectors(images, original, paraphrase)
    images, original, paraphrase = (F.normalize(x, dim=-1) for x in (images, original, paraphrase))
    delta = (images * original).sum(-1) - (images * paraphrase).sum(-1)
    return source_macro_mean(delta.square(), source_ids, valid)


def semantic_total(method, global_loss, sentence_loss, unit_loss, para_loss,
                   sentence_weight=0.2, unit_weight=0.2, para_weight=0.1):
    if method not in ("N0", "N1", "N2", "N3"):
        raise ValueError("unknown semantic experiment")
    for weight in (sentence_weight, unit_weight, para_weight):
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("semantic weights must be finite and non-negative")
    if not all(x.ndim == 0 and torch.isfinite(x) for x in (global_loss, sentence_loss, unit_loss, para_loss)):
        raise ValueError("loss terms must be finite scalars")
    if method == "N0":
        return global_loss
    if method == "N1":
        return global_loss + sentence_weight * sentence_loss
    total = global_loss + unit_weight * unit_loss
    return total + para_weight * para_loss if method == "N3" else total
