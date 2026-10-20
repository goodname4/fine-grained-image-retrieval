"""Multi-positive symmetric InfoNCE with explicit unknown-pair exclusion."""

import torch
from torch.nn import functional as F


def pair_masks(image_ids, text_image_ids, groups=None, excluded_by_text=None, device="cpu"):
    """Unknown same-scene/caption matches are ignored, never invented positives."""
    groups, excluded_by_text = groups or {}, excluded_by_text or [set() for _ in text_image_ids]
    positive = torch.tensor([[image_id == source for source in text_image_ids]
                             for image_id in image_ids], dtype=torch.bool, device=device)
    allowed = torch.tensor([[
        image_id == source or (groups.get(image_id, image_id) != groups.get(source, source)
                               and image_id not in excluded_by_text[j])
        for j, source in enumerate(text_image_ids)] for image_id in image_ids],
        dtype=torch.bool, device=device)
    return positive, allowed


def symmetric_contrastive_loss(images, texts, positive, allowed=None, temperature=0.07):
    if not 0 < temperature <= 1:
        raise ValueError("temperature must be in (0, 1]")
    expected = (len(images), len(texts))
    if positive.shape != expected or positive.dtype != torch.bool:
        raise ValueError("positive mask shape/type mismatch")
    if not positive.any(0).all() or not positive.any(1).all():
        raise ValueError("every row/column must have a positive")
    if allowed is None:
        allowed = torch.ones_like(positive)
    if allowed.shape != expected or allowed.dtype != torch.bool or (positive & ~allowed).any():
        raise ValueError("allowed mask must include all positives")
    logits = F.normalize(images, dim=-1) @ F.normalize(texts, dim=-1).T / temperature
    denominator = logits.masked_fill(~allowed, -torch.inf)
    numerator = logits.masked_fill(~positive, -torch.inf)
    image_loss = torch.logsumexp(denominator, 1) - torch.logsumexp(numerator, 1)
    text_loss = torch.logsumexp(denominator, 0) - torch.logsumexp(numerator, 0)
    return (image_loss.mean() + text_loss.mean()) / 2


def hard_negative_loss(texts, positives, negatives, valid, margin=0.2):
    """One-way hinge over valid text/negative pairs; negatives have no paired text."""
    if margin < 0 or negatives.ndim != 3 or valid.shape != negatives.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("invalid hard-negative margin/shape/mask")
    if negatives.shape[0] != len(texts) or positives.shape != texts.shape:
        raise ValueError("hard-negative batch dimensions disagree")
    text = F.normalize(texts, dim=-1)
    pos = (text * F.normalize(positives, dim=-1)).sum(-1, keepdim=True)
    neg = (text[:, None, :] * F.normalize(negatives, dim=-1)).sum(-1)
    values = F.relu(margin + neg - pos)
    return values[valid].mean() if valid.any() else (texts.sum() + negatives.sum()) * 0
