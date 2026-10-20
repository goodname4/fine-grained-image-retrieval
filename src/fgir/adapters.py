"""Frozen-feature residual adapters and portable, provenance-bound checkpoints."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import file_digest

SCHEMA = "fgir-adapter-v1"
MODEL_KEYS = ("model_id", "weights_sha256", "preprocessing", "text_processing")


class ResidualAdapter(nn.Module):
    def __init__(self, dimension, bottleneck=128, alpha=0.2):
        super().__init__()
        if type(dimension) is not int or dimension < 1 or type(bottleneck) is not int or bottleneck < 1:
            raise ValueError("adapter dimensions must be positive integers")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        self.alpha = alpha
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        features = F.normalize(features, dim=-1)
        return F.normalize(features + self.alpha * self.up(F.gelu(self.down(features))), dim=-1)


class DualAdapter(nn.Module):
    def __init__(self, dimension, bottleneck=128, alpha=0.2):
        super().__init__()
        self.structure = {"dimension": dimension, "bottleneck": bottleneck, "alpha": alpha}
        self.image = ResidualAdapter(**self.structure)
        self.text = ResidualAdapter(**self.structure)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder_trainable": 0,
                "encoder_policy": "cached detached features; encoder absent from optimizer"}


def check_model_metadata(left, right):
    for key in MODEL_KEYS:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"incompatible model metadata: {key}")


def load_adapter(path, device="cpu", expected_model=None):
    """Only tensor/basic-type payloads; callers can enforce the encoder identity."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported adapter checkpoint")
    if expected_model is not None:
        check_model_metadata(payload["model"], expected_model)
    adapter = DualAdapter(**payload["structure"])
    adapter.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(p).all() for p in adapter.parameters()):
        raise ValueError("checkpoint contains non-finite parameters")
    adapter.to(device).eval()
    return adapter, {**{k: v for k, v in payload.items() if k != "state_dict"},
                     "adapter_sha256": file_digest(path)}


def adapt_array(adapter, matrix, batch_size=1024):
    import numpy as np
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("expected a non-empty finite feature matrix")
    if (np.linalg.norm(matrix, axis=1) == 0).any():
        raise ValueError("zero feature rows are not supported")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(adapter.parameters()).device
    with torch.inference_mode():
        return np.concatenate([adapter(torch.as_tensor(matrix[start:start + batch_size],
                                                        dtype=torch.float32, device=device)).cpu().numpy()
                               for start in range(0, len(matrix), batch_size)])


def freeze_encoder(encoder):
    """For D's live encoder; offline training never loads this base model."""
    encoder.model.requires_grad_(False)
    encoder.model.eval()
    if any(p.requires_grad for p in encoder.model.parameters()):
        raise ValueError("encoder freezing failed")
    return encoder
