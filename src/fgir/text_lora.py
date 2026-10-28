"""CLIP text-only LoRA: differentiable encoding and strict standalone checkpoints.

This is a small native PyTorch LoRA implementation, not a PEFT checkpoint.
The image tower, projections and all original parameters remain frozen.
"""

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import file_digest
from .encoding import load_encoder

CHECKPOINT_SCHEMA = "fgir-text-lora-v1"
STRUCTURE_VERSION = "clip-text-qv-lora-v1"
METHODS = ("N0", "N1", "N2", "N3")
FREEZE_POLICY = {"vision_tower": "frozen", "text_base": "frozen", "projections": "frozen",
                 "logit_scale": "frozen", "trainable": "whitelisted-text-qv-lora-only"}
IDENTITY_KEYS = ("model_id", "base_weights_sha256", "tokenizer_sha256", "image_encoder_sha256",
                 "projection_dimension", "preprocessing", "text_processing")


def tensor_digest(named_tensors):
    value = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        tensor = tensor.detach().cpu().contiguous()
        value.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def frozen_state_digest(model, image_only=False):
    items = []
    for name, tensor in model.state_dict().items():
        if name.endswith((".lora_A", ".lora_B")):
            continue
        name = name.replace(".base.", ".")
        if not image_only or name.startswith(("vision_model.", "visual_projection.")):
            items.append((name, tensor))
    return tensor_digest(items)


def tokenizer_digest(model_dir):
    directory = Path(model_dir)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
             "merges.txt", "added_tokens.json")
    files = [(name, file_digest(directory / name)) for name in names if (directory / name).is_file()]
    if not files:
        raise ValueError("local tokenizer files are missing")
    return hashlib.sha256("\n".join(f"{n}:{h}" for n, h in files).encode()).hexdigest()


def validate_structure(structure, layer_count):
    required = {"version", "last_layers", "rank", "alpha", "dropout", "target_modules", "target_paths"}
    if not isinstance(structure, dict) or set(structure) != required:
        raise ValueError("unknown or missing LoRA structure fields")
    if structure["version"] != STRUCTURE_VERSION or structure["target_modules"] != ["q_proj", "v_proj"]:
        raise ValueError("unsupported LoRA structure/target modules")
    layers, rank = structure["last_layers"], structure["rank"]
    if type(layers) is not int or not 1 <= layers <= layer_count or type(rank) is not int or rank < 1:
        raise ValueError("invalid LoRA layer count/rank")
    for key in ("alpha", "dropout"):
        if type(structure[key]) not in (int, float) or not math.isfinite(structure[key]):
            raise ValueError(f"invalid LoRA {key}")
    if structure["alpha"] <= 0 or not 0 <= structure["dropout"] < 1:
        raise ValueError("LoRA alpha must be positive and dropout must be in [0,1)")
    expected = [f"text_model.encoder.layers.{i}.self_attn.{module}"
                for i in range(layer_count - layers, layer_count) for module in ("q_proj", "v_proj")]
    if structure["target_paths"] != expected:
        raise ValueError("LoRA target paths differ from the exact text-only whitelist")
    return structure


def make_structure(layer_count, last_layers=2, rank=4, alpha=8.0, dropout=0.0):
    structure = {"version": STRUCTURE_VERSION, "last_layers": last_layers, "rank": rank,
                 "alpha": alpha, "dropout": dropout, "target_modules": ["q_proj", "v_proj"],
                 "target_paths": [f"text_model.encoder.layers.{i}.self_attn.{module}"
                                  for i in range(layer_count - last_layers, layer_count)
                                  for module in ("q_proj", "v_proj")]}
    return validate_structure(structure, layer_count)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if type(base) is not nn.Linear or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA only wraps supported Linear projections with an admissible rank")
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = F.linear(F.linear(self.dropout(features), self.lora_A), self.lora_B)
        return self.base(features) + self.scale * residual


def inject_text_lora(model, structure):
    if type(model).__name__ != "CLIPModel" or not hasattr(model, "text_model"):
        raise ValueError("only the verified transformers CLIPModel architecture is supported")
    validate_structure(structure, len(model.text_model.encoder.layers))
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("refuse to inject LoRA into an already adapted/shared model")
    # Validate every path before changing any of them.
    targets = [(path, model.get_submodule(path)) for path in structure["target_paths"]]
    if any(type(module) is not nn.Linear or structure["rank"] > min(module.in_features, module.out_features)
           for _, module in targets):
        raise ValueError("unexpected attention projection structure")
    model.requires_grad_(False)
    for path, module in targets:
        parent, name = path.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, LoRALinear(module, structure["rank"],
                                                           structure["alpha"], structure["dropout"]))
    return model


def lora_parameters(model):
    return {name: param for name, param in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}


def trainable_audit(model):
    expected = lora_parameters(model)
    actual = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if not expected or set(actual) != set(expected):
        raise ValueError("trainable parameters escape the text LoRA whitelist")
    return {"trainable_parameters": sum(p.numel() for p in actual.values()),
            "base_trainable_parameters": 0, "image_trainable_parameters": 0,
            "trainable_names": sorted(actual)}


class TextLoRAEncoder(nn.Module):
    def __init__(self, model, tokenizer, metadata, structure):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.metadata, self.structure = metadata, structure
        self.max_length = model.config.text_config.max_position_embeddings

    def training_mode(self, enabled=True):
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train(enabled)

    def token_ids(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("texts must be a non-empty list of non-empty strings")
        return self.tokenizer(list(texts), truncation=False, padding=False)["input_ids"]

    def forward(self, texts):
        ids = self.token_ids(texts)
        if any(len(row) > self.max_length for row in ids):
            raise ValueError("text exceeds CLIP token limit; no silent truncation is permitted")
        inputs = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        result = self.model.get_text_features(**{k: v.to(device) for k, v in inputs.items()
                                                 if k in ("input_ids", "attention_mask")})
        if not isinstance(result, torch.Tensor):
            result = getattr(result, "pooler_output", None)
        if not isinstance(result, torch.Tensor) or result.ndim != 2 or not torch.isfinite(result).all():
            raise ValueError("unsupported or non-finite differentiable CLIP text output")
        if (result.norm(dim=-1) == 0).any():
            raise ValueError("CLIP returned zero text features")
        return F.normalize(result.float(), dim=-1)


def create_text_lora(model_dir, device="cpu", last_layers=2, rank=4, alpha=8.0, dropout=0.0, structure=None):
    # Always load a fresh model: never mutate B/D's shared E0 encoder instance.
    base = load_encoder(model_dir, device)
    if type(base.model).__name__ != "CLIPModel":
        raise ValueError("only the verified transformers CLIPModel architecture is supported")
    layer_count = len(base.model.text_model.encoder.layers)
    structure = structure if structure is not None else make_structure(layer_count, last_layers, rank, alpha, dropout)
    validate_structure(structure, layer_count)
    metadata = {"model_id": base.model_id, "base_weights_sha256": base.weights_sha256,
                "tokenizer_sha256": tokenizer_digest(model_dir),
                "image_encoder_sha256": frozen_state_digest(base.model, image_only=True),
                "projection_dimension": base.model.config.projection_dim,
                "preprocessing": base.preprocessing, "text_processing": base.text_processing}
    inject_text_lora(base.model, structure)
    encoder = TextLoRAEncoder(base.model, base.processor.tokenizer, metadata, structure)
    trainable_audit(encoder.model)
    encoder.training_mode(False)
    return encoder


def save_text_lora(path, encoder, method, seed, config, provenance, selection, semantic_dataset_sha256=None):
    path = Path(path)
    if method not in METHODS or type(seed) is not int:
        raise ValueError("checkpoint must identify a real N method and integer seed")
    state = {name: param.detach().cpu().clone() for name, param in lora_parameters(encoder.model).items()}
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("cannot save empty/non-finite LoRA state")
    payload = {"schema_version": CHECKPOINT_SCHEMA, "model_type": "text_lora", "method": method,
               "freeze_policy": FREEZE_POLICY,
               "seed": seed, "structure": encoder.structure, "model": encoder.metadata,
               "semantic_dataset_sha256": semantic_dataset_sha256,
               "training_config": config, "provenance": provenance, "selection": selection,
               "state_dict": state}
    torch.save(payload, path)
    return file_digest(path)


def load_text_lora(path, model_dir, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "model_type", "method", "seed", "structure", "model", "freeze_policy",
                "semantic_dataset_sha256", "training_config", "provenance", "selection", "state_dict"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unknown or incomplete text LoRA checkpoint schema")
    if payload["model_type"] != "text_lora" or payload["method"] not in METHODS or type(payload["seed"]) is not int:
        raise ValueError("invalid checkpoint model type/method/seed")
    if payload["freeze_policy"] != FREEZE_POLICY:
        raise ValueError("unsupported checkpoint freezing policy")
    if not isinstance(payload["training_config"], dict) or not isinstance(payload["provenance"], dict):
        raise ValueError("checkpoint training configuration/provenance must be objects")
    mode = payload["training_config"].get("mode")
    if mode not in ("formal", "development_smoke"):
        raise ValueError("checkpoint must declare formal or development mode")
    if mode == "development_smoke" and (payload["method"] != "N0" or payload["semantic_dataset_sha256"] is not None):
        raise ValueError("invalid development checkpoint identity")
    if mode == "formal":
        from .semantic_data import digest
        digest(payload["semantic_dataset_sha256"], "semantic training samples digest")
        digest(payload["provenance"].get("semantic_protocol_sha256"), "validation protocol digest")
    if not isinstance(payload["model"], dict) or set(payload["model"]) != set(IDENTITY_KEYS):
        raise ValueError("incomplete base model identity")
    encoder = create_text_lora(model_dir, device, structure=payload["structure"])
    for key in IDENTITY_KEYS:
        if encoder.metadata[key] != payload["model"][key]:
            raise ValueError(f"checkpoint/base model identity mismatch: {key}")
    expected, state = lora_parameters(encoder.model), payload["state_dict"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("missing or unexpected LoRA parameter names")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"invalid LoRA parameter shape/dtype/value: {name}")
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state[name].to(param.device))
    encoder.training_mode(False)
    receipt = {k: v for k, v in payload.items() if k != "state_dict"}
    receipt["text_lora_sha256"] = file_digest(path)
    return encoder, receipt
