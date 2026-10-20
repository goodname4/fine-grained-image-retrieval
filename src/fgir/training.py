"""Reproducible E1/E2 training over B's frozen raw feature caches."""

import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .adapters import DualAdapter, MODEL_KEYS, SCHEMA, adapt_array, check_model_metadata
from .data import file_digest
from .evaluation import load_protocol
from .features import FeatureIndex, normalize_rows
from .files import read_json, write_json
from .handoff import training_pairs_for_epoch
from .losses import hard_negative_loss, pair_masks, symmetric_contrastive_loss
from .runs import code_revision
from .training_data import CachedSplit, load_training_negatives

DEFAULTS = {"method": "E1", "seed": 42, "epochs": 20, "batch_size": 128,
            "learning_rate": 0.001, "weight_decay": 0.0001, "temperature": 0.07,
            "bottleneck": 128, "alpha": 0.2, "hard_weight": 0.0,
            "hard_margin": 0.2, "hard_per_query": 2, "patience": 5,
            "gradient_clip": 1.0, "device": "auto", "threads": 4}


def training_config(path):
    values = read_json(path)
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("unknown training configuration keys")
    config = DEFAULTS | values
    if config["method"] not in ("E1", "E2") or type(config["seed"]) is not int:
        raise ValueError("training needs E1/E2 and an integer seed")
    for key in ("epochs", "batch_size", "bottleneck", "hard_per_query", "patience", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay", "temperature", "alpha", "hard_weight", "hard_margin", "gradient_clip"):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    if not 0 < config["temperature"] <= 1 or not 0 < config["alpha"] <= 1:
        raise ValueError("temperature and alpha must be in (0, 1]")
    if config["learning_rate"] <= 0 or config["gradient_clip"] <= 0 or config["batch_size"] < 2:
        raise ValueError("learning_rate/gradient_clip must be positive; batch_size must be at least 2")
    if (config["method"] == "E1") != (config["hard_weight"] == 0):
        raise ValueError("E1 requires hard_weight=0; E2 requires hard_weight>0")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return config


def seed_everything(seed, threads=4):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def validate_adapter(adapter, split, protocol_path):
    protocol, candidates, queries, qrels = load_protocol(protocol_path)
    if protocol["split"] != "val" or set(candidates) != set(split.image_ids):
        raise ValueError("model selection requires the fixed validation gallery")
    if {r["query_id"] for r in queries} != set(split.query_ids):
        raise ValueError("validation queries do not match cached texts")
    for row in queries:
        if row["text"] != split.pair_by_id[row["query_id"]]["text"]:
            raise ValueError("validation protocol text differs from cache input")
    adapter.eval()
    images = adapt_array(adapter.image, split.images.numpy())
    texts = adapt_array(adapter.text, split.texts.numpy())
    index = FeatureIndex(split.image_ids, images)
    # Rank the complete gallery for MRR, preserving B's tie breaking semantics.
    sums = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    positions = {image_id: i for i, image_id in enumerate(index.ids)}
    for start in range(0, len(texts), 256):
        scores = normalize_rows(texts[start:start + 256]) @ index.matrix.T
        for query_id, row_scores in zip(split.query_ids[start:start + 256], scores):
            # Count strictly earlier items, including B's lexicographic tie rule,
            # instead of allocating millions of dictionaries on every epoch.
            rank = min(1 + int(np.count_nonzero((row_scores > row_scores[positions[p]]) |
                       ((row_scores == row_scores[positions[p]]) & (index._id_rank < index._id_rank[positions[p]]))))
                       for p in qrels[query_id])
            for k in (1, 5, 10):
                sums[f"recall@{k}"] += rank <= k
            sums["mrr"] += 1 / rank
    return {key: value / len(texts) for key, value in sums.items()}


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train_adapter(bundle, features, config_path, output, negatives=None):
    config = training_config(config_path)
    if (config["method"] == "E2") != (negatives is not None):
        raise ValueError("provide negatives for E2 only")
    output, bundle, features = Path(output), Path(bundle), Path(features)
    if output.exists():
        raise ValueError(f"training output already exists: {output}")
    seed_everything(config["seed"], config["threads"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    train = CachedSplit(bundle, features, "train")
    val = CachedSplit(bundle, features, "val")
    check_model_metadata(train.meta, val.meta)
    if set(train.image_ids) & set(val.image_ids):
        raise ValueError("train/val image leakage")
    protocol_path = bundle / "val/protocol.json"
    protocol = load_protocol(protocol_path)[0]
    dataset = read_json(bundle / "dataset.json")
    if protocol["manifest_sha256"] != dataset["manifest_sha256"]:
        raise ValueError("validation protocol belongs to another dataset")
    mined, negative_audit = ({}, None) if negatives is None else load_training_negatives(negatives, train)
    if negatives is not None and negative_audit["eligible_negative_pairs"] == 0:
        raise ValueError("no eligible hard negatives remain")
    adapter = DualAdapter(train.dimension, config["bottleneck"], config["alpha"]).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    if any(p.requires_grad for p in (train.images, train.texts, val.images, val.texts)):
        raise ValueError("cached encoder features must be detached")
    revision = code_revision()
    model = {key: train.meta[key] for key in MODEL_KEYS}
    provenance = {"dataset_sha256": file_digest(bundle / "dataset.json"),
                  "manifest_sha256": dataset["manifest_sha256"],
                  "val_protocol_sha256": file_digest(protocol_path),
                  "feature_sha256": {f"{s}.{k}": file_digest(features / f"{s}.{k}.npz")
                                     for s in ("train", "val") for k in ("images", "texts")},
                  "negative_sha256": file_digest(negatives) if negatives else None, **revision}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", provenance)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "device": device, "threads": config["threads"],
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "deterministic_algorithms": True}
    write_json(output / "environment.json", environment)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    started = time.perf_counter()
    best_key, best_epoch, best_metrics = None, 0, None
    history, training_seconds = [], 0.0

    def record(epoch, losses, elapsed):
        nonlocal best_key, best_epoch, best_metrics
        validation_started = time.perf_counter()
        metrics = validate_adapter(adapter, val, protocol_path)
        key = (metrics["recall@1"], metrics["mrr"])
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, best_metrics = key, epoch, metrics
            torch.save({"schema_version": SCHEMA, "structure": adapter.structure,
                        "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                        "model": model, "config": config, "epoch": epoch,
                        "validation": metrics, "provenance": provenance}, output / "best.pt")
        entry = {"epoch": epoch, **losses, "training_seconds": elapsed, "validation": metrics,
                 "validation_seconds": time.perf_counter() - validation_started, "selected": improved}
        history.append(entry)
        with (output / "history.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(f"{config['method']} epoch={epoch} val_R@1={metrics['recall@1']:.4f} best={best_epoch}",
              file=sys.stderr, flush=True)

    # Include the identity model, so failed adaptation is visible and reproducible.
    record(0, {"loss": None, "contrastive_loss": None, "hard_loss": None}, 0.0)
    for epoch in range(config["epochs"]):
        pairs = training_pairs_for_epoch(bundle, epoch, config["seed"])
        adapter.train()
        totals, seen = np.zeros(3, dtype=np.float64), 0
        _sync(device)
        epoch_started = time.perf_counter()
        for start in range(0, len(pairs), config["batch_size"]):
            batch = pairs[start:start + config["batch_size"]]
            image_ids, query_ids = [p["image_id"] for p in batch], [p["query_id"] for p in batch]
            raw_images = train.images[[train.image_position[i] for i in image_ids]].to(device)
            raw_texts = train.texts[[train.query_position[q] for q in query_ids]].to(device)
            images, texts = adapter.image(raw_images), adapter.text(raw_texts)
            positive, allowed = pair_masks(image_ids, image_ids, train.groups,
                                           [train.excluded[q] for q in query_ids], device)
            contrastive = symmetric_contrastive_loss(images, texts, positive, allowed, config["temperature"])
            hard = contrastive * 0
            if negatives is not None:
                count = config["hard_per_query"]
                selections = []
                for q in query_ids:
                    eligible = mined[q]
                    # Cycle a reproducible window rather than using only the first candidate forever.
                    offset = (epoch * count) % max(len(eligible), 1)
                    selections.append((eligible[offset:] + eligible[:offset])[:count])
                valid = torch.tensor([[j < len(items) for j in range(count)] for items in selections],
                                     dtype=torch.bool, device=device)
                indices = [[train.image_position[items[j]] if j < len(items) else train.image_position[source]
                            for j in range(count)] for items, source in zip(selections, image_ids)]
                negative_features = train.images[torch.tensor(indices)].to(device)
                negative_vectors = adapter.image(negative_features.flatten(0, 1)).reshape(len(batch), count, -1)
                hard = hard_negative_loss(texts, images, negative_vectors, valid, config["hard_margin"])
            loss = contrastive + config["hard_weight"] * hard
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss; partial logs preserved")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            totals += np.array([loss.item(), contrastive.item(), hard.item()]) * len(batch)
            seen += len(batch)
        _sync(device)
        elapsed = time.perf_counter() - epoch_started
        training_seconds += elapsed
        record(epoch + 1, dict(zip(("loss", "contrastive_loss", "hard_loss"), (totals / seen).tolist())), elapsed)
        if epoch + 1 - best_epoch >= config["patience"]:
            break
    _sync(device)
    torch.save({"schema_version": SCHEMA, "structure": adapter.structure,
                "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                "model": model, "config": config, "epoch": history[-1]["epoch"],
                "validation": history[-1]["validation"], "provenance": provenance}, output / "last.pt")
    summary = {"schema_version": "fgir-training-v1", "status": "completed", "method": config["method"],
               "seed": config["seed"], "best_epoch": best_epoch, "epochs_completed": len(history) - 1,
               "selection": "validation recall@1, then validation full-gallery mrr; ties keep earlier checkpoint",
               "baseline_validation": history[0]["validation"], "best_validation": best_metrics,
               "training_seconds": training_seconds, "wall_seconds": time.perf_counter() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
               "feature_cache_bytes": sum((features / f"{s}.{k}.npz").stat().st_size
                                          for s in ("train", "val") for k in ("images", "texts")),
               "parameters": adapter.parameter_counts(), "negative_audit": negative_audit,
               "checkpoint": "best.pt", "adapter_sha256": file_digest(output / "best.pt"),
               "model": model, "provenance": provenance, "environment": environment}
    write_json(output / "summary.json", summary)
    write_json(output / "checkpoints.json", [
        {"path": "best.pt", "sha256": summary["adapter_sha256"], "epoch": best_epoch, "validation": best_metrics},
        {"path": "last.pt", "sha256": file_digest(output / "last.pt"), "epoch": history[-1]["epoch"],
         "validation": history[-1]["validation"]}])
    return summary
