"""C's research-v2 trainer. Formal selection requires reviewed data and D's driver."""

import importlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .contracts import image_path
from .data import file_digest
from .files import read_json, write_json, write_jsonl
from .runs import code_revision
from .semantic_evaluation import load_semantic_protocol
from .semantic_losses import (component_pair_loss, global_alignment_loss, paraphrase_consistency_loss,
                              semantic_total, sentence_pair_loss)
from .text_lora import (create_text_lora, frozen_state_digest, load_text_lora,
                        lora_parameters, save_text_lora, trainable_audit)
from .text_training_data import ImageFeaturePool, TextTrainingCorpus
from .training import seed_everything

DEFAULTS = {"method": "N0", "mode": "formal", "seed": 42, "epochs": 10,
            "micro_batch_size": 8, "gradient_accumulation": 4, "learning_rate": 0.0001,
            "weight_decay": 0.0001, "temperature": 0.07, "last_layers": 2, "rank": 4,
            "lora_alpha": 8.0, "lora_dropout": 0.0, "sentence_weight": 0.2, "unit_weight": 0.2,
            "para_weight": 0.1, "include_relations": True, "gradient_clip": 1.0,
            "device": "auto", "threads": 4, "max_optimizer_steps": None,
            "development_max_images": None, "retrieval_tolerance": 0.01,
            "reload_atol": 0.00001, "validation_driver": "fgir.text_export:validate_text_lora_checkpoint"}


def text_training_config(path):
    provided = read_json(path)
    if not isinstance(provided, dict) or set(provided) - set(DEFAULTS):
        raise ValueError("unknown text LoRA training configuration fields")
    config = DEFAULTS | provided
    if config["method"] not in ("N0", "N1", "N2", "N3") or config["mode"] not in ("formal", "development_smoke"):
        raise ValueError("invalid method/mode")
    if type(config["seed"]) is not int or type(config["include_relations"]) is not bool:
        raise ValueError("seed/include_relations type mismatch")
    for key in ("epochs", "micro_batch_size", "gradient_accumulation", "last_layers", "rank", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["micro_batch_size"] < 2:
        raise ValueError("contrastive micro-batch must allow at least two source images")
    for key in ("learning_rate", "weight_decay", "temperature", "lora_alpha", "lora_dropout", "sentence_weight",
                "unit_weight", "para_weight", "gradient_clip", "retrieval_tolerance", "reload_atol"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if any(config[k] <= 0 for k in ("learning_rate", "temperature", "lora_alpha", "gradient_clip", "reload_atol")):
        raise ValueError("learning rate/temperature/alpha/clip/tolerance must be positive")
    if config["lora_dropout"] >= 1 or config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("invalid dropout/device")
    for key in ("max_optimizer_steps", "development_max_images"):
        if config[key] is not None and (type(config[key]) is not int or config[key] < 2):
            raise ValueError(f"{key} must be null or an integer >=2")
    if config["mode"] == "development_smoke":
        if (config["method"] != "N0" or config["development_max_images"] is None
                or config["max_optimizer_steps"] is None):
            raise ValueError("development smoke needs N0 and explicit image/update limits")
    elif config["development_max_images"] is not None:
        raise ValueError("formal training must not silently subsample its frozen cohort")
    if not isinstance(config["validation_driver"], str) or ":" not in config["validation_driver"]:
        raise ValueError("validation_driver must be a module:function reference")
    return config


def batch_objective(encoder, corpus, records, config, device):
    batch = corpus.batch(records, device)
    vectors = encoder(batch["texts"])
    global_loss = global_alignment_loss(batch["images"], vectors[batch["global_positions"]],
        batch["image_ids"], batch["global_sources"], corpus.pool.groups,
        batch["global_exclusions"], config["temperature"])
    zero = vectors.sum() * 0

    def paired(items, component=False, paraphrase=False):
        if not items:
            return zero
        sources = [item[0] for item in items]
        images = corpus.pool.images(sources, device)
        right, wrong = vectors[[item[1] for item in items]], vectors[[item[2] for item in items]]
        if component:
            return component_pair_loss(images, right, wrong, sources, [item[3] for item in items],
                                       config["temperature"], config["include_relations"])
        if paraphrase:
            return paraphrase_consistency_loss(images, right, wrong, sources)
        return sentence_pair_loss(images, right, wrong, sources, config["temperature"])

    # All N methods run the same text forward pool. Only objective coefficients differ.
    sentence = paired(batch["sentence_pairs"])
    unit = paired(batch["unit_pairs"], component=True)
    para = paired(batch["paraphrase_pairs"], paraphrase=True)
    total = semantic_total(config["method"], global_loss, sentence, unit, para,
                           config["sentence_weight"], config["unit_weight"], config["para_weight"])
    metrics = {"loss": total.detach().item(), "global_loss": global_loss.detach().item(),
               "sentence_loss": sentence.detach().item(), "unit_loss": unit.detach().item(),
               "para_loss": para.detach().item(), "source_images": len(records),
               "global_texts": len(batch["global_positions"]), "forward_texts": len(batch["texts"]),
               "forward_tokens": sum(len(ids) for ids in encoder.token_ids(batch["texts"])),
               "sentence_pairs": len(batch["sentence_pairs"]), "unit_pairs": len(batch["unit_pairs"]),
               "paraphrase_pairs": len(batch["paraphrase_pairs"])}
    return total, metrics


def selection_key(method, metrics, reference_recall=None, tolerance=0.01):
    required = ("recall@1", "mrr", "attribute_binding", "relation_direction")
    if not isinstance(metrics, dict) or set(metrics) != set(required):
        raise ValueError("validation driver must return exact named selection metrics")
    for key, value in metrics.items():
        if value is None and key in ("attribute_binding", "relation_direction"):
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid validation metric: {key}")
    if method == "N0":
        return (metrics["recall@1"], metrics["mrr"])
    if reference_recall is None or metrics["recall@1"] < reference_recall - tolerance:
        return None
    diagnostics = [metrics[k] for k in ("attribute_binding", "relation_direction") if metrics[k] is not None]
    if not diagnostics:
        raise ValueError("formal component selection lacks applicable binding/relation diagnostics")
    return (sum(diagnostics) / len(diagnostics), metrics["recall@1"], metrics["mrr"])


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _image_probe(encoder, corpus, data_root, processor):
    from PIL import Image
    ids = sorted(corpus.by_image)[:2]
    images = []
    try:
        for image_id in ids:
            with Image.open(image_path(data_root, corpus.pool.rows[image_id]["path"])) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        device = next(encoder.model.parameters()).device
        with torch.no_grad():
            values = encoder.model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
            if not isinstance(values, torch.Tensor):
                values = values.pooler_output
            return ids, F.normalize(values.float(), dim=-1).detach().cpu().clone()
    finally:
        for image in images:
            image.close()


def train_text_lora(bundle, model_dir, image_features, config_path, output, samples=None,
                    data_root="data", training_audit=None, val_protocol=None, reference_summary=None,
                    validation_callback=None):
    config = text_training_config(config_path)
    formal = config["mode"] == "formal"
    output, bundle = Path(output), Path(bundle)
    if output.exists():
        raise ValueError(f"training output already exists: {output}")
    if formal and (samples is None or val_protocol is None):
        raise ValueError("formal training requires verified --samples and independently reviewed --val-protocol")
    if not formal and any(x is not None for x in (samples, val_protocol, training_audit, reference_summary)):
        raise ValueError("development smoke accepts original train captions only, no evaluation gold")
    seed_everything(config["seed"], config["threads"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    encoder = create_text_lora(model_dir, device, config["last_layers"], config["rank"],
                               config["lora_alpha"], config["lora_dropout"])
    pool = ImageFeaturePool(bundle, image_features, encoder.metadata)
    corpus = TextTrainingCorpus(pool, encoder, samples, data_root, training_audit, config["development_max_images"])
    semantic_protocol_sha256, reference_recall = None, None
    if formal:
        protocol, val_rows, candidates = load_semantic_protocol(val_protocol)
        if (protocol["split"] != "val" or protocol["source_manifest_sha256"] != file_digest(bundle / "manifest.jsonl")
                or protocol["data_kind"] != pool.data_kind):
            raise ValueError("formal selection requires source-bound validation gold, never test")
        if set(pool.rows) & set(candidates):
            raise ValueError("train and validation galleries overlap")
        semantic_protocol_sha256 = file_digest(val_protocol)
        if config["method"] != "N0":
            if not corpus.stats["unit_phenomena"]:
                raise ValueError("N1/N2/N3 require shared verified minimal edits and linked components")
            if config["method"] == "N3" and not corpus.stats["variant_types"].get("paraphrase"):
                raise ValueError("N3 requires verified paraphrases also exposed to N2")
            if reference_summary is None:
                raise ValueError("formal N1/N2/N3 require the same-seed N0 --reference-summary")
            reference = read_json(reference_summary)
            if (reference.get("method") != "N0" or reference.get("mode") != "formal"
                    or reference.get("status") != "completed" or reference.get("seed") != config["seed"]
                    or reference.get("semantic_dataset_sha256") != corpus.semantic_dataset_sha256
                    or reference.get("semantic_protocol_sha256") != semantic_protocol_sha256
                    or reference.get("model") != encoder.metadata or reference.get("structure") != encoder.structure):
                raise ValueError("N0 reference is incompatible with this paired experiment")
            reference_recall = reference["best_validation"]["recall@1"]
        if validation_callback is None:
            module, name = config["validation_driver"].split(":", 1)
            try:
                validation_callback = getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError) as exc:
                raise ValueError("D-V2-01/02 validation driver missing: " + config["validation_driver"]) from exc
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    probe_ids, image_before = _image_probe(encoder, corpus, data_root, processor)
    cache_delta = (image_before - pool.images(probe_ids)).abs().max().item()
    if cache_delta > config["reload_atol"]:
        raise ValueError("cached images differ from the actual frozen image encoder")
    frozen_before = frozen_state_digest(encoder.model)
    initial_lora = {name: p.detach().cpu().clone() for name, p in lora_parameters(encoder.model).items()}
    audit = trainable_audit(encoder.model)
    optimizer = torch.optim.AdamW(lora_parameters(encoder.model).values(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    revision = code_revision()
    provenance = {**revision, "source_manifest_sha256": file_digest(bundle / "manifest.jsonl"),
                  "source_dataset_sha256": file_digest(bundle / "dataset.json"),
                  "image_features_sha256": pool.feature_sha256,
                  "semantic_dataset_sha256": corpus.semantic_dataset_sha256,
                  "semantic_protocol_sha256": semantic_protocol_sha256,
                  "training_audit_sha256": file_digest(training_audit) if training_audit else None,
                  "reference_summary_sha256": file_digest(reference_summary) if reference_summary else None,
                  "data_kind": pool.data_kind, "mode": config["mode"],
                  "test_access": "No test captions/images used by optimizer or selection. Source semantic validation may inspect split metadata; historical test access remains disclosed."}
    environment = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                   "transformers": __import__("transformers").__version__, "numpy": __import__("numpy").__version__,
                   "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                   "cuda": torch.version.cuda, "deterministic_algorithms": True,
                   "lora_implementation": "native-pytorch-not-peft"}
    output.mkdir(parents=True)
    for name, value in (("config.json", config), ("provenance.json", provenance), ("environment.json", environment),
                        ("corpus.json", {**corpus.stats, "semantic_audit": corpus.audit})):
        write_json(output / name, value)
    write_jsonl(output / "token-filter.jsonl", corpus.rejections)
    updates, microsteps, validation_attempts = 0, 0, []
    best_key, best_selection, best_metrics = None, None, None
    gradient_nonzero = {name: False for name in initial_lora}
    observed_tokens, observed_texts, skipped_batches, omitted_singletons = 0, 0, 0, 0
    empty_supervision = {"sentence_batches": 0, "unit_batches": 0, "paraphrase_batches": 0}
    training_seconds = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    started = time.perf_counter()

    def checkpoint(path, selection):
        return save_text_lora(path, encoder, config["method"], config["seed"], config, provenance,
                              selection, corpus.semantic_dataset_sha256)

    def validate(epoch):
        nonlocal best_key, best_selection, best_metrics
        encoder.training_mode(False)
        selection = {"epoch": epoch, "optimizer_steps": updates, "mode": config["mode"]}
        if not formal:
            # Development best is explicitly the final audited development state,
            # not a checkpoint selected against validation or a retrieval score.
            checkpoint(output / "best.pt", selection | {"rule": "latest-development-step-no-validation"})
            best_selection = selection
            return
        candidate = output / f"candidate-epoch{epoch:03d}.pt"
        candidate_sha = checkpoint(candidate, selection)
        result = validation_callback(checkpoint_path=str(candidate), model_dir=str(model_dir),
                                     bundle=str(bundle), val_protocol=str(val_protocol),
                                     output=str(output / "validation" / f"epoch{epoch:03d}"))
        if (not isinstance(result, dict) or result.get("text_lora_sha256") != candidate_sha
                or result.get("semantic_protocol_sha256") != semantic_protocol_sha256
                or result.get("base_weights_sha256") != encoder.metadata["base_weights_sha256"]
                or result.get("tokenizer_sha256") != encoder.metadata["tokenizer_sha256"]):
            raise ValueError("D validation result is not bound to this actual checkpoint/protocol/model")
        metrics = result.get("metrics")
        key = selection_key(config["method"], metrics, reference_recall, config["retrieval_tolerance"])
        selected = key is not None and (best_key is None or key > best_key)
        attempt = {**selection, "checkpoint_sha256": candidate_sha, "metrics": metrics,
                   "eligible": key is not None, "selected": selected, "driver_result": result}
        validation_attempts.append(attempt)
        write_json(output / "validation-history.json", validation_attempts)
        if selected:
            best_key, best_selection, best_metrics = key, selection, metrics
            checkpoint(output / "best.pt", selection | {"validation": metrics})

    validate(0)
    for epoch in range(config["epochs"]):
        rows = corpus.epoch(epoch, config["seed"])
        batches = [rows[i:i + config["micro_batch_size"]] for i in range(0, len(rows), config["micro_batch_size"])]
        if len(batches[-1]) == 1:
            batches.pop()
            omitted_singletons += 1
        for window_start in range(0, len(batches), config["gradient_accumulation"]):
            window = batches[window_start:window_start + config["gradient_accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            valid_microsteps = 0
            encoder.training_mode(True)
            _sync(device)
            update_started = time.perf_counter()
            entries = []
            for records in window:
                try:
                    loss, metrics = batch_objective(encoder, corpus, records, config, device)
                except ValueError as exc:
                    if "no informative negative pairs" not in str(exc):
                        raise
                    skipped_batches += 1
                    continue
                if not torch.isfinite(loss) or not loss.requires_grad:
                    raise ValueError("non-finite or detached text LoRA loss")
                loss.backward()
                valid_microsteps += 1
                microsteps += 1
                observed_tokens += metrics["forward_tokens"]
                observed_texts += metrics["forward_texts"]
                for term in ("sentence", "unit", "paraphrase"):
                    empty_supervision[f"{term}_batches"] += metrics[f"{term}_pairs"] == 0
                entries.append(metrics)
            if valid_microsteps == 0:
                raise ValueError("all-empty supervision window; refusing a fake zero-loss update")
            gradients = {}
            for name, param in lora_parameters(encoder.model).items():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise ValueError(f"missing/non-finite LoRA gradient: {name}")
                param.grad.div_(valid_microsteps)
                norm = param.grad.norm().item()
                gradients[name] = norm
                gradient_nonzero[name] |= norm > 0
            if any(p.grad is not None for name, p in encoder.model.named_parameters() if name not in initial_lora):
                raise ValueError("gradient leaked into frozen base parameters")
            torch.nn.utils.clip_grad_norm_(lora_parameters(encoder.model).values(), config["gradient_clip"],
                                           error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            _sync(device)
            elapsed = time.perf_counter() - update_started
            training_seconds += elapsed
            entry = {"epoch": epoch + 1, "optimizer_step": updates, "microsteps": valid_microsteps,
                     "gradient_norms": gradients, "training_seconds": elapsed, "batches": entries,
                     "contrastive_pool": "current micro-batch only; accumulation does not enlarge negatives"}
            with (output / "history.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            print(f"{config['method']} {config['mode']} epoch={epoch+1} update={updates} "
                  f"loss={sum(e['loss'] for e in entries)/len(entries):.4f}", file=sys.stderr, flush=True)
            if config["max_optimizer_steps"] is not None and updates >= config["max_optimizer_steps"]:
                break
        validate(epoch + 1)
        if config["max_optimizer_steps"] is not None and updates >= config["max_optimizer_steps"]:
            break
    encoder.training_mode(False)
    if updates < 2:
        raise ValueError("at least two actual optimizer steps are required for LoRA audit")
    frozen_after = frozen_state_digest(encoder.model)
    _, image_after = _image_probe(encoder, corpus, data_root, processor)
    image_delta = (image_before - image_after).abs().max().item()
    changes = {name: (param.detach().cpu() - initial_lora[name]).abs().max().item()
               for name, param in lora_parameters(encoder.model).items()}
    freeze_audit = {**audit, "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_after,
                    "all_original_parameters_unchanged": frozen_before == frozen_after,
                    "image_probe_ids": probe_ids, "image_output_max_abs_diff": image_delta,
                    "image_cache_max_abs_diff": cache_delta, "lora_parameter_max_changes": changes,
                    "lora_nonzero_gradient_observed": gradient_nonzero,
                    "base_files_sha256_after": __import__("fgir.encoding", fromlist=["weights_digest"]).weights_digest(model_dir)}
    if (frozen_before != frozen_after or image_delta != 0
            or freeze_audit["base_files_sha256_after"] != encoder.metadata["base_weights_sha256"]
            or not all(value > 0 for value in changes.values()) or not all(gradient_nonzero.values())):
        raise ValueError("LoRA/frozen/image invariance audit failed")
    last_sha = checkpoint(output / "last.pt", {"epoch": epoch + 1, "optimizer_steps": updates, "mode": config["mode"]})
    # Compare the same actual final state across a fresh CPU load. No driver involved.
    probe_texts = [corpus.records[0]["text"], corpus.records[-1]["text"]]
    with torch.no_grad():
        text_before = encoder(probe_texts).cpu()
    reloaded, receipt = load_text_lora(output / "last.pt", model_dir, device="cpu")
    with torch.no_grad():
        text_after = reloaded(probe_texts)
    text_delta = (text_before - text_after).abs().max().item()
    score_delta = ((image_before @ text_before.T) - (image_before @ text_after.T)).abs().max().item()
    if text_delta > config["reload_atol"] or score_delta > config["reload_atol"]:
        raise ValueError("checkpoint reload differs from actual trained text features/scores")
    write_json(output / "freeze-audit.json", freeze_audit)
    reload_audit = {"checkpoint_sha256": receipt["text_lora_sha256"], "reload_device": "cpu",
                    "text_feature_max_abs_diff": text_delta, "score_max_abs_diff": score_delta,
                    "atol": config["reload_atol"], "passed": True}
    write_json(output / "reload-audit.json", reload_audit)
    _sync(device)
    summary = {"schema_version": "fgir-text-training-v1", "status": "completed" if best_selection else "failed_retrieval_guard",
               "method": config["method"], "mode": config["mode"], "data_kind": pool.data_kind, "seed": config["seed"],
               "formal_comparison_eligible": formal and pool.data_kind == "real" and best_selection is not None,
               "model": encoder.metadata, "structure": encoder.structure,
               "semantic_dataset_sha256": corpus.semantic_dataset_sha256,
               "semantic_protocol_sha256": semantic_protocol_sha256,
               "optimizer_steps": updates, "microsteps": microsteps, "best_selection": best_selection,
               "best_validation": best_metrics, "best_sha256": file_digest(output / "best.pt") if best_selection else None,
               "last_sha256": last_sha, "freeze_audit": freeze_audit, "reload_audit": reload_audit,
               "corpus": corpus.stats, "skipped_batches": skipped_batches, "omitted_singleton_images": omitted_singletons,
               "empty_supervision": empty_supervision, "nonfinite_batches": 0,
               "forward_texts": observed_texts, "forward_tokens": observed_tokens,
               "training_seconds": training_seconds, "wall_seconds": time.perf_counter() - started,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
               "selection_policy": "fixed paired training budget; N0 val retrieval; N1-N3 diagnostic macro with N0 retrieval guard; earlier wins ties",
               "development_notice": None if formal else "Real train-caption gradient/reload smoke only; no reviewed semantic supervision, no validation selection, no N-series effectiveness claim.",
               "environment": environment, "provenance": provenance}
    write_json(output / "summary.json", summary)
    write_json(output / "checkpoints.json", [{"path": name, "sha256": file_digest(output / name)}
                                             for name in ("best.pt", "last.pt") if (output / name).exists()])
    return summary
