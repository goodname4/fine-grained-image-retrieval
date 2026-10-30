"""D: independent LoRA export and C's fixed-validation callback."""

from pathlib import Path
import shutil

from .data import file_digest
from .files import new_directory, read_json, read_jsonl, write_json


def encode_texts(encoder, texts, batch_size=64):
    import numpy as np
    import torch
    if type(batch_size) is not int or batch_size < 1 or not texts:
        raise ValueError("nonempty texts and positive batch size required")
    with torch.inference_mode():
        return np.concatenate([encoder(texts[i:i+batch_size]).detach().cpu().numpy()
                               for i in range(0, len(texts), batch_size)])


def export_text_lora(checkpoint_path, model_dir, index_path, protocol_path, output, *,
                     semantic_protocol=None, device="cpu", batch_size=64, experiment_id=None):
    """Encode real texts, bind portable artifacts and export complete rankings.

    Development checkpoints produce explicitly nonformal artifacts only; semantic
    evaluation is reserved for formal checkpoints with reviewed training identity.
    """
    from .text_lora import load_text_lora
    from .text_artifacts import MODEL_FIELDS, CHECKPOINT_FIELDS, RUN_SCHEMA, check_raw_index, write_receipt
    from .features import FeatureIndex
    from .encoding import save_features
    from .evaluation import load_protocol, evaluate_run
    from .runs import code_revision
    protocol, candidates, queries, _ = load_protocol(protocol_path)
    encoder, checkpoint = load_text_lora(checkpoint_path, model_dir, device)
    index = FeatureIndex.load(index_path)
    check_raw_index(index, encoder.metadata, candidates, protocol["split"])
    mode = checkpoint["training_config"]["mode"]
    semantic = None
    if semantic_protocol is not None:
        from .semantic_evaluation import load_semantic_protocol
        if mode != "formal":
            raise ValueError("development checkpoint cannot produce formal semantic scores")
        semantic, gold, semantic_candidates = load_semantic_protocol(semantic_protocol)
        if (semantic["split"] != protocol["split"] or semantic["data_kind"] != protocol["data_kind"]
                or semantic["source_manifest_sha256"] != protocol["manifest_sha256"]
                or set(semantic_candidates) != set(candidates)):
            raise ValueError("semantic and retrieval protocols must bind the same split/manifest/gallery")
        if (protocol["split"] == "val" and checkpoint["provenance"].get("semantic_protocol_sha256")
                != file_digest(semantic_protocol)):
            raise ValueError("checkpoint selected against a different semantic validation protocol")
    experiment_id = experiment_id or (checkpoint["method"] + ("-no-relation" if
                         checkpoint["training_config"].get("include_relations") is False else ""))
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a nonempty string")
    revision = code_revision()
    with new_directory(output) as target:
        # Byte-identical snapshots make the export portable and atomic, including
        # after the staging directory is renamed. The shared source is untouched.
        shutil.copy2(checkpoint_path, target / "checkpoint.pt")
        shutil.copy2(index_path, target / "index.npz")
        shutil.copy2(str(index_path)+".meta.json", target / "index.npz.meta.json")
        vectors = encode_texts(encoder, [q["text"] for q in queries], batch_size)
        meta = {**encoder.metadata, **{k: checkpoint[k] for k in CHECKPOINT_FIELDS},
                "kind": "text", "split": protocol["split"], "mode": mode,
                "text_lora_sha256": checkpoint["text_lora_sha256"],
                "data_kind": protocol["data_kind"], "protocol_sha256": file_digest(protocol_path)}
        save_features(target / "texts.npz", [q["query_id"] for q in queries], vectors, meta)
        receipt = write_receipt(target / "receipt.json", checkpoint, target / "checkpoint.pt",
                                target / "index.npz", target / "texts.npz", experiment_id)
        hits = index.search_many(vectors, len(candidates))
        run = {"schema_version": RUN_SCHEMA, "run_id": f"{experiment_id}-seed{checkpoint['seed']}-{checkpoint['text_lora_sha256'][:12]}-{protocol['split']}",
               "data_kind": protocol["data_kind"], "protocol_sha256": file_digest(protocol_path),
               "code_revision": revision["code_revision"], "model_type": "text_lora", "mode": mode,
               "experiment_id": experiment_id,
               **{k: receipt[k] for k in (*MODEL_FIELDS, *CHECKPOINT_FIELDS, "text_lora_sha256", "image_index_sha256")},
               "retrieval_receipt": "receipt.json", "retrieval_receipt_sha256": file_digest(target / "receipt.json"),
               "ranking_depth": len(candidates), "rankings": {q["query_id"]: [h["image_id"] for h in row]
                                                            for q, row in zip(queries, hits)},
               "config": {"similarity": "cosine", "split": protocol["split"], "batch_size": batch_size,
                          "device": str(next(encoder.parameters()).device), "structure": encoder.structure,
                          "include_relations": checkpoint["training_config"].get("include_relations")}}
        write_json(target / "run.json", run)
        if mode == "formal":
            result = evaluate_run(protocol_path, target / "run.json")
            write_json(target / "retrieval-evaluation.json", result)
        else:
            from .text_artifacts import validate_run_v2
            validate_run_v2(target / "run.json", protocol_path, allow_development=True)
        if semantic is not None:
            positions = {image_id: i for i, image_id in enumerate(index.ids)}
            predictions = []
            for sample in gold:
                variants = sample["variants"]
                ids = ["original", *[v["id"] for v in variants]]
                texts = [sample["text"], *[v["text"] for v in variants]]
                encoded = encode_texts(encoder, texts, batch_size)
                scores = encoded @ index.matrix[positions[sample["source_image_id"]]]
                eligible = {"original", *[v["id"] for v in variants if v["type"] == "paraphrase"]}
                ranks = index.search_many(encoded[[i for i, vid in enumerate(ids) if vid in eligible]], len(candidates))
                predictions.append({"sample_id": sample["sample_id"], "scores": dict(zip(ids, map(float, scores))),
                                    "rankings": {vid: [h["image_id"] for h in row] for vid, row in
                                                 zip([vid for vid in ids if vid in eligible], ranks)}})
            score_run = {"schema_version": "fgir-semantic-scores-v1", "run_id": run["run_id"] + "-semantics",
                         "method": checkpoint["method"], "seed": checkpoint["seed"],
                         "protocol_sha256": file_digest(semantic_protocol), "data_kind": semantic["data_kind"],
                         "code_revision": revision["code_revision"], "ranking_depth": len(candidates),
                         "experiment_id": experiment_id,
                         "model": {"model_type": "text_lora", **{k: run[k] for k in MODEL_FIELDS},
                                   "text_lora_sha256": run["text_lora_sha256"],
                                   "retrieval_receipt_sha256": run["retrieval_receipt_sha256"],
                                   "semantic_dataset_sha256": run["semantic_dataset_sha256"]},
                         "predictions": predictions}
            write_json(target / "semantic-scores.json", score_run)
            from .semantic_evaluation import evaluate_semantics
            evaluate_semantics(semantic_protocol, target / "semantic-scores.json", target / "semantic-evaluation", bootstrap_samples=0)
        write_json(target / "export.json", {"mode": mode, "data_kind": protocol["data_kind"],
                    "code": revision, "text_lora_sha256": checkpoint["text_lora_sha256"],
                    "receipt_sha256": file_digest(target / "receipt.json"),
                    "parsing": "not supplied; B predictions required", "formal_result": mode == "formal"})
    return {"output": str(output), "mode": mode, "run": str(Path(output) / "run.json"),
            "text_lora_sha256": checkpoint["text_lora_sha256"],
            "retrieval_receipt_sha256": file_digest(Path(output) / "receipt.json")}


def validate_text_lora_checkpoint(*, checkpoint_path, model_dir, bundle, val_protocol, output):
    """C's selection callback: independent model, val only, exact four metrics."""
    from .evaluation import load_protocol
    from .semantic_evaluation import load_semantic_protocol
    bundle = Path(bundle)
    semantic, _, _ = load_semantic_protocol(val_protocol)
    standard_path = bundle / "val" / "protocol.json"
    standard, _, standard_queries, standard_qrels = load_protocol(standard_path)
    if (semantic["split"] != "val" or standard["split"] != "val"
            or semantic["source_manifest_sha256"] != file_digest(bundle / "manifest.jsonl")
            or standard["manifest_sha256"] != file_digest(bundle / "manifest.jsonl")):
        raise ValueError("validation callback requires val protocols from the same source bundle; never test")
    pairs = read_jsonl(bundle / "val" / "pairs.jsonl")
    expected_queries = {p["query_id"]: p["text"] for p in pairs}
    if (standard["annotation_scope"] != "source_pair"
            or {q["query_id"]: q["text"] for q in standard_queries} != expected_queries
            or standard_qrels != {p["query_id"]: [p["image_id"]] for p in pairs}):
        raise ValueError("selection R@1/MRR require all original val pairs, not the semantic subset")
    config_path = bundle / "retrieval-v2.json"
    config = read_json(config_path) if config_path.exists() else {}
    if config:
        unknown = set(config) - {"image_index", "device", "batch_size"}
        if unknown or not config.get("image_index"):
            raise ValueError("retrieval-v2.json requires image_index; only device/batch_size are optional")
    index = Path(config.get("image_index", "artifacts/c-experiments/seed42/retrieval/E0/val/index.npz"))
    if config and not index.is_absolute():
        index = bundle / index
    if not index.is_file():
        raise ValueError("D callback needs a raw val index: set image_index in bundle/retrieval-v2.json")
    export_text_lora(checkpoint_path, model_dir, index, standard_path, output,
                     semantic_protocol=val_protocol, device=config.get("device", "cpu"),
                     batch_size=config.get("batch_size", 64))
    output = Path(output)
    run = read_json(output / "run.json")
    retrieval = read_json(output / "retrieval-evaluation.json")["metrics"]
    diagnostics = read_json(output / "semantic-evaluation" / "metrics.json")["metrics"]
    result = {"text_lora_sha256": file_digest(checkpoint_path),
              "semantic_protocol_sha256": file_digest(val_protocol),
              "base_weights_sha256": run["base_weights_sha256"], "tokenizer_sha256": run["tokenizer_sha256"],
              "metrics": {"recall@1": retrieval["recall@1"], "mrr": retrieval["mrr"],
                          "attribute_binding": diagnostics["accuracy/attribute_binding"]["source_macro_mean"],
                          "relation_direction": diagnostics["accuracy/relation_direction"]["source_macro_mean"]},
              "artifacts": {name: {"path": name, "sha256": file_digest(output / name)} for name in
                            ("receipt.json", "run.json", "semantic-scores.json", "retrieval-evaluation.json", "semantic-evaluation/metrics.json")}}
    write_json(output / "validation.json", result)
    return result
