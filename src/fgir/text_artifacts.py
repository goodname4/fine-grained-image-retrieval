"""D v2 artifact bindings. No model dependencies until actual artifacts are read."""

import os
from pathlib import Path

from .data import file_digest
from .files import read_json, write_json
from .semantic_data import digest

RECEIPT_SCHEMA = "fgir-retrieval-receipt-v2"
RUN_SCHEMA = "fgir-run-v2"
MODEL_FIELDS = ("model_id", "base_weights_sha256", "tokenizer_sha256", "image_encoder_sha256",
                "projection_dimension", "preprocessing", "text_processing")
CHECKPOINT_FIELDS = ("method", "seed", "structure", "semantic_dataset_sha256", "training_config")
COMPARE_FIELDS = (*MODEL_FIELDS, "image_index_sha256", "structure", "semantic_dataset_sha256")


def check_raw_index(index, metadata, candidates=None, split=None):
    """Bind an unchanged image tower to B's original E0 index."""
    pairs = {"model_id": "model_id", "weights_sha256": "base_weights_sha256",
             "preprocessing": "preprocessing", "text_processing": "text_processing"}
    for left, right in pairs.items():
        if left not in index.meta or index.meta[left] != metadata[right]:
            raise ValueError(f"raw image index/base model mismatch: {left}")
    if (index.meta.get("adapter_sha256") is not None or index.meta.get("text_lora_sha256") is not None
            or index.meta.get("method") not in (None, "E0")):
        raise ValueError("text LoRA requires an unadapted E0 image index")
    if index.meta.get("kind") != "image" or (split and index.meta.get("split") != split):
        raise ValueError("raw image index kind/split mismatch")
    if (index.dimension != metadata["projection_dimension"] or index.meta.get("dimension") != index.dimension
            or index.meta.get("count") != len(index.ids)):
        raise ValueError("raw image index dimension/count mismatch")
    if index.meta.get("image_encoder_sha256", metadata["image_encoder_sha256"]) != metadata["image_encoder_sha256"]:
        raise ValueError("raw image tower digest mismatch")
    if candidates is not None and set(index.ids) != set(candidates):
        raise ValueError("raw image index gallery differs from protocol")


def write_receipt(path, checkpoint, checkpoint_path, index_path, text_path, experiment_id):
    path = Path(path)
    index_path, text_path, checkpoint_path = map(Path, (index_path, text_path, checkpoint_path))
    receipt = {"schema_version": RECEIPT_SCHEMA, "model_type": "text_lora", "experiment_id": experiment_id,
               **{k: checkpoint["model"][k] for k in MODEL_FIELDS},
               **{k: checkpoint[k] for k in CHECKPOINT_FIELDS},
               "mode": checkpoint["training_config"]["mode"],
               "text_lora_sha256": checkpoint["text_lora_sha256"],
               "image_index_sha256": file_digest(index_path), "text_features_sha256": file_digest(text_path)}
    artifacts = {"checkpoint": checkpoint_path, "image_index": index_path,
                 "image_index_meta": Path(str(index_path)+".meta.json"), "text_features": text_path,
                 "text_features_meta": Path(str(text_path)+".meta.json")}
    receipt["artifacts"] = {key: {"path": Path(os.path.relpath(value.resolve(), path.parent.resolve())).as_posix(),
                                 "sha256": file_digest(value)} for key, value in artifacts.items()}
    write_json(path, receipt)
    return receipt


def load_receipt(path, *, model_dir=None, device="cpu"):
    """Verify referenced bytes, checkpoint identity, index and query cache binding.

    model_dir additionally verifies the local base weights/tokenizer and actual
    target modules using C's independent loader. All references are relative.
    """
    path = Path(path)
    receipt = read_json(path)
    expected = {"schema_version", "model_type", "experiment_id", *MODEL_FIELDS, *CHECKPOINT_FIELDS,
                "mode", "text_lora_sha256", "image_index_sha256", "text_features_sha256", "artifacts"}
    if not isinstance(receipt, dict) or set(receipt) != expected or receipt["schema_version"] != RECEIPT_SCHEMA:
        raise ValueError("unknown or incomplete retrieval receipt")
    if receipt["model_type"] != "text_lora" or receipt["method"] not in ("N0", "N1", "N2", "N3") or type(receipt["seed"]) is not int:
        raise ValueError("invalid receipt model type/method/seed")
    if not isinstance(receipt["experiment_id"], str) or not receipt["experiment_id"].strip():
        raise ValueError("receipt requires an experiment_id")
    for field in ("base_weights_sha256", "tokenizer_sha256", "image_encoder_sha256",
                  "text_lora_sha256", "image_index_sha256", "text_features_sha256"):
        digest(receipt[field], field)
    if receipt["mode"] == "formal":
        digest(receipt["semantic_dataset_sha256"], "semantic_dataset_sha256")
    elif receipt["mode"] != "development_smoke" or receipt["method"] != "N0" or receipt["semantic_dataset_sha256"] is not None:
        raise ValueError("invalid development receipt; never invent a training digest")
    refs = receipt["artifacts"]
    names = {"checkpoint", "image_index", "image_index_meta", "text_features", "text_features_meta"}
    if not isinstance(refs, dict) or set(refs) != names:
        raise ValueError("receipt must bind checkpoint, image index and text features with sidecars")
    paths = {}
    for key, entry in refs.items():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or not isinstance(entry["path"], str):
            raise ValueError("invalid receipt artifact binding")
        if Path(entry["path"]).is_absolute() or not entry["path"]:
            raise ValueError("receipt references must be relative paths")
        paths[key] = path.parent / entry["path"]
        if file_digest(paths[key]) != entry["sha256"]:
            raise ValueError(f"receipt artifact hash mismatch: {key}")
    for name, field in (("checkpoint", "text_lora_sha256"), ("image_index", "image_index_sha256"),
                        ("text_features", "text_features_sha256")):
        if refs[name]["sha256"] != receipt[field]:
            raise ValueError(f"receipt identity differs from bound bytes: {field}")
    # Reuse C's strict schema/tensor/base validation when the base model is available.
    if model_dir is not None:
        from .text_lora import load_text_lora
        _, checkpoint = load_text_lora(paths["checkpoint"], model_dir, device)
    else:
        import torch
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        from .text_lora import CHECKPOINT_SCHEMA, FREEZE_POLICY, validate_structure
        required = {"schema_version", "model_type", "method", "seed", "structure", "model", "freeze_policy",
                    "semantic_dataset_sha256", "training_config", "provenance", "selection", "state_dict"}
        if not isinstance(checkpoint, dict) or set(checkpoint) != required:
            raise ValueError("unknown or incomplete receipt checkpoint")
        if (checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("freeze_policy") != FREEZE_POLICY
                or checkpoint.get("model_type") != "text_lora"):
            raise ValueError("receipt checkpoint schema/freeze policy mismatch")
        structure = checkpoint["structure"]
        try:
            paths_in_model = structure["target_paths"]
            layer_count = max(int(p.split(".")[3]) for p in paths_in_model) + 1
            validate_structure(structure, layer_count)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint LoRA structure") from exc
        state = checkpoint["state_dict"]
        if not isinstance(state, dict) or set(state) != {p+suffix for p in paths_in_model for suffix in (".lora_A", ".lora_B")}:
            raise ValueError("receipt checkpoint LoRA parameters are incomplete")
        for p in paths_in_model:
            a, b = state[p+".lora_A"], state[p+".lora_B"]
            if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
                    or a.ndim != 2 or b.ndim != 2 or not a.is_floating_point() or a.dtype != b.dtype
                    or a.shape[0] != structure["rank"] or b.shape != (a.shape[1], a.shape[0])
                    or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                raise ValueError("invalid receipt checkpoint LoRA parameter values/shapes")
    for key in MODEL_FIELDS:
        if checkpoint["model"].get(key) != receipt[key]:
            raise ValueError(f"receipt/checkpoint model mismatch: {key}")
    for key in CHECKPOINT_FIELDS:
        if checkpoint.get(key) != receipt[key]:
            raise ValueError(f"receipt/checkpoint mismatch: {key}")
    if checkpoint["training_config"].get("mode") != receipt["mode"]:
        raise ValueError("receipt/checkpoint mode mismatch")
    from .encoding import load_features
    from .features import FeatureIndex
    index = FeatureIndex.load(paths["image_index"])
    check_raw_index(index, receipt)
    if read_json(paths["image_index_meta"]) != index.meta:
        raise ValueError("receipt index metadata sidecar differs")
    ids, vectors, meta = load_features(paths["text_features"])
    if read_json(paths["text_features_meta"]) != meta:
        raise ValueError("receipt text metadata sidecar differs")
    for key in (*MODEL_FIELDS, "text_lora_sha256", "semantic_dataset_sha256", "structure", "method", "seed", "mode"):
        if key not in meta or meta[key] != receipt[key]:
            raise ValueError(f"receipt/text features mismatch: {key}")
    if (meta.get("kind") != "text" or meta.get("split") != index.meta.get("split")
            or vectors.ndim != 2 or vectors.shape[1] != index.dimension):
        raise ValueError("receipt text/index kind/split/dimension mismatch")
    import numpy as np
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("invalid receipt text features")
    return receipt, paths, index, (ids, vectors, meta)


def validate_run_v2(run_path, protocol_path, *, allow_development=False):
    from .evaluation import load_protocol
    run_path = Path(run_path)
    run = read_json(run_path)
    if run.get("schema_version") != RUN_SCHEMA:
        raise ValueError("unsupported v2 ranking run")
    if run.get("protocol_sha256") != file_digest(protocol_path):
        raise ValueError("run-v2 protocol mismatch")
    ref = run.get("retrieval_receipt")
    if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
        raise ValueError("run-v2 requires a relative retrieval receipt")
    receipt_path = run_path.parent / ref
    if file_digest(receipt_path) != run.get("retrieval_receipt_sha256"):
        raise ValueError("run-v2 receipt hash mismatch")
    receipt, _, index, (ids, vectors, meta) = load_receipt(receipt_path)
    if not allow_development and receipt["mode"] != "formal":
        raise ValueError("development_smoke is not a formal N-series result")
    for key in (*MODEL_FIELDS, *CHECKPOINT_FIELDS, "model_type", "mode", "experiment_id", "text_lora_sha256", "image_index_sha256"):
        if key not in run or run[key] != receipt[key]:
            raise ValueError(f"run-v2/receipt mismatch: {key}")
    protocol, candidates, queries, _ = load_protocol(protocol_path)
    check_raw_index(index, receipt, candidates, protocol["split"])
    if run.get("data_kind") != protocol["data_kind"] or meta.get("data_kind") != protocol["data_kind"]:
        raise ValueError("run-v2 data kind mismatch")
    if meta.get("protocol_sha256") != file_digest(protocol_path) or set(ids) != {q["query_id"] for q in queries}:
        raise ValueError("run-v2 text features/protocol mismatch")
    depth = run.get("ranking_depth")
    if type(depth) is not int or depth != len(candidates):
        raise ValueError("run-v2 exports complete rankings for full-gallery MRR")
    rankings = run.get("rankings")
    positions = {qid: i for i, qid in enumerate(ids)}
    expected = index.search_many(vectors[[positions[q["query_id"]] for q in queries]], depth)
    if rankings != {q["query_id"]: [h["image_id"] for h in hits] for q, hits in zip(queries, expected)}:
        raise ValueError("run-v2 rankings disagree with bound features/index")
    return run


def check_v2_comparison(runs):
    """Allow paired seeds' distinct checkpoints while holding the control inputs fixed."""
    if any(r.get("schema_version") == RUN_SCHEMA for r in runs):
        if any(r.get("schema_version") != RUN_SCHEMA for r in runs):
            raise ValueError("compare v1 and v2 runs in separate reports")
        for field in COMPARE_FIELDS:
            if any(r.get(field) != runs[0].get(field) for r in runs[1:]):
                raise ValueError(f"run-v2 comparison mismatch: {field}")
