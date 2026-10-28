"""Contract-compliant run export for A's frozen protocol evaluation."""

import json
from pathlib import Path
import subprocess

from .data import file_digest
from .encoding import load_features
from .evaluation import load_protocol
from .features import FeatureIndex

RUN_SCHEMA = "fgir-run-v1"
METHODS = ("E0", "E1", "E2")


def _git(args, cwd):
    try:
        # Git emits UTF-8 path bytes under core.quotepath=false; the default
        # locale codec (e.g. GBK on Windows) can crash the reader thread.
        result = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", cwd=cwd, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    # Strip line endings only: porcelain paths need the leading status space intact.
    return result.stdout.rstrip("\r\n")


def code_revision(cwd=None):
    """Actual commit plus source digests for any uncommitted worktree changes."""
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"code_revision": "unknown", "worktree": "unknown"}
    status = _git(["-c", "core.quotepath=false", "status", "--porcelain", "--untracked-files=all"], cwd)
    if status is None:
        return {"code_revision": commit, "worktree": "unknown"}
    changed = sorted(line[3:].strip().strip('"') for line in status.splitlines() if line.strip())
    if not changed:
        return {"code_revision": commit, "worktree": "clean"}
    root = Path(cwd) if cwd else Path.cwd()
    digests = {}
    for rel in changed:
        path = root / rel
        if path.is_file():
            digests[rel.replace("\\", "/")] = file_digest(path)
    return {"code_revision": f"{commit}-dirty", "worktree": "dirty",
            "uncommitted_source_sha256": digests}


def _slug(value):
    return value.rstrip("/").rsplit("/", 1)[-1]


def export_run(protocol_path, index_path, query_features_path, output, method="E0",
               seed=42, run_id=None, depth=None, adapter_sha256=None, config_extra=None):
    """Rank every protocol query with one cached index and write a run JSON."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if method == "E0" and adapter_sha256 is not None:
        raise ValueError("unadapted E0 must not declare an adapter checkpoint")
    if method in ("E1", "E2"):
        if not isinstance(adapter_sha256, str) or len(adapter_sha256) != 64 \
                or any(c not in "0123456789abcdef" for c in adapter_sha256):
            raise ValueError("adapted E1/E2 runs must record a lowercase SHA-256 adapter checkpoint")
    protocol, candidates, queries, qrels = load_protocol(protocol_path)
    index = FeatureIndex.load(index_path)
    if sorted(index.ids) != sorted(candidates):
        raise ValueError("index gallery differs from the protocol candidates")
    query_ids = [row["query_id"] for row in queries]
    ids, vectors, meta = load_features(query_features_path)
    if sorted(ids) != sorted(query_ids):
        raise ValueError("query features must cover exactly the protocol queries")
    if meta.get("weights_sha256") != index.meta.get("weights_sha256"):
        raise ValueError("query and index features come from different weights")
    if meta.get("adapter_sha256") != index.meta.get("adapter_sha256"):
        raise ValueError("query and index use different adapter checkpoints")
    if meta.get("adapter_sha256") != adapter_sha256:
        raise ValueError("declared adapter does not match the actual query/index features")
    for key in ("model_id", "preprocessing", "text_processing"):
        if meta.get(key) != index.meta.get(key):
            raise ValueError(f"query/index model metadata mismatch: {key}")
    gallery = len(candidates)
    if depth is None:
        depth = gallery
    if type(depth) is not int or not 1 <= depth <= gallery:
        raise ValueError("depth must be between 1 and the candidate count")
    if depth < min(max(protocol["recall_ks"]), gallery):
        raise ValueError("ranking depth is insufficient for the requested Recall@K")
    position = {query_id: i for i, query_id in enumerate(ids)}
    ordered_vectors = vectors[[position[query_id] for query_id in query_ids]]
    hits = index.search_many(ordered_vectors, top_k=depth)
    rankings = {query_id: [hit["image_id"] for hit in row] for query_id, row in zip(query_ids, hits)}
    revision = code_revision()
    run = {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id or f"{method}-seed{seed}-{_slug(str(meta.get('model_id')))}-{protocol['split']}-d{depth}",
        "method": method, "model_id": meta.get("model_id"),
        "data_kind": protocol["data_kind"], "protocol_sha256": file_digest(protocol_path),
        "weights_sha256": meta.get("weights_sha256"), "adapter_sha256": adapter_sha256,
        "seed": seed, "code_revision": revision["code_revision"],
        "config": {
            "device": meta.get("device"), "preprocessing": meta.get("preprocessing"),
            "text_processing": meta.get("text_processing"),
            "text_rule": "original captions, no truncation or translation",
            "similarity": "cosine on L2-normalized features",
            "split": protocol["split"], "manifest_sha256": protocol["manifest_sha256"],
            "candidate_count": gallery, "query_count": len(query_ids),
            "feature_meta": {key: meta.get(key) for key in ("kind", "split", "count", "dimension")},
            "index_meta": {key: index.meta.get(key) for key in ("count", "dimension", "normalization")},
            "worktree": revision.get("worktree"),
            **({"uncommitted_source_sha256": revision["uncommitted_source_sha256"]}
               if "uncommitted_source_sha256" in revision else {}),
            **(config_extra or {}),
        },
        "ranking_depth": depth, "rankings": rankings,
    }
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(run, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    return {"run_id": run["run_id"], "method": method, "split": protocol["split"],
            "output": str(output), "bytes": output.stat().st_size,
            "query_count": len(query_ids), "ranking_depth": depth,
            "weights_sha256": run["weights_sha256"], "code_revision": run["code_revision"],
            "note": "similarity ranks candidates only; scores are not probabilities"}


def search_index(index_path, query_features_path, output, top_k=10):
    """Batch query entry point shared by D's demo path and E0 sanity checks."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    index = FeatureIndex.load(index_path)
    ids, vectors, meta = load_features(query_features_path)
    if meta.get("text_lora_sha256") is not None:
        from .text_artifacts import load_receipt
        receipt, _, _, _ = load_receipt(Path(query_features_path).parent / "receipt.json")
        if (receipt["image_index_sha256"] != file_digest(index_path)
                or receipt["text_features_sha256"] != file_digest(query_features_path)):
            raise ValueError("LoRA search index/text cache differs from receipt")
    else:
        if meta.get("weights_sha256") != index.meta.get("weights_sha256"):
            raise ValueError("query and index features come from different weights")
        if meta.get("adapter_sha256") != index.meta.get("adapter_sha256"):
            raise ValueError("query and index use different adapter checkpoints")
    hits = index.search_many(vectors, top_k=min(top_k, len(index.ids)))
    results = {query_id: hits_row for query_id, hits_row in zip(ids, hits)}
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"schema_version": "fgir-search-v1", "top_k": min(top_k, len(index.ids)),
                   "query_count": len(ids), "model_id": meta.get("model_id"),
                   "index": str(index_path), "results": results},
                  stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return {"output": str(output), "query_count": len(ids), "top_k": top_k}
