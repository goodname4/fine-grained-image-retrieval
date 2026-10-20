"""Export actual E0/E1/E2 Top-K for A's pending common-pool annotation."""

import argparse
from pathlib import Path

from fgir.adapters import adapt_array, check_model_metadata, freeze_encoder, load_adapter
from fgir.contracts import image_path
from fgir.data import file_digest
from fgir.encoding import load_encoder
from fgir.features import FeatureIndex
from fgir.files import new_directory, read_json, read_jsonl, write_json


def export_pool(request_path, model_dir, experiments, output, device="cpu", depth=10):
    request_path, experiments = Path(request_path), Path(experiments)
    request = read_json(request_path)
    if request.get("schema_version") != "fgir-pool-request-v1":
        raise ValueError("unsupported pool request")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be positive")
    paths = {}
    for name in ("candidates.json", "queries.jsonl"):
        paths[name] = image_path(request_path.parent, request["files"][name]["path"])
        if file_digest(paths[name]) != request["files"][name]["sha256"]:
            raise ValueError("pool request file hash mismatch")
    queries, candidates = read_jsonl(paths["queries.jsonl"]), read_json(paths["candidates.json"])
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("duplicate pool query IDs")
    encoder = freeze_encoder(load_encoder(model_dir, device))
    raw_texts = encoder.encode_texts([q["text"] for q in queries])
    with new_directory(output) as target:
        for method in ("E0", "E1", "E2"):
            index = FeatureIndex.load(experiments / "retrieval" / method / "test/index.npz")
            if set(index.ids) != set(candidates):
                raise ValueError("pool gallery differs from the test index")
            check_model_metadata(index.meta, encoder.metadata())
            texts, digest = raw_texts, None
            if method != "E0":
                adapter, receipt = load_adapter(experiments / method / "best.pt", expected_model=encoder.metadata())
                digest = receipt["adapter_sha256"]
                if index.meta.get("adapter_sha256") != digest:
                    raise ValueError("pool index and checkpoint differ")
                texts = adapt_array(adapter.text, raw_texts)
            elif index.meta.get("adapter_sha256") is not None:
                raise ValueError("E0 index must be unadapted")
            hits = index.search_many(texts, min(depth, len(candidates)))
            write_json(target / f"{method.lower()}.pool.json", {
                "schema_version": "fgir-pool-run-v1", "request_sha256": file_digest(request_path),
                "model_id": f"{method}-{encoder.model_id}-{digest[:12] if digest else 'unadapted'}",
                "weights_sha256": encoder.weights_sha256, "adapter_sha256": digest,
                "purpose": "annotation candidates only, not formal model evaluation",
                "rankings": {q["query_id"]: [h["image_id"] for h in result] for q, result in zip(queries, hits)}})
    return {"output": str(output), "query_count": len(queries), "depth": min(depth, len(candidates))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()
    print(export_pool(args.request, args.model_dir, args.experiments, args.output, args.device, args.depth))
