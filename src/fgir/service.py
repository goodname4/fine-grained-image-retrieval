"""Local query service used by the D demo.

The service keeps one encoder, adapter and image index per model.  An adapted
model is only accepted when its checkpoint digest is recorded by the index,
which prevents accidentally querying an E1 index with E2 (or E0) features.
"""

from pathlib import Path
from time import perf_counter
import json

from .adapters import adapt_array, check_model_metadata, freeze_encoder, load_adapter
from .contracts import image_path
from .encoding import load_encoder
from .features import FeatureIndex
from .files import read_json
from .handoff import load_split


class RetrievalService:
    """Lazy-loading text-to-image retrieval for E0, E1 and E2."""

    def __init__(self, root, config=None, device="cpu", split="test"):
        self.root = Path(root).resolve()
        self.config = read_json(config) if config else {}
        self.device, self.split = device, split
        self._models = {}
        self._encoders = {}
        self._image_paths = None

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _settings(self, model):
        settings = self.config.get("models", {}).get(model, {})
        if model not in {"E0", "E1", "E2"}:
            raise ValueError("model must be E0, E1 or E2")
        required = ("index", "encoder") if model == "E0" else ("index", "checkpoint", "encoder")
        for key in required:
            if not settings.get(key):
                raise ValueError(f"missing {key} configuration for {model}")
        return settings

    def _load(self, model):
        if model in self._models:
            return self._models[model]
        settings = self._settings(model)
        index = FeatureIndex.load(self._path(settings["index"]))
        encoder_path = self._path(settings["encoder"]).resolve()
        encoder = self._encoders.get(encoder_path)
        if encoder is None:
            encoder = freeze_encoder(load_encoder(encoder_path, self.device))
            self._encoders[encoder_path] = encoder
        check_model_metadata(index.meta, encoder.metadata())
        if index.meta.get("kind") != "image" or index.meta.get("split") != self.split:
            raise ValueError("index kind/split does not match the configured gallery")
        if index.meta.get("dimension") != index.dimension or index.meta.get("count") != len(index.ids):
            raise ValueError("index metadata count/dimension differs from actual vectors")
        if set(index.ids) != set(self.image_paths()):
            raise ValueError("index IDs differ from the configured gallery")
        adapter = receipt = None
        if model == "E0":
            if index.meta.get("adapter_sha256") or index.meta.get("method") not in (None, "E0"):
                raise ValueError("E0 requires an unadapted index")
        else:
            adapter, receipt = load_adapter(self._path(settings["checkpoint"]), encoder.device,
                                            expected_model=encoder.metadata())
            actual = index.meta.get("adapter_sha256")
            if actual != receipt["adapter_sha256"]:
                raise ValueError(f"{model} index/checkpoint mismatch; rebuild the matching index")
            if receipt["config"]["method"] != model or index.meta.get("method") != model:
                raise ValueError("checkpoint/index method differs from selected model")
            if adapter.structure["dimension"] != index.dimension:
                raise ValueError("adapter/index dimensions differ")
        self._models[model] = (encoder, adapter, receipt, index)
        return self._models[model]

    def image_paths(self):
        if self._image_paths is None:
            bundle = self.config.get("bundle")
            data_root = self.config.get("data_root", ".")
            if not bundle:
                raise ValueError("bundle is required to display image paths")
            rows = load_split(self._path(bundle), self.split)
            self._image_paths = {row["image_id"]: str(image_path(self._path(data_root), row["path"])) for row in rows}
        return self._image_paths

    def search(self, query, model="E0", top_k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        start = perf_counter()
        encoder, adapter, receipt, index = self._load(model)
        loaded = perf_counter()
        vector = encoder.encode_texts([query])
        if adapter is not None:
            vector = adapt_array(adapter.text, vector)
        encoded = perf_counter()
        hits = index.search(vector[0], top_k=top_k)
        searched = perf_counter()
        paths = self.image_paths()
        resolved_hits = [{**hit, "rank": rank, "path": paths[hit["image_id"]],
                          "image_available": Path(paths[hit["image_id"]]).is_file()}
                         for rank, hit in enumerate(hits, 1)]
        finished = perf_counter()
        return {"model": model, "query": query, "top_k": top_k,
                "elapsed_ms": (finished - start) * 1000,
                "timing_ms": {"load": (loaded-start)*1000, "encoding": (encoded-loaded)*1000,
                              "ranking": (searched-encoded)*1000, "end_to_end": (finished-start)*1000},
                "provenance": {"weights_sha256": index.meta["weights_sha256"],
                               "adapter_sha256": index.meta.get("adapter_sha256"),
                               "index_sha256": index.meta["index_sha256"], "split": self.split,
                               "gallery_size": len(index.ids)},
                "hits": resolved_hits}

    def compare(self, query, models=("E0", "E1", "E2"), top_k=5):
        if not models or len(set(models)) != len(models):
            raise ValueError("select at least one model without duplicates")
        return {model: self.search(query, model, top_k) for model in models}


def export_results(results):
    """Portable download: retain image IDs and exclude machine paths."""
    portable = {model: {**result, "hits": [{key: value for key, value in hit.items() if key != "path"}
                                         for hit in result["hits"]]} for model, result in results.items()}
    return json.dumps({"schema_version": "fgir-demo-query-v1", "results": portable},
                      ensure_ascii=False, indent=2, allow_nan=False)
