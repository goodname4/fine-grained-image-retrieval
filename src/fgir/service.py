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
from .data import file_digest
from .encoding import load_encoder, weights_digest
from .features import FeatureIndex
from .files import read_json
from .handoff import load_split


class RetrievalService:
    """Lazy-loading text-to-image retrieval for E0, E1 and E2."""

    def __init__(self, root, config=None, device="cpu", split="test"):
        self.root = Path(root).resolve()
        self.config = read_json(config) if config else {}
        self.device, self.split = device, self.config.get("split", split)
        self._models = {}
        self._encoders = {}
        self._base_digests = {}
        self._image_paths = None

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _settings(self, model):
        settings = self.config.get("models", {}).get(model, {})
        if not settings:
            raise ValueError(f"model is not registered: {model}")
        settings = dict(settings)
        settings.setdefault("method", model)
        settings.setdefault("model_type", "raw_clip" if model == "E0" else "feature_mlp")
        kind, method = settings["model_type"], settings["method"]
        if not ((kind == "raw_clip" and method == "E0") or (kind == "feature_mlp" and method in ("E1", "E2"))
                or (kind == "text_lora" and method in ("N0", "N1", "N2", "N3"))):
            raise ValueError("registered method/model_type mismatch")
        required = ["index", "encoder"]
        if kind != "raw_clip":
            required.append("checkpoint")
        if kind == "text_lora":
            required.append("receipt")
        for key in required:
            if not settings.get(key):
                raise ValueError(f"missing {key} configuration for {model}")
        return settings

    def _cache_key(self, model, settings):
        # Content digests for artifacts, plus base file identities. Replacing a
        # checkpoint at the same path cannot silently reuse an old N instance.
        fingerprints = tuple((key, file_digest(self._path(settings[key])))
                             for key in ("index", "checkpoint", "receipt") if key in settings)
        fingerprints += (("index_meta", file_digest(str(self._path(settings["index"]))+".meta.json")),)
        if "receipt" in settings:
            receipt_path = self._path(settings["receipt"])
            payload = read_json(receipt_path)
            refs = payload.get("artifacts") if isinstance(payload, dict) else None
            if not isinstance(refs, dict) or any(not isinstance(e, dict) or not isinstance(e.get("path"), str)
                                                 for e in refs.values()):
                raise ValueError("invalid receipt artifact bindings")
            fingerprints += tuple((key, file_digest(receipt_path.parent / entry["path"]))
                                  for key, entry in sorted(refs.items()))
        directory = self._path(settings["encoder"])
        base = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(directory.glob("*")) if p.is_file())
        base_signature = (str(directory.resolve()), base)
        if base_signature not in self._base_digests:
            self._base_digests[base_signature] = weights_digest(directory) if directory.is_dir() else None
        return model, str(directory.resolve()), (self._base_digests[base_signature], base), fingerprints, self.device

    def available_models(self, validate=True):
        """Registered entries with missing/invalid artifacts remain diagnostics."""
        ready, unavailable = {}, {}
        for name in self.config.get("models", {}):
            try:
                settings = self._settings(name)
                for key in ("index", "checkpoint", "receipt"):
                    if key in settings and not self._path(settings[key]).is_file():
                        raise ValueError(f"missing {key}: {settings[key]}")
                if not self._path(settings["encoder"]).is_dir():
                    raise ValueError(f"missing local base model: {settings['encoder']}")
                if validate:
                    self._load(name)
                ready[name] = settings
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                unavailable[name] = str(exc)
        return ready, unavailable

    def _load(self, model):
        settings = self._settings(model)
        cache_key = self._cache_key(model, settings)
        if cache_key in self._models:
            return self._models[cache_key]
        if settings["model_type"] == "text_lora":
            from .text_lora import load_text_lora
            from .text_artifacts import MODEL_FIELDS, check_raw_index, load_receipt
            receipt, _, index, _ = load_receipt(self._path(settings["receipt"]))
            encoder, checkpoint = load_text_lora(self._path(settings["checkpoint"]),
                                                self._path(settings["encoder"]), self.device)
            if (receipt["text_lora_sha256"] != checkpoint["text_lora_sha256"]
                    or receipt["image_index_sha256"] != file_digest(self._path(settings["index"]))
                    or receipt["method"] != settings["method"] or receipt["experiment_id"] != model):
                raise ValueError("registered LoRA/index/receipt/experiment identity mismatch")
            if any(encoder.metadata[k] != receipt[k] for k in MODEL_FIELDS):
                raise ValueError("registered base model/receipt mismatch")
            check_raw_index(index, encoder.metadata, self.image_paths(), self.split)
            # C's loader always creates its own base instance. Never inject into _encoders.
            value = (encoder, None, receipt, index)
            self._models = {k: v for k, v in self._models.items() if k[0] != model}
            self._models[cache_key] = value
            return value
        index = FeatureIndex.load(self._path(settings["index"]))
        encoder_path = self._path(settings["encoder"]).resolve()
        base_key = (encoder_path, cache_key[2])
        encoder = self._encoders.get(base_key)
        if encoder is None:
            encoder = freeze_encoder(load_encoder(encoder_path, self.device))
            self._encoders[base_key] = encoder
        check_model_metadata(index.meta, encoder.metadata())
        if index.meta.get("kind") != "image" or index.meta.get("split") != self.split:
            raise ValueError("index kind/split does not match the configured gallery")
        if index.meta.get("dimension") != index.dimension or index.meta.get("count") != len(index.ids):
            raise ValueError("index metadata count/dimension differs from actual vectors")
        if set(index.ids) != set(self.image_paths()):
            raise ValueError("index IDs differ from the configured gallery")
        adapter = receipt = None
        if settings["model_type"] == "raw_clip":
            if index.meta.get("adapter_sha256") or index.meta.get("method") not in (None, "E0"):
                raise ValueError("E0 requires an unadapted index")
        else:
            adapter, receipt = load_adapter(self._path(settings["checkpoint"]), encoder.device,
                                            expected_model=encoder.metadata())
            actual = index.meta.get("adapter_sha256")
            if actual != receipt["adapter_sha256"]:
                raise ValueError(f"{model} index/checkpoint mismatch; rebuild the matching index")
            if receipt["config"]["method"] != settings["method"] or index.meta.get("method") != settings["method"]:
                raise ValueError("checkpoint/index method differs from selected model")
            if adapter.structure["dimension"] != index.dimension:
                raise ValueError("adapter/index dimensions differ")
        self._models = {k: v for k, v in self._models.items() if k[0] != model}
        self._models[cache_key] = (encoder, adapter, receipt, index)
        return self._models[cache_key]

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
        settings = self._settings(model)
        actual_device = str(next(encoder.model.parameters()).device)
        if actual_device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(actual_device)
        loaded = perf_counter()
        if settings["model_type"] == "text_lora":
            from .text_export import encode_texts
            vector = encode_texts(encoder, [query])
        else:
            vector = encoder.encode_texts([query])
        if adapter is not None:
            vector = adapt_array(adapter.text, vector)
        if actual_device.startswith("cuda"):
            torch.cuda.synchronize(actual_device)
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
                               "model_type": settings["model_type"], "method": settings["method"],
                               "experiment_id": model, "device": actual_device,
                               **({key: receipt[key] for key in ("base_weights_sha256", "tokenizer_sha256",
                                  "image_encoder_sha256", "text_lora_sha256", "semantic_dataset_sha256", "mode", "structure")}
                                  if settings["model_type"] == "text_lora" else {}),
                               **({"retrieval_receipt_sha256": file_digest(self._path(settings["receipt"]))}
                                  if settings["model_type"] == "text_lora" else {}),
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
