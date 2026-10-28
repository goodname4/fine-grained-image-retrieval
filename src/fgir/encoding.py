"""Local CLIP-family encoding with explicit weights provenance; torch optional."""

import json
from pathlib import Path

from .contracts import image_path
from .data import file_digest
from .files import new_directory, read_json, write_json


def _torch_modules():
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ValueError("local encoding requires: python -m pip install -e '.[model]'") from exc
    return torch, AutoModel, AutoProcessor


def _weight_files(model_dir):
    directory = Path(model_dir)
    names = sorted(path.name for path in directory.iterdir()
                   if path.suffix in (".safetensors", ".bin", ".pt") and "optimizer" not in path.name)
    if not names:
        raise ValueError(f"no model weight files in {directory}")
    return names


def weights_digest(model_dir):
    """Single weight file: its own digest; sharded: digest of the per-file digest list."""
    names = _weight_files(model_dir)
    if len(names) == 1:
        return file_digest(Path(model_dir) / names[0])
    listing = "\n".join(f"{name}:{file_digest(Path(model_dir) / name)}" for name in names)
    return file_digest_text(listing)


def file_digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_model(source, output):
    """Download pretrained weights once and store them at an explicit local path."""
    torch, AutoModel, AutoProcessor = _torch_modules()
    output = Path(output)
    if output.exists():
        raise ValueError(f"model directory already exists: {output}")
    with new_directory(output) as staging:
        model = AutoModel.from_pretrained(source)
        processor = AutoProcessor.from_pretrained(source)
        model.save_pretrained(staging)
        processor.save_pretrained(staging)
        receipt = {
            "source": str(source),
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "weight_files": {name: file_digest(staging / name) for name in _weight_files(staging)},
            "weights_sha256": weights_digest(staging),
        }
        write_json(staging / "receipt.json", receipt)
    return {**receipt, "model_dir": str(output)}


def load_encoder(model_dir, device="cpu"):
    torch, AutoModel, AutoProcessor = _torch_modules()
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ValueError(f"model directory not found: {directory}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda or auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but not available")
    model = AutoModel.from_pretrained(directory, local_files_only=True)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model.eval().to(device)
    return ClipEncoder(model, processor, device, directory)


class ClipEncoder:
    """Raw (unnormalized) projected features; official preprocessing and tokenizer."""

    def __init__(self, model, processor, device, model_dir):
        self.model = model
        self.processor = processor
        self.device = device
        self.model_dir = Path(model_dir)
        self.weights_sha256 = weights_digest(self.model_dir)
        self.model_id = self._model_id()
        self.preprocessing = self._describe(processor.image_processor)
        self.text_processing = self._describe(processor.tokenizer)

    def _model_id(self):
        receipt = self.model_dir / "receipt.json"
        if receipt.exists():
            return read_json(receipt)["source"]
        return self.model.config._name_or_path

    def _describe(self, component):
        description = {"class": type(component).__name__}
        keys = ("size", "crop_size", "resample", "image_mean", "image_std",
                "model_max_length", "do_lower_case", "pad_token_id", "vocab_size")
        # transformers 5.x tokenizers have no to_dict(); fall back to attributes.
        try:
            config = component.to_dict()
        except AttributeError:
            config = {}
        for key in keys:
            value = config.get(key) if key in config else getattr(component, key, None)
            if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
                if value is not None:
                    description[key] = value
        # Cache sidecars are JSON. Canonicalize tuples from newer processors so
        # the live descriptor compares equal to its persisted representation.
        return json.loads(json.dumps(description))

    def metadata(self):
        return {"model_id": self.model_id, "weights_sha256": self.weights_sha256,
                "model_class": type(self.model).__name__, "device": self.device,
                "preprocessing": self.preprocessing, "text_processing": self.text_processing}

    def _project(self, name, **inputs):
        import torch
        features = getattr(self.model, name)(**inputs)
        if hasattr(features, "pooler_output") and not hasattr(features, "shape"):
            features = features.pooler_output
        return features.detach().to("cpu", torch.float32).numpy()

    def encode_images(self, data_root, rel_paths, batch_size=32, progress=None):
        import torch
        from PIL import Image
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(rel_paths), batch_size):
                chunk = rel_paths[start:start + batch_size]
                images = []
                for value in chunk:
                    with Image.open(image_path(data_root, value)) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                outputs.append(self._project("get_image_features", **inputs))
                for image in images:
                    image.close()
                if progress:
                    progress(min(start + batch_size, len(rel_paths)), len(rel_paths))
        return _concat(outputs, len(rel_paths))

    def encode_texts(self, texts, batch_size=64, progress=None):
        import torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                inputs = self.processor(text=list(chunk), return_tensors="pt",
                                        padding=True, truncation=True).to(self.device)
                outputs.append(self._project("get_text_features", **inputs))
                if progress:
                    progress(min(start + batch_size, len(texts)), len(texts))
        return _concat(outputs, len(texts))


def _concat(chunks, expected):
    import numpy
    if not chunks:
        raise ValueError("nothing to encode")
    matrix = numpy.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if matrix.shape[0] != expected:
        raise ValueError("encoded row count differs from inputs")
    return matrix


def encode_bundle(bundle, data_root, model_dir, output, device="cpu", batch_size=32):
    """Cache raw image and text features for every split of a handoff bundle."""
    from .handoff import load_split
    from .files import read_jsonl
    encoder = load_encoder(model_dir, device)
    output = Path(output)
    if output.exists():
        raise ValueError(f"feature directory already exists: {output}")
    counts = {}
    with new_directory(output) as staging:
        for split in ("train", "val", "test"):
            rows = load_split(bundle, split)
            ids = [row["image_id"] for row in rows]
            matrix = encoder.encode_images(data_root, [row["path"] for row in rows], batch_size,
                                           lambda done, total: _progress(f"{split} images", done, total))
            save_features(staging / f"{split}.images.npz", ids, matrix, encoder.metadata() | {"kind": "image", "split": split})
            pairs = read_jsonl(Path(bundle) / split / "pairs.jsonl")
            query_ids = [row["query_id"] for row in pairs]
            texts = [row["text"] for row in pairs]
            text_matrix = encoder.encode_texts(texts, batch_size,
                                               lambda done, total: _progress(f"{split} texts", done, total))
            save_features(staging / f"{split}.texts.npz", query_ids, text_matrix,
                          encoder.metadata() | {"kind": "text", "split": split, "text_rule": "original captions, no truncation or translation"})
            counts[split] = {"images": len(ids), "texts": len(query_ids)}
        write_json(staging / "store.json", {"schema_version": "fgir-features-v1",
                                            "model": encoder.metadata(), "splits": counts,
                                            "bundle": str(bundle),
                                            "note": "raw projected features; normalize at index build time"})
    return {"output": str(output), "splits": counts, "model": encoder.metadata()}


def save_features(path, ids, matrix, meta):
    import numpy
    if len(ids) != matrix.shape[0]:
        raise ValueError("ids and feature rows differ")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate feature ids")
    if not numpy.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    numpy.savez(path, ids=numpy.array(ids), vectors=matrix.astype(numpy.float32))
    write_json(Path(str(path) + ".meta.json"), {**meta, "count": len(ids), "dimension": int(matrix.shape[1]),
                                                "features_sha256": file_digest(path)})


def load_features(path):
    import numpy
    path = Path(path)
    meta = read_json(Path(str(path) + ".meta.json"))
    if meta.get("features_sha256") != file_digest(path):
        raise ValueError(f"feature file hash mismatch: {path.name}")
    with numpy.load(path, allow_pickle=False) as data:
        ids = [str(value) for value in data["ids"]]
        vectors = data["vectors"].astype(numpy.float32)
    if len(ids) != vectors.shape[0] or meta.get("count") != len(ids):
        raise ValueError(f"feature counts disagree: {path.name}")
    return ids, vectors, meta


def _progress(label, done, total):
    import sys
    print(f"{label}: {done}/{total}", file=sys.stderr, flush=True)
