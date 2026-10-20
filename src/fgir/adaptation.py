"""C → B/D: adapt raw caches and build matching image/text retrieval artifacts."""

from pathlib import Path

from .data import file_digest
from .encoding import load_features, save_features
from .features import FeatureIndex
from .files import new_directory, write_json


def export_adapter(checkpoint, image_features, text_features, output, device="cpu"):
    from .adapters import adapt_array, check_model_metadata, load_adapter
    ids, images, image_meta = load_features(image_features)
    queries, texts, text_meta = load_features(text_features)
    check_model_metadata(image_meta, text_meta)
    if image_meta.get("kind") != "image" or text_meta.get("kind") != "text":
        raise ValueError("image/text cache kinds are incorrect")
    if image_meta.get("split") != text_meta.get("split"):
        raise ValueError("image/text cache splits differ")
    if image_meta.get("adapter_sha256") or text_meta.get("adapter_sha256"):
        raise ValueError("adapter export requires raw, unadapted feature caches")
    adapter, receipt = load_adapter(checkpoint, device, expected_model=image_meta)
    if images.shape[1] != adapter.structure["dimension"] or texts.shape[1] != adapter.structure["dimension"]:
        raise ValueError("checkpoint and cache feature dimensions differ")
    extra = {"adapter_sha256": receipt["adapter_sha256"], "adapter_schema": receipt["schema_version"],
             "adapter_structure": receipt["structure"], "method": receipt["config"]["method"],
             "training_seed": receipt["config"]["seed"], "normalization": "l2_rows"}
    with new_directory(output) as target:
        adapted_images = adapt_array(adapter.image, images)
        adapted_texts = adapt_array(adapter.text, texts)
        save_features(target / "images.npz", ids, adapted_images, image_meta | extra)
        save_features(target / "texts.npz", queries, adapted_texts, text_meta | extra)
        FeatureIndex(ids, adapted_images, image_meta | extra).save(target / "index.npz")
        write_json(target / "receipt.json", {**extra, "model": receipt["model"],
                                             "raw_images_sha256": file_digest(image_features),
                                             "raw_texts_sha256": file_digest(text_features),
                                             "split": image_meta.get("split")})
    return {"output": str(output), **extra, "images": len(ids), "queries": len(queries)}
