"""Prepare project-local D development assets using the existing A/B/C APIs.

Download/model preparation is explicit; this script refuses to fake C's original
checkpoint. --rebuild-development creates a separate, two-step real-data smoke.
"""

import argparse
import gc
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-development", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    # C enables deterministic algorithms; cuBLAS requires this before CUDA starts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from fgir.data import file_digest
    from fgir.encoding import load_encoder, save_features, load_features
    from fgir.features import build_index
    from fgir.files import read_json, write_json
    from fgir.handoff import load_split
    from fgir.text_artifacts import load_receipt
    from fgir.text_export import export_text_lora
    root = Path(__file__).resolve().parents[1]
    # Commands run from any location still store assets inside this project.
    os.chdir(root)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model_dir, bundle = Path("models/clip-vit-base-patch32"), Path("data/prepared/flickr8k-v1")
    encoder = load_encoder(model_dir, args.device)
    features = Path("data/features/clip-vit-base-patch32")
    features.mkdir(parents=True, exist_ok=True)
    train_features = features / "train.d-v2-deterministic-v1.images.npz"
    splits = ("val", "train") if args.rebuild_development else ("val",)
    for split in splits:
        path = train_features if split == "train" else features / f"{split}.images.npz"
        rows = load_split(bundle, split)
        ids = [r["image_id"] for r in rows]
        if path.exists():
            existing_ids, _, meta = load_features(path)
            if existing_ids != ids or meta.get("weights_sha256") != encoder.weights_sha256:
                raise ValueError(f"existing cache differs; do not overwrite: {path}")
            print(f"Reusing checked {path}", flush=True)
            continue
        print(f"Encoding {split}: {len(rows)} real images", flush=True)
        matrix = encoder.encode_images("data", [r["path"] for r in rows], args.batch_size,
                    lambda done, total: print(f"{split}: {done}/{total}", flush=True))
        save_features(path, ids, matrix, encoder.metadata() | {"kind": "image", "split": split,
                      "source_manifest_sha256": file_digest(bundle / "manifest.jsonl"),
                      "precision_policy": "float32; TF32 disabled; deterministic algorithms"})
    index = Path("artifacts/d-v2/e0-val/index.npz")
    if not index.exists():
        build_index(features / "val.images.npz", index)
    del encoder
    gc.collect()
    result = {"data_kind": "real", "base_model": str(model_dir), "image_index": str(index),
              "image_index_sha256": file_digest(index), "formal_result": False}
    if args.rebuild_development:
        development = Path("artifacts/c-v2/n0-smoke-seed42-d-rebuild-v1")
        if not (development / "summary.json").exists():
            from fgir.text_training import train_text_lora
            print("Rebuilding two-step development smoke; not C's original checkpoint", flush=True)
            config_path = Path("artifacts/d-v2/local-smoke-config.json")
            config = read_json("configs/training/n0-smoke.json") | {"device": args.device}
            write_json(config_path, config)
            train_text_lora(bundle, model_dir, train_features, config_path,
                            development, data_root="data")
        summary = read_json(development / "summary.json")
        if summary["mode"] != "development_smoke" or summary["optimizer_steps"] != 2:
            raise ValueError("expected a two-step development checkpoint")
        export = Path("artifacts/d-v2/n0-development")
        if not export.exists():
            export_text_lora(development / "best.pt", model_dir, index, bundle / "val/protocol.json", export,
                             experiment_id="N0-development", device=args.device, batch_size=args.batch_size)
        load_receipt(export / "receipt.json", model_dir=model_dir)
        result.update({"checkpoint": str(development / "best.pt"), "checkpoint_sha256": summary["best_sha256"],
                       "checkpoint_origin": "D local reconstruction using C's published two-step config; not C's original artifact",
                       "optimizer_steps": summary["optimizer_steps"], "export": str(export),
                       "freeze_audit": summary["freeze_audit"], "reload_audit": summary["reload_audit"]})
    write_json(Path("artifacts/d-v2/local-assets.json"), result)
    print("Project-local assets prepared; formal N0-N3 remains gated by reviewed data.", flush=True)


if __name__ == "__main__":
    main()
