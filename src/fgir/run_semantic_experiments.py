"""Paired-seed N0-N3 orchestration with explicit external input gates."""

import importlib
from pathlib import Path

from .files import read_json, write_json


def experiment_preflight(bundle, model_dir, image_features, samples, val_protocol, config_root,
                         training_audit=None, data_root="data", validation_callback=None):
    from .text_training import text_training_config
    configs = {method: text_training_config(Path(config_root) / f"{method.lower()}.json")
               for method in ("N0", "N1", "N2", "N3")}
    shared = {k: v for k, v in configs["N0"].items() if k != "method"}
    if any(config["method"] != name or config["mode"] != "formal"
           or {k: v for k, v in config.items() if k != "method"} != shared for name, config in configs.items()):
        raise ValueError("controlled N configs must differ only in method")
    ablation = text_training_config(Path(config_root) / "n2-no-relation.json")
    if {k: v for k, v in ablation.items() if k != "include_relations"} != {
            k: v for k, v in configs["N2"].items() if k != "include_relations"}:
        raise ValueError("relation ablation changes unrelated configuration")
    if ablation["include_relations"] is not False:
        raise ValueError("relation ablation must disable relation supervision")
    missing = [{"owner": owner, "path": str(path), "needed_for": reason}
               for owner, path, reason in (
                   ("A", Path(bundle) / "dataset.json", "source image protocol"),
                   ("B/C", Path(model_dir), "explicit local CLIP weights/tokenizer"),
                   ("B", Path(image_features), "raw train image cache"),
                   ("B-V2-02", Path(samples), "verified train semantic samples"),
                   ("A-V2-04", Path(val_protocol), "independently reviewed validation gold"))
               if not path.exists()]
    if validation_callback is None:
        module, function = shared["validation_driver"].split(":", 1)
        try:
            validation_callback = getattr(importlib.import_module(module), function)
        except (ImportError, AttributeError):
            missing.append({"owner": "D-V2-01/02", "path": shared["validation_driver"],
                            "needed_for": "checkpoint-bound real validation scores/receipt"})
    result = {"status": "BLOCKED" if missing else "READY_FOR_DATA_VALIDATION", "missing": missing,
              "methods": ["N0", "N1", "N2", "N3", "N2-no-relation"],
              "control_policy": "same verified cohort, source/paraphrase pool, edit IDs, token forwards, LoRA structure and fixed update/validation budget; only objective changes"}
    if not missing:
        from .semantic_data import validate_semantics
        from .semantic_evaluation import load_semantic_protocol
        from .text_training_data import verify_training_audit
        audit = validate_semantics(samples, bundle, "train", data_root)
        verify_training_audit(training_audit, samples, __import__("fgir.files", fromlist=["read_jsonl"]).read_jsonl(samples))
        protocol, _, _ = load_semantic_protocol(val_protocol)
        if protocol["split"] != "val" or audit["data_kind"] != "real" or protocol["data_kind"] != "real":
            raise ValueError("paired real experiments require real train and independently reviewed val, never test/synthetic")
        result.update({"status": "READY", "training_audit": audit})
    return result, configs, ablation, validation_callback


def run_text_experiments(bundle, model_dir, image_features, samples, val_protocol, output,
                         config_root="configs/training", data_root="data", training_audit=None,
                         seeds=(42, 43, 44), check_only=False, validation_callback=None):
    preflight, configs, ablation, driver = experiment_preflight(bundle, model_dir, image_features, samples,
        val_protocol, config_root, training_audit, data_root, validation_callback)
    if check_only:
        return preflight
    if preflight["status"] != "READY":
        raise ValueError("formal N experiments remain blocked: " + "; ".join(f"{m['owner']}: {m['path']}" for m in preflight["missing"]))
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct integers")
    output = Path(output)
    if output.exists():
        raise ValueError("experiment output must be a new directory")
    output.mkdir(parents=True)
    write_json(output / "preflight.json", preflight)
    results = []
    from .text_training import train_text_lora
    for seed in seeds:
        reference = None
        for label in ("N0", "N1", "N2", "N3", "N2-no-relation"):
            config = (ablation if label == "N2-no-relation" else configs[label]) | {"seed": seed}
            config_path = output / f"{label}-seed{seed}.config.json"
            write_json(config_path, config)
            destination = output / f"{label}-seed{seed}"
            summary = train_text_lora(bundle, model_dir, image_features, config_path, destination,
                                      samples, data_root, training_audit, val_protocol,
                                      reference if label != "N0" else None, driver)
            results.append({"experiment": label, "seed": seed, "summary": str(destination / "summary.json"),
                            "status": summary["status"], "best_sha256": summary["best_sha256"]})
            write_json(output / "experiments.json", results)
            if label == "N0":
                reference = destination / "summary.json"
    return {"status": "completed" if all(r["status"] == "completed" for r in results) else "contains_failed_methods",
            "experiments": results, "formal_test_scored": False}
