"""D v2 end-to-end contracts with local random tiny-CLIP, never real results."""

import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import subprocess
import sys

HAS_STACK = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "numpy", "PIL"))
if HAS_STACK:
    import numpy as np
    import torch
    from test_text_lora import tiny_model, training_fixture
    from fgir.data import file_digest
    from fgir.dataset import write_protocol
    from fgir.encoding import load_encoder
    from fgir.evaluation import compare_runs, evaluate_run, summarize_seeds
    from fgir.features import FeatureIndex
    from fgir.files import read_json, read_jsonl, write_json
    from fgir.service import RetrievalService, export_results
    from fgir.text_artifacts import check_raw_index, load_receipt, validate_run_v2
    from fgir.text_export import export_text_lora, validate_text_lora_checkpoint
    from fgir.text_lora import create_text_lora, save_text_lora


@unittest.skipUnless(HAS_STACK, "install .[semantic-training] for D v2 export tests")
class TextExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model_dir = tiny_model(cls.root)
        cls.bundle, cls.data, _, cls.samples, cls.semantic = training_fixture(cls.root, cls.model_dir)
        cls.protocol = cls.bundle / "val/protocol.json"
        rows = read_jsonl(cls.bundle / "manifest.jsonl")
        write_protocol(cls.protocol.parent, rows, file_digest(cls.bundle / "manifest.jsonl"), "val")
        protocol = read_json(cls.protocol)
        write_json(cls.protocol, protocol | {"data_kind": "synthetic"})
        encoder = load_encoder(cls.model_dir)
        subset = [r for r in rows if r["split"] == "val"]
        cls.index = cls.root / "raw-val.npz"
        FeatureIndex([r["image_id"] for r in subset], encoder.encode_images(cls.data, [r["path"] for r in subset]),
                     encoder.metadata() | {"kind": "image", "split": "val"}).save(cls.index)
        write_json(cls.bundle / "retrieval-v2.json", {"image_index": "../raw-val.npz", "device": "cpu", "batch_size": 2})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        cls.temp.cleanup()

    def checkpoint(self, name, method="N0", seed=42, development=False, relations=True):
        encoder = create_text_lora(self.model_dir, rank=2)
        with torch.no_grad():
            for name, value in encoder.model.named_parameters():
                if name.endswith("lora_B"):
                    value.add_(.001 if method == "N2" else .003)
        path = self.root / (name + ".pt")
        save_text_lora(path, encoder, method, seed,
                       {"mode": "development_smoke" if development else "formal", "include_relations": relations},
                       {"semantic_protocol_sha256": file_digest(self.semantic)}, {"epoch": 0},
                       semantic_dataset_sha256=None if development else file_digest(self.samples))
        return path

    def export(self, name, **kwargs):
        checkpoint = self.checkpoint(name, **kwargs)
        output = self.root / name
        export_text_lora(checkpoint, self.model_dir, self.index, self.protocol, output,
                         semantic_protocol=None if kwargs.get("development") else self.semantic)
        return output

    def test_formal_callback_exact_metrics_and_artifact_chain(self):
        checkpoint = self.checkpoint("callback")
        result = validate_text_lora_checkpoint(checkpoint_path=checkpoint, model_dir=self.model_dir,
                    bundle=self.bundle, val_protocol=self.semantic, output=self.root / "callback")
        self.assertEqual(set(result["metrics"]), {"recall@1", "mrr", "attribute_binding", "relation_direction"})
        self.assertEqual(result["text_lora_sha256"], file_digest(checkpoint))
        output = self.root / "callback"
        receipt, _, _, _ = load_receipt(output / "receipt.json", model_dir=self.model_dir)
        self.assertEqual(receipt["mode"], "formal")
        scored = evaluate_run(self.protocol, output / "run.json")
        self.assertEqual(scored["run"]["schema_version"], "fgir-run-v2")
        self.assertIn("mrr", scored["metrics"])
        from fgir.runs import search_index
        search_index(output / "index.npz", output / "texts.npz", self.root / "search-v2.json", top_k=1)
        self.assertEqual(read_json(self.root / "search-v2.json")["query_count"], len(scored["per_query"]))
        semantic = read_json(output / "semantic-scores.json")
        self.assertNotIn("parsing", semantic["predictions"][0])
        self.assertNotIn("binding", semantic["predictions"][0]["rankings"])
        # All exported artifact references remain valid after moving the bundle.
        moved = self.root / "moved-callback"
        shutil.copytree(output, moved)
        load_receipt(moved / "receipt.json")

    def test_development_null_digest_never_becomes_formal(self):
        output = self.export("development", development=True)
        receipt, _, _, _ = load_receipt(output / "receipt.json")
        self.assertIsNone(receipt["semantic_dataset_sha256"])
        validate_run_v2(output / "run.json", self.protocol, allow_development=True)
        with self.assertRaisesRegex(ValueError, "development_smoke"):
            evaluate_run(self.protocol, output / "run.json")
        with self.assertRaisesRegex(ValueError, "development"):
            validate_text_lora_checkpoint(checkpoint_path=output / "checkpoint.pt", model_dir=self.model_dir,
                    bundle=self.bundle, val_protocol=self.semantic, output=self.root / "development-callback")
        self.assertFalse((self.root / "development-callback").exists())

    def test_receipt_and_run_identity_rejections(self):
        output = self.export("identity")
        receipt_path, run_path = output / "receipt.json", output / "run.json"
        original_receipt, original_run = read_json(receipt_path), read_json(run_path)
        for key in ("base_weights_sha256", "tokenizer_sha256", "text_lora_sha256", "image_index_sha256",
                    "image_encoder_sha256", "retrieval_receipt_sha256", "structure", "semantic_dataset_sha256"):
            run = copy.deepcopy(original_run)
            run.pop(key)
            write_json(run_path, run)
            with self.assertRaises(ValueError, msg=key):
                evaluate_run(self.protocol, run_path)
        write_json(run_path, original_run)
        for key in ("tokenizer_sha256", "base_weights_sha256", "image_index_sha256", "text_lora_sha256"):
            receipt = copy.deepcopy(original_receipt)
            receipt[key] = "0" * 64
            write_json(receipt_path, receipt)
            with self.assertRaises(ValueError, msg=key):
                load_receipt(receipt_path)
        write_json(receipt_path, original_receipt)
        original_run["rankings"] = {}
        write_json(run_path, original_run)
        with self.assertRaisesRegex(ValueError, "rankings"):
            evaluate_run(self.protocol, run_path)

    def test_shared_raw_index_and_distinct_seed_checkpoint_comparison(self):
        left = self.export("paired42", seed=42)
        right = self.export("paired43", seed=43)
        compare_runs(self.protocol, [left / "run.json", right / "run.json"], self.root / "comparison", bootstrap_samples=0)
        summarize_seeds(self.protocol, [left / "run.json", right / "run.json"], self.root / "seeds")
        receipts = [read_json(p / "receipt.json") for p in (left, right)]
        self.assertEqual(receipts[0]["image_index_sha256"], receipts[1]["image_index_sha256"])
        self.assertNotEqual(receipts[0]["text_lora_sha256"], receipts[1]["text_lora_sha256"])
        ablation = self.export("ablation", method="N2", relations=False)
        self.assertEqual(read_json(ablation / "run.json")["experiment_id"], "N2-no-relation")

    def test_val_callback_rejects_test_protocol(self):
        checkpoint = self.checkpoint("never-test")
        with self.assertRaisesRegex(ValueError, "never test"):
            validate_text_lora_checkpoint(checkpoint_path=checkpoint, model_dir=self.model_dir, bundle=self.bundle,
                    val_protocol=self.root / "frozen-test/protocol.json", output=self.root / "test-output")
        self.assertFalse((self.root / "test-output").exists())

    def test_comparison_rejects_changed_lora_structure(self):
        left = self.export("structure-left")
        encoder = create_text_lora(self.model_dir, rank=1)
        checkpoint = self.root / "rank1.pt"
        save_text_lora(checkpoint, encoder, "N1", 42, {"mode": "formal", "include_relations": True},
                       {"semantic_protocol_sha256": file_digest(self.semantic)}, {"epoch": 0}, file_digest(self.samples))
        right = self.root / "structure-right"
        export_text_lora(checkpoint, self.model_dir, self.index, self.protocol, right)
        with self.assertRaisesRegex(ValueError, "structure"):
            compare_runs(self.protocol, [left / "run.json", right / "run.json"], self.root / "invalid-comparison")
        self.assertFalse((self.root / "invalid-comparison").exists())

    def test_service_n2_e0_n3_n2_isolation_and_missing_models(self):
        n2, n3 = self.export("service-n2", method="N2"), self.export("service-n3", method="N3")
        config = {"bundle": str(self.bundle), "data_root": str(self.data), "split": "val", "models": {
            "E0": {"encoder": str(self.model_dir), "index": str(self.index)}}}
        for method, output in (("N2", n2), ("N3", n3)):
            config["models"][method] = {"model_type": "text_lora", "method": method,
                "encoder": str(self.model_dir), "index": str(self.index),
                "checkpoint": str(output / "checkpoint.pt"), "receipt": str(output / "receipt.json")}
        config["models"]["N1"] = {**config["models"]["N2"], "method": "N1", "checkpoint": "not-present.pt"}
        path = self.root / "service.json"
        write_json(path, config)
        service = RetrievalService(self.root, path)
        before = service.search("A man walks a dog.", "N2")
        e0 = service.search("A man walks a dog.", "E0")
        service.search("A man walks a dog.", "N3")
        after = service.search("A man walks a dog.", "N2")
        self.assertEqual(before["hits"], after["hits"])
        self.assertEqual(e0["provenance"]["model_type"], "raw_clip")
        ready, missing = service.available_models()
        self.assertEqual(set(ready), {"N2", "N3", "E0"})
        self.assertIn("N1", missing)
        self.assertNotIn(str(self.root).replace('\\', '\\\\'), export_results({"N2": after}))
        n_encoder = service._load("N2")[0].model
        e_encoder = service._load("E0")[0].model
        self.assertIsNot(n_encoder, e_encoder)
        self.assertFalse(any("lora_" in name for name, _ in e_encoder.named_parameters()))

    def test_cli_exports_and_benchmark_identifies_synthetic_input(self):
        checkpoint = self.checkpoint("cli", development=True)
        output = self.root / "cli-export"
        process = subprocess.run([sys.executable, "-m", "fgir", "export-text-lora", "--checkpoint", str(checkpoint),
                    "--model-dir", str(self.model_dir), "--index", str(self.index), "--protocol", str(self.protocol),
                    "--output", str(output)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("development_smoke", process.stdout)
        formal_checkpoint = self.checkpoint("cli-validation")
        validated = self.root / "cli-validation-export"
        process = subprocess.run([sys.executable, "-m", "fgir", "validate-text-lora", "--checkpoint", str(formal_checkpoint),
                    "--model-dir", str(self.model_dir), "--bundle", str(self.bundle), "--val-protocol", str(self.semantic),
                    "--output", str(validated)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(set(read_json(validated / "validation.json")["metrics"]),
                         {"recall@1", "mrr", "attribute_binding", "relation_direction"})
        from fgir.benchmark import benchmark
        config = self.root / "benchmark.json"
        write_json(config, {"bundle": str(self.bundle), "data_root": str(self.data), "split": "val",
                          "models": {"E0": {"encoder": str(self.model_dir), "index": str(self.index)}}})
        result = benchmark(RetrievalService(self.root, config), ["A man walks."], ["E0"], repeats=2)
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["models"]["E0"]["sample_count"], 2)
        self.assertIn("p95", result["models"]["E0"]["warm_ms"]["encoding"])

    def test_wrong_text_cache_and_index_metadata_are_rejected(self):
        output = self.export("cache-validation")
        meta_path = output / "texts.npz.meta.json"
        meta = read_json(meta_path)
        write_json(meta_path, meta | {"text_lora_sha256": "0" * 64})
        # Bind the new sidecar bytes so validation reaches the identity check.
        receipt_path = output / "receipt.json"
        receipt = read_json(receipt_path)
        receipt["artifacts"]["text_features_meta"]["sha256"] = file_digest(meta_path)
        write_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "text features mismatch"):
            load_receipt(receipt_path)
        index = FeatureIndex.load(self.index)
        index.meta["adapter_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "unadapted"):
            check_raw_index(index, receipt)


if __name__ == "__main__":
    unittest.main()
