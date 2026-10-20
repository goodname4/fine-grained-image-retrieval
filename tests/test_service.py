"""D integration tests. All vectors/checkpoints here are synthetic, not model results."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_MODEL = all(importlib.util.find_spec(name) for name in ("torch", "numpy"))
if HAS_MODEL:
    import numpy as np
    import torch
    from fgir.adapters import DualAdapter, SCHEMA
    from fgir.data import file_digest
    from fgir.features import FeatureIndex
    from fgir.files import write_json
    from fgir.service import RetrievalService, export_results


@unittest.skipUnless(HAS_MODEL, "install .[training] for D service tests")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = {"model_id": "synthetic", "weights_sha256": "synthetic-weights",
                     "preprocessing": {"name": "synthetic"}, "text_processing": {"name": "synthetic"}}
        self.encoder = type("FakeEncoder", (), {
            "model": torch.nn.Linear(2, 2), "device": "cpu",
            "metadata": lambda _: self.meta,
            "encode_texts": lambda _, texts: np.array([[1., 0.] for text in texts], dtype=np.float32)})()
        self.rows = [{"image_id": key, "path": f"images/{key}.png"} for key in ("a", "b")]
        self.config = {"bundle": "bundle", "data_root": "data", "models": {}}
        for model in ("E0", "E1", "E2"):
            settings = {"encoder": "encoder", "index": f"{model}.npz"}
            extra = {}
            if model != "E0":
                adapter = DualAdapter(2, 2)
                # Make E1/E2 carry distinct payloads and E2 visibly change query direction.
                with torch.no_grad():
                    adapter.text.up.bias.copy_(torch.tensor([0., 10. if model == "E2" else 0.]))
                checkpoint = self.root / f"{model}.pt"
                torch.save({"schema_version": SCHEMA, "structure": adapter.structure,
                            "state_dict": adapter.state_dict(), "model": self.meta,
                            "config": {"method": model, "seed": 42}}, checkpoint)
                settings["checkpoint"] = checkpoint.name
                extra = {"adapter_sha256": file_digest(checkpoint), "method": model}
            self.save_index(model, extra)
            self.config["models"][model] = settings
        write_json(self.root / "demo.json", self.config)
        self.load = self.enterContext(patch("fgir.service.load_encoder", return_value=self.encoder))
        self.enterContext(patch("fgir.service.load_split", return_value=self.rows))

    def save_index(self, model, extra=None):
        FeatureIndex(["a", "b"], np.eye(2, dtype=np.float32),
                     self.meta | {"kind": "image", "split": "test"} | (extra or {})).save(self.root / f"{model}.npz")

    def service(self):
        return RetrievalService(self.root, self.root / "demo.json", device="auto")

    def test_switches_adapter_and_index_and_reuses_encoder(self):
        service = self.service()
        results = service.compare("synthetic query", top_k=9)
        self.assertEqual(results["E0"]["hits"][0]["image_id"], "a")
        self.assertEqual(results["E2"]["hits"][0]["image_id"], "b")
        self.assertEqual(len(results["E0"]["hits"]), 2)
        self.assertEqual(self.load.call_count, 1)
        self.assertFalse(any(p.requires_grad for p in self.encoder.model.parameters()))
        self.assertEqual(service.search("again", "E0")["hits"][0]["image_id"], "a")
        self.assertIn("encoding", results["E2"]["timing_ms"])
        self.assertFalse(results["E0"]["hits"][0]["image_available"])
        self.assertNotIn(str(self.root), export_results(results))
        self.assertEqual(json.loads(export_results(results))["results"]["E2"]["hits"][0]["rank"], 1)

    def test_rejects_e0_index_for_adapted_model(self):
        self.config["models"]["E1"]["index"] = "E0.npz"
        write_json(self.root / "demo.json", self.config)
        with self.assertRaisesRegex(ValueError, "index/checkpoint mismatch"):
            self.service().search("test", "E1")

    def test_rejects_wrong_checkpoint_and_e0_with_adapter(self):
        self.config["models"]["E1"]["checkpoint"] = "E2.pt"
        self.config["models"]["E0"]["index"] = "E2.npz"
        write_json(self.root / "demo.json", self.config)
        for model in ("E0", "E1"):
            with self.assertRaises(ValueError):
                self.service().search("test", model)

    def test_rejects_model_gallery_split_and_hash_drift(self):
        for extra in ({"weights_sha256": "other"}, {"split": "train"}, {"count": 99}):
            self.save_index("E0", extra)
            if "count" in extra:  # save() writes actual count, so mutate the sidecar.
                from fgir.files import read_json
                sidecar = self.root / "E0.npz.meta.json"
                write_json(sidecar, read_json(sidecar) | extra)
            with self.assertRaises(ValueError):
                self.service().search("test")
        self.save_index("E0")
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "gallery"):
            self.service().search("test")
        self.rows.append({"image_id": "b", "path": "images/b.png"})
        with (self.root / "E0.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.service().search("test")

    def test_validation_missing_assets_and_path_escape(self):
        service = self.service()
        for query, k in (("", 2), (" ", 2), ("test", 0), ("test", True)):
            with self.assertRaises(ValueError):
                service.search(query, top_k=k)
        with self.assertRaises(ValueError):
            service.search("test", model="unknown")
        with self.assertRaises(ValueError):
            service.compare("test", [])
        (self.root / "E0.npz").unlink()
        with self.assertRaises(OSError):
            service.search("test")
        self.rows[0]["path"] = "../escape.png"
        with self.assertRaises(ValueError):
            self.service().image_paths()

    def test_restart_produces_same_rankings(self):
        first = self.service().search("test", "E2")
        restarted = self.service().search("test", "E2")
        self.assertEqual(first["hits"], restarted["hits"])
        self.assertEqual(first["provenance"], restarted["provenance"])


if __name__ == "__main__":
    unittest.main()
