"""C's gradient, data-leakage, provenance and train/export integration checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fgir.data import file_digest
from fgir.files import read_json, read_jsonl, write_json, write_jsonl

HAS_TRAINING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_TRAINING:
    import numpy as np
    import torch
    from fgir.adaptation import export_adapter
    from fgir.adapters import DualAdapter, ResidualAdapter, adapt_array, freeze_encoder, load_adapter
    from fgir.dataset import write_protocol
    from fgir.encoding import load_features, save_features
    from fgir.evaluation import evaluate_run
    from fgir.experiments_c import run_c_experiments
    from fgir.features import FeatureIndex
    from fgir.handoff import training_pairs_for_epoch
    from fgir.losses import hard_negative_loss, pair_masks, symmetric_contrastive_loss
    from fgir.runs import export_run, search_index
    from fgir.training import train_adapter, training_config
    from fgir.training_data import CachedSplit, load_training_negatives


@unittest.skipUnless(HAS_TRAINING, "install .[training] for C tests")
class LossTests(unittest.TestCase):
    def test_zero_init_is_identity_and_gradients_reach_both_branches(self):
        torch.manual_seed(7)
        adapter = DualAdapter(8, 3)
        x = torch.randn(4, 8)
        torch.testing.assert_close(adapter.image(x), torch.nn.functional.normalize(x, dim=-1))
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            loss = (adapter.image(x) * torch.randn(4, 8)).sum() + (adapter.text(x) * torch.randn(4, 8)).sum()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))
            optimizer.step()
        self.assertGreater(adapter.image.down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.text.down.weight.grad.abs().sum().item(), 0)
        self.assertEqual(adapter.parameter_counts()["encoder_trainable"], 0)
        self.assertEqual(DualAdapter(512, 128).parameter_counts()["trainable"], 263424)

    def test_single_positive_equals_symmetric_cross_entropy(self):
        torch.manual_seed(3)
        images, texts = torch.randn(4, 8), torch.randn(4, 8)
        logits = torch.nn.functional.normalize(images, dim=-1) @ torch.nn.functional.normalize(texts, dim=-1).T / 0.07
        target = torch.arange(4)
        expected = (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target)) / 2
        actual = symmetric_contrastive_loss(images, texts, torch.eye(4, dtype=torch.bool))
        torch.testing.assert_close(actual, expected)

    def test_multiple_captions_are_positive_and_unknown_scene_pairs_ignored(self):
        positive, allowed = pair_masks(["a", "b", "c"], ["a", "a", "b", "c"], {"a": "g", "b": "g"})
        self.assertEqual(positive.sum().item(), 4)
        self.assertFalse(allowed[0, 2])
        self.assertFalse(positive[0, 2])
        images, texts = torch.randn(3, 5, requires_grad=True), torch.randn(4, 5, requires_grad=True)
        loss = symmetric_contrastive_loss(images, texts, positive, allowed)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue(torch.isfinite(texts.grad).all())

    def test_identical_caption_other_image_is_ignored(self):
        positive, allowed = pair_masks(["a", "b"], ["a", "b"], excluded_by_text=[{"b"}, set()])
        self.assertFalse(allowed[1, 0])
        self.assertTrue(allowed[0, 1])
        self.assertTrue(positive[0, 0])

    def test_hard_loss_margin_masks_and_gradient_to_negative_adapter(self):
        text = torch.tensor([[1., 0.]], requires_grad=True)
        positive = torch.tensor([[0., 1.]], requires_grad=True)
        negatives = torch.tensor([[[1., 0.], [0., -1.]]], requires_grad=True)
        value = hard_negative_loss(text, positive, negatives, torch.tensor([[True, False]]), 0.2)
        self.assertAlmostEqual(value.item(), 1.2, places=6)
        value.backward()
        self.assertIsNotNone(negatives.grad)
        self.assertEqual(negatives.grad[0, 1].abs().sum(), 0)
        zero = hard_negative_loss(text, positive, negatives, torch.tensor([[False, False]]))
        self.assertEqual(zero.item(), 0)

    def test_no_positive_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_contrastive_loss(torch.ones(2, 3), torch.ones(2, 3), torch.zeros(2, 2, dtype=torch.bool))

    def test_freeze_encoder(self):
        class Encoder:
            model = torch.nn.Linear(3, 3)
        encoder = freeze_encoder(Encoder())
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))


@unittest.skipUnless(HAS_TRAINING, "install .[training] for C tests")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle, self.features = self.root / "bundle", self.root / "features"
        self.bundle.mkdir()
        self.features.mkdir()
        self.meta = {"model_id": "synthetic-training-fixture", "weights_sha256": "a" * 64,
                     "preprocessing": {"fixture": True}, "text_processing": {"fixture": True}, "device": "cpu"}
        rows = [{"image_id": f"{split}{i}", "path": f"images/{split}{i}.jpg", "split": split,
                 "captions": [f"caption {split} {i} {j}" for j in range(2)]}
                for split, count in (("train", 8), ("val", 4), ("test", 4)) for i in range(count)]
        write_jsonl(self.bundle / "manifest.jsonl", rows)
        write_json(self.bundle / "duplicate-review.json", {"pairs": []})
        rng = np.random.default_rng(3)
        for split in ("train", "val", "test"):
            directory = self.bundle / split
            directory.mkdir()
            subset = [r for r in rows if r["split"] == split]
            pairs = [{"query_id": f"{r['image_id']}#{j}", "image_id": r["image_id"],
                      "path": r["path"], "text": text, "split": split}
                     for r in subset for j, text in enumerate(r["captions"])]
            write_jsonl(directory / "images.jsonl", subset)
            write_jsonl(directory / "pairs.jsonl", pairs)
            images = rng.normal(size=(len(subset), 8)).astype(np.float32)
            texts = np.repeat(images, 2, axis=0) + rng.normal(0, 0.1, (len(pairs), 8)).astype(np.float32)
            save_features(self.features / f"{split}.images.npz", [r["image_id"] for r in subset], images,
                          self.meta | {"split": split, "kind": "image"})
            save_features(self.features / f"{split}.texts.npz", [p["query_id"] for p in pairs], texts,
                          self.meta | {"split": split, "kind": "text"})
            if split != "train":
                protocol = write_protocol(directory, rows, file_digest(self.bundle / "manifest.jsonl"), split)
                # Explicitly label numeric fixtures as synthetic, including exported runs.
                protocol["data_kind"] = "synthetic"
                write_json(directory / "protocol.json", protocol)
        files = {p.relative_to(self.bundle).as_posix(): file_digest(p) for p in self.bundle.rglob("*") if p.is_file()}
        write_json(self.bundle / "dataset.json", {"schema_version": "fgir-dataset-v1", "files": files,
                                                  "manifest_sha256": file_digest(self.bundle / "manifest.jsonl")})
        self.config_path = self.root / "config.json"
        self.config = {"method": "E1", "epochs": 2, "batch_size": 4, "bottleneck": 3,
                       "device": "cpu", "threads": 1, "patience": 2}
        write_json(self.config_path, self.config)
        self.output = self.root / "training"
        self.negatives_path = self.root / "negatives.jsonl"
        train = CachedSplit(self.bundle, self.features, "train")
        write_jsonl(self.negatives_path, [{"query_id": p["query_id"], "positive_image_ids": [p["image_id"]],
                                          "negative_image_ids": [n for n in train.image_ids if n != p["image_id"]][:2],
                                          "reviewed": False} for p in train.pairs])

    def test_caption_rotation_and_no_repeat_images(self):
        epochs = [training_pairs_for_epoch(self.bundle, i, 42) for i in range(2)]
        for pairs in epochs:
            self.assertEqual(len({p["image_id"] for p in pairs}), 8)
        self.assertEqual(len({p["query_id"] for pairs in epochs for p in pairs}), 16)

    def test_train_export_round_trip_and_formal_contract(self):
        summary = train_adapter(self.bundle, self.features, self.config_path, self.output)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(read_jsonl(self.output / "history.jsonl")), 3)
        self.assertEqual(summary["parameters"]["encoder_trainable"], 0)
        target = self.root / "export"
        export_adapter(self.output / "best.pt", self.features / "val.images.npz", self.features / "val.texts.npz", target)
        run = self.root / "run.json"
        export_run(self.bundle / "val/protocol.json", target / "index.npz", target / "texts.npz", run,
                   method="E1", adapter_sha256=summary["adapter_sha256"])
        result = evaluate_run(self.bundle / "val/protocol.json", run)
        self.assertEqual(result["data_kind"], "synthetic")
        self.assertEqual(result["metrics"]["recall@1"], summary["best_validation"]["recall@1"])
        self.assertEqual(result["metrics"]["mrr"], summary["best_validation"]["mrr"])
        adapter, _ = load_adapter(self.output / "best.pt", expected_model=self.meta)
        _, raw, _ = load_features(self.features / "val.texts.npz")
        _, adapted, _ = load_features(target / "texts.npz")
        np.testing.assert_allclose(adapt_array(adapter.text, raw), adapted, atol=1e-6)
        with self.assertRaises(ValueError):
            load_adapter(self.output / "best.pt", expected_model=self.meta | {"weights_sha256": "b" * 64})
        with self.assertRaises(ValueError):
            search_index(target / "index.npz", self.features / "val.texts.npz", self.root / "invalid.json")
        with self.assertRaises(ValueError):
            export_run(self.bundle / "val/protocol.json", target / "index.npz", target / "texts.npz",
                       self.root / "fake.json", method="E1", adapter_sha256="c" * 64)

    def test_e2_uses_real_hard_loss_and_filters_reviewed_false_negatives(self):
        records = read_jsonl(self.negatives_path)
        records[0]["false_negative_image_ids"] = [records[0]["negative_image_ids"][0]]
        write_jsonl(self.negatives_path, records)
        config = self.config | {"method": "E2", "hard_weight": 0.2, "hard_margin": 2.0}
        write_json(self.config_path, config)
        summary = train_adapter(self.bundle, self.features, self.config_path, self.output, self.negatives_path)
        self.assertEqual(summary["negative_audit"]["removed_false_negatives"], 1)
        self.assertGreater(read_jsonl(self.output / "history.jsonl")[1]["hard_loss"], 0)

    def test_training_does_not_read_test_features(self):
        for path in self.features.glob("test.*"):
            path.unlink()
        summary = train_adapter(self.bundle, self.features, self.config_path, self.output)
        self.assertEqual(summary["status"], "completed")

    def test_seed_repeats_exactly(self):
        train_adapter(self.bundle, self.features, self.config_path, self.output)
        second = self.root / "repeat"
        train_adapter(self.bundle, self.features, self.config_path, second)
        first_model, _ = load_adapter(self.output / "last.pt")
        second_model, _ = load_adapter(second / "last.pt")
        self.assertGreater(first_model.image.up.weight.abs().sum().item(), 0)
        self.assertGreater(first_model.text.up.weight.abs().sum().item(), 0)
        for key, value in first_model.state_dict().items():
            torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0, atol=0)

    def test_bad_cache_metadata_and_negative_leakage_rejected(self):
        path = self.features / "train.texts.npz.meta.json"
        metadata = read_json(path)
        write_json(path, metadata | {"split": "test"})
        with self.assertRaises(ValueError):
            CachedSplit(self.bundle, self.features, "train")
        write_json(path, metadata)
        records = read_jsonl(self.negatives_path)
        records[0]["negative_image_ids"] = ["test0"]
        write_jsonl(self.negatives_path, records)
        with self.assertRaises(ValueError):
            load_training_negatives(self.negatives_path, CachedSplit(self.bundle, self.features, "train"))

    def test_unknown_config_keys_and_invalid_e2_rejected(self):
        for config in ({"method": "E2"}, {"typo_lr": 0.1}, {"batch_size": True}, {"alpha": 0}):
            write_json(self.config_path, config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                training_config(self.config_path)

    def test_paired_experiment_end_to_end(self):
        configs = self.root / "configs"
        configs.mkdir()
        write_json(configs / "e1.json", self.config)
        write_json(configs / "e2.json", self.config | {"method": "E2", "hard_weight": 0.2})
        result = run_c_experiments(self.bundle, self.features, self.negatives_path, self.output, configs)
        self.assertEqual(set(result["validation"]), {"E0", "E1", "E2"})
        self.assertEqual(len(result["test_handoff"]), 3)
        self.assertTrue(all(not r["scored_here"] for r in result["test_handoff"]))
        self.assertFalse((self.output / "test-evaluation").exists())
        self.assertTrue((self.output / "handoff.json").exists())


if __name__ == "__main__":
    unittest.main()
