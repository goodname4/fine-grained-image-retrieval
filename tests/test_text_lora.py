"""Offline tiny-CLIP fixtures test C v2 math and artifacts, not real accuracy."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from fgir.data import file_digest
from fgir.files import read_json, read_jsonl, write_json, write_jsonl

HAS_STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "numpy", "PIL"))
if HAS_STACK:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPTokenizerFast, CLIPImageProcessor
    from fgir.encoding import load_encoder, save_features, weights_digest
    from fgir.semantic_data import freeze_semantics
    from fgir.semantic_losses import (component_pair_loss, global_alignment_loss, paraphrase_consistency_loss,
                                      semantic_total, sentence_pair_loss, source_macro_mean)
    from fgir.text_lora import (create_text_lora, frozen_state_digest, inject_text_lora, load_text_lora,
                                lora_parameters, make_structure, save_text_lora, trainable_audit)
    from fgir.text_training import batch_objective, selection_key, text_training_config, train_text_lora
    from fgir.text_training_data import ImageFeaturePool, TextTrainingCorpus, verify_training_audit
    from fgir.run_semantic_experiments import experiment_preflight


def tiny_model(root):
    """Local randomly initialized CLIP + word tokenizer, with no downloads."""
    words = ["<pad>", "<bos>", "<unk>", "A", "man", "in", "a", "red", "shirt", "walks", "black", "dog",
             ".", "blue", "cat", "two", "runs", "green", "the", "is", "walked", "by", "The", "sky", "white",
             "sits", "yellow", "small", "on", "grass", "with", "<eos>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    tokenizer.train_from_iterator([" ".join(words[3:-1])], trainers.BpeTrainer(vocab_size=256,
        special_tokens=["<pad>", "<bos>", "<unk>", "<eos>"], end_of_word_suffix="</w>"))
    tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 3)])
    fast = CLIPTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>",
                             unk_token="<unk>", pad_token="<pad>", model_max_length=32)
    config = CLIPConfig(text_config={"vocab_size": len(fast), "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2, "max_position_embeddings": 32,
        "bos_token_id": 1, "eos_token_id": 3, "pad_token_id": 0},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 8, "patch_size": 4}, projection_dim=8)
    torch.manual_seed(9)
    model = CLIPModel(config)
    directory = root / "model"
    model.save_pretrained(directory)
    image_processor = CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
    CLIPProcessor(image_processor=image_processor, tokenizer=fast).save_pretrained(directory)
    write_json(directory / "receipt.json", {"source": "synthetic-random-tiny-clip"})
    return directory


def training_fixture(root, model_dir):
    from test_semantics import make_fixture
    template_root = root / "template"
    template_root.mkdir()
    _, _, _, template, _, _, _ = make_fixture(template_root)
    bundle, data = root / "bundle", root / "data"
    bundle.mkdir()
    (data / "images").mkdir(parents=True)
    descriptions = {"train-a": template["text"], "train-b": "The blue cat runs.",
                    "train-c": "A small dog sits on grass.", "train-d": "A man runs with a white dog.",
                    "val-a": template["text"], "test-a": template["text"]}
    manifest = []
    for n, (image_id, text) in enumerate(descriptions.items()):
        split = image_id.split("-")[0]
        path = data / "images" / f"{image_id}.ppm"
        pixels = bytes(((n * 33 + i * 7) % 256 for i in range(8 * 8 * 3)))
        path.write_bytes(b"P6\n8 8\n255\n" + pixels)
        manifest.append({"image_id": image_id, "split": split, "path": f"images/{image_id}.ppm", "captions": [text]})
    write_jsonl(bundle / "manifest.jsonl", manifest)
    write_jsonl(bundle / "inventory.jsonl", [{**{k: r[k] for k in ("image_id", "path", "split")},
                "file_sha256": file_digest(data / r["path"])} for r in manifest])
    write_json(bundle / "duplicate-review.json", {"pairs": []})
    for split in ("train", "val", "test"):
        (bundle / split).mkdir()
        subset = [r for r in manifest if r["split"] == split]
        write_jsonl(bundle / split / "images.jsonl", subset)
        write_jsonl(bundle / split / "pairs.jsonl", [{"query_id": r["image_id"] + "#0", "image_id": r["image_id"],
                      "split": split, "path": r["path"], "text": r["captions"][0]} for r in subset])
    files = {p.relative_to(bundle).as_posix(): file_digest(p) for p in bundle.rglob("*") if p.is_file()}
    write_json(bundle / "dataset.json", {"schema_version": "fgir-dataset-v1", "data_kind": "synthetic", "files": files,
               "manifest_sha256": file_digest(bundle / "manifest.jsonl"), "image_inventory_sha256": files["inventory.jsonl"]})
    encoder = load_encoder(model_dir)
    rows = [r for r in manifest if r["split"] == "train"]
    vectors = encoder.encode_images(data, [r["path"] for r in rows])
    features = root / "train.images.npz"
    save_features(features, [r["image_id"] for r in rows], vectors, encoder.metadata() | {"kind": "image", "split": "train"})

    def semantic_record(image_id):
        row = copy.deepcopy(template)
        row.update({"sample_id": "sample-" + image_id, "source_image_id": image_id, "source_query_id": image_id + "#0",
                    "group_id": image_id, "scene_group_id": image_id, "split": image_id.split("-")[0],
                    "text": descriptions[image_id]})
        row["provenance"]["source_manifest_sha256"] = file_digest(bundle / "manifest.jsonl")
        if descriptions[image_id] != template["text"]:
            row.update({"entities": [], "relations": [], "units": [], "variants": []})
        else:
            row["units"].append({**copy.deepcopy(row["units"][0]), "id": "u2", "type": "relation",
                                  "variant_id": "direction", "foil": row["variants"][1]["text"]})
        reviews = [row["review"]] + [x["review"] for x in row["units"] + row["variants"]]
        for review in reviews:
            review["image_sha256"] = file_digest(data / f"images/{image_id}.ppm")
        return row

    samples = root / "train.jsonl"
    write_jsonl(samples, [semantic_record(r["image_id"]) for r in rows])
    val = root / "val.jsonl"
    write_jsonl(val, [semantic_record("val-a")])
    freeze_semantics(val, bundle, data, root / "frozen-val", "synthetic", "Synthetic fixtures only, not real evaluation")
    test_gold = root / "test.jsonl"
    write_jsonl(test_gold, [semantic_record("test-a")])
    freeze_semantics(test_gold, bundle, data, root / "frozen-test", "synthetic", "Synthetic test-boundary fixture only")
    return bundle, data, features, samples, root / "frozen-val/protocol.json"


@unittest.skipUnless(HAS_STACK, "install .[semantic-training] for text LoRA tests")
class SemanticLossTests(unittest.TestCase):
    def test_pair_gradient_increases_positive_negative_gap(self):
        image = torch.tensor([[1., 0.]])
        right = torch.tensor([[.2, 1.]], requires_grad=True)
        wrong = torch.tensor([[.8, 1.]], requires_grad=True)
        before = ((image * F.normalize(right, dim=-1)).sum() - (image * F.normalize(wrong, dim=-1)).sum()).item()
        loss = sentence_pair_loss(image, right, wrong, ["a"])
        loss.backward()
        with torch.no_grad():
            right -= .01 * right.grad
            wrong -= .01 * wrong.grad
        after = ((image * F.normalize(right, dim=-1)).sum() - (image * F.normalize(wrong, dim=-1)).sum()).item()
        self.assertGreater(after, before)

    def test_source_macro_and_empty_mask(self):
        values = torch.tensor([0., 2., 10.], requires_grad=True)
        self.assertEqual(source_macro_mean(values, ["a", "a", "b"]).item(), 5.5)
        empty = source_macro_mean(values, ["a", "a", "b"], torch.zeros(3, dtype=torch.bool))
        self.assertEqual(empty.item(), 0)
        empty.backward()
        self.assertTrue(torch.equal(values.grad, torch.zeros_like(values)))

    def test_relation_ablation_masks_its_gradient(self):
        images = torch.tensor([[1., 0.], [1., 0.]])
        right, wrong = torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True)
        loss = component_pair_loss(images, right, wrong, ["a", "b"],
                                   ["attribute_binding", "relation_direction"], include_relations=False)
        loss.backward()
        self.assertEqual(wrong.grad[1].abs().sum().item(), 0)
        self.assertGreater(wrong.grad[0].abs().sum().item(), 0)

    def test_paraphrase_consistency_does_not_force_identical_vectors(self):
        image = torch.tensor([[1., 0.]])
        original, para = torch.tensor([[0., 1.]]), torch.tensor([[0., -1.]])
        self.assertEqual(paraphrase_consistency_loss(image, original, para, ["a"]).item(), 0)

    def test_global_multi_positive_and_no_negatives_rejected(self):
        images = torch.randn(2, 4)
        texts = torch.randn(3, 4, requires_grad=True)
        value = global_alignment_loss(images, texts, ["a", "b"], ["a", "a", "b"])
        value.backward()
        self.assertTrue(torch.isfinite(texts.grad).all())
        with self.assertRaisesRegex(ValueError, "no informative"):
            global_alignment_loss(images, texts, ["a", "b"], ["a", "a", "b"], {"a": "same", "b": "same"})

    def test_methods_only_change_declared_loss(self):
        values = [torch.tensor(float(i)) for i in (1, 2, 3, 4)]
        self.assertAlmostEqual(semantic_total("N0", *values).item(), 1.)
        self.assertAlmostEqual(semantic_total("N1", *values).item(), 1.4, places=6)
        self.assertAlmostEqual(semantic_total("N2", *values).item(), 1.6, places=6)
        self.assertAlmostEqual(semantic_total("N3", *values).item(), 2., places=6)

    def test_selection_guard_and_tie_keys(self):
        metrics = {"recall@1": .6, "mrr": .7, "attribute_binding": .8, "relation_direction": .6}
        self.assertEqual(selection_key("N0", metrics), (.6, .7))
        self.assertEqual(selection_key("N2", metrics, .6), (.7, .6, .7))
        self.assertIsNone(selection_key("N2", metrics, .62))
        with self.assertRaises(ValueError):
            selection_key("N0", metrics | {"mrr": float("nan")})


@unittest.skipUnless(HAS_STACK, "install .[semantic-training] for text LoRA tests")
class TextLoRATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        self.model_dir = tiny_model(self.root)
        self.encoder = create_text_lora(self.model_dir, rank=2)
        self.texts = ["A man in a red shirt walks a black dog.", "The blue cat runs."]

    def _save(self, path=None):
        path = path or self.root / "adapter.pt"
        save_text_lora(path, self.encoder, "N0", 42, {"mode": "development_smoke"},
                       {"data_kind": "synthetic"}, {"optimizer_steps": 2})
        return path

    def test_zero_init_identity_and_two_step_frozen_audit(self):
        base = load_encoder(self.model_dir)
        with torch.no_grad():
            expected = torch.from_numpy(base.encode_texts(self.texts))
            torch.testing.assert_close(self.encoder(self.texts), F.normalize(expected, dim=-1))
        before = frozen_state_digest(self.encoder.model)
        initial = {k: v.detach().clone() for k, v in lora_parameters(self.encoder.model).items()}
        optimizer = torch.optim.AdamW(lora_parameters(self.encoder.model).values(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            values = self.encoder(self.texts)
            self.assertTrue(values.requires_grad)
            (values[:, 0].sum() - values[:, 1].sum()).backward()
            optimizer.step()
        self.assertEqual(before, frozen_state_digest(self.encoder.model))
        self.assertTrue(all(not torch.equal(initial[k], v) for k, v in lora_parameters(self.encoder.model).items()))
        audit = trainable_audit(self.encoder.model)
        self.assertEqual(audit["trainable_parameters"], 256)
        self.assertEqual(audit["image_trainable_parameters"], 0)

    def test_checkpoint_roundtrip_and_missing_or_nonfinite_state_rejected(self):
        path = self._save()
        reloaded, receipt = load_text_lora(path, self.model_dir)
        with torch.no_grad():
            torch.testing.assert_close(reloaded(self.texts), self.encoder(self.texts), rtol=0, atol=0)
        self.assertEqual(receipt["text_lora_sha256"], file_digest(path))
        payload = torch.load(path, weights_only=True)
        original = copy.deepcopy(payload)
        payload["state_dict"].pop(next(iter(payload["state_dict"])))
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "parameter names"):
            load_text_lora(path, self.model_dir)
        original["state_dict"][next(iter(original["state_dict"]))].fill_(float("nan"))
        torch.save(original, path)
        with self.assertRaisesRegex(ValueError, "parameter shape/dtype/value"):
            load_text_lora(path, self.model_dir)

    def test_unknown_structure_and_base_identity_rejected(self):
        path = self._save()
        payload = torch.load(path, weights_only=True)
        payload["structure"]["target_paths"][0] = "vision_model.encoder.layers.0.self_attn.q_proj"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            load_text_lora(path, self.model_dir)
        payload["structure"] = self.encoder.structure
        payload["model"]["base_weights_sha256"] = "f" * 64
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_text_lora(path, self.model_dir)

    def test_double_injection_and_unfrozen_base_rejected(self):
        with self.assertRaisesRegex(ValueError, "already adapted"):
            inject_text_lora(self.encoder.model, self.encoder.structure)
        self.encoder.model.text_projection.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            trainable_audit(self.encoder.model)

    def test_overlong_text_cannot_silently_truncate(self):
        with self.assertRaisesRegex(ValueError, "token limit"):
            self.encoder(["dog " * 40])

    def test_tokenizer_digest_mismatch_rejected(self):
        path = self._save()
        config_path = self.model_dir / "tokenizer_config.json"
        write_json(config_path, read_json(config_path) | {"verified_test_note": "changed tokenizer identity"})
        with self.assertRaisesRegex(ValueError, "tokenizer_sha256"):
            load_text_lora(path, self.model_dir)

    def test_corpus_exposure_and_review_gate(self):
        bundle, data, features, samples, val = training_fixture(self.root, self.model_dir)
        pool = ImageFeaturePool(bundle, features, self.encoder.metadata)
        corpus = TextTrainingCorpus(pool, self.encoder, samples, data)
        self.assertEqual(corpus.stats["unit_phenomena"], {"attribute_binding": 1, "relation_direction": 1})
        records = corpus.epoch(0, 42)
        config = {"method": "N2", "temperature": .07, "include_relations": True,
                  "sentence_weight": .2, "unit_weight": .2, "para_weight": .1}
        _, n2 = batch_objective(self.encoder, corpus, records, config, "cpu")
        _, n3 = batch_objective(self.encoder, corpus, records, config | {"method": "N3"}, "cpu")
        self.assertEqual(n2["forward_texts"], n3["forward_texts"])
        self.assertEqual(n2["forward_tokens"], n3["forward_tokens"])
        rows = read_jsonl(samples)
        rows[0]["review"]["annotator_kind"] = "assistant"
        rows[0]["review"]["status"] = "annotated"
        write_jsonl(samples, rows)
        with self.assertRaisesRegex(ValueError, "assistant annotations"):
            TextTrainingCorpus(pool, self.encoder, samples, data)

    def test_auto_verified_supervision_needs_human_audit(self):
        rows = [{"sample_id": "a", "review": {"annotator_kind": "automatic"}, "variants": [], "units": []}]
        path = self.root / "training.jsonl"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "human spot-check"):
            verify_training_audit(None, path, rows)
        audit = self.root / "audit.json"
        write_json(audit, {"schema_version": "fgir-semantic-training-audit-v1", "samples_sha256": file_digest(path),
                           "reviewer_kind": "human", "reviewer": "synthetic-human-reviewer", "notes": "Synthetic test fixture",
                           "sampled_sample_ids": ["a"], "unresolved_disputes": []})
        self.assertEqual(verify_training_audit(audit, path, rows)["checked_samples"], 1)

    def test_real_training_path_on_synthetic_model_with_reload(self):
        bundle, data, features, _, _ = training_fixture(self.root, self.model_dir)
        config = self.root / "smoke.json"
        write_json(config, {"mode": "development_smoke", "epochs": 2, "micro_batch_size": 2,
                           "gradient_accumulation": 1, "rank": 2, "max_optimizer_steps": 2,
                           "development_max_images": 4, "threads": 1, "device": "cpu"})
        output = self.root / "smoke"
        summary = train_text_lora(bundle, self.model_dir, features, config, output, data_root=data)
        self.assertEqual(summary["optimizer_steps"], 2)
        self.assertEqual(summary["data_kind"], "synthetic")
        self.assertFalse(summary["formal_comparison_eligible"])
        self.assertTrue(summary["freeze_audit"]["all_original_parameters_unchanged"])
        self.assertTrue(summary["reload_audit"]["passed"])
        self.assertTrue((output / "best.pt").exists())
        self.assertTrue((output / "last.pt").exists())

    def test_formal_training_refuses_missing_reviewed_inputs(self):
        path = self.root / "config.json"
        write_json(path, {})
        with self.assertRaisesRegex(ValueError, "verified --samples"):
            train_text_lora("missing", self.model_dir, "missing", path, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_formal_selection_uses_bound_callback_and_refuses_test(self):
        bundle, data, features, samples, val = training_fixture(self.root, self.model_dir)
        config = self.root / "formal.json"
        write_json(config, {"epochs": 1, "micro_batch_size": 2, "gradient_accumulation": 1,
                           "rank": 2, "threads": 1, "device": "cpu"})

        def synthetic_callback(**kwargs):
            payload = torch.load(kwargs["checkpoint_path"], weights_only=True)
            return {"text_lora_sha256": file_digest(kwargs["checkpoint_path"]),
                    "semantic_protocol_sha256": file_digest(kwargs["val_protocol"]),
                    "base_weights_sha256": payload["model"]["base_weights_sha256"],
                    "tokenizer_sha256": payload["model"]["tokenizer_sha256"],
                    "metrics": {"recall@1": .5, "mrr": .6, "attribute_binding": .5, "relation_direction": .5},
                    "synthetic_fixture_only": True}

        summary = train_text_lora(bundle, self.model_dir, features, config, self.root / "formal", samples,
                                  data, val_protocol=val, validation_callback=synthetic_callback)
        self.assertEqual(summary["best_selection"]["epoch"], 0)
        self.assertEqual(summary["data_kind"], "synthetic")
        self.assertEqual(len(read_json(self.root / "formal/validation-history.json")), 2)
        with self.assertRaisesRegex(ValueError, "never test"):
            train_text_lora(bundle, self.model_dir, features, config, self.root / "bad-test", samples,
                            data, val_protocol=self.root / "frozen-test/protocol.json", validation_callback=synthetic_callback)
        self.assertFalse((self.root / "bad-test").exists())

    def test_all_objectives_share_forward_exposure_and_reload(self):
        bundle, data, features, samples, val = training_fixture(self.root, self.model_dir)
        config_path = self.root / "paired.json"

        def callback(**kwargs):
            payload = torch.load(kwargs["checkpoint_path"], weights_only=True)
            return {"text_lora_sha256": file_digest(kwargs["checkpoint_path"]),
                    "semantic_protocol_sha256": file_digest(kwargs["val_protocol"]),
                    "base_weights_sha256": payload["model"]["base_weights_sha256"],
                    "tokenizer_sha256": payload["model"]["tokenizer_sha256"],
                    "metrics": {"recall@1": .5, "mrr": .6, "attribute_binding": .5, "relation_direction": .5},
                    "synthetic_fixture_only": True}

        summaries, reference = [], None
        for method in ("N0", "N1", "N2", "N3", "N2-no-relation"):
            write_json(config_path, {"method": method.split("-")[0], "epochs": 1, "micro_batch_size": 2,
                "gradient_accumulation": 1, "rank": 2, "threads": 1, "device": "cpu",
                "include_relations": method != "N2-no-relation"})
            output = self.root / method
            summary = train_text_lora(bundle, self.model_dir, features, config_path, output,
                samples, data, val_protocol=val, reference_summary=reference, validation_callback=callback)
            summaries.append(summary)
            if method == "N0":
                reference = output / "summary.json"
            self.assertTrue(summary["reload_audit"]["passed"])
            self.assertFalse(summary["formal_comparison_eligible"])
        self.assertEqual(len({s["forward_tokens"] for s in summaries}), 1)
        self.assertEqual(len({s["forward_texts"] for s in summaries}), 1)
        self.assertEqual(len({s["corpus"]["cohort_sha256"] for s in summaries}), 1)

    def test_preflight_reports_external_missing_inputs_without_writing(self):
        root = Path(__file__).resolve().parents[1]
        result, _, _, _ = experiment_preflight("missing", self.model_dir, "missing.npz", "train-missing.jsonl",
                                               "val-missing.json", root / "configs/training")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue({"B-V2-02", "A-V2-04", "D-V2-01/02"} <= {m["owner"] for m in result["missing"]})

    def test_config_typos_and_invalid_modes_rejected(self):
        path = self.root / "config.json"
        for value in ({"rank": True}, {"typo": 1}, {"mode": "development_smoke"}, {"temperature": 0},
                      {"lora_dropout": 1}, {"micro_batch_size": 1}):
            write_json(path, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                text_training_config(path)


if __name__ == "__main__":
    unittest.main()
