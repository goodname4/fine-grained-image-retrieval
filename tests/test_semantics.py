"""Synthetic fixtures exercise A's gates and math, never model quality."""

import copy
from pathlib import Path
import tempfile
import unittest

from fgir.data import file_digest
from fgir.files import read_json, read_jsonl, write_json, write_jsonl
from fgir.semantic_data import apply_semantic_source_checks, freeze_semantics, prepare_semantic_pilot, source_context, validate_semantics
from fgir.semantic_evaluation import compare_semantics, evaluate_semantics, load_semantic_protocol, parsing_scores
from fgir.semantic_review import render_semantics


def make_fixture(root):
    root = Path(root)
    bundle, data = root / "source", root / "data"
    bundle.mkdir(parents=True)
    (data / "images").mkdir(parents=True)
    caption = "A man in a red shirt walks a black dog."
    manifest = []
    for image_id, split in (("train-a", "train"), ("val-a", "val"), ("val-b", "val"), ("val-c", "val"), ("test-a", "test")):
        image = data / "images" / f"{image_id}.ppm"
        image.write_bytes(b"P6\n2 1\n255\n\xff\x00\x00\x00\x00\xff")
        manifest.append({"image_id": image_id, "split": split, "path": f"images/{image_id}.ppm", "captions": [caption]})
    write_jsonl(bundle / "manifest.jsonl", manifest)
    write_jsonl(bundle / "inventory.jsonl", [{"image_id": r["image_id"], "path": r["path"], "split": r["split"],
                "file_sha256": file_digest(data / r["path"])} for r in manifest])
    write_json(bundle / "duplicate-review.json", {"pairs": []})
    names = ["manifest.jsonl", "inventory.jsonl", "duplicate-review.json"]
    for split in ("train", "val", "test"):
        (bundle / split).mkdir()
        name = f"{split}/pairs.jsonl"
        names.append(name)
        write_jsonl(bundle / name, [{"query_id": f"{r['image_id']}#0", "image_id": r["image_id"], "path": r["path"],
                                    "text": caption, "split": split} for r in manifest if r["split"] == split])
    write_json(bundle / "dataset.json", {"schema_version": "fgir-dataset-v1", "data_kind": "synthetic",
               "image_inventory_sha256": file_digest(bundle / "inventory.jsonl"), "files": {n: file_digest(bundle / n) for n in names}})
    review = {"status": "reviewed", "annotator": "synthetic-human-A", "annotator_kind": "human",
              "reviewer": "synthetic-human-B", "reviewer_kind": "human", "notes": "synthetic fixture, no real annotations",
              "review_notes": "synthetic independent review fixture", "image_evidence": "synthetic RGB swatch, arbitrary test labels",
              "image_sha256": file_digest(data / "images/val-a.ppm"), "disputes": []}
    record = {"schema_version": "fgir-semantic-v1", "sample_id": "sample-val-a", "group_id": "val-a", "scene_group_id": "val-a",
              "source_image_id": "val-a", "source_query_id": "val-a#0", "split": "val", "text": caption,
              "source_visual_label": "supported", "review": review,
              "provenance": {"source_manifest_sha256": file_digest(bundle / "manifest.jsonl"), "seed": 42,
                             "generator": {"name": "human", "weights_sha256": None, "prompt_sha256": None}, "code_revision": "synthetic-fixture"},
              "entities": [{"id": "e1", "head": "man", "span": [2, 5], "attributes": [{"entity_id": "e1", "name": "shirt_color", "value": "red", "span": [11, 14]}]},
                           {"id": "e2", "head": "dog", "span": [35, 38], "attributes": [{"entity_id": "e2", "name": "color", "value": "black", "span": [29, 34]}]}],
              "relations": [{"id": "r1", "subject": "e1", "predicate": "walks", "object": "e2", "span": [21, 26]}],
              "units": [], "variants": []}

    def variant(vid, kind, phrase, span, after, target="sentence", phenomenon="false_addition"):
        start, end = span
        return {"id": vid, "type": kind, "text": phrase, "target_unit_id": target, "phenomenon": phenomenon,
                "semantic_relation": "equivalent" if kind == "paraphrase" else "changed",
                "visual_label": "supported" if kind in ("paraphrase", "true_addition") else "contradicted",
                "naturalness": "acceptable", "control_pair_id": "addition-pair",
                "edit": {"field": target, "span": span, "before": caption[start:end], "after": after}, "review": copy.deepcopy(review)}

    record["variants"] = [
        variant("binding", "minimal_edit", "A man in a black shirt walks a red dog.", [11, 34], "black shirt walks a red", "e1", "attribute_binding"),
        variant("direction", "minimal_edit", "A black dog walks a man in a red shirt.", [0, len(caption)],
                "A black dog walks a man in a red shirt.", "r1", "relation_direction"),
        variant("para", "paraphrase", "A black dog is walked by a man in a red shirt.", [0, len(caption)],
                "A black dog is walked by a man in a red shirt.", phenomenon="paraphrase"),
        variant("true", "true_addition", caption + " The sky is blue.", [len(caption), len(caption)], " The sky is blue."),
        variant("false", "false_addition", caption + " The sky is green.", [len(caption), len(caption)], " The sky is green.")]
    record["units"] = [{"id": "u1", "type": "entity", "entity_ids": ["e1", "e2"], "span": [0, len(caption)],
                        "text": caption, "foil": "A man in a black shirt walks a red dog.", "variant_id": "binding", "review": copy.deepcopy(review)}]
    samples = root / "gold.jsonl"
    write_jsonl(samples, [record])
    frozen = root / "frozen"
    freeze_semantics(samples, bundle, data, frozen, "synthetic", "Synthetic fixtures; no real test evaluation")
    protocol = frozen / "protocol.json"
    run = {"schema_version": "fgir-semantic-scores-v1", "data_kind": "synthetic", "protocol_sha256": file_digest(protocol),
           "run_id": "synthetic-E0-42", "method": "E0", "seed": 42, "code_revision": "synthetic-fixture",
           "model": {"model_type": "raw_clip", "model_id": "synthetic-no-model", "base_weights_sha256": "0" * 64, "tokenizer_sha256": "1" * 64},
           "ranking_depth": 3,
           "predictions": [{"sample_id": record["sample_id"], "scores": {"original": .5, "binding": .5, "direction": .6, "para": .5, "true": .7, "false": .1},
                            "parsing": {"entities": copy.deepcopy(record["entities"]), "relations": copy.deepcopy(record["relations"])},
                            "rankings": {"original": ["val-a", "val-b", "val-c"], "para": ["val-b", "val-a", "val-c"]}}]}
    run_path = root / "run.json"
    write_json(run_path, run)
    return bundle, data, samples, record, protocol, run_path, run


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle, self.data, self.samples, self.row, self.protocol, self.run_path, self.run = make_fixture(self.root)

    def validate(self, purpose="draft"):
        write_jsonl(self.samples, [self.row])
        return validate_semantics(self.samples, self.bundle, purpose, self.data)

    def evaluate(self):
        write_json(self.run_path, self.run)
        return evaluate_semantics(self.protocol, self.run_path, self.root / "evaluation", 50)

    def test_valid_source_and_real_image_hash_gate(self):
        self.assertEqual(self.validate("eval")["source_image_count"], 1)
        (self.data / "images/val-a.ppm").write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "different image bytes"):
            self.validate("eval")

    def test_source_caption_split_manifest_scene_and_group_binding(self):
        original = copy.deepcopy(self.row)
        for field, value in (("text", "altered"), ("split", "train"), ("source_image_id", "test-a"),
                             ("source_query_id", "test-a#0"), ("scene_group_id", "other"), ("group_id", "other")):
            self.row = {**original, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()
        self.row = copy.deepcopy(original)
        self.row["provenance"]["source_manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.validate()

    def test_no_val_optimization(self):
        with self.assertRaisesRegex(ValueError, "only from train"):
            self.validate("train")

    def test_invalid_spans_owners_refs_and_undeclared_edits(self):
        original = copy.deepcopy(self.row)
        mutations = [lambda r: r["entities"][0].update(span=[True, 5]),
                     lambda r: r["entities"][0].update(span=[3, 6]),
                     lambda r: r["entities"][0]["attributes"][0].update(entity_id="e2"),
                     lambda r: r["relations"][0].update(subject="missing"),
                     lambda r: r["variants"][0].update(text="unrecorded edit"),
                     lambda r: r["variants"][0].update(target_unit_id="missing"),
                     lambda r: r["variants"][0].update(id="original"),
                     lambda r: r["units"][0].update(foil="a different edit")]
        for mutation in mutations:
            self.row = copy.deepcopy(original)
            mutation(self.row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate()

    def test_draft_auto_assistant_and_same_reviewer_cannot_be_gold(self):
        original = copy.deepcopy(self.row)
        for change in ({"status": "draft"}, {"status": "auto_verified", "annotator_kind": "automatic"},
                       {"annotator_kind": "assistant"}, {"reviewer": " SYNTHETIC-HUMAN-A "},
                       {"reviewer_kind": "assistant"}, {"disputes": ["ambiguous"]}):
            self.row = copy.deepcopy(original)
            self.row["review"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate("eval")

    def test_each_variant_and_unit_need_review_and_visual_evidence(self):
        original = copy.deepcopy(self.row)
        for item in ("variants", "units"):
            self.row = copy.deepcopy(original)
            self.row[item][0]["review"]["status"] = "draft"
            with self.assertRaisesRegex(ValueError, "not independently"):
                self.validate("eval")
        for field, value in (("visual_label", "uncertain"), ("naturalness", "uncertain")):
            self.row = copy.deepcopy(original)
            self.row["variants"][0][field] = value
            with self.assertRaises(ValueError):
                self.validate("eval")
        self.row = copy.deepcopy(original)
        del self.row["variants"][0]["review"]["image_evidence"]
        with self.assertRaisesRegex(ValueError, "image evidence"):
            self.validate()

    def test_duplicate_sample_rejected(self):
        write_jsonl(self.samples, [self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            validate_semantics(self.samples, self.bundle)

    def test_known_scene_leakage_is_rejected(self):
        write_json(self.bundle / "duplicate-review.json", {"pairs": [{"left": "val-a", "right": "test-a", "decision": "same_scene"}]})
        dataset = read_json(self.bundle / "dataset.json")
        dataset["files"]["duplicate-review.json"] = file_digest(self.bundle / "duplicate-review.json")
        write_json(self.bundle / "dataset.json", dataset)
        with self.assertRaisesRegex(ValueError, "scene crosses splits"):
            source_context(self.bundle)

    def test_source_pair_tampering(self):
        (self.bundle / "val/pairs.jsonl").write_text('{}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source file hash mismatch"):
            source_context(self.bundle)

    def test_pilot_deterministic_val_only_no_reviews(self):
        for name in ("pilot-1", "pilot-2"):
            result = prepare_semantic_pilot(self.bundle, self.data, self.root / name, 3)
            self.assertEqual(result["review_status"], {"draft": 3})
            self.assertEqual(result["splits"], {"val": 3})
        self.assertEqual(file_digest(self.root / "pilot-1/samples.work.jsonl"), file_digest(self.root / "pilot-2/samples.work.jsonl"))
        with self.assertRaisesRegex(ValueError, "not independently"):
            freeze_semantics(self.root / "pilot-1/samples.work.jsonl", self.bundle, self.data, self.root / "bad-freeze", "synthetic", "fixture")
        self.assertFalse((self.root / "bad-freeze").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_semantic_pilot(self.bundle, self.data, self.root / "pilot-1", 3)

    def test_offline_board_has_local_images_and_safe_payload(self):
        self.row["review"]["notes"] = '</script><script>alert("unsafe")</script>'
        self.validate()
        target = self.root / "board"
        render_semantics(self.samples, self.bundle, self.data, target)
        html = (target / "board.html").read_text(encoding="utf-8")
        self.assertNotIn(self.row["review"]["notes"], html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertTrue((target / "images/0.ppm").exists())

    def test_protocol_tamper_and_traversal(self):
        (self.protocol.parent / "samples.jsonl").write_text('{}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_semantic_protocol(self.protocol)
        protocol = read_json(self.protocol)
        protocol["files"]["samples.jsonl"]["path"] = "../gold.jsonl"
        write_json(self.protocol, protocol)
        with self.assertRaises(ValueError):
            load_semantic_protocol(self.protocol)

    def test_tie_direction_addition_and_length_controls(self):
        result = self.evaluate()
        m = result["metrics"]
        self.assertEqual(m["accuracy/attribute_binding"]["source_macro_mean"], .5)
        self.assertEqual(m["accuracy/relation_direction"]["source_macro_mean"], 0)
        self.assertEqual(m["accuracy/false_addition"]["source_macro_mean"], 1)
        self.assertAlmostEqual(m["score_delta/true_addition_minus_original"]["source_macro_mean"], .2)
        self.assertEqual(m["accuracy/addition_length_matched"]["source_macro_mean"], 1)
        self.assertEqual(result["coverage"]["addition_control_status"]["length_matched"], 1)

    def test_common_failure_not_stability_success(self):
        self.run["predictions"][0]["rankings"] = {"original": [], "para": []}
        result = self.evaluate()
        self.assertEqual(result["metrics"]["paraphrase/joint_recall@5"]["source_macro_mean"], 0)
        self.assertIsNone(result["metrics"]["paraphrase/absolute_rank_gap_full_rankings"]["source_macro_mean"])

    def test_paraphrase_joint_not_one_sided_success(self):
        result = self.evaluate()
        self.assertEqual(result["metrics"]["paraphrase/joint_recall@1"]["source_macro_mean"], 0)
        self.assertEqual(result["metrics"]["paraphrase/original_recall@1"]["source_macro_mean"], 1)
        self.assertEqual(result["metrics"]["paraphrase/joint_recall@5"]["source_macro_mean"], 1)
        self.assertEqual(result["metrics"]["paraphrase/absolute_rank_gap_full_rankings"]["source_macro_mean"], 1)

    def test_parser_ids_are_local_but_direction_and_owner_matter(self):
        pred = copy.deepcopy(self.run["predictions"][0]["parsing"])
        pred["entities"][0]["id"] = "person-99"
        pred["entities"][0]["attributes"][0]["entity_id"] = "person-99"
        pred["relations"][0]["subject"] = "person-99"
        correct = parsing_scores(self.row, pred)
        self.assertTrue(all(x["f1"] == 1 for x in correct.values()))
        pred["relations"][0].update(subject="e2", object="person-99")
        self.assertEqual(parsing_scores(self.row, pred)["directed_relation"]["f1"], 0)
        attr = pred["entities"][0]["attributes"].pop()
        attr["entity_id"] = "e2"
        pred["entities"][1]["attributes"].append(attr)
        self.assertEqual(parsing_scores(self.row, pred)["attribute_binding"]["f1"], .5)

    def test_missing_extra_nonfinite_or_boolean_scores_rejected(self):
        original = copy.deepcopy(self.run)
        for kind in ("missing", "extra", "bool", "infinite", "missing-sample", "duplicate-sample"):
            self.run = copy.deepcopy(original)
            scores = self.run["predictions"][0]["scores"]
            if kind == "missing":
                del scores["para"]
            elif kind == "extra":
                scores["unexpected"] = .2
            elif kind == "bool":
                scores["original"] = True
            elif kind == "infinite":
                scores["original"] = "NaN"
            elif kind == "missing-sample":
                self.run["predictions"] = []
            else:
                self.run["predictions"] *= 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.evaluate()

    def test_wrong_gallery_duplicate_rankings_and_foil_qrels_rejected(self):
        original = copy.deepcopy(self.run)
        for rankings in ({"original": ["outside", "val-b", "val-c"], "para": []},
                         {"original": ["val-a", "val-a", "val-c"], "para": []},
                         {"original": [], "para": [], "binding": []}):
            self.run = copy.deepcopy(original)
            self.run["predictions"][0]["rankings"] = rankings
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_omitted_predictions_produce_null_not_zero(self):
        del self.run["predictions"][0]["rankings"]
        del self.run["predictions"][0]["parsing"]
        result = self.evaluate()
        self.assertIsNone(result["parsing_micro"])
        self.assertIsNone(result["metrics"]["paraphrase/joint_recall@5"]["source_macro_mean"])

    def test_n_method_needs_lora_identity_and_digests(self):
        self.run["method"] = "N2"
        with self.assertRaisesRegex(ValueError, "text_lora identity"):
            self.evaluate()
        self.run["model"]["model_type"] = "text_lora"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.evaluate()

    def test_image_macro_not_variant_micro(self):
        rows = [copy.deepcopy(self.row) for _ in range(3)]
        rows[1]["sample_id"] = "same-source-caption-2"
        rows[2].update(sample_id="sample-val-b", source_image_id="val-b", source_query_id="val-b#0", group_id="val-b", scene_group_id="val-b")
        write_jsonl(self.samples, rows)
        target = self.root / "macro-frozen"
        freeze_semantics(self.samples, self.bundle, self.data, target, "synthetic", "fixture")
        self.protocol = target / "protocol.json"
        self.run["protocol_sha256"] = file_digest(self.protocol)
        predictions = [copy.deepcopy(self.run["predictions"][0]) for _ in rows]
        for r, pred in zip(rows, predictions):
            pred["sample_id"] = r["sample_id"]
            pred["scores"]["binding"] = .4 if r["source_image_id"] == "val-a" else .6
        self.run["predictions"] = predictions
        result = self.evaluate()["metrics"]["accuracy/attribute_binding"]
        self.assertEqual(result["source_macro_mean"], .5)
        self.assertEqual(result["bootstrap"]["cluster_count"], 2)

    def test_paired_comparison_keeps_regressions_and_seed_sd_separate(self):
        runs = []
        for method in ("E0", "N2"):
            for training_seed in (42, 43):
                run = copy.deepcopy(self.run)
                run.update(method=method, seed=training_seed, run_id=f"synthetic-{method}-{training_seed}")
                if method == "N2":
                    run["model"].update(model_type="text_lora", text_lora_sha256="2" * 64, retrieval_receipt_sha256="3" * 64, semantic_dataset_sha256="4" * 64)
                    run["predictions"][0]["scores"].update(binding=.4, false=.9)
                path = self.root / f"{method}-{training_seed}.json"
                write_json(path, run)
                runs.append(path)
        result = compare_semantics(self.protocol, runs, self.root / "comparison", 20)
        self.assertEqual(len(result["comparisons"]), 2)
        cases = read_jsonl(self.root / "comparison/cases.unreviewed.jsonl")
        self.assertEqual({r["outcome"] for r in cases}, {"improved", "regressed", "unchanged"})
        self.assertTrue(all(r["cause"] is None for r in cases))
        self.assertEqual({r["phenomenon"] for r in cases}, {"attribute_binding", "relation_direction", "false_addition", "paraphrase"})
        self.assertEqual(result["seed_summary"]["N2"]["accuracy/attribute_binding"]["sample_sd"], 0)
        with self.assertRaisesRegex(ValueError, "paired seeds"):
            compare_semantics(self.protocol, runs[:-1], self.root / "unpaired", 20)

    def test_comparison_allows_failed_retrieval_and_discloses_rank_gap_exclusions(self):
        self.run["predictions"][0]["rankings"] = {"original": [], "para": []}
        self.run.update(run_id="synthetic-failure", method="E1")
        self.run["model"].update(model_type="feature_mlp", adapter_sha256="a" * 64)
        candidate = self.root / "failed.json"
        write_json(candidate, self.run)
        result = compare_semantics(self.protocol, [self.run_path, candidate], self.root / "failure-comparison", 20)
        entry = result["comparisons"][0]
        self.assertEqual(entry["source_macro_deltas"]["paraphrase/joint_recall@5"]["source_macro_mean"], -1)
        self.assertEqual(entry["unpaired_metric_sources"]["paraphrase/absolute_rank_gap_full_rankings"]["baseline_only"], ["val-a"])

    def test_assisted_source_checks_preserve_uncertainty_and_cannot_become_gold(self):
        import hashlib
        pilot = self.root / "check-pilot"
        prepare_semantic_pilot(self.bundle, self.data, pilot, 3)
        row = read_jsonl(pilot / "samples.work.jsonl")[0]
        checks = {"schema_version": "fgir-semantic-source-checks-v1", "source_manifest_sha256": row["provenance"]["source_manifest_sha256"],
                  "checks": [{"sample_id": row["sample_id"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                              "image_sha256": row["review"]["image_sha256"], "source_visual_label": "uncertain", "image_evidence": "synthetic ambiguity"}]}
        check_path = self.root / "source-checks.json"
        write_json(check_path, checks)
        target = self.root / "assisted"
        result = apply_semantic_source_checks(pilot / "samples.work.jsonl", self.bundle, self.data, check_path, target)
        self.assertEqual(result["review_status"], {"annotated": 1, "draft": 2})
        updated = read_jsonl(target / "samples.work.jsonl")[0]
        self.assertEqual(updated["source_visual_label"], "uncertain")
        self.assertEqual(updated["review"]["annotator_kind"], "assistant")
        with self.assertRaises(ValueError):
            validate_semantics(target / "samples.work.jsonl", self.bundle, "eval", self.data)
        checks["checks"][0]["image_sha256"] = "f" * 64
        write_json(check_path, checks)
        with self.assertRaisesRegex(ValueError, "image hash mismatch"):
            apply_semantic_source_checks(pilot / "samples.work.jsonl", self.bundle, self.data, check_path, self.root / "bad-check")

    def test_length_mismatch_and_missing_controls_are_disclosed(self):
        rows = copy.deepcopy(self.row)
        rows["variants"][3]["text"] += " today"
        rows["variants"][3]["edit"]["after"] += " today"
        write_jsonl(self.samples, [rows])
        target = self.root / "unequal-length"
        freeze_semantics(self.samples, self.bundle, self.data, target, "synthetic", "fixture")
        self.protocol = target / "protocol.json"
        self.run["protocol_sha256"] = file_digest(self.protocol)
        result = self.evaluate()
        self.assertEqual(result["coverage"]["addition_control_status"]["length_mismatch"], 1)
        self.assertIsNone(result["metrics"]["accuracy/addition_length_matched"]["source_macro_mean"])

    def test_cli_fixture_runs_as_synthetic(self):
        import subprocess
        import sys
        fixture = Path(__file__).resolve().parents[1] / "examples/semantic-evaluation"
        output = self.root / "cli-example"
        result = subprocess.run([sys.executable, "-m", "fgir", "evaluate-semantics", "--protocol", str(fixture / "frozen/protocol.json"),
                                 "--run", str(fixture / "run.json"), "--output", str(output), "--bootstrap-samples", "0"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(output / "metrics.json")["data_kind"], "synthetic")

    def test_synthetic_source_cannot_freeze_as_real(self):
        with self.assertRaisesRegex(ValueError, "cannot relabel"):
            freeze_semantics(self.samples, self.bundle, self.data, self.root / "fake-real", "real", "fixture")
        self.assertFalse((self.root / "fake-real").exists())

    def test_replaced_image_and_updated_reviews_rejected_by_inventory(self):
        image = self.data / "images/val-a.ppm"
        image.write_bytes(b"P6\n2 1\n255\n\x00\xff\x00\xff\xff\x00")
        changed_hash = file_digest(image)
        for review in [self.row["review"], *[v["review"] for v in self.row["variants"] + self.row["units"]]]:
            review["image_sha256"] = changed_hash
        write_jsonl(self.samples, [self.row])
        for data_root in (None, self.data):
            with self.subTest(data_root=data_root), self.assertRaisesRegex(ValueError, "source inventory"):
                validate_semantics(self.samples, self.bundle, "eval", data_root)
        target = self.root / "must-not-freeze"
        with self.assertRaisesRegex(ValueError, "source inventory"):
            freeze_semantics(self.samples, self.bundle, self.data, target, "synthetic", "fixture")
        self.assertFalse(target.exists())
        target = self.root / "must-not-prepare"
        with self.assertRaisesRegex(ValueError, "source inventory"):
            prepare_semantic_pilot(self.bundle, self.data, target, 3)
        self.assertFalse(target.exists())

    def test_inventory_binding_is_required_and_hash_checked(self):
        path = self.bundle / "inventory.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source file hash mismatch: inventory"):
            source_context(self.bundle)
        metadata = read_json(self.bundle / "dataset.json")
        del metadata["files"]["inventory.jsonl"]
        write_json(self.bundle / "dataset.json", metadata)
        with self.assertRaisesRegex(ValueError, "source file binding: inventory"):
            source_context(self.bundle)

    def test_inventory_coverage_metadata_and_digests_rejected(self):
        original = read_jsonl(self.bundle / "inventory.jsonl")
        mutations = [lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
                     lambda r: r[0].update(path="images/other.ppm"),
                     lambda r: r[0].update(split="val"), lambda r: r[0].update(image_id="extra-image"),
                     lambda r: r[0].update(file_sha256="invalid")]
        for mutate in mutations:
            rows = copy.deepcopy(original)
            mutate(rows)
            write_jsonl(self.bundle / "inventory.jsonl", rows)
            metadata = read_json(self.bundle / "dataset.json")
            metadata["files"]["inventory.jsonl"] = metadata["image_inventory_sha256"] = file_digest(self.bundle / "inventory.jsonl")
            write_json(self.bundle / "dataset.json", metadata)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                source_context(self.bundle)
        write_jsonl(self.bundle / "inventory.jsonl", original)
        metadata["files"]["inventory.jsonl"] = file_digest(self.bundle / "inventory.jsonl")
        metadata["image_inventory_sha256"] = "f" * 64
        write_json(self.bundle / "dataset.json", metadata)
        with self.assertRaisesRegex(ValueError, "inventory digest fields disagree"):
            source_context(self.bundle)

    def test_frozen_context_binds_source_inventory(self):
        context = read_json(self.protocol.parent / "source-context.json")
        self.assertEqual(set(context["inventory"]), {"val-a"})
        self.assertEqual(context["source_files"]["inventory.jsonl"], file_digest(self.bundle / "inventory.jsonl"))
        self.assertEqual(context["inventory"]["val-a"]["file_sha256"], self.row["review"]["image_sha256"])
        del context["inventory"]
        write_json(self.protocol.parent / "source-context.json", context)
        protocol = read_json(self.protocol)
        protocol["files"]["source-context.json"]["sha256"] = file_digest(self.protocol.parent / "source-context.json")
        write_json(self.protocol, protocol)
        with self.assertRaisesRegex(ValueError, "must bind inventory"):
            load_semantic_protocol(self.protocol)

    def test_duplicate_gold_mentions_attributes_and_relations_rejected_before_freeze(self):
        original = copy.deepcopy(self.row)
        for kind in ("entity mention", "attribute", "relation"):
            self.row = copy.deepcopy(original)
            if kind == "entity mention":
                duplicate = copy.deepcopy(self.row["entities"][0])
                duplicate["id"] = "person-copy"
                for attr in duplicate["attributes"]:
                    attr["entity_id"] = duplicate["id"]
                self.row["entities"].append(duplicate)
            elif kind == "attribute":
                duplicate = copy.deepcopy(self.row["entities"][0]["attributes"][0])
                duplicate["name"] = " SHIRT_COLOR "
                self.row["entities"][0]["attributes"].append(duplicate)
            else:
                duplicate = {**self.row["relations"][0], "id": "relation-copy"}
                self.row["relations"].append(duplicate)
            write_jsonl(self.samples, [self.row])
            target = self.root / f"bad-{kind.replace(' ', '-')}"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f"gold parsing.*duplicate {kind}"):
                freeze_semantics(self.samples, self.bundle, self.data, target, "synthetic", "fixture")
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError, f"gold parsing.*duplicate {kind}"):
                parsing_scores(self.row, self.run["predictions"][0]["parsing"])

    def test_duplicate_prediction_error_is_distinct_from_gold_error(self):
        prediction = copy.deepcopy(self.run["predictions"][0]["parsing"])
        duplicate = copy.deepcopy(prediction["entities"][0])
        duplicate["id"] = "person-copy"
        for attr in duplicate["attributes"]:
            attr["entity_id"] = duplicate["id"]
        prediction["entities"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "prediction parsing: duplicate entity mention"):
            parsing_scores(self.row, prediction)

    def _rank_gap_comparison(self, baseline_rankings, candidate_rankings):
        original_para = next(v for v in self.row["variants"] if v["type"] == "paraphrase")
        for vid, text in (("para2", "A man wearing a red shirt is walking a black dog."),
                          ("para3", "A red-shirted man is walking a black dog.")):
            variant = copy.deepcopy(original_para)
            variant.update(id=vid, text=text)
            variant["edit"]["after"] = text
            self.row["variants"].append(variant)
        write_jsonl(self.samples, [self.row])
        frozen = self.root / "gap-frozen"
        freeze_semantics(self.samples, self.bundle, self.data, frozen, "synthetic", "fixture")
        self.protocol = frozen / "protocol.json"
        self.run["protocol_sha256"] = file_digest(self.protocol)
        self.run["predictions"][0]["scores"].update(para2=.5, para3=.5)
        self.run["predictions"][0]["rankings"] = baseline_rankings
        candidate = copy.deepcopy(self.run)
        candidate.update(run_id="gap-candidate", method="E1")
        candidate["model"].update(model_type="feature_mlp", adapter_sha256="a" * 64)
        candidate["predictions"][0]["rankings"] = candidate_rankings
        write_json(self.run_path, self.run)
        path = self.root / "gap-candidate.json"
        write_json(path, candidate)
        return compare_semantics(self.protocol, [self.run_path, path], self.root / "gap-comparison", 20)

    def test_disjoint_paraphrases_have_no_paired_rank_gap_delta(self):
        first = ["val-a", "val-b", "val-c"]
        last = ["val-b", "val-c", "val-a"]
        result = self._rank_gap_comparison({"original": first, "para": first, "para2": [], "para3": []},
                                           {"original": first, "para": [], "para2": last, "para3": []})
        entry = result["comparisons"][0]
        metric = "paraphrase/absolute_rank_gap_full_rankings"
        self.assertIsNone(entry["source_macro_deltas"][metric]["source_macro_mean"])
        self.assertIsNone(entry["source_macro_deltas"][metric]["bootstrap"])
        self.assertEqual(entry["rank_gap_pairing"]["shared_pair_count"], 0)
        self.assertEqual(entry["rank_gap_pairing"]["baseline_only_pairs"][0]["variant_id"], "para")
        self.assertEqual(entry["rank_gap_pairing"]["candidate_only_pairs"][0]["variant_id"], "para2")
        self.assertEqual(entry["unpaired_metric_sources"][metric], {"baseline_only": ["val-a"], "candidate_only": ["val-a"]})
        self.assertEqual(result["seed_summary"]["E0"][metric]["mean"], 0)
        self.assertEqual(result["seed_summary"]["E1"][metric]["mean"], 2)

    def test_rank_gap_delta_uses_only_shared_sample_variant_ids(self):
        first = ["val-a", "val-b", "val-c"]
        middle = ["val-b", "val-a", "val-c"]
        last = ["val-b", "val-c", "val-a"]
        result = self._rank_gap_comparison({"original": first, "para": middle, "para2": first, "para3": []},
                                           {"original": first, "para": last, "para2": [], "para3": first})
        entry = result["comparisons"][0]
        metric = "paraphrase/absolute_rank_gap_full_rankings"
        self.assertEqual(entry["source_macro_deltas"][metric]["source_macro_mean"], 1)
        self.assertEqual(entry["source_macro_deltas"][metric]["bootstrap"]["percentile_95"], [1, 1])
        self.assertEqual(entry["rank_gap_pairing"]["shared_pairs"], [{"sample_id": self.row["sample_id"], "variant_id": "para", "source_image_id": "val-a"}])
        # Per-run descriptive means remain available, but their difference is .5.
        self.assertEqual(result["seed_summary"]["E0"][metric]["mean"], .5)
        self.assertEqual(result["seed_summary"]["E1"][metric]["mean"], 1)

    def test_paired_rank_gaps_average_within_source_before_bootstrap(self):
        para = copy.deepcopy(next(v for v in self.row["variants"] if v["type"] == "paraphrase"))
        para.update(id="para2", text="A man wearing a red shirt is walking a black dog.")
        para["edit"]["after"] = para["text"]
        self.row["variants"].append(para)
        other = copy.deepcopy(self.row)
        other.update(sample_id="sample-val-b", group_id="val-b", scene_group_id="val-b", source_image_id="val-b", source_query_id="val-b#0")
        rows = [self.row, other]
        write_jsonl(self.samples, rows)
        target = self.root / "source-macro-pairs"
        freeze_semantics(self.samples, self.bundle, self.data, target, "synthetic", "fixture")
        protocol = target / "protocol.json"
        baseline = copy.deepcopy(self.run)
        baseline["protocol_sha256"] = file_digest(protocol)
        first = ["val-a", "val-b", "val-c"]
        last = ["val-b", "val-c", "val-a"]
        b_first = ["val-b", "val-a", "val-c"]
        b_last = ["val-a", "val-c", "val-b"]
        baseline["predictions"] = [copy.deepcopy(self.run["predictions"][0]) for _ in rows]
        for row, pred in zip(rows, baseline["predictions"]):
            pred["sample_id"] = row["sample_id"]
            pred["scores"]["para2"] = .5
        baseline["predictions"][0]["rankings"] = {"original": first, "para": first, "para2": first}
        baseline["predictions"][1]["rankings"] = {"original": b_first, "para": b_last, "para2": []}
        candidate = copy.deepcopy(baseline)
        candidate.update(method="E1", run_id="source-macro-candidate")
        candidate["model"].update(model_type="feature_mlp", adapter_sha256="a" * 64)
        candidate["predictions"][0]["rankings"]["para2"] = last
        candidate["predictions"][1]["rankings"]["para"] = b_first
        write_json(self.run_path, baseline)
        path = self.root / "source-macro-candidate.json"
        write_json(path, candidate)
        result = compare_semantics(protocol, [self.run_path, path], self.root / "source-macro-comparison", 20)
        entry = result["comparisons"][0]
        metric = entry["source_macro_deltas"]["paraphrase/absolute_rank_gap_full_rankings"]
        # Source A has deltas 0 and +2; source B has delta -2: mean(1,-2)=-.5.
        self.assertEqual(metric["source_macro_mean"], -.5)
        self.assertEqual(metric["item_count"], 3)
        self.assertEqual(metric["bootstrap"]["cluster_count"], 2)
        self.assertEqual(entry["rank_gap_pairing"]["shared_pair_count"], 3)


if __name__ == "__main__":
    unittest.main()
