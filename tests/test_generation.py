"""B's generation pipeline tests: scripted backend, no model weights needed."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from fgir.data import file_digest
from fgir.files import read_jsonl, write_json, write_jsonl
from fgir.semantic_data import validate_semantics
from fgir.semantic_generation import GenerationBackend, LocalLmBackend, generate_semantics

CAPTION = "A man in a red shirt walks a black dog."
PROMPTS = {"schema_version": "fgir-semantic-generation-prompts-v1",
           "parse": {"system": "Parse this caption.", "examples": [], "max_new_tokens": 200},
           "paraphrase": {"system": "Rewrite this caption.", "examples": [], "max_new_tokens": 96}}


class ScriptedBackend:
    name = "scripted-test-backend"

    def __init__(self):
        self.weights_sha256 = hashlib.sha256(b"scripted test backend, no real weights").hexdigest()
        self.parses = {}
        self.paraphrases = {}

    def generate(self, messages, max_new_tokens, seed):
        caption = messages[-1]["content"]
        if "Rewrite" in messages[0]["content"]:
            return self.paraphrases.get(caption, "")
        return self.parses.get(caption, "NOTHING")

    def provenance(self):
        return {"name": self.name, "weights_sha256": self.weights_sha256, "prompt_sha256": None}

    def describe(self):
        return {}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "source"
        self.data = self.root / "data"
        self._make_fixture()
        self.prompts_path = self.root / "prompts.json"
        write_json(self.prompts_path, PROMPTS)

    def _make_fixture(self):
        bundle, data = self.bundle, self.data
        bundle.mkdir(parents=True)
        (data / "images").mkdir(parents=True)
        captions = {"val-b": ["A black dog runs in a park."]}
        manifest = []
        for image_id, split in (("train-a", "train"), ("val-a", "val"), ("val-b", "val"), ("test-a", "test")):
            image = data / "images" / f"{image_id}.ppm"
            image.write_bytes(b"P6\n2 1\n255\n\xff\x00\x00\x00\x00\xff")
            manifest.append({"image_id": image_id, "split": split, "path": f"images/{image_id}.ppm",
                             "captions": captions.get(image_id, [CAPTION])})
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
                                        "text": r["captions"][0], "split": split}
                                       for r in manifest if r["split"] == split])
        write_json(bundle / "dataset.json", {"schema_version": "fgir-dataset-v1", "data_kind": "synthetic",
                   "image_inventory_sha256": file_digest(bundle / "inventory.jsonl"),
                   "files": {n: file_digest(bundle / n) for n in names}})

    def _pilot_row(self, sample_id="semantic:val-a#0", image_id="val-a", query_id="val-a#0"):
        return {"schema_version": "fgir-semantic-v1", "sample_id": sample_id, "group_id": image_id,
                "scene_group_id": image_id, "source_image_id": image_id, "source_query_id": query_id, "split": "val",
                "text": CAPTION, "source_visual_label": "uncertain", "entities": [], "relations": [],
                "variants": [], "units": [], "assigned_to": "B", "independent_reviewer": "A",
                "provenance": {"source_manifest_sha256": file_digest(self.bundle / "manifest.jsonl"),
                               "generator": {"name": "source-only", "weights_sha256": None, "prompt_sha256": None},
                               "seed": 42, "code_revision": "pilot-fixture"},
                "review": {"status": "draft", "image_sha256": file_digest(self.data / "images/val-a.ppm"),
                           "disputes": []}}

    def _run(self, backend, rows=None, limit=None, seed=42):
        self._run_counter = getattr(self, "_run_counter", 0) + 1
        samples = self.root / "pilot.jsonl"
        write_jsonl(samples, rows if rows is not None else [self._pilot_row()])
        output = self.root / f"candidates-{self._run_counter}"
        result = generate_semantics(samples, self.bundle, self.data, output, backend, self.prompts_path,
                                    seed=seed, code_revision="test-revision", limit=limit)
        return samples, output, result

    def _good_backend(self):
        backend = ScriptedBackend()
        backend.parses[CAPTION] = ("ENTITY: man\nENTITY: dog\nATTRIBUTE: man | shirt_color | red\n"
                                   "ATTRIBUTE: dog | color | black\nRELATION: man | walks | dog")
        backend.paraphrases[CAPTION] = "A black dog is walked by a man in a red shirt."
        return backend

    def test_full_pipeline_yields_valid_draft_records(self):
        samples, output, result = self._run(self._good_backend())
        self.assertEqual(result["records"], 1)
        self.assertEqual((result["entities"], result["relations"], result["variants"], result["units"]), (2, 1, 5, 2))
        validate_semantics(output / "samples.work.jsonl", self.bundle, "draft", self.data)
        row = read_jsonl(output / "samples.work.jsonl")[0]
        self.assertEqual(row["assigned_to"], "B")
        self.assertEqual(row["review"]["status"], "draft")
        self.assertEqual(row["entities"][0], {"id": "e1", "head": "man", "span": [2, 5], "attributes": [
            {"name": "shirt_color", "value": "red", "span": [11, 14], "entity_id": "e1"}]})
        self.assertEqual(row["relations"][0], {"id": "r1", "subject": "e1", "object": "e2",
                                               "predicate": "walks", "span": [21, 26]})
        by_id = {v["id"]: v for v in row["variants"]}
        self.assertEqual(by_id["v-binding"]["text"], "A man in a black shirt walks a red dog.")
        self.assertEqual(by_id["v-direction"]["text"], "A black dog walks a man in a red shirt.")
        self.assertEqual(by_id["v-paraphrase"]["semantic_relation"], "equivalent")
        self.assertEqual(by_id["v-add-true"]["control_pair_id"], by_id["v-add-false"]["control_pair_id"])
        provenance = row["provenance"]["generator"]
        self.assertEqual(provenance["name"], ScriptedBackend.name)
        self.assertTrue(all(isinstance(provenance[k], str) and len(provenance[k]) == 64 for k in ("weights_sha256", "prompt_sha256")))

    def test_each_foil_corresponds_to_the_linked_single_edit(self):
        _, output, _ = self._run(self._good_backend())
        row = read_jsonl(output / "samples.work.jsonl")[0]
        variants = {v["id"]: v for v in row["variants"]}
        for unit in row["units"]:
            variant = variants[unit["variant_id"]]
            a, b = unit["span"]
            c, d = variant["edit"]["span"]
            self.assertLessEqual(a, c)
            self.assertLessEqual(d, b)
            self.assertEqual(CAPTION[a:c] + variant["edit"]["after"] + CAPTION[d:b], unit["foil"])
            self.assertEqual(CAPTION[:c] + variant["edit"]["after"] + CAPTION[d:], variant["text"])

    def test_garbage_ambiguous_and_missing_anchors_are_excluded_with_reasons(self):
        backend = ScriptedBackend()
        backend.parses[CAPTION] = ("ENTITY: man\nENTITY: dog\nENTITY: the\nENTITY: man\n"
                                   "ATTRIBUTE: man | shirt_color | green\nATTRIBUTE: unknown | color | red\n"
                                   "RELATION: man | sits | dog\nRELATION: with leaves | covers | lawn\n"
                                   "RELATION: man | walks | dog\nRELATION: red | dances | shirt")
        backend.paraphrases[CAPTION] = CAPTION  # identical: must be excluded
        _, output, result = self._run(backend)
        self.assertEqual(result["rejected_items"] > 0, True)
        rejects = read_jsonl(output / "rejections.jsonl")[0]["items"]
        reasons = " | ".join(f"{item['item']}: {item['reason']}" for item in rejects)
        self.assertIn("green", reasons)
        self.assertIn("unknown", reasons)
        self.assertIn("function word", reasons)
        self.assertIn("dances", reasons)
        self.assertIn("identical to the original caption", reasons)
        row = read_jsonl(output / "samples.work.jsonl")[0]
        self.assertNotIn("v-paraphrase", {v["id"] for v in row["variants"]})
        # The hallucinated predicate 'sits' is dropped, but the actual verb
        # 'walks' is recovered deterministically from the gap between the
        # anchored arguments; duplicate relations collapse to one record.
        self.assertEqual([r["predicate"] for r in row["relations"]], ["walks"])
        validate_semantics(output / "samples.work.jsonl", self.bundle, "draft", self.data)

    def test_lexicon_supplements_attributes_the_model_missed(self):
        caption = "A black dog runs in a park."
        row = dict(self._pilot_row(image_id="val-b", query_id="val-b#0"))
        row.update(text=caption)
        backend = ScriptedBackend()
        backend.parses[caption] = "ENTITY: dog\nENTITY: park\nRELATION: dog | runs | park"
        backend.paraphrases[caption] = "In a park, a black dog runs."
        samples = self.root / "pilot.jsonl"
        write_jsonl(samples, [row])
        output = self.root / "candidates-lexicon"
        generate_semantics(samples, self.bundle, self.data, output, backend, self.prompts_path,
                           seed=42, code_revision="test-revision")
        record = read_jsonl(output / "samples.work.jsonl")[0]
        dog = next(e for e in record["entities"] if e["head"] == "dog")
        attrs = {(a["name"], a["value"], a["span"][0]) for a in dog["attributes"]}
        self.assertIn(("dog_color", "black", caption.index("black")), attrs)
        validate_semantics(output / "samples.work.jsonl", self.bundle, "draft", self.data)

    def test_assistant_review_and_assignment_fields_survive_generation(self):
        row = self._pilot_row()
        row["source_visual_label"] = "supported"
        row["review"] = {"status": "annotated", "annotator": "A-assisted (scripted test)", "annotator_kind": "assistant",
                         "image_evidence": "scripted precheck", "image_sha256": file_digest(self.data / "images/val-a.ppm"),
                         "notes": "source precheck only", "disputes": []}
        _, output, _ = self._run(self._good_backend(), rows=[row])
        record = read_jsonl(output / "samples.work.jsonl")[0]
        self.assertEqual(record["review"]["annotator_kind"], "assistant")
        self.assertEqual(record["source_visual_label"], "supported")
        self.assertEqual(record["independent_reviewer"], "A")

    def test_same_seed_is_deterministic(self):
        first = self._run(self._good_backend(), seed=42)[2]
        second = self._run(self._good_backend(), seed=42)[2]
        self.assertEqual(first["samples_sha256"], second["samples_sha256"])

    def test_source_mismatch_rejects_whole_output(self):
        row = self._pilot_row(query_id="val-b#0")
        output = self.root / "bad-output"
        samples = self.root / "bad-pilot.jsonl"
        write_jsonl(samples, [row])
        with self.assertRaises(ValueError):
            generate_semantics(samples, self.bundle, self.data, output, self._good_backend(), self.prompts_path,
                               seed=42, code_revision="test-revision")
        self.assertFalse(output.exists())

    def test_missing_weights_reports_next_steps_without_fabrication(self):
        with self.assertRaises(ValueError) as caught:
            LocalLmBackend("org/definitely-not-a-real-model-xyz", allow_download=False)
        message = str(caught.exception)
        self.assertIn("org/definitely-not-a-real-model-xyz", message)
        self.assertIn("Next steps", message)
        self.assertIn("No model call", message)

    def test_backend_weight_digest_must_be_real_sha256(self):
        with self.assertRaises(ValueError):
            GenerationBackend("fake", "not-a-sha256")
        with self.assertRaises(ValueError):
            GenerationBackend("fake", "z" * 64)

    def test_additions_are_paired_and_always_present(self):
        backend = ScriptedBackend()
        backend.parses[CAPTION] = "NOTHING"
        backend.paraphrases[CAPTION] = ""
        _, output, _ = self._run(backend)
        row = read_jsonl(output / "samples.work.jsonl")[0]
        additions = [v for v in row["variants"] if v["type"].endswith("addition")]
        self.assertEqual({v["type"] for v in additions}, {"true_addition", "false_addition"})
        self.assertEqual(len({v["control_pair_id"] for v in additions}), 1)
        self.assertEqual(len({v["edit"]["span"][0] for v in additions}), 1)
        validate_semantics(output / "samples.work.jsonl", self.bundle, "draft", self.data)


if __name__ == "__main__":
    unittest.main()
