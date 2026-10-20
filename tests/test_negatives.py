"""Negative mining tests: exclusion rules and validate-negatives compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy
except ImportError:
    numpy = None

from fgir.analysis import validate_negatives
from fgir.data import file_digest
from fgir.encoding import save_features
from fgir.files import write_json, write_jsonl
from fgir.negatives import exclusion_groups, mine_negatives

requires_numpy = unittest.skipUnless(numpy is not None, "numpy not installed")


def make_bundle(root):
    """Minimal train-only bundle with one same-scene group and one repeated caption."""
    bundle = root / "bundle"
    (bundle / "train").mkdir(parents=True)
    rows = [
        {"image_id": "a", "path": "images/a.jpg", "split": "train", "captions": ["A dog runs .", "Second a ."]},
        {"image_id": "b", "path": "images/b.jpg", "split": "train", "captions": ["A dog runs .", "Second b ."]},
        {"image_id": "c", "path": "images/c.jpg", "split": "train", "captions": ["Third c ."]},
        {"image_id": "d", "path": "images/d.jpg", "split": "train", "captions": ["Fourth d ."]},
        {"image_id": "e", "path": "images/e.jpg", "split": "train", "captions": ["Fifth e ."]},
    ]
    write_jsonl(bundle / "train" / "images.jsonl", rows)
    pairs = [{"query_id": f"{row['image_id']}#{i}", "image_id": row["image_id"],
              "path": row["path"], "text": caption, "split": "train"}
             for row in rows for i, caption in enumerate(row["captions"])]
    write_jsonl(bundle / "train" / "pairs.jsonl", pairs)
    review = {"pairs": [{"left": "c", "right": "d", "decision": "same_scene"}]}
    write_json(bundle / "duplicate-review.json", review)
    write_json(bundle / "dataset.json", {"schema_version": "fgir-dataset-v1",
                                        "files": {"duplicate-review.json": file_digest(bundle / "duplicate-review.json")}})
    return bundle, rows, pairs


@requires_numpy
class NegativesMiningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle, self.rows, self.pairs = make_bundle(self.root)
        texts = {"a#0": [1.0, 0.0], "a#1": [0.9, 0.1], "b#0": [1.0, 0.0], "b#1": [0.2, 1.0],
                 "c#0": [0.8, 0.2], "d#0": [0.7, 0.3], "e#0": [0.3, 0.7]}
        images = {"a": [1.0, 0.0], "b": [0.95, 0.05], "c": [0.8, 0.2], "d": [0.75, 0.25], "e": [0.2, 1.0]}
        meta = {"model_id": "test-model", "weights_sha256": "a" * 64}
        self.text_path = self.root / "texts.npz"
        self.image_path = self.root / "images.npz"
        save_features(self.text_path, list(texts), numpy.asarray(list(texts.values()), dtype=numpy.float32), meta)
        save_features(self.image_path, list(images), numpy.asarray(list(images.values()), dtype=numpy.float32), meta)
        self.output = self.root / "negatives.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _mine(self, **kwargs):
        return mine_negatives(self.bundle, self.text_path, self.image_path, self.output,
                              top_m=3, per_query=2, seed=1, review_sample=2, **kwargs)

    def test_mined_records_pass_validate_negatives(self):
        self._mine()
        result = validate_negatives(self._manifest(), self.output)
        self.assertTrue(result["ok"])
        self.assertEqual(result["query_count"], len(self.pairs))
        self.assertEqual(result["reviewed_query_count"], 0)

    def _manifest(self):
        path = self.root / "manifest.jsonl"
        if not path.exists():
            write_jsonl(path, self.rows)
        return path

    def test_source_never_appears_as_negative(self):
        self._mine()
        for row in json.loads("[" + ",".join(self.output.read_text(encoding="utf-8").strip().splitlines()) + "]"):
            self.assertNotIn(row["positive_image_ids"][0], row["negative_image_ids"])

    def test_identical_caption_images_excluded(self):
        self._mine()
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["query_id"] in ("a#0", "b#0"):
                self.assertEqual(row["excluded_image_ids"].get("b" if row["query_id"] == "a#0" else "a"),
                                 "identical_caption")
                other = "b" if row["query_id"] == "a#0" else "a"
                self.assertNotIn(other, row["negative_image_ids"])

    def test_same_scene_group_excluded(self):
        self._mine()
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        c_rows = [row for row in rows if row["query_id"] == "c#0"]
        self.assertTrue(c_rows)
        self.assertEqual(c_rows[0]["excluded_image_ids"].get("d"), "same_scene_group")
        self.assertNotIn("d", c_rows[0]["negative_image_ids"])

    def test_group_ids_match_handoff_rule(self):
        groups = exclusion_groups(self.bundle)
        self.assertEqual(groups["c"], groups["d"])
        self.assertNotEqual(groups.get("a", "a"), groups.get("c", "c"))

    def test_review_sample_written_with_false_negatives_slot(self):
        self._mine()
        sample = self.output.with_suffix(".review-sample.jsonl")
        self.assertTrue(sample.exists())
        rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
        self.assertLessEqual(len(rows), 2)
        self.assertIn("false_negative_image_ids", rows[0])
        self.assertIsNone(rows[0]["decision"])

    def test_refuses_existing_output(self):
        self.output.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._mine()

    def test_text_and_image_weights_must_match(self):
        mismatch = self.root / "other.npz"
        save_features(mismatch, ["a"], numpy.asarray([[1.0, 0.0]], dtype=numpy.float32),
                      {"model_id": "test-model", "weights_sha256": "b" * 64})
        with self.assertRaises(ValueError):
            mine_negatives(self.bundle, self.text_path, mismatch, self.output, top_m=3, per_query=2, seed=1)


if __name__ == "__main__":
    unittest.main()
