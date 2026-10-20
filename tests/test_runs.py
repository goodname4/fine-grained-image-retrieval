"""Run export tests against the synthetic evaluation fixtures in examples/."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy
except ImportError:
    numpy = None

from fgir.data import file_digest
from fgir.encoding import save_features
from fgir.evaluation import evaluate_run
from fgir.features import build_index
from fgir.files import read_json, read_jsonl
from fgir.runs import export_run, search_index

requires_numpy = unittest.skipUnless(numpy is not None, "numpy not installed")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "evaluation"


@requires_numpy
class ExportRunTests(unittest.TestCase):
    """End-to-end contract check using the repo's synthetic protocol fixtures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        protocol = read_json(EXAMPLES / "protocol.json")
        candidates = read_json(EXAMPLES / "candidates.json")
        queries = read_jsonl(EXAMPLES / "queries.jsonl")
        self.protocol_path = EXAMPLES / "protocol.json"
        rng = numpy.random.default_rng(7)
        image_vectors = rng.normal(size=(len(candidates), 8)).astype(numpy.float32)
        query_vectors = rng.normal(size=(len(queries), 8)).astype(numpy.float32)
        meta = {"model_id": "synthetic-model", "weights_sha256": "a" * 64, "device": "cpu"}
        self.index_path = self.root / "index.npz"
        self.texts_path = self.root / "texts.npz"
        save_features(self.root / "gallery.npz", candidates, image_vectors, meta)
        save_features(self.texts_path, [row["query_id"] for row in queries], query_vectors, meta)
        build_index(self.root / "gallery.npz", self.index_path)
        self.output = self.root / "run.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_exported_run_passes_evaluate_run(self):
        result = export_run(self.protocol_path, self.index_path, self.texts_path,
                            self.output, method="E0", seed=3, depth=10)
        run = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(run["schema_version"], "fgir-run-v1")
        self.assertIsNone(run["adapter_sha256"])
        self.assertEqual(run["protocol_sha256"], file_digest(self.protocol_path))
        self.assertEqual(result["query_count"], 16)
        evaluation = evaluate_run(self.protocol_path, self.output)
        self.assertIn("mrr@10", evaluation["metrics"])
        self.assertEqual(evaluation["metrics"]["declared_ranking_depth"], 10)

    def test_full_gallery_depth_labels_true_mrr(self):
        export_run(self.protocol_path, self.index_path, self.texts_path, self.output, method="E0")
        evaluation = evaluate_run(self.protocol_path, self.output)
        self.assertIn("mrr", evaluation["metrics"])
        self.assertEqual(evaluation["metrics"]["declared_ranking_depth"], 12)

    def test_e0_rejects_adapter(self):
        with self.assertRaises(ValueError):
            export_run(self.protocol_path, self.index_path, self.texts_path,
                       self.output, method="E0", adapter_sha256="b" * 64)

    def test_e1_requires_adapter(self):
        with self.assertRaises(ValueError):
            export_run(self.protocol_path, self.index_path, self.texts_path,
                       self.output, method="E1")

    def test_rankings_cover_exactly_the_protocol_queries(self):
        export_run(self.protocol_path, self.index_path, self.texts_path, self.output, depth=10)
        run = json.loads(self.output.read_text(encoding="utf-8"))
        qrels = read_json(EXAMPLES / "qrels.json")
        self.assertEqual(set(run["rankings"]), set(qrels))
        self.assertTrue(all(len(row) == 10 for row in run["rankings"].values()))

    def test_refuses_existing_output(self):
        self.output.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            export_run(self.protocol_path, self.index_path, self.texts_path, self.output)

    def test_search_index_batch_output(self):
        out = self.root / "search.json"
        result = search_index(self.index_path, self.texts_path, out, top_k=3)
        self.assertEqual(result["query_count"], 16)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["top_k"], 3)
        self.assertTrue(all(len(hits) == 3 for hits in payload["results"].values()))


if __name__ == "__main__":
    unittest.main()
