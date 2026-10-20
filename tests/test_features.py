"""Feature cache, index and run export tests; numpy-dependent parts self-skip."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy
except ImportError:
    numpy = None

from fgir.features import FeatureIndex, build_index
from fgir.encoding import save_features, load_features
from fgir.runs import code_revision

requires_numpy = unittest.skipUnless(numpy is not None, "numpy not installed")


@requires_numpy
class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, name, ids, vectors, extra=None):
        path = self.root / name
        save_features(path, ids, numpy.asarray(vectors, dtype=numpy.float32),
                      {"model_id": "test-model", "weights_sha256": "a" * 64, **(extra or {})})
        return path

    def test_roundtrip(self):
        path = self._save("f.npz", ["x", "y"], [[1.0, 0.0], [0.0, 2.0]])
        ids, vectors, meta = load_features(path)
        self.assertEqual(ids, ["x", "y"])
        self.assertEqual(meta["dimension"], 2)
        self.assertTrue(numpy.allclose(vectors, [[1, 0], [0, 2]]))

    def test_hash_mismatch_detected(self):
        path = self._save("f.npz", ["x"], [[1.0, 0.0]])
        path.write_bytes(path.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            load_features(path)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            self._save("f.npz", ["x", "x"], [[1.0, 0.0], [0.0, 1.0]])

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            self._save("f.npz", ["x"], [[float("nan"), 0.0]])

    def test_index_search_orders_by_score_then_id(self):
        index = FeatureIndex(["b", "a"], numpy.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=numpy.float32))
        results = index.search([1.0, 0.0], top_k=2)
        self.assertEqual([row["image_id"] for row in results], ["a", "b"])
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_index_rejects_dimension_mismatch(self):
        index = FeatureIndex(["a"], numpy.asarray([[1.0, 0.0]], dtype=numpy.float32))
        with self.assertRaises(ValueError):
            index.search([1.0, 0.0, 0.0], top_k=1)

    def test_index_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            FeatureIndex(["a"], numpy.asarray([[0.0, 0.0]], dtype=numpy.float32))

    def test_build_index_adapter_reuses_raw_features(self):
        path = self._save("raw.npz", ["p", "q"], [[1.0, 0.0], [0.0, 1.0]])
        out = self.root / "idx.npz"
        result = build_index(path, out)
        self.assertEqual(result["count"], 2)
        loaded = FeatureIndex.load(out)
        self.assertEqual(loaded.ids, ["p", "q"])
        self.assertEqual(loaded.search([0.0, 1.0], top_k=1)[0]["image_id"], "q")
        adapted = self.root / "adapted.npz"
        build_index(path, adapted, adapter=lambda rows: rows[:, ::-1])
        hits = FeatureIndex.load(adapted).search([0.0, 1.0], top_k=1)
        self.assertEqual(hits[0]["image_id"], "p")

    def test_index_dimension_check_on_load(self):
        path = self._save("raw.npz", ["p"], [[1.0, 0.0]])
        out = self.root / "idx.npz"
        build_index(path, out)
        loaded = FeatureIndex.load(out)
        self.assertEqual(loaded.dimension, 2)


class CodeRevisionTests(unittest.TestCase):
    def test_code_revision_reports_commit(self):
        revision = code_revision(str(Path(__file__).resolve().parents[1]))
        self.assertIn(revision["worktree"], ("clean", "dirty", "unknown"))
        self.assertTrue(revision["code_revision"])

    @unittest.skipUnless(shutil.which("git"), "git not installed")
    def test_every_unstaged_file_is_digested(self):
        # Regression: stripping porcelain output ate the first line's status space,
        # so that path parsed wrong and its digest was silently dropped.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for args in (["init", "-q"], ["config", "user.email", "t@example.com"],
                         ["config", "user.name", "test"], ["config", "commit.gpgsign", "false"]):
                subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
            (repo / "aaa.txt").write_text("one\n", encoding="utf-8")
            (repo / "zzz.txt").write_text("two\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True,
                           capture_output=True)
            (repo / "aaa.txt").write_text("changed\n", encoding="utf-8")
            (repo / "zzz.txt").write_text("changed\n", encoding="utf-8")
            revision = code_revision(str(repo))
            self.assertTrue(revision["code_revision"].endswith("-dirty"))
            self.assertEqual(sorted(revision["uncommitted_source_sha256"]), ["aaa.txt", "zzz.txt"])


if __name__ == "__main__":
    unittest.main()
