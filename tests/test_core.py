import math
import unittest

from fgir.metrics import evaluate
from fgir.retrieval import ExactIndex, normalize


class RetrievalTests(unittest.TestCase):
    def test_cosine_not_magnitude_and_k_clamps(self):
        index = ExactIndex([
            {"image_id": "b", "embedding": [6, 8]},
            {"image_id": "a", "embedding": [100, 0]},
        ])
        result = index.search([2, 0], 99)
        self.assertEqual([x["image_id"] for x in result], ["a", "b"])
        self.assertAlmostEqual(result[1]["score"], 0.6)

    def test_ties_are_deterministic(self):
        index = ExactIndex([
            {"image_id": "b", "embedding": [1, 0]},
            {"image_id": "a", "embedding": [1, 0]},
        ])
        self.assertEqual(index.search([1, 0], 1)[0]["image_id"], "a")

    def test_reject_invalid_vectors(self):
        for vector in ([], [0, 0], [math.nan, 1], [math.inf, 0], [True, 0], ["1", 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalize(vector)

    def test_extreme_finite_vectors(self):
        vector = normalize([1e308, 1e308])
        self.assertAlmostEqual(sum(x * x for x in vector), 1.0)

    def test_duplicate_and_dimension_mismatch(self):
        for records in (
            [{"image_id": "a", "embedding": [1]}, {"image_id": "a", "embedding": [1]}],
            [{"image_id": "a", "embedding": [1]}, {"image_id": "b", "embedding": [1, 0]}],
            [],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                ExactIndex(records)

    def test_query_validation(self):
        index = ExactIndex([{"image_id": "a", "embedding": [1, 0]}])
        with self.assertRaises(ValueError):
            index.search([1], 1)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                index.search([1, 0], k)


class MetricTests(unittest.TestCase):
    def test_multiple_relevant_and_empty_result(self):
        result = evaluate(
            {"q1": ["x", "b"], "q2": []},
            {"q1": ["a", "b"], "q2": ["c"]},
        )
        self.assertEqual(result["recall@1"], 0)
        self.assertEqual(result["recall@5"], 0.5)
        self.assertEqual(result["mrr_over_supplied_rankings"], 0.25)
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["ranking_depth_min"], 0)

    def test_missing_and_extra_queries_rejected(self):
        for rankings in ({}, {"q": [], "extra": []}):
            with self.subTest(rankings=rankings), self.assertRaises(ValueError):
                evaluate(rankings, {"q": ["a"]})

    def test_duplicate_results_and_empty_qrels_rejected(self):
        for rankings, qrels in (
            ({"q": ["a", "a"]}, {"q": ["a"]}),
            ({"q": []}, {"q": []}),
            ({"q": []}, {"q": ["a", "a"]}),
            ({}, {}),
        ):
            with self.subTest(rankings=rankings, qrels=qrels), self.assertRaises(ValueError):
                evaluate(rankings, qrels)

    def test_example_and_k_boundary(self):
        result = evaluate({"q1": ["a", "b"], "q2": ["a", "b"]}, {"q1": ["a"], "q2": ["b"]}, (1, 2))
        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@2"], 1.0)
        self.assertEqual(result["mrr_over_supplied_rankings"], 0.75)

    def test_invalid_k(self):
        for ks in ((), (0,), (True,), (1.5,)):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                evaluate({"q": []}, {"q": ["a"]}, ks)


if __name__ == "__main__":
    unittest.main()
