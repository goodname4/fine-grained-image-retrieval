"""Small explicit rankings test protocol integrity and ranking math independently."""

import copy
from pathlib import Path
import tempfile
import unittest

from fgir.contracts import validate_queries
from fgir.data import file_digest
from fgir.evaluation import compare_runs, evaluate_run, load_protocol, paired_bootstrap, save_evaluation, summarize_seeds
from fgir.files import new_directory, parse_json, read_json, read_jsonl, write_json, write_jsonl


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = self.root / "protocol.json"
        self.run_path = self.root / "run.json"
        self.queries = [
            {"query_id": "q1", "text": "two red dogs", "categories": ["color", "count"], "relevant_image_ids": ["a", "b"]},
            {"query_id": "q2", "text": "cat below table", "categories": ["relation"], "relevant_image_ids": ["c"]},
            {"query_id": "q3", "text": "dog running", "categories": ["action"], "relevant_image_ids": ["d"]},
            {"query_id": "q4", "text": "one blue dog", "categories": ["color"], "relevant_image_ids": ["b"]},
        ]
        self.write_protocol()
        self.run = {"schema_version": "fgir-run-v1", "run_id": "fixture-baseline", "method": "E0", "model_id": "synthetic-fixture",
                    "data_kind": "synthetic", "protocol_sha256": file_digest(self.protocol), "seed": 42, "config": {"fixture": True},
                    "weights_sha256": "0" * 64, "adapter_sha256": None, "code_revision": "synthetic-test",
                    "ranking_depth": 4, "rankings": {"q1": ["c", "a", "b", "d"], "q2": ["c", "a", "b", "d"], "q3": [], "q4": ["b", "a", "c", "d"]}}
        self.save_run()

    def write_protocol(self, scope="source_pair", ks=None):
        write_json(self.root / "candidates.json", ["a", "b", "c", "d"])
        write_jsonl(self.root / "queries.jsonl", self.queries)
        write_json(self.root / "qrels.json", {r["query_id"]: r["relevant_image_ids"] for r in self.queries})
        write_json(self.protocol, {"schema_version": "fgir-protocol-v1", "task": "text_to_image", "split": "test", "data_kind": "synthetic",
                                  "annotation_scope": scope, "independently_reviewed": scope != "source_pair", "manifest_sha256": "0" * 64,
                                  "candidate_count": 4, "query_count": 4, "recall_ks": ks or [1, 5, 10],
                                  "files": {n: {"path": n, "sha256": file_digest(self.root / n)} for n in ("candidates.json", "queries.jsonl", "qrels.json")}})

    def save_run(self):
        write_json(self.run_path, self.run)

    def test_known_multiple_relevance_and_empty_results(self):
        result = evaluate_run(self.protocol, self.run_path)
        self.assertEqual(result["metrics"]["recall@1"], .5)
        self.assertEqual(result["metrics"]["recall@5"], .75)
        self.assertEqual(result["metrics"]["mrr"], .625)
        self.assertEqual(result["metrics"]["empty_result_count"], 1)
        self.assertEqual(result["by_category"]["color"]["query_count"], 2)
        self.assertEqual(result["by_category"]["color"]["mrr"], .75)
        self.assertEqual(result["by_category"]["action"]["recall@10"], 0)
        self.assertIsNone(result["per_query"][2]["first_relevant_rank"])

    def test_truncated_mrr_is_named_and_short_lists_rejected(self):
        self.write_protocol(ks=[1, 2])
        self.run["protocol_sha256"] = file_digest(self.protocol)
        self.run["ranking_depth"] = 2
        self.run["rankings"] = {q: values[:2] for q, values in self.run["rankings"].items()}
        self.save_run()
        self.assertEqual(evaluate_run(self.protocol, self.run_path)["metrics"]["mrr@2"], .625)
        self.run["rankings"]["q1"] = ["a"]
        self.save_run()
        with self.assertRaisesRegex(ValueError, "declared depth"):
            evaluate_run(self.protocol, self.run_path)

    def test_reject_query_drift_duplicates_and_unknown_images(self):
        original = copy.deepcopy(self.run)
        for rankings in ({"q1": []}, {**original["rankings"], "extra": []},
                         {**original["rankings"], "q1": ["a", "a", "b", "c"]},
                         {**original["rankings"], "q1": ["a", "b", "c", "outside"]}):
            self.run = {**original, "rankings": rankings}
            self.save_run()
            with self.subTest(rankings=rankings), self.assertRaises(ValueError):
                evaluate_run(self.protocol, self.run_path)

    def test_reject_protocol_or_bound_file_tampering(self):
        self.run["protocol_sha256"] = "f" * 64
        self.save_run()
        with self.assertRaisesRegex(ValueError, "different protocol"):
            evaluate_run(self.protocol, self.run_path)
        write_json(self.root / "candidates.json", ["d", "c", "b", "a"])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_protocol(self.protocol)

    def test_metadata_and_depth_validation(self):
        for key, value in (("data_kind", "real"), ("seed", True), ("weights_sha256", "placeholder"),
                           ("config", []), ("ranking_depth", 0), ("ranking_depth", 3), ("code_revision", "")):
            with self.subTest(key=key):
                original = copy.deepcopy(self.run)
                self.run[key] = value
                self.save_run()
                with self.assertRaises(ValueError):
                    evaluate_run(self.protocol, self.run_path)
                self.run = original

    def test_pooled_evaluation_rejects_unjudged_returns(self):
        for row in self.queries:
            row.update(annotation_status="reviewed", annotator="fixture-author", reviewer="fixture-peer", review_notes="synthetic fixture",
                       annotation_scope="candidate_pool", judged_image_ids=row["relevant_image_ids"], pool_method="synthetic test pool")
        self.write_protocol(scope="candidate_pool")
        self.run["protocol_sha256"] = file_digest(self.protocol)
        self.save_run()
        with self.assertRaisesRegex(ValueError, "unjudged"):
            evaluate_run(self.protocol, self.run_path)

    def test_independent_review_and_coverage_gate(self):
        row = dict(self.queries[0], annotation_status="reviewed", annotation_scope="full_gallery", annotator="same", reviewer="same",
                   judged_image_ids=["a", "b"], review_notes="test")
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_queries([row], ["a", "b", "c"], require_review=True, fine_grained=True)
        row["reviewer"] = "other"
        with self.assertRaisesRegex(ValueError, "every candidate"):
            validate_queries([row], ["a", "b", "c"], require_review=True, fine_grained=True)
        row["annotation_status"] = "draft"
        with self.assertRaisesRegex(ValueError, "not been independently"):
            validate_queries([row], ["a", "b", "c"], require_review=True, fine_grained=True)

    def test_comparison_keeps_regressions_and_per_query_exports(self):
        candidate = self.root / "candidate.json"
        run = copy.deepcopy(self.run)
        run.update(run_id="fixture-candidate", method="fixture-variant", rankings={"q1": ["a", "b", "c", "d"], "q2": ["a", "c", "b", "d"], "q3": ["d", "a", "b", "c"], "q4": []})
        write_json(candidate, run)
        output = self.root / "comparison"
        compare_runs(self.protocol, [self.run_path, candidate], output, bootstrap_samples=50)
        result = read_json(output / "comparison.json")["comparisons"][0]
        self.assertEqual(result["outcome_counts"]["improved"], 2)
        self.assertEqual(result["outcome_counts"]["regressed"], 2)
        self.assertEqual(result["metric_deltas"]["mrr"], 0)
        self.assertEqual(len(read_jsonl(output / "cases-1.jsonl")), 4)
        self.assertTrue(all(row["analysis_status"] == "needs_visual_review" for row in read_jsonl(output / "cases-1.jsonl")))
        saved = self.root / "evaluation"
        save_evaluation(self.protocol, self.run_path, saved)
        self.assertEqual(len(read_jsonl(saved / "per-query.jsonl")), 4)

    def test_bootstrap_is_paired_deterministic_and_not_seed_variation(self):
        self.assertEqual(paired_bootstrap([1, 1], 25)["percentile_95"], [1, 1])
        self.assertEqual(paired_bootstrap([1, 0, -1], 50), paired_bootstrap([1, 0, -1], 50))
        self.assertIsNone(paired_bootstrap([1], 0))
        result = paired_bootstrap([1, 1, -1, -1], 100, groups=["a", "a", "b", "b"])
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["percentile_95"], [-1, 1])

    def test_seed_summary_requires_distinct_seeds_and_fixed_configuration(self):
        paths = []
        for i in range(3):
            run = copy.deepcopy(self.run)
            run.update(run_id=f"fixture-seed-{i}", seed=i)
            path = self.root / f"seed-{i}.json"
            write_json(path, run)
            paths.append(path)
        result = summarize_seeds(self.protocol, paths, self.root / "seeds")
        self.assertEqual(result["groups"][0]["seed_count"], 3)
        self.assertEqual(result["groups"][0]["metrics"]["mrr"], {"mean": .625, "sample_std": 0})
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_seeds(self.protocol, [paths[0], paths[0]], self.root / "duplicate")
        run = read_json(paths[1])
        run["config"] = {"changed_learning_rate": True}
        write_json(paths[1], run)
        with self.assertRaisesRegex(ValueError, "configuration"):
            summarize_seeds(self.protocol, paths, self.root / "mixed")

    def test_checked_in_cli_fixtures_cannot_be_used_as_real_data(self):
        import subprocess
        import sys
        examples = Path(__file__).resolve().parents[1] / "examples/evaluation"
        output = self.root / "cli-comparison"
        result = subprocess.run([sys.executable, "-m", "fgir", "compare-runs", "--protocol", str(examples / "protocol.json"),
                                 "--runs", *[str(examples / f"run-{method}.json") for method in ("e0", "e1", "e2")],
                                 "--output", str(output), "--bootstrap-samples", "20"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = read_json(output / "run-0.metrics.json")
        self.assertEqual(summary["data_kind"], "synthetic")
        self.assertEqual(summary["metrics"]["recall@1"], .25)
        self.assertTrue(read_json(output / "case-coverage-1.json")["minimum_12_met"])

    def test_strict_json_and_atomic_output(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_json(value)
        output = self.root / "partial"
        with self.assertRaises(ValueError):
            with new_directory(output) as staging:
                write_json(staging / "example.json", {})
                raise ValueError("abort")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
