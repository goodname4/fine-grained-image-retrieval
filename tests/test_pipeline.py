"""Synthetic corpus exercises data handoff and annotation lifecycle without network."""

import copy
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

from fgir.analysis import render_cases, summarize_latency, validate_negatives
from fgir.annotations import apply_source_checks, draft_annotations, expand_annotation_pool, freeze_annotations, make_pool_request, merge_annotations, validate_annotations
from fgir.data import file_digest
from fgir.dataset import build_handoff, duplicate_report, inventory_images, verify_handoff
from fgir.download import extract_archive
from fgir.evaluation import load_protocol
from fgir.files import read_json, read_jsonl, write_json, write_jsonl
from fgir.handoff import training_pairs_for_epoch
from fgir.statistics import dataset_statistics


class PipelineTests(unittest.TestCase):
    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow data extra not installed")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        self.manifest = self.root / "manifest.jsonl"
        self.rows = []
        for i, split in enumerate(["train", "train", "val", "val", *(["test"] * 8)]):
            name = f"image-{i}"
            rng = random.Random(i)
            (self.root / "images" / (name + ".ppm")).write_bytes(b"P6\n24 24\n255\n" + rng.randbytes(24 * 24 * 3))
            self.rows.append({"image_id": name, "path": f"images/{name}.ppm", "split": split,
                              "captions": [f"two red dogs running under table {i}", f"caption number {i}"]})
        write_jsonl(self.manifest, self.rows)
        self.review = self.root / "review.json"
        inventory = inventory_images(self.rows, self.root)
        review = duplicate_report(inventory, 8)
        for pair in review["pairs"]:
            pair.update(decision="distinct", reviewer="synthetic-fixture", notes="independently generated synthetic random pixels")
        review["manifest_sha256"] = file_digest(self.manifest)
        write_json(self.review, review)

    def test_handoff_pairs_protocol_and_mutation_check(self):
        output = self.root / "bundle"
        summary = build_handoff(self.manifest, self.root, output, duplicate_review=self.review)
        self.assertEqual(summary["splits"]["train"]["caption_count"], 4)
        self.assertEqual(len(read_jsonl(output / "train/pairs.jsonl")), 4)
        self.assertTrue(verify_handoff(output, self.root)["ok"])
        self.assertEqual(load_protocol(output / "test/protocol.json")[0]["query_count"], 16)
        self.assertFalse((output / "train/protocol.json").exists())
        first = training_pairs_for_epoch(output, 0, seed=7)
        second = training_pairs_for_epoch(output, 1, seed=7)
        self.assertEqual(first, training_pairs_for_epoch(output, 0, seed=7))
        self.assertEqual({row["image_id"] for row in first}, {"image-0", "image-1"})
        self.assertEqual(len({row["image_id"] for row in first}), len(first))
        self.assertEqual({row["query_id"] for row in first} & {row["query_id"] for row in second}, set())
        statistics = dataset_statistics(output, self.root / "statistics")
        self.assertEqual(statistics["total_caption_count"], 24)
        (self.root / self.rows[0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image file hash mismatch"):
            verify_handoff(output, self.root)

    def test_handoff_exclusions_are_explicit_and_unknown_ids_rejected(self):
        exclusions = self.root / "excluded.json"
        write_json(exclusions, [{"image_id": "unknown", "reason": "test"}])
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_handoff(self.manifest, self.root, self.root / "bundle", exclusions, duplicate_review=self.review)
        write_json(exclusions, [{"image_id": "image-0", "reason": "synthetic exclusion"}])
        summary = build_handoff(self.manifest, self.root, self.root / "bundle", exclusions, duplicate_review=self.review)
        self.assertEqual(summary["excluded_image_count"], 1)
        self.assertEqual(summary["splits"]["train"]["image_count"], 1)

    def test_decoded_duplicates_block_even_with_changed_file_metadata(self):
        from PIL import Image, PngImagePlugin
        first = self.root / "images/a.png"
        second = self.root / "images/b.png"
        image = Image.open(self.root / self.rows[0]["path"])
        image.save(first)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("changed", "metadata")
        image.save(second, pnginfo=metadata)
        self.rows[0]["path"] = "images/a.png"
        self.rows[1]["path"] = "images/b.png"
        write_jsonl(self.manifest, self.rows)
        review = read_json(self.review)
        review["manifest_sha256"] = file_digest(self.manifest)
        write_json(self.review, review)
        with self.assertRaisesRegex(ValueError, "decoded duplicate"):
            build_handoff(self.manifest, self.root, self.root / "bundle", duplicate_review=self.review)
        self.assertFalse((self.root / "bundle").exists())

    def test_draft_allocation_source_checks_and_review_freeze(self):
        output = self.root / "drafts"
        summary = draft_annotations(self.manifest, self.root, output, per_category=1)
        self.assertEqual(summary["query_count"], 4)
        self.assertEqual(summary["assignment_counts"], dict.fromkeys("ABCD", 1))
        annotations = output / "annotations.jsonl"
        with self.assertRaisesRegex(ValueError, "independently reviewed"):
            freeze_annotations(self.manifest, annotations, self.root / "frozen", 4, 1)
        rows = read_jsonl(annotations)
        self.assertEqual(len({row["source_image_id"] for row in rows}), 4)
        checks = self.root / "checks.json"
        write_json(checks, {"manifest_sha256": file_digest(self.manifest), "actor": "fixture-assistant",
                            "judged_pool_image_ids": [rows[0]["source_image_id"]], "pool_method": "synthetic fixture",
                            "checks": [dict(rows[0], kind="initial_annotation", notes="synthetic", relevant_image_ids=[rows[0]["source_image_id"]])]})
        work = output / "work.jsonl"
        summary = apply_source_checks(self.manifest, annotations, checks, work)
        self.assertEqual(summary["status_counts"]["annotated"], 1)
        candidates = [row["image_id"] for row in self.rows if row["split"] == "test"]
        for row in rows:
            row.update(annotation_status="reviewed", annotation_scope="full_gallery", judged_image_ids=candidates,
                       annotator="fixture-author", reviewer="fixture-peer", review_notes="synthetic full gallery")
        write_jsonl(work, rows)
        summary = validate_annotations(self.manifest, work, True)
        self.assertEqual(summary["status_counts"]["reviewed"], 4)
        frozen = self.root / "frozen"
        freeze_annotations(self.manifest, work, frozen, 4, 1)
        self.assertEqual(load_protocol(frozen / "protocol.json")[0]["annotation_scope"], "full_gallery")

    def test_negative_mining_never_accepts_val_test_or_positive_as_negative(self):
        negatives = self.root / "negatives.jsonl"
        row = {"query_id": "image-0#0", "positive_image_ids": ["image-0"], "negative_image_ids": ["image-1"],
               "mining_model": "fixture", "reviewed": False}
        write_jsonl(negatives, [row])
        self.assertEqual(validate_negatives(self.manifest, negatives)["negative_count"], 1)
        for changed in ({"negative_image_ids": ["image-0"]}, {"negative_image_ids": ["image-4"]},
                        {"query_id": "image-4#0"}, {"reviewed": True}):
            write_jsonl(negatives, [{**row, **changed}])
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_negatives(self.manifest, negatives)

    def test_member_merge_rejects_conflicts_instead_of_overwriting(self):
        directory = self.root / "draft"
        draft_annotations(self.manifest, self.root, directory, per_category=1)
        base = directory / "annotations.jsonl"
        original = read_jsonl(base)
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        first[0]["annotation_notes"] = "member first check"
        second[1]["annotation_notes"] = "member second check"
        left, right = self.root / "left.jsonl", self.root / "right.jsonl"
        write_jsonl(left, first)
        write_jsonl(right, second)
        output = self.root / "merged"
        merge_annotations(self.manifest, base, [left, right], output)
        merged = read_jsonl(output / "annotations.jsonl")
        self.assertEqual(merged[0]["annotation_notes"], "member first check")
        self.assertEqual(merged[1]["annotation_notes"], "member second check")
        second[0]["annotation_notes"] = "conflicting check"
        write_jsonl(right, second)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            merge_annotations(self.manifest, base, [left, right], self.root / "conflict")
        self.assertFalse((self.root / "conflict").exists())

    def test_member_merge_rejects_cross_export_duplicate_text(self):
        directory = self.root / "draft"
        draft_annotations(self.manifest, self.root, directory, per_category=1)
        base = directory / "annotations.jsonl"
        original = read_jsonl(base)
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        first[0]["text"] = "Two dogs beside a red ball."
        second[1]["text"] = "  two dogs beside a RED ball.  "
        left, right = self.root / "left.jsonl", self.root / "right.jsonl"
        write_jsonl(left, first)
        write_jsonl(right, second)
        # Each member export is valid against the common base. Only their union
        # violates the normalized-text uniqueness rule across query IDs.
        validate_annotations(self.manifest, left)
        validate_annotations(self.manifest, right)
        output = self.root / "invalid-merge"
        with self.assertRaisesRegex(ValueError, "duplicate fine-grained query text"):
            merge_annotations(self.manifest, base, [left, right], output)
        self.assertFalse(output.exists())
        self.assertEqual(read_jsonl(base), original)
        self.assertEqual(read_jsonl(left), first)
        self.assertEqual(read_jsonl(right), second)

    def test_common_pool_union_keeps_new_images_unjudged_and_binds_query_text(self):
        drafts = self.root / "drafts"
        draft_annotations(self.manifest, self.root, drafts, per_category=1)
        annotations = drafts / "annotations.jsonl"
        rows = read_jsonl(annotations)
        request = self.root / "request"
        make_pool_request(self.manifest, annotations, request)
        run_path = self.root / "pool-run.json"
        run = {"schema_version": "fgir-pool-run-v1", "request_sha256": file_digest(request / "request.json"),
               "model_id": "synthetic-fixture", "rankings": {row["query_id"]: ["image-4", "image-5"] for row in rows}}
        write_json(run_path, run)
        output = self.root / "expanded"
        expand_annotation_pool(self.manifest, annotations, request / "request.json", [run_path], output, depth=2)
        expanded = read_jsonl(output / "annotations.jsonl")
        self.assertTrue(all({"image-4", "image-5"} <= set(row["pool_candidate_image_ids"]) for row in expanded))
        self.assertTrue(all(row["judged_image_ids"] == [] for row in expanded))
        rows[0]["text"] = "changed query text"
        write_jsonl(annotations, rows)
        with self.assertRaisesRegex(ValueError, "texts changed"):
            expand_annotation_pool(self.manifest, annotations, request / "request.json", [run_path], self.root / "stale", depth=2)

    def test_latency_excludes_warmup_and_separates_cold_and_devices(self):
        path = self.root / "latency.jsonl"
        base = {"stage": "end_to_end", "device": "cpu", "gallery_size": 8, "top_k": 5,
                "batch_size": 1, "cache_state": "warm", "warmup": False, "cold_start": False}
        rows = [{**base, "milliseconds": value} for value in (1, 2, 3, 4, 5)]
        rows += [{**base, "milliseconds": 999, "warmup": True}, {**base, "milliseconds": 99, "cold_start": True}]
        write_jsonl(path, rows)
        result = summarize_latency(path)
        steady = next(row for row in result["groups"] if row["stage"] == "end_to_end" and not row["cold_start"])
        self.assertEqual(steady["p50_ms"], 3)
        self.assertAlmostEqual(steady["p95_ms"], 4.8)
        self.assertEqual(steady["sample_count"], 5)
        cold = next(row for row in result["groups"] if row["cold_start"])
        self.assertEqual(cold["stage"], "end_to_end")
        self.assertEqual(cold["sample_count"], 1)
        self.assertEqual(cold["p50_ms"], 99)
        self.assertEqual(result["excluded_from_steady_state"], {"warmup": 1, "cold_start": 1})
        write_jsonl(path, [{**base, "device": "cuda:0", "milliseconds": 1}])
        with self.assertRaisesRegex(ValueError, "synchronization"):
            summarize_latency(path)

    def test_latency_preserves_cold_stages_and_legacy_cold_start_stage(self):
        path = self.root / "latency.jsonl"
        base = {"device": "cpu", "gallery_size": 1000, "top_k": 10,
                "batch_size": 1, "cache_state": "cold", "warmup": False, "cold_start": True}
        rows = [{**base, "stage": stage, "milliseconds": value} for stage, value in
                (("text_encoding", 100), ("vector_sort", 5), ("end_to_end", 110))]
        # The old explicit cold_start stage still denotes a cold measurement,
        # even when callers supplied false in the separate flag.
        rows.append({**base, "stage": "cold_start", "cold_start": False, "milliseconds": 150})
        write_jsonl(path, rows)
        result = summarize_latency(path)
        self.assertEqual(result["excluded_from_steady_state"], {"cold_start": 4})
        groups = {row["stage"]: row for row in result["groups"]}
        self.assertEqual(set(groups), {"text_encoding", "vector_sort", "end_to_end", "cold_start"})
        for stage, expected in (("text_encoding", 100), ("vector_sort", 5), ("end_to_end", 110), ("cold_start", 150)):
            with self.subTest(stage=stage):
                self.assertTrue(groups[stage]["cold_start"])
                self.assertEqual(groups[stage]["sample_count"], 1)
                self.assertEqual(groups[stage]["p50_ms"], expected)
                self.assertEqual(groups[stage]["p95_ms"], expected)

    def test_zip_rejects_traversal_and_unexpected_existing_content(self):
        archive = self.root / "a.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../bad.txt", "escape")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            extract_archive(archive, self.root / "raw")
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("folder/good.txt", "good")
        self.assertEqual(extract_archive(archive, self.root / "raw"), 1)
        self.assertEqual(extract_archive(archive, self.root / "raw"), 1)
        (self.root / "raw/good.txt").write_text("bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            extract_archive(archive, self.root / "raw")

    def test_visual_case_review_embeds_images_and_rejects_unknown_ids(self):
        cases = self.root / "cases.jsonl"
        row = {"query_id": "synthetic-case", "text": "synthetic <script> label", "categories": ["count"],
               "relevant_image_ids": ["image-4"], "baseline_ranking": ["image-5", "image-4"],
               "candidate_ranking": ["image-4", "image-5"], "outcome": "improved",
               "analysis_notes": "", "error_type": None}
        write_jsonl(cases, [row])
        render_cases(self.manifest, cases, self.root, self.root / "cases-review")
        page = (self.root / "cases-review/review.html").read_text(encoding="utf-8")
        self.assertIn("data:image/jpeg;base64,", page)
        self.assertIn("&lt;script&gt;", page)
        row["candidate_ranking"] = ["unknown"]
        write_jsonl(cases, [row])
        with self.assertRaisesRegex(ValueError, "unknown image ID"):
            render_cases(self.manifest, cases, self.root, self.root / "bad-cases")
        self.assertFalse((self.root / "bad-cases").exists())

    def test_cli_handoff_and_annotation_validation(self):
        output = self.root / "bundle"
        result = subprocess.run([sys.executable, "-m", "fgir", "build-handoff", "--manifest", str(self.manifest),
                                 "--data-root", str(self.root), "--output", str(output), "--duplicate-review", str(self.review)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-m", "fgir", "verify-handoff", "--bundle", str(output),
                                 "--data-root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
