import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fgir.data import audit_manifest, prepare_flickr8k


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        self.manifest = self.root / "manifest.jsonl"

    def write_rows(self, rows):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    def row(self, image_id="a", split="train", path="images/a.ppm"):
        return {"image_id": image_id, "split": split, "path": path, "captions": ["a red pixel", "one pixel"]}

    def image(self, name="a.ppm", pixel=b"\xff\x00\x00"):
        # Real, distinct, tiny PPM images; no model or external dataset required.
        (self.root / "images" / name).write_bytes(b"P6\n1 1\n255\n" + pixel)

    def codes(self, report):
        return {issue["code"] for issue in report["issues"]}

    def test_valid_counts_and_stable_digest(self):
        self.image()
        self.write_rows([self.row()])
        result = audit_manifest(self.manifest, self.root)
        self.assertTrue(result["ok"])
        self.assertEqual(result["splits"]["train"], {"image_count": 1, "caption_count": 2})
        self.assertFalse(result["image_decoding_checked"])
        self.assertFalse(result["near_duplicates_checked"])
        self.assertEqual(result["manifest_sha256"], audit_manifest(self.manifest, self.root)["manifest_sha256"])

    def test_cross_split_content_and_id_leakage(self):
        self.image()
        self.image("b.ppm")
        self.write_rows([self.row(), self.row("a", "test", "images/b.ppm")])
        result = audit_manifest(self.manifest, self.root)
        self.assertFalse(result["ok"])
        self.assertEqual(self.codes(result), {"duplicate_id", "content_leakage"})

    def test_alias_path_and_same_split_duplicates(self):
        self.image()
        self.write_rows([self.row(), self.row("b", path="images/./a.ppm")])
        result = audit_manifest(self.manifest, self.root)
        self.assertEqual(self.codes(result), {"duplicate_path", "duplicate_content"})
        self.write_rows([self.row(), self.row("b", "test")])
        self.assertIn("path_leakage", self.codes(audit_manifest(self.manifest, self.root)))

    def test_invalid_schema_and_missing_file(self):
        self.write_rows([{}, {"image_id": [], "split": [], "captions": [" "]}, self.row()])
        self.assertEqual(self.codes(audit_manifest(self.manifest, self.root)), {
            "invalid_id", "invalid_split", "invalid_captions", "unsafe_path", "unreadable_image",
        })

    def test_reject_nonportable_or_escaping_paths(self):
        for path in ("../outside.jpg", "/tmp/image.jpg", "C:/image.jpg", "C:image.jpg",
                     "images\\a.jpg", "//server/share/image.jpg", "images/a.jpg:stream", "."):
            with self.subTest(path=path):
                self.write_rows([self.row(path=path)])
                self.assertIn("unsafe_path", self.codes(audit_manifest(self.manifest, self.root)))

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.ppm"
            target.write_bytes(b"outside")
            try:
                (self.root / "images" / "link.ppm").symlink_to(target)
            except OSError:
                self.skipTest("symlink creation unavailable on this platform")
            self.write_rows([self.row(path="images/link.ppm")])
            self.assertIn("unsafe_path", self.codes(audit_manifest(self.manifest, self.root)))

    def test_empty_and_bad_json(self):
        self.manifest.write_text("\n", encoding="utf-8")
        self.assertIn("empty_manifest", self.codes(audit_manifest(self.manifest, self.root)))
        self.manifest.write_text('not-json\n[]\n', encoding="utf-8")
        self.assertEqual(self.codes(audit_manifest(self.manifest, self.root)), {"invalid_json", "invalid_record"})

    def test_image_decode_detects_corrupt_and_truncated_image(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("optional Pillow not installed")
        self.image()
        self.write_rows([self.row()])
        self.assertTrue(audit_manifest(self.manifest, self.root, check_images=True)["ok"])
        for data in (b"not an image", b"P6\n1 1\n255\n\xff"):
            (self.root / "images" / "a.ppm").write_bytes(data)
            self.assertIn("invalid_image", self.codes(audit_manifest(self.manifest, self.root, True)))

    def split_inputs(self):
        tokens = self.root / "Flickr8k.token.txt"
        tokens.write_text("a.jpg#1\tsecond\na.jpg#0\tfirst\nb.jpg#0\tval\nc.jpg#0\ttest\n"
                          "unused.jpg#0\tunused\n", encoding="utf-8")
        splits = {}
        for split, filename in zip(("train", "val", "test"), ("a.jpg", "b.jpg", "c.jpg")):
            splits[split] = self.root / f"{split}.txt"
            splits[split].write_text(filename + "\n", encoding="utf-8")
        return tokens, splits

    def test_import_keeps_official_splits_and_caption_order(self):
        tokens, splits = self.split_inputs()
        records, receipt = prepare_flickr8k(tokens, splits)
        self.assertEqual([row["split"] for row in records], ["train", "val", "test"])
        self.assertEqual(records[0], {"image_id": "a", "path": "images/a.jpg", "split": "train",
                                      "captions": ["first", "second"]})
        self.assertEqual(receipt["unused_caption_image_count"], 1)
        self.assertEqual((records, receipt), prepare_flickr8k(tokens, splits))

    def test_import_rejects_leakage_missing_captions_and_empty_split(self):
        tokens, splits = self.split_inputs()
        for content in ("a.jpg\n", "missing.jpg\n", "\n"):
            splits["test"].write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                prepare_flickr8k(tokens, splits)

    def test_import_rejects_duplicate_or_malformed_captions(self):
        tokens, splits = self.split_inputs()
        for content in ("a.jpg#0\tx\na.jpg#00\ty", "a.jpg#0 no tab", "a.jpg#0\t ",
                        "../a.jpg#0\tx", "..#0\tx", "a.jpg#-1\tx"):
            tokens.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                prepare_flickr8k(tokens, splits)

    def test_cli_audit_exit_status_and_report(self):
        self.write_rows([self.row()])
        output = self.root / "audit.json"
        result = subprocess.run([sys.executable, "-m", "fgir", "validate-data", "--manifest",
                                 str(self.manifest), "--data-root", str(self.root), "--output", str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["ok"])
        self.image()
        result = subprocess.run([sys.executable, "-m", "fgir", "validate-data", "--manifest",
                                 str(self.manifest), "--data-root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_prepare_receipt_and_refuse_overwrite(self):
        tokens, splits = self.split_inputs()
        output = self.root / "prepared.jsonl"
        command = [sys.executable, "-m", "fgir", "prepare-flickr8k", "--tokens", str(tokens),
                   "--source-url", "https://example.org/test-fixture", "--terms-note", "synthetic fixture",
                   "--output", str(output)]
        for split, path in splits.items():
            command.extend([f"--{split}-list", str(path)])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(output.with_suffix(".jsonl.receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["source_url"], "https://example.org/test-fixture")
        self.assertEqual(len(receipt["manifest_sha256"]), 64)
        original = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
