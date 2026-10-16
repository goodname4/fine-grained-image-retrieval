"""Image inventory, near-duplicate screening, and B/C/D data handoff bundles."""

import hashlib
from pathlib import Path

from .contracts import image_path, load_manifest
from .data import audit_manifest, file_digest, SPLITS
from .files import new_directory, read_json, write_json, write_jsonl


def inventory_images(rows, data_root):
    from PIL import Image, ImageOps
    inventory = []
    for row in rows:
        path = image_path(data_root, row["path"])
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            pixels = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
            small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            values = list(small.get_flattened_data()) if hasattr(small, "get_flattened_data") else list(small.getdata())
            difference_hash = 0
            for y in range(8):
                for x in range(8):
                    difference_hash = (difference_hash << 1) | (values[y * 9 + x] > values[y * 9 + x + 1])
            inventory.append({"image_id": row["image_id"], "split": row["split"],
                              "path": row["path"], "file_sha256": file_digest(path),
                              "pixel_sha256": pixels, "width": image.width, "height": image.height,
                              "dhash64": f"{difference_hash:016x}"})
    return inventory


def duplicate_report(inventory, threshold=4):
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 10:
        raise ValueError("dHash threshold must be an integer in [0, 10]")
    pairs = []
    hashes = [int(row["dhash64"], 16) for row in inventory]
    for i, left in enumerate(inventory):
        for j in range(i):
            right = inventory[j]
            distance = (hashes[i] ^ hashes[j]).bit_count()
            exact = left["pixel_sha256"] == right["pixel_sha256"]
            if exact or distance <= threshold:
                pairs.append({"left": right["image_id"], "right": left["image_id"],
                              "left_split": right["split"], "right_split": left["split"],
                              "cross_split": left["split"] != right["split"],
                              "pixel_identical": exact, "dhash_distance": distance,
                              "review_status": "pending"})
    return {"algorithm": "exif_transposed_rgb_sha256_and_dhash64", "dhash_threshold": threshold,
            "image_count": len(inventory), "pair_count": len(pairs),
            "cross_split_pair_count": sum(pair["cross_split"] for pair in pairs),
            "limitation": "dHash proposes candidates; misses some crops/transforms and may flag unrelated images. Visual review is required.",
            "pairs": pairs}


def screen_duplicates(manifest, data_root, output, threshold=4):
    rows = load_manifest(manifest)
    inventory = inventory_images(rows, data_root)
    report = duplicate_report(inventory, threshold)
    with new_directory(output) as target:
        write_jsonl(target / "inventory.jsonl", inventory)
        write_json(target / "duplicates.json", report)
        write_json(target / "provenance.json", {"manifest_sha256": file_digest(manifest)})
    return {key: value for key, value in report.items() if key != "pairs"}


def write_protocol(directory, rows, manifest_sha256, split, scope="source_pair", reviewed=False):
    """Write a self-contained, hashed text-to-image evaluation protocol."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [row["image_id"] for row in rows if row["split"] == split]
    if not candidates:
        raise ValueError(f"empty candidate gallery: {split}")
    queries = [{"query_id": f"{row['image_id']}#{i}", "text": caption,
                "relevant_image_ids": [row["image_id"]], "categories": []}
               for row in rows if row["split"] == split for i, caption in enumerate(row["captions"])]
    write_json(directory / "candidates.json", candidates)
    write_jsonl(directory / "queries.jsonl", queries)
    write_json(directory / "qrels.json", {row["query_id"]: row["relevant_image_ids"] for row in queries})
    protocol = {"schema_version": "fgir-protocol-v1", "task": "text_to_image", "split": split,
                "manifest_sha256": manifest_sha256, "annotation_scope": scope,
                "independently_reviewed": reviewed, "data_kind": "real",
                "recall_ks": [1, 5, 10], "candidate_count": len(candidates), "query_count": len(queries),
                "files": {name: {"path": name, "sha256": file_digest(directory / name)}
                          for name in ("candidates.json", "queries.jsonl", "qrels.json")},
                "interpretation": "Success against the annotated source image; other valid images may be unlabelled."}
    write_json(directory / "protocol.json", protocol)
    return protocol


def build_handoff(manifest, data_root, output, exclusions=None, source_receipt=None, duplicate_review=None):
    """Fail closed on bad data, record explicit exclusions, and freeze file hashes."""
    original = load_manifest(manifest)
    excluded = [] if exclusions is None else read_json(exclusions)
    if not isinstance(excluded, list):
        raise ValueError("exclusions must be a list of image_id/reason records")
    seen = set()
    for row in excluded:
        if (not isinstance(row, dict) or not isinstance(row.get("image_id"), str)
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or row["image_id"] in seen):
            raise ValueError("each exclusion needs a unique image_id and a non-empty reason")
        seen.add(row["image_id"])
    if not seen <= {row["image_id"] for row in original}:
        raise ValueError("exclusions contain unknown image IDs")
    rows = [row for row in original if row["image_id"] not in seen]
    review = read_json(duplicate_review) if duplicate_review else None
    if not isinstance(review, dict) or review.get("manifest_sha256") != file_digest(manifest):
        raise ValueError("a duplicate review bound to the source manifest SHA-256 is required")
    reviewed_pairs = {}
    for pair in review.get("pairs", []):
        if pair.get("decision") not in ("distinct", "same_scene", "identical") or not pair.get("reviewer") or not pair.get("notes"):
            raise ValueError("every duplicate candidate needs a decision, reviewer and notes")
        key = tuple(sorted((pair["left"], pair["right"])))
        if key in reviewed_pairs:
            raise ValueError("duplicate review contains repeated pair")
        reviewed_pairs[key] = pair
    with new_directory(output) as target:
        write_jsonl(target / "manifest.jsonl", rows)
        audit = audit_manifest(target / "manifest.jsonl", data_root, check_images=True)
        write_json(target / "audit.json", audit)
        if not audit["ok"]:
            raise ValueError("manifest audit failed; run validate-data and document exclusions first")
        inventory = inventory_images(rows, data_root)
        pixel_ids = {}
        for row in inventory:
            if row["pixel_sha256"] in pixel_ids:
                raise ValueError(f"decoded duplicate images require explicit exclusions: {pixel_ids[row['pixel_sha256']]} / {row['image_id']}")
            pixel_ids[row["pixel_sha256"]] = row["image_id"]
        write_jsonl(target / "inventory.jsonl", inventory)
        write_json(target / "exclusions.json", excluded)
        screen = duplicate_report(inventory, review.get("dhash_threshold", 8))
        for pair in screen["pairs"]:
            decision = reviewed_pairs.get(tuple(sorted((pair["left"], pair["right"]))))
            if decision is None:
                raise ValueError(f"unreviewed near-duplicate candidate: {pair['left']} / {pair['right']}")
            if pair["cross_split"] and decision["decision"] != "distinct":
                raise ValueError("visually related cross-split images must be excluded before handoff")
        write_json(target / "duplicate-review.json", review)
        write_json(target / "duplicate-screen.json", screen)
        digest = file_digest(target / "manifest.jsonl")
        for split in SPLITS:
            subset = [row for row in rows if row["split"] == split]
            if not subset:
                raise ValueError(f"empty split after exclusions: {split}")
            directory = target / split
            directory.mkdir()
            write_jsonl(directory / "images.jsonl", subset)
            write_jsonl(directory / "pairs.jsonl", [
                {"query_id": f"{row['image_id']}#{i}", "image_id": row["image_id"],
                 "path": row["path"], "text": caption, "split": split}
                for row in subset for i, caption in enumerate(row["captions"])
            ])
            if split != "train":
                write_protocol(directory, rows, digest, split)
        if source_receipt:
            write_json(target / "source-receipt.json", read_json(source_receipt))
        files = {str(path.relative_to(target).as_posix()): file_digest(path)
                 for path in sorted(target.rglob("*")) if path.is_file()}
        summary = {"schema_version": "fgir-dataset-v1", "manifest_sha256": digest,
                   "source_manifest_sha256": file_digest(manifest), "splits": audit["splits"],
                   "excluded_image_count": len(excluded), "image_inventory_sha256": file_digest(target / "inventory.jsonl"),
                   "image_path_base": "data_root_supplied_by_caller", "files": files,
                   "near_duplicate_policy": "Decoded exact duplicates excluded; all dHash candidates at the recorded threshold visually reviewed; related cross-split candidates excluded. Heuristic screening is not a proof of no leakage.",
                   "training_rule": "Only train/ images and pairs may optimize parameters or mine negatives; val selects settings; test is held out."}
        write_json(target / "dataset.json", summary)
    return summary


def verify_handoff(bundle, data_root):
    bundle = Path(bundle)
    summary = read_json(bundle / "dataset.json")
    if summary.get("schema_version") != "fgir-dataset-v1":
        raise ValueError("unsupported dataset bundle")
    for name, digest in summary["files"].items():
        if file_digest(image_path(bundle, name)) != digest:
            raise ValueError(f"bundle file hash mismatch: {name}")
    if file_digest(bundle / "manifest.jsonl") != summary["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    from .files import read_jsonl
    images = load_manifest(bundle / "manifest.jsonl")
    inventory = read_jsonl(bundle / "inventory.jsonl")
    if [row["image_id"] for row in images] != [row["image_id"] for row in inventory]:
        raise ValueError("manifest and inventory image order disagree")
    for row, item in zip(images, inventory):
        if row["path"] != item["path"] or row["split"] != item["split"]:
            raise ValueError("inventory path/split disagrees with manifest")
        if file_digest(image_path(data_root, row["path"])) != item["file_sha256"]:
            raise ValueError(f"image file hash mismatch: {row['image_id']}")
    return {"ok": True, "image_count": len(images), "manifest_sha256": summary["manifest_sha256"], "splits": summary["splits"]}
