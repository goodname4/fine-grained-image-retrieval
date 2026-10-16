"""Local manifests, official Flickr8k splits, and reproducible data audits."""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from .files import parse_json


SPLITS = ("train", "val", "test")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    """Use portable relative paths, including when checking on another OS."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or "\\" in value
            or ".." in path.parts or not path.parts or ":" in value):
        raise ValueError("path must be relative, without backslashes, drive or '..'")
    return path


def audit_manifest(manifest, data_root, check_images=False):
    """Report all row errors; hash bytes to detect exact duplicates and leakage.

    Decoding requires Pillow and is explicitly recorded. Near duplicates require
    separate review; a passing audit does not establish semantic uniqueness.
    """
    if check_images:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("image decoding requires: pip install -e '.[data]'") from exc
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("data root must be a directory")
    issues = []
    stats = {split: {"image_count": 0, "caption_count": 0} for split in SPLITS}
    ids, paths, digests = {}, {}, {}
    row_count = 0

    def issue(line, code, message):
        issues.append({"line": line, "code": code, "message": message})

    with Path(manifest).open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            row_count += 1
            try:
                row = parse_json(text)
            except ValueError:
                issue(line, "invalid_json", "expected one JSON object per line")
                continue
            if not isinstance(row, dict):
                issue(line, "invalid_record", "record must be an object")
                continue
            image_id, split = row.get("image_id"), row.get("split")
            captions = row.get("captions")
            if not isinstance(image_id, str) or not image_id.strip():
                issue(line, "invalid_id", "image_id must be a non-empty string")
            elif image_id in ids:
                issue(line, "duplicate_id", f"image_id duplicates line {ids[image_id]}")
            else:
                ids[image_id] = line
            valid_split = isinstance(split, str) and split in SPLITS
            if not valid_split:
                issue(line, "invalid_split", "split must be train, val or test")
            valid_captions = (isinstance(captions, list) and bool(captions)
                              and all(isinstance(c, str) and c.strip() for c in captions))
            if not valid_captions:
                issue(line, "invalid_captions", "captions must be a non-empty list of non-empty strings")
            if valid_split:
                stats[split]["image_count"] += 1
                if valid_captions:
                    stats[split]["caption_count"] += len(captions)
            try:
                rel = relative_path(row.get("path"))
                path = (root / str(rel)).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("resolved path escapes data root (possibly through a symlink)")
            except (ValueError, OSError) as exc:
                issue(line, "unsafe_path", str(exc))
                continue
            if path in paths:
                previous_line, previous_split = paths[path]
                code = "path_leakage" if split != previous_split else "duplicate_path"
                issue(line, code, f"same resolved image path as line {previous_line}")
            else:
                paths[path] = (line, split)
            try:
                digest = file_digest(path)
            except OSError:
                issue(line, "unreadable_image", "image is missing, not a file, or unreadable")
                continue
            if digest in digests:
                previous_line, previous_split = digests[digest]
                code = "content_leakage" if split != previous_split else "duplicate_content"
                issue(line, code, f"identical file bytes to line {previous_line}")
            else:
                digests[digest] = (line, split)
            if check_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                    issue(line, "invalid_image", "image verification or decoding failed")
    if not row_count:
        issue(0, "empty_manifest", "manifest contains no records")
    return {
        "ok": not issues,
        "manifest_sha256": file_digest(manifest),
        "record_count": row_count,
        "splits": stats,
        "image_decoding_checked": check_images,
        "duplicate_check": "sha256_file_bytes",
        "near_duplicates_checked": False,
        "issue_count": len(issues),
        "issues": issues,
    }


def prepare_flickr8k(tokens, split_files, image_dir="images"):
    """Import Flickr8k.token.txt and explicit official image lists without resplitting.

    Returns records and a receipt; acquiring data and confirming its terms remain
    the caller's responsibility. Extra caption images outside the lists are counted.
    """
    directory = relative_path(image_dir)
    if set(split_files) != set(SPLITS):
        raise ValueError("provide exactly train, val and test split files")
    captions = {}
    for line, text in enumerate(Path(tokens).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        key, separator, caption = text.partition("\t")
        filename, marker, number = key.rpartition("#")
        if (not separator or not marker or not number.isascii() or not number.isdigit()
                or not caption.strip() or not filename or filename in (".", "..")
                or PurePosixPath(filename).name != filename
                or "\\" in filename or ":" in filename):
            raise ValueError(f"invalid Flickr8k token at line {line}; expected filename#N<TAB>caption")
        items = captions.setdefault(filename, {})
        index = int(number)
        if index in items:
            raise ValueError(f"duplicate caption index at token line {line}")
        items[index] = caption.strip()
    records, assigned, ids = [], set(), set()
    for split in SPLITS:
        filenames = Path(split_files[split]).read_text(encoding="utf-8-sig").splitlines()
        if not any(name.strip() for name in filenames):
            raise ValueError(f"{split} split list must not be empty")
        for filename in filenames:
            filename = filename.strip()
            if not filename:
                continue
            if filename in assigned:
                raise ValueError(f"image repeated within/across split lists: {filename}")
            if filename not in captions:
                raise ValueError(f"image has no captions: {filename}")
            image_id = PurePosixPath(filename).stem
            if image_id in ids:
                raise ValueError(f"image_id collision: {image_id}")
            assigned.add(filename)
            ids.add(image_id)
            records.append({
                "image_id": image_id, "path": str(directory / filename), "split": split,
                "captions": [caption for _, caption in sorted(captions[filename].items())],
            })
    receipt = {
        "format": "flickr8k_token_and_explicit_splits_v1",
        "tokens_sha256": file_digest(tokens),
        "split_files_sha256": {split: file_digest(split_files[split]) for split in SPLITS},
        "unused_caption_image_count": len(set(captions) - assigned),
        "record_count": len(records),
    }
    return records, receipt
