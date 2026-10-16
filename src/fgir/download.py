"""Reproducible Flickr8k mirror acquisition; no inference service is used."""

import json
import shutil
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from .data import file_digest


SOURCE = "https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k"
TERMS = "https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html"
ARCHIVES = {"Flickr8k_text.zip": 2340801, "Flickr8k_Dataset.zip": 1115419746}
ARCHIVE_SHA256 = {
    "Flickr8k_text.zip": "4992ddc8110e9aa49da5bf698522b0c8f11c448814a488584ee6bf040e5137e7",
    "Flickr8k_Dataset.zip": "61e4b111d32b24a55b69dafd91f4c3aec07391b7b9217face15dd35d517fe6de",
}


def download_archive(url, output, expected_size):
    """Resume partial transfers only when the server honors the requested range."""
    output = Path(output)
    if output.exists():
        if output.stat().st_size != expected_size or not zipfile.is_zipfile(output):
            raise ValueError(f"existing archive is invalid: {output.name}")
        return
    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "fgir-coursework-data/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                resume = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                if not resume:
                    offset = 0
                with partial.open("ab" if resume else "wb") as stream:
                    last = time.monotonic()
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last > 5:
                            print(f"{output.name}: {offset / 1048576:.1f}/{expected_size / 1048576:.1f} MiB", file=sys.stderr, flush=True)
                            last = time.monotonic()
            if partial.stat().st_size != expected_size or not zipfile.is_zipfile(partial):
                raise ValueError(f"size or ZIP header mismatch: {output.name}")
            partial.replace(output)
            return
        except OSError:
            if attempt == 2:
                raise


def extract_archive(archive, output, images=False):
    """Flatten only the intended corpus files and check ZIP CRCs on extraction."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or "__MACOSX" in path.parts or path.name.startswith("."):
                continue
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("unsafe archive member")
            if images and path.suffix.lower() != ".jpg":
                continue
            if not images and path.suffix.lower() != ".txt":
                continue
            if path.name in seen:
                raise ValueError(f"duplicate archive filename: {path.name}")
            seen.add(path.name)
            destination = output / path.name
            if destination.exists():
                crc = 0
                with destination.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if destination.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError(f"existing extracted file differs: {path.name}")
            else:
                with source.open(info) as reader, destination.open("xb") as writer:
                    try:
                        shutil.copyfileobj(reader, writer)
                    except Exception:
                        writer.close()
                        destination.unlink()
                        raise
            count += 1
    return count


def fetch_flickr8k(data_root):
    """Download the verified public mirror for this educational project."""
    root = Path(data_root)
    archives = root / "downloads"
    archives.mkdir(parents=True, exist_ok=True)
    receipt = {
        "dataset": "Flickr8k", "mirror_url": SOURCE, "terms_url": TERMS,
        "terms_note": "Non-commercial research/education; original image rights and Flickr terms apply. Images are not redistributed by this repository.",
        "archives": {},
    }
    for name, size in ARCHIVES.items():
        url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/" + name
        archive = archives / name
        download_archive(url, archive, size)
        if file_digest(archive) != ARCHIVE_SHA256[name]:
            raise ValueError(f"archive differs from the pinned mirror version: {name}")
        count = extract_archive(archive, root / ("images" if "Dataset" in name else "raw"), images="Dataset" in name)
        receipt["archives"][name] = {"url": url, "bytes": size, "sha256": file_digest(archive), "extracted_file_count": count}
        print(f"{name}: extracted {count} files", file=sys.stderr, flush=True)
    (root / "download-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    print(json.dumps(fetch_flickr8k(args.data_root), ensure_ascii=False, indent=2))
