"""One-command reproducible preparation using the team's pinned review decisions."""

from pathlib import Path

from .annotations import apply_source_checks, draft_annotations
from .data import file_digest, prepare_flickr8k
from .dataset import build_handoff, verify_handoff
from .download import fetch_flickr8k, SOURCE, TERMS
from .files import read_json, read_jsonl, write_json, write_jsonl


def prepare_a_data(data_root, bundle, annotations, config_root="configs", skip_download=False):
    root, bundle, annotations = Path(data_root), Path(bundle), Path(annotations)
    config = Path(config_root)
    if not skip_download:
        fetch_flickr8k(root)
    source = root / "manifest.official.jsonl"
    records, receipt = prepare_flickr8k(root / "raw/Flickr8k.token.txt", {
        split: root / "raw" / name for split, name in (
            ("train", "Flickr_8k.trainImages.txt"), ("val", "Flickr_8k.devImages.txt"), ("test", "Flickr_8k.testImages.txt"))})
    if source.exists():
        if read_jsonl(source) != records:
            raise ValueError("existing official manifest differs from pinned input; choose another data root")
    else:
        write_jsonl(source, records)
    receipt.update({"source_url": SOURCE,
                    "terms_note": "Non-commercial coursework and research; original image rights and Flickr terms apply; " + TERMS,
                    "manifest_sha256": file_digest(source)})
    receipt_path = source.with_suffix(source.suffix + ".receipt.json")
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError("existing source receipt differs; resolve provenance before proceeding")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    if not bundle.exists():
        build_handoff(source, root, bundle, config / "data/flickr8k-exclusions.json", receipt_path,
                      config / "data/flickr8k-duplicate-review.json")
    status = verify_handoff(bundle, root)
    if not annotations.exists():
        draft_annotations(bundle / "manifest.jsonl", root, annotations)
    allocation = read_json(annotations / "allocation.json")
    if allocation.get("manifest_sha256") != status["manifest_sha256"]:
        raise ValueError("existing annotation allocation belongs to another dataset")
    work = annotations / "annotations.work.jsonl"
    if not work.exists():
        apply_source_checks(bundle / "manifest.jsonl", annotations / "annotations.jsonl",
                            config / "annotations/a-source-checks.json", work)
    return {"data": status, "bundle": str(bundle), "annotations": str(annotations),
            "work_file": str(work), "next_step": "B/C can use train and val now; independent annotation review and real model ranking runs remain outstanding."}
