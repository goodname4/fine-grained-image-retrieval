"""Local data preparation, vector search and evaluation commands."""

import argparse
import json
from pathlib import Path

from .data import audit_manifest, file_digest, prepare_flickr8k
from .files import read_json
from .metrics import evaluate
from .retrieval import ExactIndex


def main():
    parser = argparse.ArgumentParser(description="Local retrieval data and evaluation tools")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="search an existing numeric vector index")
    search.add_argument("--index", required=True)
    search.add_argument("--query-vector", required=True)
    search.add_argument("--top-k", type=int, default=10)
    score = commands.add_parser("evaluate", help="evaluate ranked IDs against relevant IDs")
    score.add_argument("--rankings", required=True)
    score.add_argument("--qrels", required=True)
    audit = commands.add_parser("validate-data", help="audit a JSONL image/caption manifest")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--check-images", action="store_true", help="verify and decode images with Pillow")
    audit.add_argument("--output", help="save the JSON audit report (exit 1 when issues exist)")
    prepare = commands.add_parser("prepare-flickr8k", help="import token captions and fixed official splits")
    prepare.add_argument("--tokens", required=True)
    for split in ("train", "val", "test"):
        prepare.add_argument(f"--{split}-list", required=True)
    prepare.add_argument("--image-dir", default="images", help="image directory relative to data root")
    prepare.add_argument("--source-url", required=True, help="actual dataset source, recorded in receipt")
    prepare.add_argument("--terms-note", required=True, help="record the verified usage terms")
    prepare.add_argument("--output", required=True, help="new manifest path; also writes .receipt.json")
    from .commands import add_commands, execute
    add_commands(commands)
    from .semantic_commands import COMMANDS, add_commands as add_semantic_commands, execute as execute_semantic
    add_semantic_commands(commands)
    args = parser.parse_args()
    try:
        if args.command == "search":
            result = ExactIndex(read_json(args.index)).search(read_json(args.query_vector), args.top_k)
        elif args.command == "evaluate":
            result = evaluate(read_json(args.rankings), read_json(args.qrels))
        elif args.command == "validate-data":
            result = audit_manifest(args.manifest, args.data_root, args.check_images)
            if args.output:
                output = Path(args.output)
                if output.resolve() == Path(args.manifest).resolve():
                    raise ValueError("audit output must not overwrite the manifest")
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif args.command == "prepare-flickr8k":
            if not args.source_url.strip() or not args.terms_note.strip():
                raise ValueError("source URL and usage terms must not be empty")
            output = Path(args.output)
            receipt_path = output.with_suffix(output.suffix + ".receipt.json")
            if output.exists() or receipt_path.exists():
                raise ValueError("manifest or receipt already exists; choose a new output path")
            records, receipt = prepare_flickr8k(args.tokens, {
                split: getattr(args, f"{split}_list") for split in ("train", "val", "test")
            }, args.image_dir)
            receipt.update({"source_url": args.source_url, "terms_note": args.terms_note})
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            receipt["manifest_sha256"] = file_digest(output)
            with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            result = {"manifest": str(output), "receipt": str(receipt_path), **receipt}
        else:
            result = execute_semantic(args) if args.command in COMMANDS else execute(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if args.command == "validate-data" and not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
