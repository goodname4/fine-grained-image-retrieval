"""Vector-only CLI; natural language encoding is a planned feature."""

import argparse
import json
from pathlib import Path

from .metrics import evaluate
from .retrieval import ExactIndex


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Local vector search and evaluation foundation")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="search an existing numeric vector index")
    search.add_argument("--index", required=True)
    search.add_argument("--query-vector", required=True)
    search.add_argument("--top-k", type=int, default=10)
    score = commands.add_parser("evaluate", help="evaluate ranked IDs against relevant IDs")
    score.add_argument("--rankings", required=True)
    score.add_argument("--qrels", required=True)
    args = parser.parse_args()
    try:
        if args.command == "search":
            result = ExactIndex(read_json(args.index)).search(read_json(args.query_vector), args.top_k)
        else:
            result = evaluate(read_json(args.rankings), read_json(args.qrels))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
