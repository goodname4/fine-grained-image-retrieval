"""Strict JSON and atomic directory outputs shared by data/evaluation tools."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                value = parse_json(text)
                if not isinstance(value, dict):
                    raise ValueError("expected an object")
            except ValueError as exc:
                raise ValueError(f"{Path(path).name}:{line}: {exc}") from exc
            records.append(value)
    if not records:
        raise ValueError(f"empty JSONL file: {Path(path).name}")
    return records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@contextmanager
def new_directory(path):
    """Publish a complete new bundle; on failure keep no partial output bundle."""
    output = Path(path)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fgir-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        yield staging
        if output.exists():
            raise ValueError(f"output directory already exists: {output}")
        staging.rename(output)
