"""Portable offline semantic review board; never impersonates a reviewer."""

import json
from pathlib import Path
import shutil

from .contracts import image_path
from .data import file_digest
from .files import new_directory, read_jsonl, write_json
from .semantic_data import source_context, validate_semantics


def render_semantics(samples, bundle, data_root, output):
    audit = validate_semantics(samples, bundle, data_root=data_root)
    context = source_context(bundle)
    rows = read_jsonl(samples)
    template = Path(__file__).with_name("semantic-board.html").read_text(encoding="utf-8")
    with new_directory(output) as target:
        (target / "images").mkdir()
        display = []
        for i, row in enumerate(rows):
            source = image_path(data_root, context["images"][row["source_image_id"]]["path"])
            name = f"images/{i}{source.suffix.lower()}"
            shutil.copyfile(source, target / name)
            display.append({"record": row, "image": name, "image_sha256": file_digest(source)})
        payload = json.dumps(display, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        (target / "board.html").write_text(template.replace("__SEMANTIC_DATA__", payload), encoding="utf-8", newline="\n")
        write_json(target / "receipt.json", {"schema_version": "fgir-semantic-board-v1", "samples_sha256": file_digest(samples),
                   "audit": audit, "interpretation": "Source-only tasks may contain no parses or variants. Editing or exporting does not confer independent review."})
    return {"output": str(output), "board": str(Path(output) / "board.html"), **audit}
