"""Make source-image contact sheets for disclosed visual annotation assistance."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from fgir.contracts import image_path, load_manifest
from fgir.files import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--members", default="CD")
    args = parser.parse_args()
    images = {r["image_id"]: r for r in load_manifest(args.manifest)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for member in args.members:
        rows = [r for r in read_jsonl(args.annotations) if r["assigned_to"] == member]
        for start in range(0, len(rows), 15):
            chunk = rows[start:start + 15]
            sheet = Image.new("RGB", (1600, ((len(chunk) + 4) // 5) * 290), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(chunk):
                x, y = (i % 5) * 320, (i // 5) * 290
                path = image_path(args.data_root, images[row["source_image_id"]]["path"])
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    image.thumbnail((312, 255))
                    sheet.paste(image, (x + (320 - image.width) // 2, y + 24))
                draw.text((x + 5, y + 5), row["query_id"], fill="black")
                print(f"{member} {row['query_id']} {row['source_image_id']} {row['text']}")
            sheet.save(output / f"{member}-{start // 15 + 1}.jpg", quality=95)


if __name__ == "__main__":
    main()
