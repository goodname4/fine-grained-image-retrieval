"""Build a demo ZIP and library wheel, optionally including local v2 demo assets."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-v2-assets", action="store_true",
                        help="include local CLIP, val images and checked development exports")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "dist"
    package_version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    output.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(output)],
                   cwd=root, check=True)
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    assets = set()
    if args.include_v2_assets:
        from fgir.service import RetrievalService
        service = RetrievalService(root, root / "configs/demo-v2.json", device="cpu")
        ready, unavailable = service.available_models()
        if set(ready) != {"E0", "N0-development"} or unavailable:
            raise ValueError(f"v2 resources are not ready: {unavailable}")
        for directory in ("models/clip-vit-base-patch32", "data/prepared/flickr8k-v1",
                          "artifacts/d-v2/e0-val", "artifacts/d-v2/n0-development"):
            assets.update(p for p in (root / directory).rglob("*") if p.is_file())
        assets.add(root / "artifacts/d-v2/local-assets.json")
        rows = (root / "data/prepared/flickr8k-v1/val/images.jsonl").read_text(encoding="utf-8").splitlines()
        assets.update(root / "data" / json.loads(row)["path"] for row in rows if row.strip())
        for path in assets:
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError(f"invalid or missing package asset: {path}")
    archive = output / ("fgir-d-v2-ready-windows.zip" if args.include_v2_assets else "fgir-d-demo-windows.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(filter(None, files)):
            bundle.write(root / name, "fgir-d-demo/" + name)
        for wheel in output.glob(f"fine_grained_image_retrieval-{package_version}-*.whl"):
            bundle.write(wheel, "fgir-d-demo/wheels/" + wheel.name)
        for path in sorted(assets):
            bundle.write(path, "fgir-d-demo/" + path.relative_to(root).as_posix())
        if args.include_v2_assets:
            bundle.writestr("fgir-d-demo/V2-START-HERE.txt",
                            "Python 3.11+ required. Run install.cmd once (internet required for dependencies), "
                            "then start-v2.cmd. Includes local CLIP, 1000 real val images and a two-step "
                            "N0-development checkpoint. Not a formal N0 result. "
                            "No training images or formal N1-N3 weights are included.\n")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    print(archive)
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
