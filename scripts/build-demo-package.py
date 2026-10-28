"""Build a source/demo ZIP and library wheel without weights or personal files."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist"
    package_version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    output.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(output)],
                   cwd=root, check=True)
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    archive = output / "fgir-d-demo-windows.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(filter(None, files)):
            bundle.write(root / name, "fgir-d-demo/" + name)
        for wheel in output.glob(f"fine_grained_image_retrieval-{package_version}-*.whl"):
            bundle.write(wheel, "fgir-d-demo/wheels/" + wheel.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
    print(archive)
    print("SHA256:", digest)


if __name__ == "__main__":
    main()
