"""Verify the real, project-local development demo without reporting formal scores."""

import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    import torch
    torch.set_num_threads(4)
    from fgir.benchmark import benchmark
    from fgir.data import file_digest
    from fgir.files import read_json, write_json
    from fgir.service import RetrievalService, export_results
    from fgir.text_artifacts import validate_run_v2

    export = Path("artifacts/d-v2/n0-development")
    run = read_json(export / "run.json")
    assert run["data_kind"] == "real" and run["mode"] == "development_smoke"
    assert run["semantic_dataset_sha256"] is None
    validate_run_v2(export / "run.json", "data/prepared/flickr8k-v1/val/protocol.json", allow_development=True)
    service = RetrievalService(root, root / "configs/demo-v2.json", device="auto")
    performance = benchmark(service, read_json("examples/demo-queries.json"), ["E0", "N0-development"], 5, 10)
    write_json(Path("artifacts/d-v2/performance-real-v1.json"), performance)
    ready, unavailable = service.available_models()
    assert set(ready) == {"E0", "N0-development"} and not unavailable, unavailable
    sequence = ["E0", "N0-development", "E0", "N0-development"]
    results = [service.search("two dogs playing on grass", model, 5) for model in sequence]
    signatures = [[(h["image_id"], h["score"]) for h in result["hits"]] for result in results]
    assert signatures[0] == signatures[2] and signatures[1] == signatures[3]
    assert all(hit["image_available"] for result in results for hit in result["hits"])
    portable = export_results({r["model"]: r for r in results})
    assert '"path"' not in portable
    Path("artifacts/d-v2/query-real-development.json").write_text(portable, encoding="utf-8")

    from streamlit.testing.v1 import AppTest
    sys.argv = ["app.py", "--config", "configs/demo-v2.json", "--device", "auto"]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=180).run()
    assert not app.exception, app.exception
    assert app.multiselect[0].value == ["E0", "N0-development"]
    next(button for button in app.button if button.label == "检索").click().run()
    assert not app.exception and not app.error, (app.exception, app.error)
    image_count = len(app.get("image"))
    assert image_count == 10, image_count
    evidence = {"data_kind": "real", "formal_result": False,
                "checkpoint_origin": "D rebuild via C's published two-step training configuration",
                "original_c_package_verified": False,
                "run_sha256": file_digest(export / "run.json"), "query_count": len(run["rankings"]),
                "ranking_depth": run["ranking_depth"], "switch_sequence": sequence,
                "switch_results_exactly_restored": True, "all_topk_images_exist": True,
                "offline_mode": True, "portable_export_has_no_image_paths": True,
                "streamlit_query_passed": True, "streamlit_image_count": image_count,
                "gpu": performance["gpu"], "packages": performance["packages"],
                "models": {key: {"device": value["provenance"]["device"],
                          "sample_count": value["sample_count"], "warm_ms": value["warm_ms"]}
                           for key, value in performance["models"].items()}}
    write_json(Path("artifacts/d-v2/real-development-verification.json"), evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
