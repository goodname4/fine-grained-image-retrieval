"""A's data, annotation, evaluation and analysis command registrations."""

from .analysis import render_cases, summarize_latency, validate_negatives
from .annotations import apply_source_checks, draft_annotations, expand_annotation_pool, freeze_annotations, make_pool_request, merge_annotations, render_annotations, validate_annotations
from .dataset import build_handoff, screen_duplicates, verify_handoff
from .download import fetch_flickr8k
from .encoding import encode_bundle, prepare_model
from .evaluation import compare_runs, save_evaluation, summarize_seeds
from .features import build_index
from .negatives import mine_negatives
from .pipeline import prepare_a_data
from .runs import export_run, search_index
from .statistics import dataset_statistics


def add_commands(commands):
    def command(name, help_text, options):
        parser = commands.add_parser(name, help=help_text)
        for flag, kwargs in options:
            parser.add_argument(flag, **kwargs)
        return parser

    required = {"required": True}
    command("run-c-experiments", "train paired E1/E2, validate, export unscored test runs for A", [
        ("--bundle", {"default": "data/prepared/flickr8k-v1"}), ("--features", required),
        ("--negatives", required), ("--output", required), ("--config-root", {"default": "configs/training"})])
    command("train-adapter", "train C's frozen-feature E1/E2 residual adapters", [
        ("--bundle", {"default": "data/prepared/flickr8k-v1"}),
        ("--features", required), ("--config", required), ("--output", required), ("--negatives", {})])
    command("export-adapter", "adapt matching raw image/text caches and rebuild an index", [
        ("--checkpoint", required), ("--image-features", required), ("--text-features", required),
        ("--output", required), ("--device", {"default": "cpu"})])
    command("fetch-flickr8k", "download and verify pinned public Flickr8k archives", [("--data-root", {"default": "data"})])
    command("prepare-a-data", "reproduce A's pinned real dataset and annotation handoff", [
        ("--data-root", {"default": "data"}), ("--bundle", {"default": "data/prepared/flickr8k-v1"}),
        ("--annotations", {"default": "data/annotations/fine-v1"}), ("--config-root", {"default": "configs"}),
        ("--skip-download", {"action": "store_true"})])
    command("screen-duplicates", "decode images and screen near-duplicate candidates", [
        ("--manifest", required), ("--data-root", required), ("--output", required), ("--threshold", {"type": int, "default": 8})])
    command("build-handoff", "create an immutable cleaned B/C/D data bundle", [
        ("--manifest", required), ("--data-root", required), ("--output", required),
        ("--exclusions", {}), ("--source-receipt", {}), ("--duplicate-review", required)])
    command("verify-handoff", "verify bundle hashes and every local image before use", [
        ("--bundle", required), ("--data-root", required)])
    command("data-statistics", "summarize frozen image/caption statistics and repeated texts", [
        ("--bundle", required), ("--output", required)])
    command("draft-annotations", "draft balanced real-caption tasks and an offline review board", [
        ("--manifest", required), ("--data-root", required), ("--output", required),
        ("--seed", {"type": int, "default": 42}), ("--per-category", {"type": int, "default": 25})])
    command("validate-annotations", "check annotation schema and review progress", [
        ("--manifest", required), ("--annotations", required), ("--require-review", {"action": "store_true"})])
    command("render-annotations", "open existing annotation work in a fresh offline review board", [
        ("--manifest", required), ("--annotations", required), ("--data-root", required), ("--output", required)])
    command("apply-source-checks", "apply disclosed assisted initial annotations and pre-review notes", [
        ("--manifest", required), ("--annotations", required), ("--checks", required), ("--output", required)])
    command("merge-annotations", "merge member exports against a shared base, rejecting conflicts", [
        ("--manifest", required), ("--base", required), ("--updates", {"nargs": "+", "required": True}), ("--output", required)])
    command("make-pool-request", "freeze fine query text/gallery for pre-review method ranking", [
        ("--manifest", required), ("--annotations", required), ("--output", required)])
    command("expand-annotation-pool", "union method Top-K while keeping added images unjudged", [
        ("--manifest", required), ("--annotations", required), ("--request", required),
        ("--runs", {"nargs": "+", "required": True}), ("--output", required), ("--depth", {"type": int, "default": 10})])
    command("freeze-annotations", "freeze independently reviewed fine-grained evaluation", [
        ("--manifest", required), ("--annotations", required), ("--output", required),
        ("--expected-count", {"type": int, "default": 100}), ("--min-per-category", {"type": int, "default": 20})])
    command("evaluate-run", "evaluate a metadata-bound ranking run against a fixed protocol", [
        ("--protocol", required), ("--run", required), ("--output", required)])
    command("compare-runs", "compare fixed-protocol runs and export balanced case candidates", [
        ("--protocol", required), ("--runs", {"nargs": "+", "required": True}), ("--output", required),
        ("--bootstrap-samples", {"type": int, "default": 1000}), ("--seed", {"type": int, "default": 42})])
    command("summarize-seeds", "aggregate actual distinct training seeds with mean and sample std", [
        ("--protocol", required), ("--runs", {"nargs": "+", "required": True}), ("--output", required)])
    command("validate-negatives", "audit B's training-only negative mining records", [
        ("--manifest", required), ("--negatives", required)])
    command("summarize-latency", "summarize compatible measured timings with warmups separated", [("--measurements", required)])
    command("render-cases", "build an offline image-based error-analysis review sheet", [
        ("--manifest", required), ("--cases", required), ("--data-root", required), ("--output", required)])
    command("prepare-clip-model", "download pretrained CLIP weights once to an explicit local path", [
        ("--source", {"default": "openai/clip-vit-base-patch32"}), ("--output", required)])
    command("encode-features", "cache raw image and text features for every bundle split", [
        ("--bundle", {"default": "data/prepared/flickr8k-v1"}), ("--data-root", {"default": "data"}),
        ("--model-dir", required), ("--output", required),
        ("--device", {"default": "cpu"}), ("--batch-size", {"type": int, "default": 32})])
    command("build-index", "normalize cached features into an exact cosine index", [
        ("--features", required), ("--output", required)])
    command("search-index", "rank cached query features against a built index", [
        ("--index", required), ("--query-features", required), ("--output", required),
        ("--top-k", {"type": int, "default": 10})])
    command("export-run", "write a protocol-bound fgir-run-v1 ranking file for A", [
        ("--protocol", required), ("--index", required), ("--query-features", required), ("--output", required),
        ("--method", {"default": "E0"}), ("--seed", {"type": int, "default": 42}), ("--run-id", {}),
        ("--depth", {"type": int}), ("--adapter-sha256", {})])
    command("mine-negatives", "mine training-only hard negatives with false-negative guards", [
        ("--bundle", {"default": "data/prepared/flickr8k-v1"}),
        ("--text-features", required), ("--image-features", required), ("--output", required),
        ("--top-m", {"type": int, "default": 50}), ("--per-query", {"type": int, "default": 5}),
        ("--seed", {"type": int, "default": 42}), ("--review-sample", {"type": int, "default": 100})])


def execute(args):
    if args.command == "run-c-experiments":
        from .experiments_c import run_c_experiments
        return run_c_experiments(args.bundle, args.features, args.negatives, args.output, args.config_root)
    if args.command == "train-adapter":
        from .training import train_adapter
        return train_adapter(args.bundle, args.features, args.config, args.output, args.negatives)
    if args.command == "export-adapter":
        from .adaptation import export_adapter
        return export_adapter(args.checkpoint, args.image_features, args.text_features, args.output, args.device)
    if args.command == "fetch-flickr8k":
        return fetch_flickr8k(args.data_root)
    if args.command == "prepare-a-data":
        return prepare_a_data(args.data_root, args.bundle, args.annotations, args.config_root, args.skip_download)
    if args.command == "screen-duplicates":
        return screen_duplicates(args.manifest, args.data_root, args.output, args.threshold)
    if args.command == "build-handoff":
        return build_handoff(args.manifest, args.data_root, args.output, args.exclusions, args.source_receipt, args.duplicate_review)
    if args.command == "draft-annotations":
        return draft_annotations(args.manifest, args.data_root, args.output, args.seed, args.per_category)
    if args.command == "verify-handoff":
        return verify_handoff(args.bundle, args.data_root)
    if args.command == "data-statistics":
        return dataset_statistics(args.bundle, args.output)
    if args.command == "apply-source-checks":
        return apply_source_checks(args.manifest, args.annotations, args.checks, args.output)
    if args.command == "merge-annotations":
        return merge_annotations(args.manifest, args.base, args.updates, args.output)
    if args.command == "make-pool-request":
        return make_pool_request(args.manifest, args.annotations, args.output)
    if args.command == "expand-annotation-pool":
        return expand_annotation_pool(args.manifest, args.annotations, args.request, args.runs, args.output, args.depth)
    if args.command == "validate-annotations":
        return validate_annotations(args.manifest, args.annotations, args.require_review)
    if args.command == "render-annotations":
        return render_annotations(args.manifest, args.annotations, args.data_root, args.output)
    if args.command == "freeze-annotations":
        return freeze_annotations(args.manifest, args.annotations, args.output, args.expected_count, args.min_per_category)
    if args.command == "evaluate-run":
        return save_evaluation(args.protocol, args.run, args.output)
    if args.command == "compare-runs":
        return compare_runs(args.protocol, args.runs, args.output, args.bootstrap_samples, args.seed)
    if args.command == "summarize-seeds":
        return summarize_seeds(args.protocol, args.runs, args.output)
    if args.command == "validate-negatives":
        return validate_negatives(args.manifest, args.negatives)
    if args.command == "summarize-latency":
        return summarize_latency(args.measurements)
    if args.command == "render-cases":
        return render_cases(args.manifest, args.cases, args.data_root, args.output)
    if args.command == "prepare-clip-model":
        return prepare_model(args.source, args.output)
    if args.command == "encode-features":
        return encode_bundle(args.bundle, args.data_root, args.model_dir, args.output, args.device, args.batch_size)
    if args.command == "build-index":
        return build_index(args.features, args.output)
    if args.command == "search-index":
        return search_index(args.index, args.query_features, args.output, args.top_k)
    if args.command == "export-run":
        return export_run(args.protocol, args.index, args.query_features, args.output,
                          args.method, args.seed, args.run_id, args.depth, args.adapter_sha256)
    if args.command == "mine-negatives":
        return mine_negatives(args.bundle, args.text_features, args.image_features, args.output,
                              args.top_m, args.per_query, args.seed, args.review_sample)
    raise ValueError(f"unknown command: {args.command}")
