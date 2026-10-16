"""A's data, annotation, evaluation and analysis command registrations."""

from .analysis import render_cases, summarize_latency, validate_negatives
from .annotations import apply_source_checks, draft_annotations, expand_annotation_pool, freeze_annotations, make_pool_request, merge_annotations, render_annotations, validate_annotations
from .dataset import build_handoff, screen_duplicates, verify_handoff
from .download import fetch_flickr8k
from .evaluation import compare_runs, save_evaluation, summarize_seeds
from .pipeline import prepare_a_data
from .statistics import dataset_statistics


def add_commands(commands):
    def command(name, help_text, options):
        parser = commands.add_parser(name, help=help_text)
        for flag, kwargs in options:
            parser.add_argument(flag, **kwargs)
        return parser

    required = {"required": True}
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


def execute(args):
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
    raise ValueError(f"unknown command: {args.command}")
