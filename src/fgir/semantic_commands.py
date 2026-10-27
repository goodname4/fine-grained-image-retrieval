"""Standard-library-only commands for A's research-v2 work."""

from .semantic_data import apply_semantic_source_checks, freeze_semantics, prepare_semantic_pilot, validate_semantics
from .semantic_evaluation import evaluate_semantics
from .semantic_review import render_semantics


COMMANDS = {"prepare-semantic-pilot", "apply-semantic-source-checks", "validate-semantics", "render-semantics", "freeze-semantics", "evaluate-semantics", "compare-semantics"}


def add_commands(commands):
    for name in sorted(COMMANDS):
        parser = commands.add_parser(name, help={
            "prepare-semantic-pilot": "select source-only val tasks without inventing language labels",
            "apply-semantic-source-checks": "apply source-bound assistant prechecks, never independent gold",
            "validate-semantics": "validate semantic spans, sources and review gates",
            "render-semantics": "create a portable offline image/semantic review board",
            "freeze-semantics": "freeze independently reviewed semantic diagnostics",
            "evaluate-semantics": "score semantic pairs, parsing and paraphrase retrieval separately",
            "compare-semantics": "compare fixed semantic runs with paired source-image bootstrap",
        }[name])
        if name in ("evaluate-semantics", "compare-semantics"):
            parser.add_argument("--protocol", required=True)
            parser.add_argument("--runs" if name == "compare-semantics" else "--run", required=True,
                                **({"nargs": "+"} if name == "compare-semantics" else {}))
            parser.add_argument("--bootstrap-samples", type=int, default=1000)
        else:
            parser.add_argument("--bundle", default="data/prepared/flickr8k-v1")
            if name != "prepare-semantic-pilot":
                parser.add_argument("--samples", required=True)
            parser.add_argument("--data-root", **({} if name == "validate-semantics" else {"default": "data"}))
        if name != "validate-semantics":
            parser.add_argument("--output", required=True)
        if name in ("prepare-semantic-pilot", "evaluate-semantics", "compare-semantics"):
            parser.add_argument("--seed", type=int, default=42)
        if name == "prepare-semantic-pilot":
            parser.add_argument("--count", type=int, default=30)
            parser.add_argument("--code-revision", default="working-tree")
        elif name == "validate-semantics":
            parser.add_argument("--purpose", choices=("draft", "train", "eval"), default="draft")
        elif name == "freeze-semantics":
            parser.add_argument("--data-kind", choices=("real", "synthetic"), required=True)
            parser.add_argument("--test-access-notes", required=True)
        elif name == "apply-semantic-source-checks":
            parser.add_argument("--checks", required=True)


def execute(args):
    if args.command == "prepare-semantic-pilot":
        return prepare_semantic_pilot(args.bundle, args.data_root, args.output, args.count, args.seed, args.code_revision)
    if args.command == "validate-semantics":
        return validate_semantics(args.samples, args.bundle, args.purpose, args.data_root)
    if args.command == "apply-semantic-source-checks":
        return apply_semantic_source_checks(args.samples, args.bundle, args.data_root, args.checks, args.output)
    if args.command == "render-semantics":
        return render_semantics(args.samples, args.bundle, args.data_root, args.output)
    if args.command == "freeze-semantics":
        return freeze_semantics(args.samples, args.bundle, args.data_root, args.output, args.data_kind, args.test_access_notes)
    if args.command == "compare-semantics":
        from .semantic_evaluation import compare_semantics
        return compare_semantics(args.protocol, args.runs, args.output, args.bootstrap_samples, args.seed)
    return evaluate_semantics(args.protocol, args.run, args.output, args.bootstrap_samples, args.seed)
