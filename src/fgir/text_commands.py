"""C v2 command registrations; importing basic CLI never imports torch."""

COMMANDS = {"train-text-lora", "inspect-text-lora", "run-text-experiments", "export-text-lora", "validate-text-lora"}


def add_commands(commands):
    export = commands.add_parser("export-text-lora", help="export bound text LoRA features, receipt and run-v2")
    for name in ("checkpoint", "model-dir", "index", "protocol", "output"):
        export.add_argument(f"--{name}", required=True)
    export.add_argument("--semantic-protocol")
    export.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    export.add_argument("--batch-size", type=int, default=64)
    export.add_argument("--experiment-id")
    validate = commands.add_parser("validate-text-lora", help="run C's independent val-only checkpoint callback")
    for name in ("checkpoint", "model-dir", "bundle", "val-protocol", "output"):
        validate.add_argument(f"--{name}", required=True)
    train = commands.add_parser("train-text-lora", help="train differentiable text-only CLIP LoRA")
    for name, default in (("bundle", "data/prepared/flickr8k-v1"), ("data-root", "data")):
        train.add_argument(f"--{name}", default=default)
    for name in ("model-dir", "image-features", "config", "output"):
        train.add_argument(f"--{name}", required=True)
    for name in ("samples", "training-audit", "val-protocol", "reference-summary"):
        train.add_argument(f"--{name}")
    inspect = commands.add_parser("inspect-text-lora", help="strictly reload a text LoRA checkpoint against local CLIP")
    inspect.add_argument("--checkpoint", required=True)
    inspect.add_argument("--model-dir", required=True)
    inspect.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run = commands.add_parser("run-text-experiments", help="gated N0-N3 paired-seed experiment orchestration")
    run.add_argument("--bundle", default="data/prepared/flickr8k-v1")
    run.add_argument("--data-root", default="data")
    run.add_argument("--config-root", default="configs/training")
    for name in ("model-dir", "image-features", "samples", "val-protocol", "output"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--training-audit")
    run.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    run.add_argument("--check-only", action="store_true")


def execute(args):
    if args.command == "export-text-lora":
        from .text_export import export_text_lora
        return export_text_lora(args.checkpoint, args.model_dir, args.index, args.protocol, args.output,
                                semantic_protocol=args.semantic_protocol, device=args.device,
                                batch_size=args.batch_size, experiment_id=args.experiment_id)
    if args.command == "validate-text-lora":
        from .text_export import validate_text_lora_checkpoint
        return validate_text_lora_checkpoint(checkpoint_path=args.checkpoint, model_dir=args.model_dir,
                                             bundle=args.bundle, val_protocol=args.val_protocol, output=args.output)
    if args.command == "train-text-lora":
        from .text_training import train_text_lora
        try:
            return train_text_lora(args.bundle, args.model_dir, args.image_features, args.config, args.output,
                                   args.samples, args.data_root, args.training_audit, args.val_protocol, args.reference_summary)
        except (ValueError, RuntimeError) as exc:
            from pathlib import Path
            from .files import write_json
            output = Path(args.output)
            if (output / "config.json").exists() and not (output / "summary.json").exists():
                write_json(output / "failure.json", {"status": "failed", "exception": type(exc).__name__, "reason": str(exc)})
            raise
    if args.command == "inspect-text-lora":
        from .text_lora import load_text_lora, trainable_audit
        encoder, receipt = load_text_lora(args.checkpoint, args.model_dir, args.device)
        return {**receipt, "parameter_audit": trainable_audit(encoder.model)}
    from .run_semantic_experiments import run_text_experiments
    return run_text_experiments(args.bundle, args.model_dir, args.image_features, args.samples,
                                args.val_protocol, args.output, args.config_root, args.data_root,
                                args.training_audit, args.seeds, args.check_only)
