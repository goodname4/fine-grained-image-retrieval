# Synthetic semantic evaluation fixture

This directory tests interfaces and metric arithmetic only. There is no model
inference, real photo annotation, or model performance result here. PPM images
are two-pixel RGB swatches. Names, reviews, labels, and model digests are synthetic
test fixtures; their text has no visual relationship to those swatches.

`source/` supplies a tiny bound source dataset. `gold.jsonl` demonstrates the full
semantic record, `frozen/` is an immutable semantic protocol, and `run.json` and
`candidate.json` contain hand-written scores and rankings.

The source bundle includes a hashed `inventory.jsonl`; the frozen context binds
the canonical image bytes. Synthetic fixtures follow the same inventory checks
as real datasets. This example has been re-frozen after the inventory validation
fix; earlier semantic protocols without inventory must be frozen into a new bundle.

From the repository root, choose new output directories:

```powershell
python -m fgir validate-semantics --samples examples/semantic-evaluation/gold.jsonl --bundle examples/semantic-evaluation/source --data-root examples/semantic-evaluation/data --purpose eval
python -m fgir evaluate-semantics --protocol examples/semantic-evaluation/frozen/protocol.json --run examples/semantic-evaluation/run.json --output artifacts/semantic-example-e0 --bootstrap-samples 100
python -m fgir compare-semantics --protocol examples/semantic-evaluation/frozen/protocol.json --runs examples/semantic-evaluation/run.json examples/semantic-evaluation/candidate.json --output artifacts/semantic-example-comparison --bootstrap-samples 100
```

The baseline has attribute credit 0.5 (tie), relation credit 0, false-addition
credit 1, length-matched addition credit 1, joint recall@1 of 0, joint recall@5
of 1, and absolute source-rank gap 1. Parsing F1 is 1 because the prediction
copies the synthetic reference. The candidate improves attribute credit but
regresses on false additions. These are arithmetic expectations, not findings.
