# Retained evidence for run 0ad6556ae9189d24

This directory is a **selected evidence subset**, copied byte-for-byte from `artifacts/research_runs/frozen-upgrade`. [The results report](../../RESEARCH_RESULTS.md) describes the experiment, measured findings and limitations.

Retained original exports:

- `manifest.json`: full-run configuration, versions, source/data hashes, selection boundaries, assumptions, limitations and original export checksums.
- `forecasts.csv`: common final-test forecasts, observed outcomes and calibration fields.
- `model_summary.csv`, `ranking_summary.csv`, `portfolio_summary.csv`: separate predictive, ranking and investment metrics.
- `selection.csv`: model selection evidence.
- `model_tests.csv`, `signal_tests.csv`, `portfolio_tests.csv`: the three declared paired comparison families.
- `per_date.csv`, `period_returns.csv`: date-level predictive metrics and actual portfolio comparison intervals.
- `regime_models.csv`, `regime_portfolios.csv`: descriptive regime tables.

`baseline_before.json` records the separately measured original-commit baseline and is not an original export. This README is also supplementary. Neither is listed in the original manifest.

**The manifest's `output_sha256` describes the entire original export, including files not retained here.** This directory does not contain development forecasts, all scores/folds, aligned/source input prices, regime labels, or daily portfolio/trade/weight ledgers. Regenerate the full evidence folder with:

```sh
.venv/bin/python scripts/run_research.py --dataset frozen \
  --validation-start 2020-01-01 --test-start 2023-01-01 \
  --end 2024-12-31 --output artifacts/research_runs/frozen-upgrade
```

Use a fresh output path if that directory contains a different run. Defaults and exact measured versions are recorded in the manifest. The source frozen price CSV is already tracked under `data/`, with original bytes unchanged. Full exports also include the precise observed input prices and original metadata for portable replay; see [methods](../../RESEARCH_METHODS.md).

For each retained original CSV, its SHA-256 must equal the corresponding `output_sha256` entry. The manifest was captured before the upgrade commit and records a dirty working tree based on the prior HEAD; its `code.source_sha256` records the exact measured implementation. This provenance has been preserved rather than rewritten to imply a later measurement.
