# Quantitative research upgrade

DerivPort's existing Streamlit architecture, finance branding, Ridge baseline, formula editor, standalone balanced algorithm, calculus, historical replay, gross endpoint backtest, global explorer, downloads, and hypothetical future scenarios are preserved. New research modules sit alongside those features and consume the same validated price data. The original frozen price CSV is unchanged.

## What changed and why

- **Chronological model comparison:** Ridge, Elastic Net, Random Forest, Gradient Boosting, momentum, and zero-return forecasts share features, targets, expanding training periods, forecast dates, and final-test support. Scalers fit on training rows. Completed-label validation selects small predefined parameter grids; final-test hyperparameters freeze before testing.
- **Balanced-signal research:** the original coefficients remain a baseline. Cross-sectional standardization compares relative factor spreads; the risk-adjusted variant divides by volatility with a fixed 1 pp floor; the uncertainty-only penalty isolates forecast-error effects. Nonnegative constrained weights sum to one, cap each factor at 0.6, and use separately purged development training/validation before freezing for the test. Scores retain their distinct units.
- **Portfolio accounting:** a separate delayed-execution engine includes commissions, slippage, cash, diversification, target exposure/position caps, weight drift, and drawdowns. Trade sizes solve the post-cost self-financing equation; fees apply to full buy-plus-sell notional. Portfolio evaluation ends at the common last predictive outcome. The earlier gross research comparison remains available.
- **Evidence controls:** histories are bound to price/protocol/content fingerprints; date-only company notes use a conservative publication-day embargo; cache checksums are mandatory and failed publication rolls back prior files. Price-quality flags are disclosed without silently altering observations.
- **Research interface and exports:** model accuracy, ranking quality, regime breakdowns, portfolio risk, and approximate statistical comparisons are separate. Manifests retain data/source/code identities, configuration, software versions, seeds, and output hashes; portable exports include aligned and source observations.

Exact mathematical definitions and caveats are in [Research methods](RESEARCH_METHODS.md); the prioritized original findings and remedies are in [Audit](AUDIT.md).

## Measured result and verification

Run **`0ad6556ae9189d24`** uses the frozen five-stock sample with a 2023 final-test boundary: **23 forecast dates / 115 stock forecasts**, with observed outcomes through **December 4, 2024**. The before/after Ridge RMSE on identical dates is **11.564965 pp in both versions**, and all matched predictions agree exactly. None of the added regressors reduced RMSE on this test.

The original balanced signal's mean Spearman IC was **−0.1304**; constrained weights produced **+0.0130**. The paired IC difference was not statistically significant: its approximate 95% interval crosses zero and its Holm-adjusted p-value is **1.0**. Equal weighting also had a higher hypothetical net total return than constrained weighting in this run (**199.06% versus 178.88%**). These are conditional measured comparisons on selected survivors, not evidence of dependable prospective returns. See [Measured research results](RESEARCH_RESULTS.md) and its evidence files for full tables and assumptions.

The pre-change test suite passed **154 tests**. The complete upgraded suite passed **254 tests**; a final focused research-interface check passed **2 tests**. Coverage includes formulas, feature/return definitions, cutoff enforcement, future-data mutation invariance, parameter/weight selection, deterministic models, accounting/costs, rankings, provenance, exports, and retained application flows.

## Changed and added files

Paths below describe this upgrade relative to the audited baseline.

| Purpose | Files |
| --- | --- |
| Existing application and research APIs | `app.py`, `derivport/analysis.py`, `derivport/backtest.py` |
| New research engine | `derivport/research_models.py`, `derivport/research_signals.py`, `derivport/research_metrics.py`, `derivport/portfolio.py`, `derivport/research.py` |
| New research interface and CLI | `derivport/research_ui.py`, `scripts/run_research.py` |
| Data/simulation hardening and provenance | `derivport/market_data.py`, `derivport/simulation.py`, `data/CREDITS.md`, `data/provenance.json` |
| Existing tests extended | `tests/test_data_integrity.py`, `tests/test_market_data.py`, `tests/test_simulation.py` |
| New tests | `tests/test_history_provenance.py`, `tests/test_portfolio.py`, `tests/test_research.py`, `tests/test_research_metrics.py`, `tests/test_research_models.py`, `tests/test_research_signals.py`, `tests/test_research_ui.py` |
| Repository guidance/configuration | `.gitignore`, `README.md`, `docs/HOSTING.md`, `docs/AUDIT.md`, `docs/RESEARCH_METHODS.md`, `docs/RESEARCH_RESULTS.md`, `docs/UPGRADE_SUMMARY.md` |

The committed evidence subset is under `docs/research_results/frozen_2024/`:

```text
README.md
baseline_before.json
manifest.json
forecasts.csv
model_summary.csv
ranking_summary.csv
portfolio_summary.csv
selection.csv
model_tests.csv
signal_tests.csv
portfolio_tests.csv
per_date.csv
period_returns.csv
regime_models.csv
regime_portfolios.csv
```

Its README distinguishes this selected evidence from the complete manifest's output inventory. Full generated runs, including portable price inputs and daily portfolio/trade/weight files, remain ignored under `artifacts/research_runs/` and can be regenerated.

## Reproduce

From the repository root, using the pinned dependencies:

```sh
.venv/bin/python scripts/run_research.py --dataset frozen \
  --symbols AAPL,AMZN,GOOGL,MSFT,NVDA \
  --validation-start 2020-01-01 --test-start 2023-01-01 \
  --end 2024-12-31 --seed 2026 --rebalance-sessions 21 \
  --top-n 3 --max-weight 0.4 --gross-exposure 1.0 --execution-lag 1 \
  --commission-bps 10.0 --slippage-bps 5.0 --annual-rf 0.0 --k 1.0 \
  --bootstrap-samples 1000 --block-length 3 \
  --output artifacts/research_runs/frozen-upgrade
```

Run verification with `.venv/bin/python -m pytest -q`. The methods document also provides an API replay from exported observations without a provider request.

## Remaining limitations

The data are retrospectively adjusted snapshots of selected surviving listings, with no historical constituent archive or delisting-return coverage. Only two price features and a small number of independent test periods are available. Cross-market research uses shared sessions and local returns; portfolio comparisons require one known currency. Execution, costs, cash returns, and 252-session annualization are explicit simplifying assumptions rather than measured trading conditions. Approximate date-block intervals and adjusted tests do not eliminate selection, nonstationarity, or repeated-experiment bias.

More independent periods, broader historically constituted universes, point-in-time data vintages, and realistic execution evidence are required before making general claims of statistical or investment superiority. No experimental model or signal is promoted automatically on this run's results.
