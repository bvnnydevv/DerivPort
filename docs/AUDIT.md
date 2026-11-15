# DerivPort technical audit and resolution

The audit started from commit `534b7f2`, before implementation. The upgrade extends the existing Streamlit application and retains its Ridge baseline, formula builder, calculus, historical replay, original gross backtest, downloads, global explorer, and hypothetical scenarios. Numerical comparisons belong to exported experiments; adding a model is not evidence that it improves performance.

## Coverage

The review covered `app.py`; all original `derivport` modules (`analysis`, `backtest`, `formula`, `calculus`, `calculus_ui`, `market_data`, `simulation`, `projections`, and package exports); the standalone algorithm; the fetch script; and every existing test. Data inspection covered CSV schemas, dates, positivity, duplicates, checksums, all tracked cache sidecars, the catalog, notes, credits, and provenance. README, evaluation artifacts, demo/design/hosting documentation, dependencies, configuration, licensing, and tracked design assets were inspected at their relevant level.

The new research modules, CLI, interface, and tests were reviewed during implementation. Source and dataset identities are recorded in each run manifest.

## Prioritized findings

| Priority | Baseline finding | Implemented treatment and remaining limits |
| --- | --- | --- |
| P1 | Date-only company notes could include an after-close release on its publication date. | Eligibility now requires `published < cutoff`. This conservative embargo avoids assuming a release time; timestamped vendor evidence would support a more precise policy. |
| P1 | Reused forecast histories checked universe membership but could silently belong to another price snapshot or contain edited forecasts. | Histories now carry price, protocol, horizon, and content fingerprints. Analysis and legacy backtests reject mismatches. This guards accidental reuse, not malicious attestation. |
| P1 | Existing linked returns selected at the same closing observation and omitted trading frictions. | They remain an explicitly gross endpoint research comparison. A separate portfolio engine delays execution, tracks holdings/cash, and charges commissions and slippage on actual buy-plus-sell notional. Adjusted-close fills remain hypothetical. |
| P1 | One fitted model and limited metrics could not establish model or signal superiority. | Common-support walk-forward comparisons add Elastic Net, Random Forest, Gradient Boosting, momentum and zero-return baselines; selection is confined to development outcomes. Prediction, ranking, and portfolio metrics remain separate. |
| P1 | Raw balanced-score coefficients were readable hypotheses, not fitted or calibrated weights. | The original formula remains unchanged. Standardized, risk-adjusted, uncertainty-penalized, and development-fitted constrained variants are selectable research comparisons. None automatically replaces it. |
| P1 limitation | The fixed stock universe consists of selected surviving listings; retrospective adjusted data are not point-in-time archives. | Every experiment discloses this selection/vintage limitation. No code change can manufacture historical constituents, delisting returns, or an unavailable vendor archive. |
| P1 limitation | International shared dates and local-currency returns can be mistaken for synchronous common-currency portfolio results. | Shared-session horizons and exclusions are recorded. Portfolio comparisons require one known currency; global local-return predictions and rankings remain available. Annualization still assumes 252 shared sessions/year. |
| P2 | Statistical support and uncertainty calibration were insufficiently characterized. | Completed residual diagnostics, empirical interval coverage, date-level block bootstrap comparisons, sample counts, and Holm adjustments expose uncertainty. They do not establish distribution-free coverage or prospective validity. |
| P2 | Extended-cache hashes were optional; a failure between CSV and JSON replacement could invalidate the previous disk pair. | Recorded checksums are mandatory. Frozen reads verify the original digest, and failed cache publication rolls back prior files. Checksums establish file identity, not economic correctness. |
| P2 | Nonfinite simulation inputs and extreme numerical results needed stronger rejection. | Simulation inputs, identifiers, seeds, path counts, and finite positive outputs are validated. Existing scenarios and fixed-seed behavior remain available. |
| P2 limitation | Vendor snapshots contain extreme historical changes requiring reconciliation. | The manifest flags absolute one-session changes above 50%; observations are never clipped, fabricated, or silently removed. |

The price audit verified all 25 tracked extended-cache checksums and found no invalid, nonpositive, or duplicate-date records. It nevertheless found large moves, including approximately +211.6% in ASML.AS, +397.6% in NOVO-B.CO, and +337.5% in RELIANCE.NS at older historical dates. These are reconciliation flags, not proof that a record is erroneous. The original `prices.csv` remains byte-for-byte unchanged, with SHA-256 `812922aec7ee2a760d2e8cb91a447e8ef36099c796477c15d1ddd272a20132a1`.

## Interpretation and verification

The original return features, train-only scaler fitting, and completed-label Ridge construction were preserved. The main improvement is a reproducible comparison protocol and stronger evidence controls. Tests address mathematical identities, future-data mutation invariance, chronological selection, cache integrity/rollback, costs, holdings, and existing application interactions; the final verification report supplies the measured test results.

The sample remains small and concentrated. Correlated assets do not create independent repetitions of market history. More independent periods, broader historically constituted universes, delisting data, point-in-time vintages, and execution evidence are required before asserting reliable market-wide superiority. A favorable metric or adjusted p-value is conditional evidence to investigate, not a profitable-strategy claim.

See [Research methods](RESEARCH_METHODS.md) for exact definitions and reproduction.
