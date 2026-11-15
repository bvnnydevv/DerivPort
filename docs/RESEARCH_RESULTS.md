# Measured frozen-data research results

The upgrade preserves the original Ridge predictions exactly. None of the added regressors improves final-test RMSE in this experiment. The constrained signal improves sample rank correlation over the original balanced rule, but its confidence interval includes no improvement and its adjusted p-value is 1.0. More independent periods and broader point-in-time data are required to establish reliable superiority.

These are retrospective, conditional results for five selected technology-stock survivors, not an unbiased market study or evidence of prospective profitability. The 2023–2024 rally produces unusually large hypothetical portfolio returns; equal weight outperforms the constrained signal in this sample.

## Experiment and evidence

Run **`0ad6556ae9189d24`** uses the verified frozen AAPL, AMZN, GOOGL, MSFT and NVDA cache, 1,761 common sessions from 2018-01-02 through 2024-12-31. Development forecasts start in 2020; the final-test boundary is 2023-01-01. There are **23 common final-test origins and 115 forecasts per model**, from 2023-01-03 through 2024-11-04, with the last 21-session outcome on **2024-12-04**. No model receives a different test calendar or target.

Hyperparameters are frozen before the test; the last parameter-selection outcome is 2022-11-01. Expanding refits during the test use only then-completed labels. Constrained signal weights use development out-of-sample predictions with outcomes strictly before 2023-01-01, and are frozen for the test. The latest weight-fitting outcome is 2022-12-01. This code boundary prevents selection leakage; it cannot turn a previously viewed historical dataset into an untouched prospective experiment.

The [retained evidence](research_results/frozen_2024/README.md) includes exact CSV results, selection records, paired tests, forecasts and the original full-run manifest. The manifest records measured source hashes, data hashes, versions, seed, assumptions and limitations. It was generated before the upgrade commit, so its `repository_commit` is the prior commit and `working_tree_dirty` is true; the source hashes identify the measured implementation. Documentation and evidence copies do not alter those measured source hashes.

## Before and after

The prior commit `534b7f2d3ad45534bd3fa41667456532bf73104c` was independently executed from an isolated archive on the same 23 origins. Its 154 existing tests passed. The upgraded implementation passes **254 tests**, including the old suite, with **2 UI tests** rerun successfully after final explanatory text changes.

| Matched measurement | Prior commit | Upgrade |
|---|---:|---:|
| Ridge RMSE (percentage points) | 11.5649650167 | 11.5649650167 |
| Original balanced mean Spearman IC | -0.1304347826 | -0.1304347826 |
| Matched Ridge predictions | 115 | 115 |
| Maximum absolute prediction difference (pp) | — | 0.0 |

The former replay reports gross forward-return observations. It does not have the new delayed-execution, self-financing holdings and cost ledger, so there is no comparable old net portfolio result. New portfolio results below apply all strategies to the same new engine; they are not claimed as a before/after investment improvement.

## Predictive accuracy

RMSE and MAE are in percentage points. Pearson is pooled across asset/date observations; IC is the mean within-date cross-sectional Spearman correlation. Undefined correlations are shown as —, never replaced with zero. Gradient Boosting has nonconstant rankings on only 11 of 23 dates; Elastic Net has none, so its cross-sectional IC is undefined.

| Forecast | RMSE | MAE | R² | Pearson | Direction accuracy | Mean Spearman IC |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 11.5650 | 7.7779 | -0.0600 | 0.1213 | 72.17% | 0.1217 |
| Elastic Net | 11.6402 | 7.8249 | -0.0738 | -0.2897 | 72.17% | — |
| Random Forest | 11.6296 | 7.8440 | -0.0718 | 0.0150 | 70.43% | 0.0048 |
| Gradient Boosting | 11.6689 | 7.8192 | -0.0791 | -0.1652 | 72.17% | -0.1156 |
| Momentum | 17.8810 | 13.4699 | -1.5339 | -0.2061 | 46.96% | 0.0826 |
| Zero return | 12.3808 | 8.6206 | -0.2148 | — | 0.00% | — |

The observed positive-return fraction is also **72.17%**. Always predicting a positive direction achieves that fraction; directional accuracy alone does not establish skill. Negative R² means worse squared error than the ex-post test-sample mean reference, which is not a tradable forecast. A zero prediction is counted as zero direction, not as a positive guess.

Ridge's approximate date-block 95% RMSE interval is **[8.3152, 14.2472] pp**. Its empirical nominal-80% past-residual prediction bands cover **80.87%** of the 115 outcomes. This measured coverage is not a guarantee of future calibration or a confidence interval for expected return.

Model selection chooses Elastic Net `alpha=1, l1_ratio=0.75`, Random Forest `max_depth=3`, and Gradient Boosting `max_depth=1` within small predefined grids. Ridge retains `alpha=25`. Full estimator settings and folds are recorded by the manifest and reproducible code.

## Signal rankings and mathematical interpretation

The original balanced formula and coefficients remain unchanged. Standardization tests whether cross-sectional factor dispersion drives rankings; the return/volatility ratio tests relative risk scaling with a 1 pp denominator floor; the uncertainty penalty isolates forecast error risk; constrained nonnegative weights test a regularized, capped combination learned exclusively on development outcomes. These are alternative research hypotheses, not automatic replacements. See [the equations and fitting procedure](RESEARCH_METHODS.md).

| Signal | Mean Spearman IC |
|---|---:|
| Original balanced | -0.1304 |
| Standardized weighted | -0.1609 |
| Risk-adjusted return | -0.2087 |
| Uncertainty-penalized | -0.2043 |
| Constrained weights | 0.0130 |

The constrained solution weights standardized predicted return **0.503336**, momentum **0.031270**, negative uncertainty **0.104100**, and negative volatility **0.361294**. Weights sum to one, each is at most 0.6, and selected regularization is 0.1. They are specific to this development sample and are not recommended universal coefficients.

## Hypothetical portfolios

All ranked portfolios use the same delayed engine: top three equally weighted targets, maximum target weight 40%, total target exposure 100%, one shared-close execution lag, 21-session rebalance spacing, and **10 bps commission plus 5 bps slippage on full buy-plus-sell notional**. Entry and final liquidation costs are included. Equal weight holds all five stocks with the same constraints/cost convention; cash earns zero. Target weights can drift between fills. All portfolios stop at the common last predictive outcome on 2024-12-04, rather than extending into an unscored tail.

| Strategy | Net total return | Gross total return | Max drawdown | Annualized Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|
| Original balanced | 130.52% | 136.26% | -14.18% | 2.017 | 8.202 |
| Standardized weighted | 138.36% | 144.81% | -14.92% | 2.158 | 8.894 |
| Risk-adjusted return | 95.09% | 98.62% | -17.11% | 1.728 | 5.978 |
| Uncertainty-penalized | 98.76% | 99.58% | -14.18% | 1.902 | 1.383 |
| Constrained weights | 178.88% | 187.64% | -18.50% | 2.286 | 10.319 |
| Ridge ranking | 261.07% | 270.92% | -18.50% | 2.595 | 8.968 |
| Elastic Net ranking | 118.71% | 119.64% | -15.71% | 1.970 | 1.424 |
| Random Forest ranking | 176.55% | 185.01% | -15.71% | 2.373 | 10.049 |
| Gradient Boosting ranking | 144.12% | 148.09% | -15.71% | 2.207 | 5.383 |
| Momentum ranking | 239.15% | 248.96% | -14.92% | 2.549 | 9.498 |
| Equal weight, five stocks | 199.06% | 200.45% | -17.42% | 2.542 | 1.547 |
| Cash | 0.00% | 0.00% | 0.00% | — | 0.000 |

Sharpe assumes 252 shared sessions per year and a zero annual risk-free rate. Turnover is cumulative half-L1 risky-weight change; cost is instead charged on full traded asset notional. The CSV also exports full notional turnover and costs in units of initial wealth. Flat cross-sectional forecasts are alphabetically tie-broken; their portfolio returns are not evidence that their model successfully ranks stocks. The zero-return forecast is represented by a separate cash portfolio rather than an arbitrary top-three allocation.

The constrained strategy has higher sample return than original balanced, but also a deeper drawdown and more turnover. It underperforms the simple equal-weight survivor portfolio. These findings do not establish a profitable strategy.

## Statistical evidence

All comparisons pair identical dates and outcome endpoints. Approximate circular moving-date-block bootstrap uses seed 2026, 1,000 samples, block length three origins, and at least `max(12, 2 × block_length)` dates. The reported 95% percentile intervals are unadjusted; two-sided centered-null p-values receive Holm correction within each separately declared comparison family. Confidence intervals and adjusted p-values need not give identical decisions.

| Comparison | Mean difference | Approximate 95% interval | Holm-adjusted p |
|---|---:|---:|---:|
| Elastic Net − Ridge, date MSE (pp²) | 1.7461 | [-1.1423, 5.0962] | 0.5934 |
| Random Forest − Ridge, date MSE (pp²) | 1.4999 | [-1.3604, 5.9280] | 0.5934 |
| Gradient Boosting − Ridge, date MSE (pp²) | 2.4140 | [-0.6497, 6.7030] | 0.5934 |
| Zero − Ridge, date MSE (pp²) | 19.5346 | [6.1811, 34.0140] | 0.0300 |
| Constrained − original, Spearman IC | 0.1435 | [-0.1435, 0.4437] | 1.0000 |
| Constrained − original, net interval return | 1.0004 pp | [-1.0804, 2.8676] pp | 1.0000 |
| Equal weight − original, net interval return | 1.2811 pp | [0.2708, 2.2613] pp | 0.1079 |

There is conditional evidence of lower Ridge MSE than zero in this fixed sample. None of the learned regressor alternatives improves Ridge RMSE, and no experimental balanced-signal alternative establishes a rank or net-return improvement over original balanced after the declared adjustments. Portfolio comparisons test mean decision-interval returns, not differences in compounded terminal wealth or Sharpe. The exported interval table identifies each start, outcome mark and session count.

## Regime results and limits

Regimes use only trailing prices of the selected-universe equal-weight proxy. They are descriptive strata, not independently selected market-index regimes. Ridge RMSE is 21.4309 pp on **2 bear** origins, 10.1714 pp on **20 bull** origins, and 9.4063 pp on **1 sideways** origin; the bear band coverage is only 40%. It is 13.4157 pp across **7 high-volatility** origins versus 10.6546 pp across **16 low-volatility** origins. These tiny and unequal subsets cannot establish regime robustness. All models' results and the corresponding portfolio daily-return strata are retained in the evidence CSVs; portfolio strata use the prior close's regime.

The largest unresolved limitations are survivorship and retrospective universe choice, adjusted vendor snapshots rather than point-in-time vintages, no delisting returns, only two price-derived features, a short final test dominated by a technology rally, approximate small-sample inference, and hypothetical fills without measured liquidity/impact/taxes. Mixed-currency universes can compare returns and rankings but cannot produce portfolios without an FX accounting model. Broader vintage data, historical constituents, delisted securities, more independent regimes and external prospective evaluation are needed.

## Reproduce

From the repository root with the documented dependencies installed:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_research.py --dataset frozen \
  --validation-start 2020-01-01 --test-start 2023-01-01 \
  --end 2024-12-31 --output artifacts/research_runs/frozen-upgrade
```

The runner is offline: it verifies the bundled cache or reads existing checksum-verified local cache files and never silently downloads a replacement. The manifest contains the fully expanded command and exact package versions. Use a fresh output directory if an earlier experiment occupies that path; runs with different content are not overwritten. Reproducing with the same source/data/version/config hashes yields the same numerical results; metadata such as creation time and clean commit state can differ.

Run `streamlit run app.py`, select **Frozen demo (2018–2024)** and a main cutoff of **2024-12-31**, open **Research & Benchmarking**, retain the defaults, then click **Run research experiment**. Its ZIP contains full inputs, scores, daily ledgers and checksums. Earlier main cutoffs deliberately produce shorter tests. See [the audit](AUDIT.md), [methods](RESEARCH_METHODS.md), and [complete change list](UPGRADE_SUMMARY.md).
