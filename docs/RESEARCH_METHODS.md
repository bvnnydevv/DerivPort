# Research and benchmarking methods

DerivPort separates three questions: how accurately a model predicts returns, how well a score orders assets, and how a hypothetical portfolio behaves after costs. Scores with different units do not receive return-forecast RMSE. All comparisons are retrospective and conditional on the selected fixed universe and data vintage.

## Data, features, and forecast timing

Let $P_{i,t}$ be an observed adjusted close and $r_{i,t}=P_{i,t}/P_{i,t-1}-1$. The unchanged features and target, in numerical percentage points, are

$$
M_{i,t}=100(P_{i,t}/P_{i,t-21}-1),\qquad
\sigma_{i,t}=100\sqrt{21}\,\operatorname{sd}_{ddof=1}(r_{i,t-20:t}),\qquad
y_{i,t}=100(P_{i,t+21}/P_{i,t}-1).
$$

Thus `3.5` means 3.5%, not 350%. The volatility estimate is a scaled trailing sample standard deviation, not a forecast interval. Square-root scaling assumes a reference session clock and does not model return autocorrelation.

Only actual common observed dates are retained; prices are never forward-filled. With international listings, 21 means shared observations rather than 21 native exchange sessions or calendar days. Local-currency returns omit FX. Adjusted closes approximate total-return history but are retrospective vendor snapshots and hypothetical execution marks.

At origin $t$, fitting includes only feature rows with `label_end <= t`. All current cross-sectional features are known after the cutoff close; portfolio fills occur later. Origins are at least 21 shared sessions apart, avoiding overlapping evaluation targets. Date-only company notes separately require `published < cutoff`, because their intraday publication time is unknown.

## Forecast models and selection

The pooled two-feature Ridge baseline remains `StandardScaler + Ridge(alpha=25)`. The scaler is fitted separately on each expanding training set. Elastic Net also uses a training-only scaler; trees do not need that scaling. This follows scikit-learn's [preprocessing/leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

| Model | Fixed research specification / candidate grid |
| --- | --- |
| Ridge | `alpha=25`; unchanged baseline |
| [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) | `(alpha,l1_ratio)` = `(0.1,0.25)` or `(1,0.75)`; cyclic coordinate selection, `max_iter=10000`, `tol=1e-6` |
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | Depth 3 or 5; 32 trees, leaf minimum 20, all features, one worker |
| [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | Depth 1 or 2; 32 stages, learning rate 0.05, leaf minimum 20, squared loss, no early stopping |
| Momentum | Forecast next return equals trailing momentum; no fitted coefficients |
| Zero return | Forecast exactly zero; no fitted coefficients |

At most six inner origins are selected: the first eligible scheduled origin at or after 18, 15, 12, 9, 6, and 3 months before the declared final-test boundary. Their labels must finish strictly before that boundary. Candidate selection minimizes pooled validation MSE, equivalently RMSE; grid order breaks ties. A development forecast can use only validation folds completed by its own origin. Final-test hyperparameters freeze before testing. Model coefficients continue expanding during testing as earlier outcomes become observable; that is scheduled updating, not test-set hyperparameter selection.

Grouped dates and completed-label purges are implemented explicitly. A generic row-wise split could separate contemporaneous assets or admit an unfinished 21-session target; simply applying [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) to pooled stock rows would not supply these safeguards.

For each model, past residuals use the latest 24 completed stock forecasts when at least six exist; otherwise the latest 120 pooled completed errors are used. Their RMSE is $\varepsilon$; mean actual-minus-predicted residual is bias. Adding residual 10th/90th percentiles to the current prediction produces a nominal 80% empirical interval. Coverage and support are measured separately: neither RMSE nor these quantiles guarantee calibrated future probabilities.

## Ranking hypotheses

All five experimental signals use Ridge's forecast inputs. Let $z_t(x)$ standardize the current asset cross section with population standard deviation; a constant factor contributes zero. No future dates fit this transformation. Changing the universe changes these relative scores.

| Signal | Definition | Units / rationale |
| --- | --- | --- |
| Balanced original | $0.6\widehat y+0.4M-k(0.5\varepsilon+0.25\sigma)$ | Original percentage-point score; explicit fixed hypothesis |
| Standardized weighted | $0.6z_t(\widehat y)+0.4z_t(M)-k[0.5z_t(\varepsilon)+0.25z_t(\sigma)]$ | Dimensionless; reduces dominance from different cross-sectional factor spreads |
| Risk-adjusted return | $\widehat y/\max(\sigma,1\text{ pp})$ | Dimensionless; fixed volatility floor avoids unstable division |
| Uncertainty-penalized | $\widehat y-k\varepsilon$ | Percentage-point score; tests error penalty without momentum/volatility terms |
| Constrained weights | $w^\top[z_t(\widehat y),z_t(M),-z_t(\varepsilon),-z_t(\sigma)]$ | Dimensionless; development-fitted, bounded factor mixture |

The original terms have compatible nominal horizons and percentage-point units, but compatible units do not establish suitable coefficients. RMSE includes bias and dispersion; volatility and error may be correlated, so the original formula can penalize related risks twice and heavily suppress volatile assets. Alternatives investigate these choices rather than presume a correction improves results.

Constrained weights minimize $\operatorname{mean}(Xw-z_t(y))^2+\lambda\lVert w\rVert_2^2$, subject to $w_j\ge0$, $\sum_jw_j=1$, and $w_j\le0.6$. Development OOS origins must have `label_end < test_start`. The first approximately two thirds train weights; the last third validate $\lambda\in\{0.1,1,10\}$. Weight-training targets crossing the validation boundary are purged. The selected penalty is refitted on all eligible development rows, then weights freeze for the final test. Deterministic projected gradient and capped-simplex projection report convergence. Fewer than 12 development dates, fewer than four validation dates, or failed convergence makes this variant unavailable. Development scores using these fitted weights are not OOS weight-selection evidence.

Only complete common model/asset dates enter comparisons. Model predictions also serve as ranking scores. A zero-return forecast is distinct from the cash portfolio; equal weighting is a separate diversification baseline. Constant scores have undefined rank correlation, not zero correlation.

## Portfolio accounting and regimes

The legacy backtest remains a gross linked top-stock endpoint comparison. The new `portfolio.py` consumes dated scores, executes at least one later shared observed close, and earns no return before the fill. Targets select up to `top_n` assets equally, each at $\min(\text{cap},\text{exposure}/n)$; unused exposure is zero-interest cash. Targets are long only and unleveraged. Caps apply at rebalances; subsequent price drift can exceed them. Portfolio evaluation ends at the common last predictive `label_end`, without extending the last score into an unscored cache tail. Execution lag can shorten the final holding interval.

For pre-trade asset values $a_i$, cash $c$, and $V^-=c+\sum_i a_i$, post-cost NAV solves

$$
V^+ + q\sum_i|w_iV^+-a_i|=V^-,\qquad q=(\text{commission bps}+\text{slippage bps})/10000.
$$

Bisection sizes post-cost targets self-financingly. Costs apply to full buy-plus-sell notional, including entry and optional final liquidation. Reported half-L1 risky-weight turnover excludes cash; full notional turnover is also reported and is the cost basis. A fee-free shadow portfolio uses identical targets. Default costs are 10 bps commission plus 5 bps slippage, assumptions rather than measured execution estimates.

Daily net NAV supplies drawdown $V_t/\max(1,V_0,\ldots,V_t)-1$ and Sharpe $\sqrt{252}\,\overline{(r-r_f)}/\operatorname{sd}_{ddof=1}(r-r_f)$. Annual risk-free input converts to a daily compounded rate; cash still earns zero. Zero variance yields unavailable Sharpe. Annualization assumes 252 shared sessions/year. Portfolios require one known common currency; no FX conversion is invented. Spreads varying over time, market impact, taxes, liquidity and actual bid/ask fills remain unmodeled.

Regimes use a fixed-universe equal-weight daily-rebalanced return proxy. Trailing 126-session compounded return above +5% is Bull, below −5% Bear, otherwise Sideways. Trailing 21-session annualized volatility is High only above a strictly past expanding median with 126 prior volatility observations; otherwise Low. Warm-up is Unknown. Forecast comparisons use the cutoff regime; portfolio daily returns use the preceding observed close's regime. This is a selected-stock proxy, not an independent market index. Regime tables are descriptive; no regime-specific confidence intervals are exported.

## Evaluation, inference, and reproduction

Forecast metrics include pooled RMSE, MAE, $R^2$, Pearson/Spearman correlations, sign agreement, empirical interval coverage, and mean squared error normalized by past RMSE. Information coefficients are cross-sectional correlations at each date; their time means differ from pooled correlations. Average ranks handle ties. Ranking scores receive ICs, not return RMSE. Portfolio results separately report net/gross returns, daily Sharpe, drawdown, turnover, and costs.

Paired comparisons match dates and horizons: model-minus-Ridge MSE, signal-minus-original-balanced IC, and portfolio-minus-original-balanced net **decision-interval** return. Portfolio intervals partition daily accounting returns over `(cutoff, next cutoff]`, ending at the last predictive `label_end` for the final interval. They include delayed fills, old holdings, and fees, and are not individual-forecast holding returns. Exports record `interval_start`, `label_end`, and `n_sessions`.

Circular moving-date-block resampling preserves contemporaneous asset dependence; defaults are three origins per block and 1,000 seeded draws. Gaps or excluded undefined correlations produce blocks in retained event time, not equal calendar-time blocks. Percentile 95% intervals are unadjusted, approximate conditional intervals. Centered-null two-sided p-values use a +1 resampling correction; Holm adjustment applies within each comparison family, retaining unavailable comparisons in family size. Inference requires at least $\max(12,2L)$ finite dates for block length $L$; otherwise it reports `more_data_required`. Small samples, nonstationarity, repeated exploration, and survivor selection limit inference even when an adjusted p-value is small. More independent periods and broader vintage data are required for reliable generalization.

Reproduce the frozen experiment from the repository root:

```sh
.venv/bin/python scripts/run_research.py --dataset frozen \
  --validation-start 2020-01-01 --test-start 2023-01-01 \
  --end 2024-12-31 --output artifacts/research_runs/frozen-demo
```

Exports include forecasts, folds, rankings, regimes, comparisons, daily NAV, trades, and weights. Portable evidence also includes wide date-indexed `aligned_prices.csv` and long `source_observations.csv` (`date,ticker,adjusted_close`) containing each source series through the run's as-of end. `manifest.json` preserves exact `source_metadata`, raw/observed and aligned hashes, missing-date exclusions, unrepaired quality flags, software versions, seeds, configuration, source-code hashes, Git identity/dirty state, and output checksums. The stable run ID excludes creation time.

Replay the exported evidence without a provider request or the original runtime cache:

```python
import json
from pathlib import Path
import pandas as pd
from derivport.research import ExperimentConfig, run_experiment

directory = Path('artifacts/research_runs/frozen-demo')
manifest = json.loads((directory / 'manifest.json').read_text())
prices = pd.read_csv(directory / 'aligned_prices.csv', index_col='date',
    parse_dates=['date'], float_precision='round_trip')
source = pd.read_csv(directory / 'source_observations.csv',
    parse_dates=['date'], float_precision='round_trip')
raw = {ticker: group.set_index('date')['adjusted_close'].sort_index()
       for ticker, group in source.groupby('ticker')}
result = run_experiment(prices, ExperimentConfig(**manifest['config']),
    manifest['source_metadata'], raw)
```

Use the recorded code/software and compare hashes and numerical outputs; the creation timestamp changes. Matched-date legacy/new Ridge parity is measured, not invented uplift. The interface separately identifies historical observations, OOS predictions, analytical score curves, and simulated future scenarios.
