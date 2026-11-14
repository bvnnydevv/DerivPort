# DerivPort

An educational stock-research and algorithm lab for the Veritas AI finance challenge. Compare dated company evidence with a **quant analyst — model and ranking**, edit a mathematical rule, and reveal what happened next. The interface has two focused tabs, **Research** and **Algorithm builder**. No brokerage connections, balances, position sizing, trades, or buy/sell recommendations.

## Run locally

Tested with Python **3.13**:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**. Installation requires internet. Once installed, all bundled and previously downloaded stock histories work offline. Explicit provider search, adding an uncached stock, refreshing history, and opening external source links require internet. No API key is required. Dependencies include Streamlit, pandas, NumPy, scikit-learn, Plotly, and `yfinance==1.8.0`.

## Explore stocks and dates

The curated starting example is **Apple on May 3, 2024**, after its May 2 earnings release. It was chosen for clear, mixed business evidence, not to demonstrate a successful forecast. The initial five-stock comparison uses AAPL, MSFT, NVDA, AMZN, and GOOGL with the frozen 2018–2024 demo cache.

Select **2–10 stocks** for a comparison. The offline starter catalog contains **109 symbols**; an explicit online search queries Yahoo Finance equity listings worldwide. Enter a Yahoo-format symbol directly for listings outside the catalog, such as `0700.HK`, `SAP.DE`, `7203.T`, `ASML.AS`, or `RELIANCE.NS`. Provider coverage varies: this is not a guarantee that every current, historical, or delisted security is available.

**25 full histories are supplied**, containing **212,969 adjusted daily observations** through November 13, 2026. They include the original five, ten additional US stocks, and ten international listings. Enable **Use extended downloaded histories** to use these longer histories for the original stocks; adding another stock enables extended history automatically. The frozen demo file remains unchanged. Further downloads stay in the local cache and are ignored by Git.

Choose a daily historical cutoff. Weekends and other missing dates normalize to the previous shared trading session. Available cutoffs begin after at least 252 shared observations and no earlier than April 1, 2020; the latest cutoff must have 21 subsequent shared observations available. Different exchange holidays mean a global comparison uses **shared sessions**, without forward filling. Returns are measured in each listing's local currency; there is no FX conversion.

## Research and algorithm experiments

Both analyst panels use the same stock and cutoff. Company notes separate reported facts, editorial interpretation, and open questions, with the publication date and official source link. Only notes published by the cutoff are eligible. The 20 verified notes are a limited collection; additional symbols and dates may lack suitable company evidence. No narrative or disagreement is invented, and qualitative notes receive no predictive-accuracy score.

Three algorithms rank the same comparison group:

- **Momentum:** trailing 21-session adjusted-price return.
- **AI model:** regression estimate of the next 21-session return.
- **Student algorithm:** an editable expression, initially `predicted_return - k * uncertainty`, with `k = 2`.

In **Algorithm builder**, use `predicted_return`, `uncertainty`, `momentum`, `volatility`, `k`, numeric constants, basic arithmetic, and parentheses. A restricted AST parser rejects calls, attributes, powers, division by zero, and non-finite results. Invalid edits preserve the last valid formula, k, scores, and ranks.

The score-versus-uncertainty chart holds other inputs fixed, marks the current score, and shows the local slope. The default slope is −k: at k = 2, one extra percentage point of uncertainty lowers the score by two percentage points. Custom slopes are numerical. Crossover examples distinguish a changing score from a changing relative rank.

**Research** hides subsequent returns, realized ranks, and single-date ranking accuracy until **Reveal next 21 trading days**. Reveal freezes that experiment's predictions; later rule edits are labeled **edited after seeing the outcome**. Another unrevealed cutoff provides a fresh experiment. Reveal context includes the universe, cutoff, and price-data fingerprint, so changing the experiment does not expose another experiment's result.

Run a separate date-range backtest in **Algorithm builder**. All methods use the same nonoverlapping forecast origins and completed outcomes; labels extending past the requested end date are excluded. Results include per-period Spearman correlations, top-ranked-stock returns, forecast RMSE, aggregate comparisons, and a linked research-return curve. These exploratory retrospective results stay separate from a single historical reveal; a revised rule is not untouched validation.

The expandable stress tests use **600 reproducible correlated paths per scenario** for rally, sudden reversal, and sideways/volatile assumptions. Each method freezes its initial ranking and top-stock selection, then receives the same paths. Formula edits preserve the paths. Return distributions and first-place frequencies are simulated projections, not historical evidence or model validation.

## Model, units, and evaluation

A pooled expanding-window **StandardScaler + Ridge(alpha=25)** regression uses trailing 21-session return and trailing daily-return sample standard deviation × √21. Its target is the next 21-session adjusted-price return. Training includes only rows whose complete target ends by the forecast cutoff; scaling is fitted on those training rows. There is no random split.

Out-of-sample origins are 21 sessions apart beginning in 2020. Uncertainty is the RMSE of each stock's latest 24 completed out-of-sample forecast errors, using only outcomes already known at the cutoff. Fewer than six stock errors triggers a disclosed last-120-error pooled fallback. This is typical past forecast error, not a confidence interval. Reported aggregate evaluation begins in 2022.

Return-like inputs use percentage points: `3.5` means 3.5%. The default score has those units; k is dimensionless. Custom multiplication or division can change score units. Average ranks handle ties, with alphabetical order only breaking selection/display ties. Spearman compares ranked orders: +1 means identical order and −1 means reversed order; a constant ranking is undefined. The momentum forecast baseline assumes the next return equals the preceding return.

For the **frozen original five-stock cache only**, there are **34 held-out dates / 170 forecasts** with completed outcomes through December 2024:

| Measure | AI regression | Momentum baseline | Student rule, k = 2 |
| --- | ---: | ---: | ---: |
| Return forecast RMSE | 12.11 pp | 18.42 pp | Not a return forecast |
| Mean per-date Spearman | +0.085 | −0.044 | −0.097 |

The baseline has lower RMSE on **5 of 34 dates**. These weak rank correlations demonstrate that lower prediction error does not establish useful rankings. These numbers do **not** describe expanded histories, custom universes, or new formulas. In Research, aggregate results include only outcomes completed by its selected cutoff.

Exact frozen-cache metrics and assumptions are in [docs/evaluation.json](docs/evaluation.json). The two-minute presentation follows one stock from business evidence to model rank, changes one mathematical assumption, then reveals the outcome: [docs/DEMO.md](docs/DEMO.md). **Both files contain outcome spoilers.**

## Data, integrity, and limitations

The original cache contains **8,805 adjusted prices** from Yahoo Finance via Akmal Yaasir Fauzaan's public mirror at a pinned commit. Extended caches download Yahoo Finance `Adj Close` through yfinance. Each extended file has provenance, currency, retrieval time, and a checksum; failed refreshes preserve the prior valid cache. No synthetic historical observations, interpolation, or unadjusted substitutes are used.

See [data/CREDITS.md](data/CREDITS.md), [data/provenance.json](data/provenance.json), and the sidecars in `data/market_cache/` for source details. To reproduce the frozen base cache with internet access: `.venv/bin/python scripts/fetch_data.py`.

These are retrospective adjusted snapshots, not point-in-time vendor archives. Later corporate actions can rescale earlier levels; vendor revisions and errors remain possible. Chronological modeling does not eliminate that limitation. Chosen stock groups have selection and survivorship bias; the original group is concentrated in surviving technology businesses. Source coverage is curated, global calendars differ, local-currency results exclude FX effects, and simulations depend on their assumptions. Linked research returns omit trading frictions and are not executable performance. Reveal state is session-local. The finance theme is confirmed; no detailed official scoring rubric was supplied.

The repository's existing MIT license covers application code. Market data and issuer material retain their owners' terms; the credited cache is for educational demonstration, not a commercial redistribution license.

## Verify

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests exercise the normal Streamlit entry point with outbound connections blocked: hidden outcomes, formula preservation, real k-driven rank crossings, frozen reveal predictions, post-reveal edit labels, and fresh experiment contexts. Additional checks cover dated sources, completed training labels, future-price mutation invariance, tied/constant Spearman handling, measured uncertainty, identical simulations for identical rankings, global data/cache behavior, dynamic universes, and common-date backtests.
