# DerivPort

**Interactive equity research, algorithm building, and calculus sensitivity analysis.**

DerivPort is an educational finance project for the Veritas AI Hackathon. Explore a stock, turn an idea into a transparent ranking rule, and test it against historical evidence—all in one workspace.

[Quick start](#quick-start) · [Two-minute tour](#two-minute-tour) · [How it works](#how-it-works) · [Documentation](#documentation)

![Research workspace: real candlesticks and an integrated stock coach](docs/screenshots/research-workspace.jpg)

*Research: observed prices, customizable charts, and a coach grounded in the selected historical view.*

![Algorithm builder: the selected stock chart beside its draft algorithm-score graph](docs/screenshots/algorithm-builder.jpg)

*Algorithm builder: compare the stock chart with your recipe’s score curve. A ranking score is different from an investment return.*

## What you can explore

| Workspace | What it does |
| --- | --- |
| **Research** | Browse US-listed equities, inspect candlesticks, read dated company evidence, and compare model and algorithm rankings. |
| **Algorithm builder** | Start with a guided recipe or arrange factor blocks. Preview score contributions, explore calculus, and run a historical backtest. |
| **Future scenarios** | Explore assumption-driven paths through 2037, with adjustable growth, volatility, and correlation. |
| **Benchmarks** | Compare six forecasting methods, alternative ranking rules, and hypothetical portfolios using chronological evaluation. |

Switch between **Day** (pastel gray), **Night** (dark), and **Pause** (bright green) at the top of the page. Stock charts, fullscreen charts, and algorithm views follow the selected mode. Settings and deeper research tools stay collapsed until needed.

## Quick start

Tested with **Python 3.13**. On macOS or Linux:

```sh
git clone https://github.com/bvnnydevv/DerivPort.git
cd DerivPort
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). On Windows, use the executables in `.venv\Scripts` instead of `.venv/bin`.

**No API key is needed for the bundled demo.** After installation, bundled price history, charting, guided lessons, algorithm experiments, and simulations work offline. News, provider searches, uncached stocks, refreshes, and external links require internet access.

To publish a shareable demo, follow the [Streamlit Community Cloud hosting guide](docs/HOSTING.md). Use repository `bvnnydevv/DerivPort`, branch `main`, and entry point `app.py`.

## Two-minute tour

1. **Explore a stock.** Start with AAPL in Research. Use **Customize** for moving averages, candle intervals, colors, and volume; open **Fullscreen** to inspect the chart. Type a question into the coach or use **Chart tools** to mark a trend or range.
2. **Build an idea.** In Algorithm builder, adjust the model/recent-strength mix and risk penalty. The orange score graph updates with the draft; the stock-price data stays fixed. Inspect the contribution table, then choose **Apply blocks to algorithm**.
3. **Test and explain it.** Run a historical backtest to compare your rule, the model, momentum, and the selected stock’s buy-and-hold return. Open **Explore the formula & Calculus Lab** to see how a small input change affects slope, curvature, and accumulated score change.
4. **Explore a possible future.** In Future scenarios, change Low/Medium/High risk and simulate through a target date. The purple price bands and amber return distribution show what the assumptions produce—not known future prices.

For a repeatable historical example, open **Settings**, select **Frozen demo (2018–2024)** and the original five-stock group, then choose **AAPL · May 3, 2024**. In Research tools, reveal the next 21 sessions only after inspecting the evidence. This example was curated for an understandable earnings discussion, not as proof of a successful forecast.

## How it works

### Price charts and company news

Research and the builder share the same selected stock, chart range, candle colors, and indicators. Charts support daily/weekly/monthly candles, SMA/EMA overlays, volume, log scale, pan, zoom, drawing tools, and exports.

**Company news** searches a rolling seven-day window on demand. Each headline includes its publisher, publication date, and source link. Ten appear initially; **More headlines** expands the list, up to fifty. The app shows the actual supplied date range because the provider may return less than a full week. A conservative title filter can miss aliases or articles that omit the company name and ticker.

**Price move around an event** measures observed 1-, 5-, and 21-session changes around a dated announcement or article. It uses the first observed session after the publication date and its preceding close as the baseline; publication-day moves are excluded. Precise timestamps use the New York calendar date. Missing outcomes remain pending. These changes describe an association, not proof that the news caused the move.

### Transparent algorithms and calculus

The initial active rule is:

$$
S=\widehat r_{21}-k\varepsilon,\qquad k=2.
$$

Here, $\widehat r_{21}$ is the local regression’s estimated return over the next 21 trading sessions, and $\varepsilon$ is its typical past forecast error. Higher scores rank higher within the comparison group.

Guided recipes combine model outlook, recent strength, past forecast error, and price swings. Advanced blocks and the text editor use the same restricted arithmetic parser. Draft changes affect the preview; **Apply** activates the block recipe. The orange curve replays the draft on historical forecast inputs, rather than inventing daily predictions or treating scores as returns.

The named **DerivPort Signal** preset combines 60% model estimate and 40% momentum with error and volatility penalties. Its standalone [Python download](algorithms/derivport_signal.py) defaults to `k = 1`; guided drafts retain the current multiplier. See the [algorithm guide](algorithms/README.md) for exact inputs and usage.

Calculus Lab renders the formula in LaTeX and explores derivatives, curvature, tangent and quadratic approximations, and the integral of the derivative. These explain the ranking rule’s arithmetic, not the regression’s feature influence.

### Historical evidence and future scenarios

The default prediction model is standardized **Ridge regression** using past momentum and volatility. Training uses completed historical labels; the error estimate uses completed out-of-sample forecasts. The Benchmarks workspace adds Elastic Net, Random Forest, Gradient Boosting, momentum, and zero-return baselines.

Future scenarios use **600 correlated lognormal paths by default**, anchored to the latest observed adjusted close. Volatility comes from up to three years of past daily returns. Annual log-price drift and correlation are user assumptions; Low/Medium/High applies 0.75×/1×/1.5× historical volatility. The bands are simulated percentiles, not calibrated confidence intervals. News, the DIY ranking rule, and the 21-session regression do not generate these long-range paths.

### Optional AI coach

The local regression and the conversational coach are separate. Without a key, the coach provides clearly labeled prepared lessons and local chart guides. For AI-generated explanations, add `OPENAI_API_KEY` to the hosting service’s secrets or the ignored `.streamlit/secrets.toml`; `OPENAI_MODEL` is optional.

An explicit question submission sends the question, a bounded conversation, and selected public historical evidence to OpenAI. Current news and future outcomes stay outside that historical context. Never commit credentials. The bundled demo remains usable without this integration.

## Data and research boundaries

| Included snapshot | Coverage |
| --- | --- |
| **US equity directory** | 6,094 current listing candidates after disclosed filters, from NasdaqTrader files dated **November 13, 2026**. |
| **Offline US price history** | 15 adjusted-price histories through **November 13, 2026**. Additional listings load their available history on selection. |
| **Offline candlesticks** | 33,450 daily OHLCV bars across 15 US stocks, from **2018-01-02 to 2026-11-13**. |

The directory includes qualifying common shares, ADRs, preferred shares, and partnership equity units across exchanges. It is not an archive of every historical, delisted, or OTC security; provider coverage varies.

Candles use unadjusted OHLCV. Model inputs and return comparisons use adjusted closes, so displayed price levels can differ. Prices are cached daily observations, not a live intraday feed. No missing historical prices or candles are fabricated.

Historical cutoffs restrict displayed evidence; subsequent returns stay hidden until explicitly revealed. Current news is kept separate from historical model and coach inputs. Adjusted vendor snapshots can still change with later corporate actions, and chosen stock groups have selection and survivorship bias.

Backtests are retrospective experiments. The builder’s linked returns omit execution costs and taxes; the separate Benchmarks portfolio engine models delayed fills, commission, slippage, sizing, and drawdowns. Repeated tuning can overfit. This is an educational research app with hypothetical allocations, not brokerage access or investment advice.

## Documentation

| Resource | Contents |
| --- | --- |
| [Hosting](docs/HOSTING.md) | Deployment and presentation checks. |
| [Research methods](docs/RESEARCH_METHODS.md) | Chronological evaluation, models, signals, portfolios, and inference. |
| [Measured results](docs/RESEARCH_RESULTS.md) | Retained experiments, comparisons, and limitations. |
| [Chart data](docs/CHART_DATA.md) · [Data credits](data/CREDITS.md) | Price provenance, caching, and source terms. |
| [Algorithm guide](algorithms/README.md) | Standalone Python algorithm and input format. |
| [Figma logo](https://www.figma.com/design/ZxLybASxv0lxyUkdnQ33A2/Hackathon-DerivPort?node-id=8-8) · [Brand](docs/BRAND.md) | Editable logo source and SVG provenance. |

## Complete sources, models, and tool credits

This project is a prototype and is not financial advice. The inventory below distinguishes source datasets, models trained by this project, optional pretrained AI, development tools, and original project work.

### Datasets and data providers

| Dataset / source | Exactly how DerivPort uses it |
| --- | --- |
| [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://github.com/ranaroussi/yfinance) 1.8.0 | Adjusted-close research history, separate daily OHLCV chart history, equity search and on-demand company-news headlines. Cached snapshots are retrospective vendor data, not live quotes or point-in-time archives. |
| [Akmal Yaasir Fauzaan / Akma86 — US Tech Giants Market Analytics](https://github.com/Akma86/us-tech-giants-market-analytics/tree/363151b27e8bf71b377483b37fa49ee5ebb3f5e2) | Source mirror for the frozen five-stock dataset: 8,805 adjusted-price observations, AAPL/AMZN/GOOGL/MSFT/NVDA, 2018-01-02–2024-12-31. Pinned commit `363151b27e8bf71b377483b37fa49ee5ebb3f5e2`; exact upstream files/checksums appear in [provenance](data/provenance.json). The mirror is a data source; DerivPort does not claim authorship of it. |
| [NasdaqTrader — nasdaqlisted.txt](https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt) and [otherlisted.txt](https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt) | US equity directory, filtered to 6,094 listing candidates from source files dated 2026-11-13. [Field definitions](https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs), [TXSE notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=DTN2026-10), [NYSE Texas definitions](https://www.nyse.com/publicdocs/nyse/data/NYSE_Texas_Support_of_Primary_Listings.pdf), and [directory provenance](data/us_listings_provenance.json). |
| [Curated issuer evidence](data/notes.json) | 20 project-authored paraphrases of dated primary earnings releases/exhibits, with separately labeled interpretation and limitations. Every original source is listed below. |
| [Starter symbol catalog](data/market_catalog.csv) | Project-curated 109-symbol US/international lookup catalog, not an external dataset or complete exchange directory. International cache files remain archived; the current explorer is US-only. |
| Simulated scenario data | Generated locally by DerivPort using NumPy, observed volatility and user assumptions with fixed seeds. These are synthetic paths, not a downloaded dataset, pretrained prediction model, or known future prices. |

News titles, publishers, timestamps and article links come from Yahoo Finance search through yfinance. The publisher/source URL is shown on each fetched headline; this changing feed is not a fixed training corpus, and full articles are not scraped. News is kept separate from historical model inputs. There is no fixed list of every future publisher the provider may return.

<details>
<summary>Every committed extended price and candle dataset</summary>

The following inventory describes the committed snapshots, rather than uncommitted local refreshes. Each linked sidecar records the provider URL, retrieval time, adjustment basis and checksum. Price values use the listing's local currency; no FX dataset or conversion is supplied.

| Symbol | Adjusted-close dates | Provider | Adjusted-price provenance | OHLCV provenance |
| --- | --- | --- | --- | --- |
| 005930.KS | 2000-01-04–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/005930.KS/history/) | [Snapshot](data/market_cache/005930.KS_7599fc75.json) | Not bundled |
| 0700.HK | 2004-06-16–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/0700.HK/history/) | [Snapshot](data/market_cache/0700.HK_efe2ecdc.json) | Not bundled |
| 7203.T | 1999-05-06–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/7203.T/history/) | [Snapshot](data/market_cache/7203.T_39c0b98e.json) | Not bundled |
| AAPL | 1980-12-12–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history/) | [Snapshot](data/market_cache/AAPL_1eb44d62.json) | [Snapshot](data/candle_cache/AAPL_1eb44d62.json) |
| AMD | 1980-03-17–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/AMD/history/) | [Snapshot](data/market_cache/AMD_0fd3e4e7.json) | [Snapshot](data/candle_cache/AMD_0fd3e4e7.json) |
| AMZN | 1997-05-15–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/AMZN/history/) | [Snapshot](data/market_cache/AMZN_31ac18dc.json) | [Snapshot](data/candle_cache/AMZN_31ac18dc.json) |
| ASML.AS | 1998-07-20–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/ASML.AS/history/) | [Snapshot](data/market_cache/ASML.AS_960318b1.json) | Not bundled |
| AVGO | 2009-08-06–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/AVGO/history/) | [Snapshot](data/market_cache/AVGO_b78cca24.json) | [Snapshot](data/candle_cache/AVGO_b78cca24.json) |
| BHP.AX | 1988-01-29–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/BHP.AX/history/) | [Snapshot](data/market_cache/BHP.AX_0ab38321.json) | Not bundled |
| GOOGL | 2004-08-19–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/GOOGL/history/) | [Snapshot](data/market_cache/GOOGL_a25fc665.json) | [Snapshot](data/candle_cache/GOOGL_a25fc665.json) |
| JPM | 1980-03-17–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/JPM/history/) | [Snapshot](data/market_cache/JPM_b64a6324.json) | [Snapshot](data/candle_cache/JPM_b64a6324.json) |
| LLY | 1972-06-01–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/LLY/history/) | [Snapshot](data/market_cache/LLY_4bac0862.json) | [Snapshot](data/candle_cache/LLY_4bac0862.json) |
| META | 2012-05-18–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/META/history/) | [Snapshot](data/market_cache/META_2d1edf72.json) | [Snapshot](data/candle_cache/META_2d1edf72.json) |
| MSFT | 1986-03-13–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/MSFT/history/) | [Snapshot](data/market_cache/MSFT_8638f027.json) | [Snapshot](data/candle_cache/MSFT_8638f027.json) |
| NESN.SW | 1990-01-03–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/NESN.SW/history/) | [Snapshot](data/market_cache/NESN.SW_a19f050f.json) | Not bundled |
| NFLX | 2002-05-23–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/NFLX/history/) | [Snapshot](data/market_cache/NFLX_edf1b81f.json) | [Snapshot](data/candle_cache/NFLX_edf1b81f.json) |
| NOVO-B.CO | 2001-01-08–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/NOVO-B.CO/history/) | [Snapshot](data/market_cache/NOVO-B.CO_5b3392c5.json) | Not bundled |
| NVDA | 1999-01-22–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/NVDA/history/) | [Snapshot](data/market_cache/NVDA_426219f2.json) | [Snapshot](data/candle_cache/NVDA_426219f2.json) |
| ORCL | 1986-03-12–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/ORCL/history/) | [Snapshot](data/market_cache/ORCL_71ed90c9.json) | [Snapshot](data/candle_cache/ORCL_71ed90c9.json) |
| RELIANCE.NS | 1996-01-01–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/RELIANCE.NS/history/) | [Snapshot](data/market_cache/RELIANCE.NS_e43ba598.json) | Not bundled |
| RY.TO | 1995-01-12–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/RY.TO/history/) | [Snapshot](data/market_cache/RY.TO_e8c9dd9e.json) | Not bundled |
| SAP.DE | 1998-04-09–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/SAP.DE/history/) | [Snapshot](data/market_cache/SAP.DE_a80ae49a.json) | Not bundled |
| TSLA | 2010-06-29–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/TSLA/history/) | [Snapshot](data/market_cache/TSLA_2ff6985a.json) | [Snapshot](data/candle_cache/TSLA_2ff6985a.json) |
| WMT | 1972-08-25–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/WMT/history/) | [Snapshot](data/market_cache/WMT_dd388dd2.json) | [Snapshot](data/candle_cache/WMT_dd388dd2.json) |
| XOM | 1962-01-02–2026-11-13 | [Yahoo Finance](https://finance.yahoo.com/quote/XOM/history/) | [Snapshot](data/market_cache/XOM_b8b420a4.json) | [Snapshot](data/candle_cache/XOM_b8b420a4.json) |

</details>

<details>
<summary>All 20 original company-evidence sources</summary>

| Company | Publication date | Original release / filing |
| --- | --- | --- |
| AAPL | 2023-08-03 | [Apple reports third quarter results](https://www.apple.com/newsroom/2023/08/apple-reports-third-quarter-results/) |
| AAPL | 2023-11-02 | [Apple reports fourth quarter results](https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/) |
| AAPL | 2024-05-02 | [Apple reports second quarter results](https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/) |
| AAPL | 2024-08-01 | [Apple reports third quarter results](https://www.apple.com/newsroom/2024/08/apple-reports-third-quarter-results/) |
| AMZN | 2023-08-03 | [Amazon.com announces second quarter results](https://ir.aboutamazon.com/news-release/news-release-details/2023/Amazon.com-Announces-Second-Quarter-Results/default.aspx) |
| AMZN | 2023-10-26 | [Amazon.com announces third quarter results](https://ir.aboutamazon.com/news-release/news-release-details/2023/Amazon-com-Announces-Third-Quarter-Results/default.aspx) |
| AMZN | 2024-04-30 | [Amazon.com announces first quarter results](https://www.sec.gov/Archives/edgar/data/1018724/000101872424000081/amzn-20240331xex991.htm) |
| AMZN | 2024-08-01 | [Amazon.com announces second quarter results](https://ir.aboutamazon.com/news-release/news-release-details/2024/Amazon-com-Announces-Second-Quarter-Results/default.aspx) |
| GOOGL | 2023-07-25 | [Alphabet announces second quarter 2023 results](https://www.sec.gov/Archives/edgar/data/1652044/000165204423000067/googexhibit991q22023.htm) |
| GOOGL | 2023-10-24 | [Alphabet announces third quarter 2023 results](https://www.sec.gov/Archives/edgar/data/1652044/000165204423000088/googexhibit991q32023.htm) |
| GOOGL | 2024-04-25 | [Alphabet announces first quarter 2024 results](https://www.sec.gov/Archives/edgar/data/1652044/000165204424000047/googexhibit991q12024.htm) |
| GOOGL | 2024-07-23 | [Alphabet announces second quarter 2024 results](https://abc.xyz/assets/19/e4/3dc1d4d6439c81206370167db1bd/2024q2-alphabet-earnings-release.pdf) |
| MSFT | 2023-07-25 | [Microsoft earnings release FY2023 Q4](https://www.microsoft.com/en-us/investor/earnings/fy-2023-q4/press-release-webcast) |
| MSFT | 2023-10-24 | [Microsoft earnings release FY2024 Q1](https://www.microsoft.com/en-us/investor/earnings/fy-2024-q1/press-release-webcast) |
| MSFT | 2024-04-25 | [Microsoft earnings release FY2024 Q3](https://www.microsoft.com/en-us/investor/earnings/fy-2024-q3/press-release-webcast) |
| MSFT | 2024-07-30 | [Microsoft earnings release FY2024 Q4](https://www.microsoft.com/en-us/investor/earnings/fy-2024-q4/press-release-webcast) |
| NVDA | 2023-05-24 | [NVIDIA announces financial results for first quarter fiscal 2024](https://investor.nvidia.com/news/press-release-details/2023/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2024/default.aspx) |
| NVDA | 2023-08-23 | [NVIDIA announces financial results for second quarter fiscal 2024](https://investor.nvidia.com/news/press-release-details/2023/NVIDIA-Announces-Financial-Results-for-Second-Quarter-Fiscal-2024/default.aspx) |
| NVDA | 2024-02-21 | [NVIDIA announces financial results for fourth quarter and fiscal 2024](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2024) |
| NVDA | 2024-05-22 | [NVIDIA announces financial results for first quarter fiscal 2025](https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2025/) |

</details>

### Models and pretrained AI

| Model / method | Origin and role |
| --- | --- |
| [Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) + [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) | Trained within DerivPort on completed historical return labels; Ridge `alpha=25`. No downloaded pretrained stock-model weights. |
| [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) | Trained locally for chronological benchmark comparisons. |
| [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) | Trained locally for benchmark comparisons; 32 trees, fixed seed. |
| [Gradient Boosting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) | Trained locally for benchmark comparisons; 32 estimators, learning rate 0.05, fixed seed. |
| Momentum and zero-return baselines | Deterministic project baselines, not pretrained models. |
| DerivPort Signal, factor blocks and alternative ranking rules | Explicit mathematical scoring rules implemented in this repository. These are not externally pretrained models or proven investment strategies. See [methods](docs/RESEARCH_METHODS.md) and [algorithm source](algorithms/derivport_signal.py). |
| Optional OpenAI coach | External pretrained language model accessed through the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses). The code default is `gpt-5.4-mini`; a server may override it using `OPENAI_MODEL`. A configured key and explicit submission are required. This credits the implemented optional integration, not a claim that paid/live inference was used in the submitted demonstration. |
| Offline coach | Project-authored deterministic prepared lessons and bounded chart-guide actions. It is not a pretrained model or generated AI response. |

Exact fitted-model settings and chronological training/validation boundaries are documented in [research_models.py](derivport/research_models.py), [analysis.py](derivport/analysis.py), and [Research methods](docs/RESEARCH_METHODS.md). Actual optional-coach deployment configuration cannot be inferred from repository defaults; secrets are intentionally excluded.

### AI development tools, design tools, and tutorials

- **OpenAI ChatGPT / Codex — “ChatGPT 6.1 Sol” (team-reported tool/model label):** the team reports this was its only AI development assistant. It assisted with code, debugging, tests, UI iterations, mathematical explanations, documentation and demo preparation. The exact public product/version mapping has not been independently verified. This development assistant is distinct from the optional in-app coach. [ChatGPT](https://chatgpt.com/) · [Codex](https://openai.com/codex/).
- **Figma:** used for interface design/reference work and the original vector logo component/export. [Project file](https://www.figma.com/design/ZxLybASxv0lxyUkdnQ33A2/Hackathon-DerivPort) · [export provenance](docs/BRAND.md). The current app header uses a text wordmark; the earlier SVG remains credited in the repository.
- **Tutorials and starter code:** no third-party tutorial, course, video or starter application has been identified in the repository or reported in the team's clarification. The upstream market-data mirror is explicitly credited above. Official API/library documentation is listed below as implementation reference material; it is not presented as a copied tutorial. If a teammate used additional material, it must be added before submission.
- **Images and reference designs:** the interface was informed by reference screenshots supplied by the project owner; their original designers/fonts were not identified, so no authorship is claimed. The retained NYSE photograph is by [Ken Lund](https://commons.wikimedia.org/wiki/File:New_York_Stock_Exchange,_Manhattan,_New_York_(7236984682).jpg), licensed [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/); it is no longer displayed in the app. `rule-lab.svg` and `scenario-lab.svg` are original project illustrations. [Asset credits](assets/CREDITS.md).

### Software and documentation references

The pinned direct runtime dependencies are [Streamlit](https://docs.streamlit.io/) **1.67.0**, [pandas](https://pandas.pydata.org/docs/) **3.1.1**, [NumPy](https://numpy.org/doc/) **2.5.4**, [scikit-learn](https://scikit-learn.org/stable/) **1.9.1**, [Plotly.py](https://plotly.com/python/) **7.1.0**, and [yfinance](https://ranaroussi.github.io/yfinance/) **1.8.0**. [pytest](https://docs.pytest.org/) **9.2.0** is the direct development/test dependency. Python **3.13** and its standard library support the application and standalone algorithm. See [requirements.txt](requirements.txt) and [requirements-dev.txt](requirements-dev.txt).

Specific implementation references include [yfinance Search](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Search.html), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing), and the linked scikit-learn estimator and OpenAI API references above. The mathematical parser, differentiation, quadrature, scenario generation, scoring and backtesting logic are implemented in this repository; no external calculus dataset or downloaded financial forecasting checkpoint is used.

Application code is MIT licensed. That license does not relicense Yahoo Finance data, issuer releases, third-party reference images, or the retained Creative Commons photograph. Original provider terms continue to apply. No real brokerage accounts or private portfolio uploads are part of the submitted datasets.

## Verify and reproduce

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests cover cutoff boundaries, formula safety, draft/active isolation, matched chart data, pending outcomes, news dates, calculus, and reproducible simulations. Core dashboard tests block outbound network calls.

To reproduce the fixed-cache research experiment:

```sh
.venv/bin/python scripts/run_research.py --dataset frozen --validation-start 2020-01-01 --test-start 2023-01-01 --end 2024-12-31 --output artifacts/research_runs/frozen-demo
```

The export includes input prices, forecasts, scores, folds, trades, NAV, diagnostics, and a manifest recording hashes and settings.

Application code is [MIT licensed](LICENSE). Market data and issuer material retain their source terms; see [data credits](data/CREDITS.md) and [image credits](assets/CREDITS.md).
