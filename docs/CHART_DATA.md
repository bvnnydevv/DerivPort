# Historical candle data

The stock chart uses **real vendor daily open, high, low, close and volume observations**, downloaded from Yahoo Finance using the pinned yfinance dependency. Candles are never fabricated from close-only observations. Yahoo's historical snapshot can reflect stock splits; it is not a point-in-time archive of original quoted prices.

The interactive chart requires no live data connection for the bundled demo. Chart data, indicators and exports all use the same cutoff-safe observations; drawing annotations remain temporary browser state.

## Bundled snapshot

The repository includes **33,450 daily bars across 15 US-listed stocks**, covering **2018-01-02 through 2026-11-13**, with 2,230 observations per stock:

```text
AAPL AMZN GOOGL MSFT NVDA AMD META NFLX TSLA AVGO ORCL JPM LLY XOM WMT
```

CSV data occupy **3,082,145 bytes**; the CSV and JSON pairs together occupy **3,101,054 bytes**. Each metadata file records its exact source URL, download time, exchange, currency, field basis, observation count and CSV SHA-256. The files are under `data/candle_cache/`. The complete provider histories were downloaded and validated first, then their unchanged daily observations from 2018 onward were retained for the portable demo. Metadata records the untrimmed source coverage and the bundle restriction. An explicit refresh requests the provider's available maximum history.

Each source URL follows `https://finance.yahoo.com/quote/AAPL/history/`, substituting the listing symbol. The request uses daily regular-session data with `auto_adjust=False`, `back_adjust=False`, `repair=False`, `keepna=True`, and `actions=False`. The official [yfinance history implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/history.py) documents these switches; [the API reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html) describes the history entry point.

## Price basis and research separation

These vendor OHLC bars are **not dividend-adjusted** by yfinance. They are chart data only. Research predictions, signals, portfolio benchmarks and replay outcomes continue to use the existing, separately verified **adjusted-close** price cache. Therefore a candle close can differ from the adjusted research close on the same historical date. A candlestick's body displays its actual vendor open and close; its wick displays its actual vendor high and low.

Chart aggregation must first remove observations after the selected historical cutoff, then combine daily bars using first open, maximum high, minimum low, last close and summed volume. Weekly or monthly partial bars represent only sessions observed through the cutoff. No intraday bars or real-time streaming quotes are implied by these daily snapshots.

## Validation and cache behavior

`load_candles(symbol)` reads the existing cache without a provider call. An uncached stock requires an explicit **Download candles** action; `refresh=True` explicitly requests a refresh. The CSV and metadata checksum must agree. Invalid or duplicate session dates, missing/nonfinite fields, nonpositive OHLC, inconsistent high/low bounds, and negative volume are rejected. Provider bars on the current UTC date or later are excluded conservatively before storage, and cached incomplete/future-date bars are rejected. Missing sessions are never filled and missing OHLC fields are never synthesized.

Cache readers and publishers serialize pair access with a process-local thread lock and, on macOS/Linux, a file lock. Publication replaces temporary files and rolls back the previous pair if a replacement fails. A failed network refresh preserves and returns a valid prior cache with a visible warning. Candle cache files and lock/provider files stay inside `data/candle_cache/`; no existing analysis inputs are rewritten.

To fetch authentic candles explicitly from the repository root:

```sh
.venv/bin/python scripts/fetch_candles.py AAPL NVDA JPM
.venv/bin/python scripts/fetch_candles.py AAPL --refresh
```

Availability, exchange metadata and historical prices can change at the provider. The app discloses the cached observation date and basis; the snapshot does not claim live prices, executable quotes or complete US exchange coverage.
