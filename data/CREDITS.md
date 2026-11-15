# Historical data and source credits

`prices.csv` contains 8,805 observations: five stocks × 1,761 common US trading dates, from January 2, 2018 through December 31, 2024. The five-stock universe is fixed: AAPL, MSFT, NVDA, AMZN, GOOGL.

Prices originate from **Yahoo Finance**, downloaded via `yfinance` by **Akmal Yaasir Fauzaan**, and are cached from [Akma86/us-tech-giants-market-analytics](https://github.com/Akma86/us-tech-giants-market-analytics) at commit `363151b27e8bf71b377483b37fa49ee5ebb3f5e2`. The upstream script requests `auto_adjust=False`; DerivPort consistently selects its separate `Adj Close` field, which accounts for splits and cash dividends. No synthetic observations, interpolation, or current quotes are included.

The cache was retrieved on November 14, 2026. This is a retrospectively adjusted snapshot, not a vendor archive of the data as originally delivered at each historical cutoff. Later corporate actions can rescale earlier levels; DerivPort uses scale-invariant returns and volatility. Data-vendor revisions and mirror transcription issues cannot be ruled out. Historical adjusted prices approximate total-return history and are not executable prices.

`notes.json` contains 20 short editorial summaries of dated primary sources: official Apple Newsroom, Microsoft Investor Relations, Amazon Investor Relations, NVIDIA Investor Relations/Newsroom, Alphabet Investor Relations, and issuer earnings exhibits filed at SEC.gov. Every record includes the publication date and original URL. Reported facts, DerivPort interpretation, and open questions are separate fields. These notes make no explicit stock-return prediction and receive no accuracy score. They are a curated sample, not complete company-news coverage.

For each stock/date pair the app selects the newest curated note with `published < cutoff`, or explicitly reports that no source note is available. Date-only releases are embargoed on their publication day because their time relative to the closing price is unknown. Curated decision dates are later trading days than the release dates. Older NVIDIA notes explicitly disclose their age.

`provenance.json` records exact source URLs, source SHA-256 hashes, adjustment details, selected dates, cache checksum, and limitations. To reproduce the frozen price cache with network access, run `python scripts/fetch_data.py`. The main app never calls this downloader.

The mirror's code license is MIT. Underlying market data and issuer content retain their original owners' terms. This limited credited cache is for personal educational demonstration; it is not a commercial redistribution license.

## Expanded global explorer

`market_catalog.csv` is a manually curated starter catalog of 109 US and international stock symbols, with illustrative listing and currency labels. It is not a complete exchange directory. An explicit online search queries Yahoo Finance equity listings via `yfinance`; users can also enter a Yahoo-format symbol directly. Provider coverage and available history vary, and newly listed, delisted, renamed, and some regional securities may be absent.

`market_cache/` contains separately cached genuine `Adj Close` histories from Yahoo Finance, downloaded on explicit requests with `yfinance` 1.8.0. Each CSV has a matching JSON sidecar recording its source URL, exchange, local currency, retrieval time, date range, observation count, and SHA-256 hash. No prices are generated, interpolated, or replaced with unadjusted closes. The current UTC date and future bars are excluded to avoid incomplete daily observations. Cross-market comparisons use percentage returns in each listing's local currency, without an FX conversion. Market holidays differ; the experiment engine must align actual shared sessions without forward filling and disclose that its horizon then means shared sessions.

Cached histories are retrospective vendor snapshots, not point-in-time archives. Corporate-action revisions, vendor errors, symbol changes, and regional coverage gaps remain limitations. Company notes stay limited to the 20 verified releases above; adding a symbol does not create an earnings narrative. Unsupported stock/date evidence must remain explicitly unavailable.

The original `prices.csv` bytes remain unchanged. Its provenance describes that frozen dataset and the current source-note eligibility policy. Global caches extend independently to the latest completed available dates; an explicit refresh preserves the old cache if a request fails. Existing cached symbols need no network, while provider-wide search and an uncached symbol's first download require internet. Underlying data remains subject to Yahoo Finance's terms and is provided here for educational use.
