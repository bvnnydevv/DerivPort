"""Global provider symbol search and persistent, adjusted historical-price caches.

The offline catalog is a starter list, not an exhaustive exchange directory.
Yahoo Finance coverage is broad but varies by exchange, symbol, and history.
Only downloads/searches use the network; existing cache reads never do.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
CACHE_DIR = DATA_DIR / 'market_cache'
CATALOG_PATH = DATA_DIR / 'market_catalog.csv'
BUNDLED_PATH = DATA_DIR / 'prices.csv'
BUNDLED_PROVENANCE_PATH = DATA_DIR / 'provenance.json'
BUNDLED_SYMBOLS = ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
_LAST_SEARCH_STATUS: dict = {}


class MarketDataError(ValueError):
    """An actionable market-data problem safe to display to a student."""


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-=]{0,31}', symbol):
        raise MarketDataError('Use a stock symbol such as AAPL, BRK-B, 0700.HK, or SAP.DE (letters, numbers, dots and hyphens).')
    return symbol


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def symbol_info(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    found = next((r for r in load_catalog() if r['symbol'] == symbol), None)
    return dict(found or {'symbol': symbol, 'name': symbol, 'exchange': 'Provider listing', 'currency': 'Unknown', 'region': 'Provider listing'})


def _cache_paths(symbol: str) -> tuple[Path, Path]:
    # Hash prevents collisions and path traversal while preserving readable symbols.
    safe = re.sub(r'[^A-Z0-9.-]', '_', symbol)
    stem = f'{safe}_{hashlib.sha256(symbol.encode()).hexdigest()[:8]}'
    return CACHE_DIR / f'{stem}.csv', CACHE_DIR / f'{stem}.json'


def _validate_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if not {'date', 'adjusted_close'}.issubset(frame.columns):
        raise MarketDataError(f'{symbol}: the local cache does not contain date and adjusted_close columns.')
    dates = pd.to_datetime(frame['date'], errors='coerce')
    values = pd.to_numeric(frame['adjusted_close'], errors='coerce')
    if dates.isna().any() or values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise MarketDataError(f'{symbol}: the cache contains invalid dates or prices. Refresh it to obtain verified positive adjusted prices.')
    result = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name='adjusted_close').sort_index()
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    if result.index.duplicated().any():
        raise MarketDataError(f'{symbol}: duplicate dates were found in its cache. Refresh this symbol.')
    result.index.name = 'date'
    if result.empty:
        raise MarketDataError(f'{symbol}: no adjusted daily prices are available.')
    return result


def verify_frozen_cache() -> str:
    """Verify the original frozen CSV against its recorded byte-level digest."""
    try:
        provenance = json.loads(BUNDLED_PROVENANCE_PATH.read_text())
        expected = provenance['cache_sha256']
        digest = hashlib.sha256(BUNDLED_PATH.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise MarketDataError('The frozen cache provenance could not be verified; no unverified prices were used.') from exc
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected) or digest != expected:
        raise MarketDataError('The frozen price cache checksum failed; restore the recorded dataset before using it.')
    return digest


def _publish_cache(csv_path: Path, meta_path: Path, series: pd.Series, meta: dict) -> None:
    """Publish a cache pair with rollback if either replacement fails.

    Each rename is atomic; readers reject the short intermediate pair through
    its checksum rather than accepting unverified data. A failed second rename
    restores the previous CSV, preserving the prior valid pair on disk.
    """
    previous = {path: path.read_bytes() if path.exists() else None for path in (csv_path, meta_path)}
    token = uuid4().hex
    temp_csv = csv_path.with_suffix('.csv.' + token + '.tmp')
    temp_json = meta_path.with_suffix('.json.' + token + '.tmp')
    replaced = []
    temporary = [temp_csv, temp_json]
    try:
        series.to_csv(temp_csv, float_format='%.17g')
        meta['cache_sha256'] = hashlib.sha256(temp_csv.read_bytes()).hexdigest()
        temp_json.write_text(json.dumps(meta, indent=2) + '\n')
        temp_csv.replace(csv_path)
        replaced.append(csv_path)
        temp_json.replace(meta_path)
        replaced.append(meta_path)
    except Exception as publication_error:
        try:
            for path in reversed(replaced):
                if previous[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    rollback = path.with_suffix(path.suffix + '.' + token + '.rollback.tmp')
                    temporary.append(rollback)
                    rollback.write_bytes(previous[path])
                    rollback.replace(path)
        except Exception as rollback_error:
            failure = MarketDataError('Cache publication failed and the previous disk cache could not be restored. Restore or refresh this stock before continuing.')
            failure.cache_preserved = False
            raise failure from rollback_error
        raise MarketDataError('The verified download could not be saved to the local cache. Previous files were restored where present; no partial new cache was retained.') from publication_error
    finally:
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_cached(symbol: str) -> tuple[pd.Series, dict] | None:
    csv_path, meta_path = _cache_paths(symbol)
    if csv_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if not isinstance(meta, dict):
                raise MarketDataError(f'{symbol}: cache metadata must be an object. Refresh this symbol.')
            if meta.get('symbol') != symbol:
                raise MarketDataError(f'{symbol}: cache metadata does not match the requested symbol.')
            expected = meta.get('cache_sha256')
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise MarketDataError(f'{symbol}: its cache has no valid recorded checksum. Refresh this symbol; no unverified prices were used.')
            if hashlib.sha256(csv_path.read_bytes()).hexdigest() != expected:
                raise MarketDataError(f'{symbol}: its cached file checksum failed. Refresh this symbol; no unverified prices were used.')
            series = _validate_series(pd.read_csv(csv_path), symbol)
            meta.update(cache_status='cached', start=series.index[0].date().isoformat(), end=series.index[-1].date().isoformat(), rows=len(series))
            return series, meta
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise MarketDataError(f'{symbol}: its local cache could not be read. Refresh this symbol.') from exc
    if symbol in BUNDLED_SYMBOLS and BUNDLED_PATH.exists():
        verify_frozen_cache()
        raw = pd.read_csv(BUNDLED_PATH)
        series = _validate_series(raw.loc[raw['ticker'] == symbol], symbol)
        meta = symbol_info(symbol)
        meta.update(source='Yahoo Finance via pinned Akma86 public mirror', source_url='https://github.com/Akma86/us-tech-giants-market-analytics', cache_status='bundled', fetched_at='2026-11-14', adjusted=True, price_field='Adj Close', start=str(series.index[0].date()), end=str(series.index[-1].date()), rows=len(series), point_in_time_archive=False)
        return series, meta
    return None


def read_cached_symbols() -> list[str]:
    symbols = set(BUNDLED_SYMBOLS if BUNDLED_PATH.exists() else ())
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob('*.json'):
            try:
                meta = json.loads(path.read_text())
                symbol = normalize_symbol(meta['symbol'])
                if _cache_paths(symbol)[0].exists():
                    symbols.add(symbol)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(symbols)


def get_search_status() -> dict:
    return dict(_LAST_SEARCH_STATUS)


def _provider():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MarketDataError('Online lookup requires yfinance. Install requirements.txt; cached stocks continue to work offline.') from exc
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Keep provider timezone/cookie databases inside the project cache as well.
    yf.set_tz_cache_location(str(CACHE_DIR / '_provider'))
    return yf


def _readable_error(symbol: str, exc: Exception) -> str:
    text = str(exc).lower()
    if 'rate' in text or '429' in text:
        return f'{symbol}: Yahoo Finance is rate-limiting requests. Try again later or use a cached stock.'
    if any(word in text for word in ('resolve', 'network', 'connection', 'timed out', 'timeout', 'curl:')):
        return f'{symbol}: the market-data provider could not be reached. Check internet access or use a cached stock.'
    return f'{symbol}: adjusted price history is unavailable from Yahoo Finance. Check the listing suffix and try another stock or a cached symbol.'


def search_symbols(query: str, online: bool = True) -> list[dict]:
    """Search starter names/symbols and, when online, Yahoo's global equity listings.

    A provider failure retains offline matches. Call get_search_status() to show
    whether the provider search succeeded; absence of a result is not proof a
    company is unlisted. A known Yahoo symbol may be entered directly for loading.
    """
    global _LAST_SEARCH_STATUS
    query = str(query).strip()[:120]
    normalized = query.upper()
    local = [dict(row, search_source='starter catalog') for row in load_catalog()
             if not query or normalized in row['symbol'] or query.casefold() in row['name'].casefold()]
    local.sort(key=lambda row: (row['symbol'] != normalized, not row['symbol'].startswith(normalized), row['symbol']))
    status = {'query': query, 'online_attempted': bool(query and online), 'online_ok': False, 'message': 'Offline starter catalog. Enter a provider symbol directly to load a listing outside this list.'}
    results = {row['symbol']: row for row in local}
    if query and online:
        try:
            yf = _provider()
            search = yf.Search(query, max_results=30, news_count=0, lists_count=0, recommended=0, timeout=12, raise_errors=True)
            for quote in search.quotes:
                if quote.get('quoteType', '').upper() != 'EQUITY':
                    continue
                try:
                    symbol = normalize_symbol(quote['symbol'])
                except (KeyError, MarketDataError):
                    continue
                catalog = symbol_info(symbol)
                results[symbol] = {'symbol': symbol, 'name': quote.get('longname') or quote.get('shortname') or catalog['name'], 'exchange': quote.get('exchDisp') or quote.get('exchange') or catalog['exchange'], 'currency': quote.get('currency') or catalog['currency'], 'region': catalog['region'], 'search_source': 'Yahoo Finance'}
            status.update(online_ok=True, message='Searched Yahoo Finance equity listings. Exchange and historical coverage vary; this is not a guaranteed directory of every listing.')
        except Exception as exc:
            status['message'] = _readable_error('Search', exc) + ' Showing any matching starter-catalog entries.'
    _LAST_SEARCH_STATUS = status
    return sorted(results.values(), key=lambda row: (row['symbol'] != normalized, not row['symbol'].startswith(normalized), row['symbol']))[:50]


def load_symbol_prices(symbol: str, refresh: bool = False) -> tuple[pd.Series, dict]:
    """Return real adjusted closes and provenance; use local data without network.

    Yahoo's full available daily history is requested for uncached symbols.
    Current UTC date and future bars are excluded conservatively. Missing days
    are not interpolated. Currency is informational: returns stay in local units.
    On a failed refresh, preserve and return the previous cache with a warning.
    """
    symbol = normalize_symbol(symbol)
    try:
        cached = _read_cached(symbol)
    except MarketDataError:
        if not refresh:
            raise
        cached = None  # Explicit refresh can repair a corrupt cache.
    if cached is not None and not refresh:
        return cached
    try:
        yf = _provider()
        ticker = yf.Ticker(symbol)
        history = ticker.history(period='max', interval='1d', auto_adjust=False, back_adjust=False, repair=False, keepna=False, actions=True, timeout=20, raise_errors=True)
        if history.empty or 'Adj Close' not in history.columns:
            raise MarketDataError(f'{symbol}: Yahoo did not provide adjusted closes for this listing. No unadjusted substitute was used.')
        # Work in exchange-local calendar dates, then remove incomplete/future days.
        dates = pd.DatetimeIndex(history.index).tz_localize(None).normalize()
        values = pd.to_numeric(history['Adj Close'], errors='coerce').to_numpy()
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        valid = (dates < today) & np.isfinite(values) & (values > 0)
        frame = pd.DataFrame({'date': dates[valid], 'adjusted_close': values[valid]})
        series = _validate_series(frame, symbol)
        if len(series) < 63:
            raise MarketDataError(f'{symbol}: only {len(series)} valid daily observations are available; at least 63 are required for a historical experiment.')
        provider_meta = ticker.history_metadata or {}
        meta = symbol_info(symbol)
        meta.update(name=provider_meta.get('longName') or provider_meta.get('shortName') or meta['name'], exchange=provider_meta.get('fullExchangeName') or provider_meta.get('exchangeName') or meta['exchange'], currency=provider_meta.get('currency') or meta['currency'], source='Yahoo Finance via yfinance', source_url=f'https://finance.yahoo.com/quote/{symbol}/history/', cache_status='downloaded', fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), start=str(series.index[0].date()), end=str(series.index[-1].date()), rows=len(series), adjusted=True, price_field='Adj Close', point_in_time_archive=False, observations_excluded=int(len(history)-len(series)), note='Retrospectively adjusted snapshot; current UTC date excluded; no interpolation or currency conversion.')
        csv_path, meta_path = _cache_paths(symbol)
        _publish_cache(csv_path, meta_path, series, meta)
        return _validate_series(pd.read_csv(csv_path), symbol), meta
    except Exception as exc:
        message = str(exc) if isinstance(exc, MarketDataError) else _readable_error(symbol, exc)
        if cached is not None and getattr(exc, 'cache_preserved', True):
            series, meta = cached
            return series, dict(meta, warning=message + ' The previous cached history was preserved.', refresh_failed=True)
        raise MarketDataError(message) from exc
