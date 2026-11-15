"""Real daily OHLCV chart data, kept separate from adjusted research prices.

Provider OHLC is never inferred from closing prices. Downloads are explicit;
verified cached candles are available offline and never alter model inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

import numpy as np
import pandas as pd

from .market_data import MarketDataError, normalize_symbol, symbol_info

CANDLE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'candle_cache'
COLUMNS = ('open', 'high', 'low', 'close', 'volume')
PRICE_BASIS = 'Vendor daily OHLC; auto_adjust=False, back_adjust=False; not dividend-adjusted.'
_LOCKS = {}
_LOCKS_GUARD = Lock()


@contextmanager
def _cache_lock(csv_path: Path):
    """Serialize pair readers/writers, including separate macOS/Linux processes."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(str(csv_path.resolve()), Lock())
    with lock:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.with_suffix('.lock').open('a+b') as handle:
            try:
                import fcntl
            except ImportError:  # Windows retains in-process protection.
                fcntl = None
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _paths(symbol: str) -> tuple[Path, Path]:
    safe = re.sub(r'[^A-Z0-9.-]', '_', symbol)
    stem = f'{safe}_{hashlib.sha256(symbol.encode()).hexdigest()[:8]}'
    return CANDLE_DIR / f'{stem}.csv', CANDLE_DIR / f'{stem}.json'


def _validate_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reject incomplete/invalid bars rather than inventing or repairing OHLC."""
    if 'date' not in frame.columns or not set(COLUMNS).issubset(frame.columns):
        raise MarketDataError(f'{symbol}: candle data needs date, open, high, low, close and volume columns.')
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame['date'], format='ISO8601', errors='raise'))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        if dates.isna().any() or not dates.equals(dates.normalize()) or dates.duplicated().any():
            raise ValueError('Dates must be unique daily observations')
        values = frame.loc[:, COLUMNS].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Missing or nonfinite candle values')
        if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError('OHLC must be positive and volume nonnegative')
        o, h, low, c = (values[:, i] for i in range(4))
        tolerance = h * 1e-10
        if ((low > np.minimum(o, c) + tolerance) | (h < np.maximum(o, c) - tolerance) | (low > h + tolerance)).any():
            raise ValueError('OHLC bounds are inconsistent')
        if (dates >= pd.Timestamp(datetime.now(timezone.utc).date())).any():
            raise ValueError('Current UTC date and future sessions are not completed bars')
    except (ValueError, TypeError, OverflowError) as exc:
        raise MarketDataError(f'{symbol}: candle data contains invalid dates, OHLC bounds, prices or volume. Refresh the candle history.') from exc
    result = pd.DataFrame(values, index=dates, columns=COLUMNS).sort_index()
    result.index.name = 'date'
    return result


def _read_cached(symbol: str) -> tuple[pd.DataFrame, dict] | None:
    csv_path, meta_path = _paths(symbol)
    with _cache_lock(csv_path):
        return _read_pair(csv_path, meta_path, symbol)


def _read_pair(csv_path: Path, meta_path: Path, symbol: str) -> tuple[pd.DataFrame, dict] | None:
    if not csv_path.exists() and not meta_path.exists():
        return None
    if not csv_path.exists() or not meta_path.exists():
        raise MarketDataError(f'{symbol}: the candle cache is incomplete. Download or refresh candles to restore it.')
    try:
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict) or meta.get('symbol') != symbol or meta.get('schema_version') != 1:
            raise MarketDataError(f'{symbol}: candle metadata does not match this listing or schema. Refresh candles.')
        expected = meta.get('cache_sha256')
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected) or hashlib.sha256(csv_path.read_bytes()).hexdigest() != expected:
            raise MarketDataError(f'{symbol}: the candle cache checksum failed. Refresh candles; unverified bars were not used.')
        if meta.get('columns') != list(COLUMNS) or any(meta.get(flag) is not False for flag in ('auto_adjust', 'back_adjust', 'repair')):
            raise MarketDataError(f'{symbol}: the candle price basis is not verified. Refresh candles.')
        bars = _validate_frame(pd.read_csv(csv_path), symbol)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise MarketDataError(f'{symbol}: the candle cache could not be read. Refresh candles.') from exc
    return bars, dict(meta, cache_status='cached', start=str(bars.index[0].date()), end=str(bars.index[-1].date()), rows=len(bars))


def _publish(csv_path: Path, meta_path: Path, bars: pd.DataFrame, meta: dict) -> None:
    """Publish a checksummed pair; restore both prior files on failed replacement."""
    with _cache_lock(csv_path):
        _publish_pair(csv_path, meta_path, bars, meta)


def _publish_pair(csv_path: Path, meta_path: Path, bars: pd.DataFrame, meta: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {p: p.read_bytes() if p.exists() else None for p in (csv_path, meta_path)}
    token = uuid4().hex
    temp_csv = csv_path.with_suffix('.csv.' + token + '.tmp')
    temp_meta = meta_path.with_suffix('.json.' + token + '.tmp')
    replaced, temporary = [], [temp_csv, temp_meta]
    try:
        bars.to_csv(temp_csv, float_format='%.17g')
        meta['cache_sha256'] = hashlib.sha256(temp_csv.read_bytes()).hexdigest()
        temp_meta.write_text(json.dumps(meta, indent=2) + '\n')
        temp_csv.replace(csv_path)
        replaced.append(csv_path)
        temp_meta.replace(meta_path)
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
            failure = MarketDataError(f'{meta["symbol"]}: candle publication and restoration failed; restore or refresh its candle cache.')
            failure.cache_preserved = False
            raise failure from rollback_error
        raise MarketDataError('Candle history could not be saved; the previous cache was restored where present.') from publication_error
    finally:
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _provider():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise MarketDataError('Candle downloads require yfinance; existing candle caches still work offline.') from exc
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(CANDLE_DIR / '_provider'))
    return yf


def load_candles(symbol: str, refresh: bool = False, allow_download: bool = False) -> tuple[pd.DataFrame, dict]:
    """Load verified OHLCV, downloading only when explicitly requested.

    ``allow_download=True`` permits an uncached listing; ``refresh=True`` is an
    explicit refresh request. A failed refresh returns a valid prior cache with
    a warning. Provider current-date/future bars are excluded conservatively.
    """
    symbol = normalize_symbol(symbol)
    try:
        cached = _read_cached(symbol)
    except MarketDataError:
        if not refresh:
            raise
        cached = None
    if cached is not None and not refresh:
        return cached
    if not (allow_download or refresh):
        raise MarketDataError(f'{symbol}: daily candles have not been downloaded. Use Download candles to fetch real OHLCV history; no synthetic candles are shown.')
    try:
        yf = _provider()
        ticker = yf.Ticker(symbol)
        history = ticker.history(period='max', interval='1d', auto_adjust=False, back_adjust=False,
            repair=False, keepna=True, actions=False, timeout=20, raise_errors=True)
        if history.empty or not {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(history.columns):
            raise MarketDataError(f'{symbol}: the provider did not return complete daily OHLCV history.')
        dates = pd.DatetimeIndex(history.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        completed = dates < pd.Timestamp(datetime.now(timezone.utc).date())
        frame = history.loc[completed, ['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        frame.columns = COLUMNS
        frame.insert(0, 'date', dates[completed])
        bars = _validate_frame(frame, symbol)
        provider_meta = ticker.history_metadata or {}
        meta = symbol_info(symbol)
        meta.update(schema_version=1, columns=list(COLUMNS), source='Yahoo Finance via yfinance',
            source_url=f'https://finance.yahoo.com/quote/{symbol}/history/',
            name=provider_meta.get('longName') or provider_meta.get('shortName') or meta['name'],
            exchange=provider_meta.get('fullExchangeName') or provider_meta.get('exchangeName') or meta['exchange'],
            currency=provider_meta.get('currency') or meta['currency'],
            instrument_type=provider_meta.get('instrumentType', 'Unknown'),
            exchange_timezone=provider_meta.get('exchangeTimezoneName', 'Unknown'),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            start=str(bars.index[0].date()), end=str(bars.index[-1].date()), rows=len(bars),
            observations_excluded=int((~completed).sum()), cache_status='downloaded',
            auto_adjust=False, back_adjust=False, repair=False, price_basis=PRICE_BASIS,
            point_in_time_archive=False,
            note='Chart-only vendor snapshot; split adjustments may be reflected by Yahoo. Current UTC date excluded; no missing-day filling or synthetic candles. Research uses its separate adjusted-close cache.')
        csv_path, meta_path = _paths(symbol)
        _publish(csv_path, meta_path, bars, meta)
        verified = _read_cached(symbol)
        return verified[0], dict(verified[1], cache_status='downloaded')
    except Exception as exc:
        message = str(exc) if isinstance(exc, MarketDataError) else f'{symbol}: candle history could not be downloaded from Yahoo Finance. Check the connection or try again later.'
        if cached is not None and getattr(exc, 'cache_preserved', True):
            return cached[0], dict(cached[1], warning=message + ' The previous candle history was preserved.', refresh_failed=True)
        raise MarketDataError(message) from exc
