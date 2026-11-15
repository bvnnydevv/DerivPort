"""US-listed equity scope for the explorer, without changing research caches.

Listing venue determines eligibility, rather than issuer domicile or USD alone.
The bundled current directory comes from official NasdaqTrader listing files;
it is not a historical security master or a guarantee of provider price coverage.
Existing international APIs remain available for reproducing historical research.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
from uuid import uuid4

from derivport import market_data
from derivport.market_data import MarketDataError

US_EXCHANGES = ('NASDAQ', 'NYSE', 'NYSE American', 'NYSE Arca', 'Cboe', 'IEX', 'TXSE', 'NYSE Texas', 'LTSE')
DIRECTORY_PATH = Path(__file__).resolve().parents[1] / 'data' / 'us_listings.csv'
DIRECTORY_PROVENANCE_PATH = DIRECTORY_PATH.with_name('us_listings_provenance.json')

# Provider codes/names are matched exactly after punctuation/spacing removal.
# Do not use a NASDAQ/NYSE prefix match: foreign Nasdaq markets are not US venues.
_EXCHANGE_ALIASES = {
    'NASDAQ': 'NASDAQ', 'NAS': 'NASDAQ', 'NMS': 'NASDAQ',
    'NGM': 'NASDAQ', 'NCM': 'NASDAQ', 'NASDAQGS': 'NASDAQ',
    'NASDAQGM': 'NASDAQ', 'NASDAQCM': 'NASDAQ',
    'NASDAQGLOBALSELECT': 'NASDAQ', 'NASDAQGLOBALSELECTMARKET': 'NASDAQ',
    'NASDAQGLOBALMARKET': 'NASDAQ', 'NASDAQCAPITALMARKET': 'NASDAQ',
    'NASDAQSTOCKEXCHANGE': 'NASDAQ',
    'NYSE': 'NYSE', 'NYQ': 'NYSE', 'NEWYORKSTOCKEXCHANGE': 'NYSE',
    'AMEX': 'NYSE American', 'ASE': 'NYSE American',
    'NYSEAMERICAN': 'NYSE American', 'NYSEMKT': 'NYSE American',
    'PCX': 'NYSE Arca', 'ARCA': 'NYSE Arca', 'NYSEARCA': 'NYSE Arca',
    'BATS': 'Cboe', 'BTS': 'Cboe', 'CBOE': 'Cboe',
    'CBOEBZX': 'Cboe', 'CBOEUS': 'Cboe',
    'IEX': 'IEX', 'IEXG': 'IEX', 'INVESTORSEXCHANGE': 'IEX',
    'TXSE': 'TXSE', 'TEXASSTOCKEXCHANGE': 'TXSE',
    'NYSETEXAS': 'NYSE Texas', 'NYSECHICAGO': 'NYSE Texas',
    'LTSE': 'LTSE', 'LONGTERMSTOCKEXCHANGE': 'LTSE',
}


def canonical_us_exchange(value: str) -> str | None:
    return _EXCHANGE_ALIASES.get(re.sub(r'[^A-Z0-9]', '', str(value).upper()))


def _us_row(row: dict, *, assume_equity: bool = False) -> dict | None:
    """Canonicalize an evidenced equity row; reject unknown venues/types.

    ``assume_equity`` is only for the curated stock catalog and the existing
    provider search API, which already filters quoteType=EQUITY upstream.
    """
    if not isinstance(row, dict):
        return None
    try:
        symbol = market_data.normalize_symbol(row.get('symbol', ''))
    except MarketDataError:
        return None
    # Yahoo uses hyphens for US share classes. Dotted suffix listings and
    # derivatives/currency syntax must not enter via an incorrect venue label.
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,15}', symbol):
        return None
    exchange = canonical_us_exchange(row.get('exchange') or row.get('fullExchangeName') or row.get('exchangeName'))
    quote_type = str(row.get('quoteType') or row.get('instrumentType') or '').upper()
    if quote_type and quote_type != 'EQUITY':
        return None
    if not quote_type and not assume_equity:
        return None
    currency = str(row.get('currency') or '').upper()
    if currency not in ('', 'UNKNOWN', 'USD') or exchange is None:
        return None
    return dict(row, symbol=symbol, exchange=exchange, exchange_group=exchange,
                quoteType='EQUITY', currency='USD' if currency == 'USD' else 'Unknown')


def filter_us_listings(rows, exchange: str | None = None, *, assume_equity: bool = False) -> list[dict]:
    """Filter stock rows into a known US venue; USD alone never qualifies."""
    if exchange is not None and exchange not in US_EXCHANGES:
        raise MarketDataError('Choose one of the supported US exchanges.')
    results = {}
    for row in rows:
        found = _us_row(row, assume_equity=assume_equity)
        if found and (exchange is None or found['exchange'] == exchange):
            results[found['symbol']] = found
    order = {name: i for i, name in enumerate(US_EXCHANGES)}
    return sorted(results.values(), key=lambda row: (order[row['exchange']], row['symbol']))


@lru_cache(maxsize=2)
def _snapshot_rows(raw: bytes) -> tuple[dict, ...]:
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    required = {'symbol', 'name', 'exchange', 'currency', 'quoteType', 'listing_symbol', 'source_file', 'security_type'}
    if not rows or not required.issubset(rows[0]) or len({r['symbol'] for r in rows}) != len(rows):
        raise MarketDataError('The US directory has an invalid schema or duplicate symbols. Refresh its verified snapshot.')
    for row in rows:
        if not row['name'] or row['quoteType'] != 'EQUITY' or row['source_file'] not in ('nasdaqlisted.txt', 'otherlisted.txt') or _us_row(row) is None:
            raise MarketDataError('The US directory contains an invalid listing. Restore or refresh its verified snapshot.')
    return tuple(rows)


def _verified_snapshot() -> tuple[bytes, dict]:
    try:
        raw = DIRECTORY_PATH.read_bytes()
        provenance = json.loads(DIRECTORY_PROVENANCE_PATH.read_text())
        if provenance.get('schema_version') != 1 or not isinstance(provenance.get('directory_sha256'), str):
            raise ValueError('Invalid metadata schema')
        if hashlib.sha256(raw).hexdigest() != provenance['directory_sha256']:
            raise ValueError('Checksum mismatch')
        if len(_snapshot_rows(raw)) != provenance.get('rows'):
            raise ValueError('Snapshot row count mismatch')
        return raw, provenance
    except (OSError, ValueError, UnicodeDecodeError, TypeError, AttributeError) as exc:
        raise MarketDataError('The US listing directory could not be verified. Restore the bundled CSV/provenance pair or run scripts/refresh_us_directory.py.') from exc


def directory_provenance() -> dict:
    """Return verified current snapshot metadata, without contacting a provider."""
    return _verified_snapshot()[1]


def us_catalog(exchange: str | None = None) -> list[dict]:
    if not DIRECTORY_PATH.exists() and not DIRECTORY_PROVENANCE_PATH.exists():
        # Source checkouts predating the directory retain a small offline base;
        # a damaged or partially present snapshot never silently falls back.
        return filter_us_listings(market_data.load_catalog(), exchange, assume_equity=True)
    raw, _ = _verified_snapshot()
    return filter_us_listings(_snapshot_rows(raw), exchange)


def _cached_metadata(symbol: str) -> dict | None:
    csv_path, meta_path = market_data._cache_paths(symbol)
    if not csv_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        return meta if isinstance(meta, dict) and meta.get('symbol') == symbol else None
    except (OSError, ValueError):
        return None


def cached_us_listings(exchange: str | None = None) -> list[dict]:
    """List cached US equities from metadata only; never contact a provider.

    Curated stock identities support legacy caches without instrument types.
    Unknown legacy cached instruments require explicit online verification.
    Price checksums remain enforced by the normal price loader on selection.
    """
    catalog = {row['symbol']: row for row in us_catalog()}
    rows = []
    for symbol in market_data.read_cached_symbols():
        meta = _cached_metadata(symbol)
        known = catalog.get(symbol)
        if meta is not None:
            row = _us_row(meta, assume_equity=known is not None)
        else:
            row = known
        if row:
            rows.append(row)
    return filter_us_listings(rows, exchange)


def search_us_symbols(query: str, online: bool = True, exchange: str | None = None) -> list[dict]:
    # search_symbols excludes funds, futures, indices and currencies upstream.
    if online:
        return filter_us_listings(market_data.search_symbols(query, online), exchange, assume_equity=True)
    normalized = str(query).strip().casefold()
    rows = [row for row in us_catalog(exchange) if not normalized
        or normalized in row['symbol'].casefold() or normalized in row['name'].casefold()]
    return sorted(rows, key=lambda row: (row['symbol'].casefold() != normalized,
        not row['symbol'].casefold().startswith(normalized), row['exchange'], row['symbol']))


def resolve_us_listing(symbol: str, metadata: dict | None = None) -> dict:
    """Verify explicit additions as US-listed equities before downloading.

    Catalog/previously verified caches are offline. Unknown symbols require a
    provider instrument type and recognized US venue, not guessed geography.
    """
    symbol = market_data.normalize_symbol(symbol)
    catalog = next((row for row in us_catalog() if row['symbol'] == symbol), None)
    curated = next((row for row in filter_us_listings(market_data.load_catalog(), assume_equity=True) if row['symbol'] == symbol), None)
    candidate = dict(metadata) if metadata is not None else _cached_metadata(symbol)
    if candidate is not None:
        candidate['symbol'] = symbol
        found = _us_row(candidate, assume_equity=catalog is not None)
        if found:
            return found
        # An explicitly supplied incompatible venue/type must not fall through
        # to a catalog identity and accidentally approve a non-equity.
        if metadata is not None:
            raise MarketDataError(f'{symbol}: choose a US-listed stock on a supported exchange. Funds and international listings are excluded.')
    elif catalog is not None and (curated is not None or not catalog.get('source_file')):
        return catalog
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,15}', symbol):
        raise MarketDataError('Only US-listed stock symbols are available here, such as AAPL or BRK-B.')
    try:
        info = market_data._provider().Ticker(symbol).get_info()
        row = dict(info, symbol=symbol, name=info.get('longName') or info.get('shortName') or symbol,
                   exchange=info.get('fullExchangeName') or info.get('exchange'))
        found = _us_row(row)
        if found:
            return found
    except Exception as exc:
        raise MarketDataError(f'{symbol}: its US stock listing could not be verified. Search for a supported US listing or use a cached stock.') from exc
    raise MarketDataError(f'{symbol}: only equities listed on supported US exchanges can be added. Funds, OTC and international listings are excluded.')


def load_us_symbol_prices(symbol: str, refresh: bool = False):
    """Load a verified US equity and retain type evidence for offline reuse."""
    listing = resolve_us_listing(symbol)
    series, meta = market_data.load_symbol_prices(listing['symbol'], refresh=refresh)
    # The downloaded venue must remain eligible, even if a symbol changed venue.
    verified = resolve_us_listing(listing['symbol'], dict(meta, quoteType='EQUITY'))
    result = dict(meta, quoteType='EQUITY', exchange=verified['exchange'],
                  exchange_group=verified['exchange'],
                  source_exchange=meta.get('source_exchange') or meta.get('exchange'))
    # Save only identity/type metadata; original price bytes and their checksum
    # are untouched. Atomic replacement preserves the prior metadata on error.
    _, meta_path = market_data._cache_paths(listing['symbol'])
    if meta_path.exists() and result != meta:
        temporary = meta_path.with_suffix('.json.' + uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(meta_path)
        except OSError as exc:
            raise MarketDataError(f'{listing["symbol"]}: prices were retained, but listing verification could not be saved. Try again.') from exc
        finally:
            temporary.unlink(missing_ok=True)
    return series, result
