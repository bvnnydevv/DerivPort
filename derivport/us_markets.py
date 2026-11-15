"""US-listed equity scope for the explorer, without changing research caches.

Listing venue determines eligibility, rather than issuer domicile or USD alone.
The catalog is a starter list; it is not an exhaustive security master. Existing
international price APIs remain available for reproducing historical research.
"""
from __future__ import annotations

import json
import re
from uuid import uuid4

from derivport import market_data
from derivport.market_data import MarketDataError

US_EXCHANGES = ('NASDAQ', 'NYSE', 'NYSE American', 'NYSE Arca', 'Cboe', 'IEX')

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


def us_catalog(exchange: str | None = None) -> list[dict]:
    return filter_us_listings(market_data.load_catalog(), exchange, assume_equity=True)


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
    return filter_us_listings(market_data.search_symbols(query, online), exchange, assume_equity=True)


def resolve_us_listing(symbol: str, metadata: dict | None = None) -> dict:
    """Verify explicit additions as US-listed equities before downloading.

    Catalog/previously verified caches are offline. Unknown symbols require a
    provider instrument type and recognized US venue, not guessed geography.
    """
    symbol = market_data.normalize_symbol(symbol)
    catalog = next((row for row in us_catalog() if row['symbol'] == symbol), None)
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
    elif catalog is not None:
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
