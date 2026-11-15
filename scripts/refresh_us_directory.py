"""Refresh the official current US equity directory; never download prices.

NasdaqTrader's two daily symbol files include securities other than stocks.
ETF/test flags and disclosed name-based instrument filters remove funds,
debt, warrants, rights and packaged SPAC units. Common/ordinary shares, ADRs,
preferred equity and operating-company partnership units remain eligible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import ssl
import sys
from urllib.request import Request, urlopen
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derivport.market_data import MarketDataError

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {name: 'https://www.nasdaqtrader.com/dynamic/symdir/' + name
    for name in ('nasdaqlisted.txt', 'otherlisted.txt')}
EXCHANGES = {'N': 'NYSE', 'A': 'NYSE American', 'P': 'NYSE Arca',
    'Z': 'Cboe', 'V': 'IEX', 'M': 'NYSE Texas', 'F': 'TXSE', 'L': 'LTSE'}
COLUMNS = ('symbol', 'name', 'exchange', 'currency', 'region', 'quoteType',
    'listing_symbol', 'source_file', 'security_type')
POLICY = ('Current listed equities only. Exclude ETF/NextShares/test flags, '
    'name-identified investment funds, notes/bonds/debentures, warrants, rights '
    'and packaged units. Keep common/ordinary shares, ADR/ADS/registry shares, '
    'preferred equity and common/preferred partnership units. Security-name '
    'classification is imperfect; provider equity identity and history remain '
    'subject to verification on first download. Not a historical security master.')


def _security_type(name: str) -> str | None:
    text = name.casefold()
    if re.search(r'\b(warrants?|rights?|debentures?|bonds?|etns?)\b|\bnotes?\s+(due|to|matur)|\b(senior|subordinated|unsecured|secured|rate|reset|perpetual)\s+notes?\b', text):
        return None
    if re.search(r'\b(closed[- ]end|exchange[- ]traded)\b|\b(etfs?|funds?)\b', text):
        return None
    if re.search(r'\bunits?\b', text) and not re.search(r'\b(common|preferred)\s+units?\b|units?\s+representing\s+(limited|partnership)', text):
        return None
    if 'preferred' in text or 'preference' in text:
        return 'Preferred equity'
    if re.search(r'\b(american deposit[ao]ry|ads\b|adss\b|adrs?\b|registry shares)', text):
        return 'Depositary / registry shares'
    if re.search(r'\bunits?\b', text):
        return 'Partnership equity units'
    return 'Common / ordinary equity'


def provider_symbol(listing_symbol: str) -> str:
    """Map official US share-class and preferred suffixes to Yahoo symbols."""
    symbol = listing_symbol.strip().upper()
    if '$' in symbol:
        root, series = symbol.split('$', 1)
        symbol = root + '-P' + series
    else:
        symbol = symbol.replace('.', '-')
    if not re.fullmatch(r'[A-Z][A-Z0-9-]{0,15}', symbol):
        raise MarketDataError(f'Unsupported provider alias for official symbol {listing_symbol!r}.')
    return symbol


def parse_directory(text: str, filename: str) -> tuple[list[dict], dict]:
    """Parse a complete pipe-delimited source, retaining honest source identity."""
    if filename not in SOURCES:
        raise MarketDataError('Unknown official directory source.')
    reader = csv.DictReader(io.StringIO(text), delimiter='|')
    symbol_field = 'Symbol' if filename == 'nasdaqlisted.txt' else 'ACT Symbol'
    required = {symbol_field, 'Security Name', 'Test Issue', 'ETF'}
    if filename == 'otherlisted.txt':
        required.add('Exchange')
    if not required.issubset(reader.fieldnames or []):
        raise MarketDataError(f'{filename}: official directory columns are missing.')
    results, excluded = [], Counter()
    footer, source_rows = None, 0
    for row in reader:
        listing = row.get(symbol_field, '').strip()
        if listing.startswith('File Creation Time:'):
            if footer is not None:
                raise MarketDataError(f'{filename}: repeated source footer.')
            footer = listing.removeprefix('File Creation Time:').strip()
            continue
        if footer is not None:
            raise MarketDataError(f'{filename}: data appeared after the source footer.')
        if not listing:
            continue
        source_rows += 1
        if row.get('Test Issue') not in ('N', 'Y') or row.get('ETF') not in ('N', 'Y'):
            raise MarketDataError(f'{filename}: an instrument eligibility flag is invalid.')
        if row['Test Issue'] == 'Y':
            excluded['test_issue'] += 1
            continue
        if row['ETF'] == 'Y' or row.get('NextShares') == 'Y':
            excluded['etf_or_nextshares'] += 1
            continue
        name = row['Security Name'].strip()
        security_type = _security_type(name)
        if not name or security_type is None:
            excluded['nonstock_name_filter'] += 1
            continue
        venue = 'NASDAQ' if filename == 'nasdaqlisted.txt' else EXCHANGES.get(row['Exchange'])
        if venue is None:
            raise MarketDataError(f'{filename}: unknown live listing venue {row["Exchange"]!r}; update the verified exchange map before refreshing.')
        try:
            symbol = provider_symbol(listing)
        except MarketDataError:
            excluded['unsupported_provider_alias'] += 1
            continue
        results.append(dict(symbol=symbol, name=name, exchange=venue,
            currency='Unknown', region='United States listing', quoteType='EQUITY',
            listing_symbol=listing, source_file=filename, security_type=security_type))
    if footer is None:
        raise MarketDataError(f'{filename}: official file-creation footer is missing; incomplete source rejected.')
    try:
        source_time = datetime.strptime(footer, '%m%d%Y%H:%M').isoformat(timespec='minutes')
    except ValueError as exc:
        raise MarketDataError(f'{filename}: source file-creation timestamp is invalid.') from exc
    if len({r['symbol'] for r in results}) != len(results):
        raise MarketDataError(f'{filename}: provider aliases collided; no directory was published.')
    return results, dict(source_rows=source_rows, included_rows=len(results),
        excluded=dict(excluded), file_creation_time=source_time,
        file_creation_timezone='Unspecified by the source file')


def build_snapshot(source_bytes: dict[str, bytes], retrieved_at: str | None = None) -> tuple[bytes, dict]:
    rows, sources = [], []
    if set(source_bytes) != set(SOURCES):
        raise MarketDataError('Both official listing files are required.')
    for name, url in SOURCES.items():
        raw = source_bytes[name]
        try:
            parsed, details = parse_directory(raw.decode('utf-8-sig'), name)
        except UnicodeDecodeError as exc:
            raise MarketDataError(f'{name}: source is not valid UTF-8 text.') from exc
        rows.extend(parsed)
        sources.append(dict(filename=name, url=url, sha256=hashlib.sha256(raw).hexdigest(), **details))
    # A symbol has one canonical current primary venue; don't silently choose
    # between contradicting source identities or provider aliases.
    unique = {}
    for row in rows:
        if row['symbol'] in unique and unique[row['symbol']] != row:
            raise MarketDataError(f'{row["symbol"]}: conflicting current listings; refresh was not published.')
        unique[row['symbol']] = row
    rows = sorted(unique.values(), key=lambda r: (r['exchange'], r['symbol']))
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    raw = output.getvalue().encode()
    provenance = dict(schema_version=1, retrieved_at_utc=retrieved_at or datetime.now(timezone.utc).isoformat(timespec='seconds'),
        directory_sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
        exchange_counts=dict(sorted(Counter(r['exchange'] for r in rows).items())),
        security_type_counts=dict(sorted(Counter(r['security_type'] for r in rows).items())),
        sources=sources, policy=POLICY,
        scope='Current NasdaqTrader Nasdaq and other-exchange US listings, after disclosed instrument filters. No OTC, delisted-security or historical-constituent archive; vendor price/alias coverage is not guaranteed.',
        source_definitions='https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs',
        additional_exchange_definitions=[
            'https://www.nasdaqtrader.com/TraderNews.aspx?id=DTN2026-10',
            'https://www.nyse.com/publicdocs/nyse/data/NYSE_Texas_Support_of_Primary_Listings.pdf'],
        currency_policy='The source directory has no currency field. Currency remains Unknown until the price provider supplies it.')
    return raw, provenance


def publish_snapshot(raw: bytes, provenance: dict, directory: Path) -> None:
    """Publish the verified pair, rolling back both prior files on failure."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / 'us_listings.csv', directory / 'us_listings_provenance.json']
    if hashlib.sha256(raw).hexdigest() != provenance.get('directory_sha256'):
        raise MarketDataError('Directory bytes do not match their provenance digest.')
    previous = {p: p.read_bytes() if p.exists() else None for p in paths}
    token = uuid4().hex
    temps = [p.with_name(p.name + '.' + token + '.tmp') for p in paths]
    replaced = []
    try:
        temps[0].write_bytes(raw)
        temps[1].write_text(json.dumps(provenance, indent=2) + '\n')
        for temp, path in zip(temps, paths):
            temp.replace(path)
            replaced.append(path)
    except Exception as publication_error:
        try:
            for path in reversed(replaced):
                if previous[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    rollback = path.with_name(path.name + '.' + token + '.rollback.tmp')
                    temps.append(rollback)
                    rollback.write_bytes(previous[path])
                    rollback.replace(path)
        except Exception as rollback_error:
            raise MarketDataError('US directory publication and rollback failed; restore the last verified directory pair.') from rollback_error
        raise MarketDataError('US directory refresh could not be saved; previous files were restored.') from publication_error
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def download_sources() -> dict[str, bytes]:
    # certifi is included with the project's provider dependencies; using its
    # trusted CA roots also supports Python.org macOS installs without local CAs.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    output = {}
    for name, url in SOURCES.items():
        with urlopen(Request(url, headers={'User-Agent': 'DerivPort educational stock directory'}), context=context, timeout=30) as response:
            output[name] = response.read(5_000_001)
        if len(output[name]) > 5_000_000:
            raise MarketDataError(f'{name}: directory source exceeded its expected size.')
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, help='Parse previously downloaded official source files instead of making requests.')
    args = parser.parse_args(argv)
    try:
        sources = ({name: (args.source_directory / name).read_bytes() for name in SOURCES}
            if args.source_directory else download_sources())
        raw, provenance = build_snapshot(sources)
        if provenance['rows'] < 2500 or min(s['source_rows'] for s in provenance['sources']) < 1000:
            raise MarketDataError('Unexpectedly small US directory; no previous snapshot was replaced.')
        publish_snapshot(raw, provenance, ROOT / 'data')
        print(f"Published {provenance['rows']:,} current US equity listings: {provenance['exchange_counts']}")
        print('Listing directory only; price history is loaded separately on demand.')
        return 0
    except Exception as exc:
        print(f'US directory refresh failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
