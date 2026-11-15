"""Current US security discovery is broad, source-verified and price-independent."""
import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from derivport import market_data, us_markets
from scripts import refresh_us_directory as refresh


NASDAQ_HEADER = 'Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n'
OTHER_HEADER = 'ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n'


def sample_sources():
    return {
        'nasdaqlisted.txt': (NASDAQ_HEADER
            + 'AAA|AAA Inc. - Common Stock|Q|N|N|100|N|N\n'
            + 'ADR|ADR plc - American Depositary Shares|G|N|N|100|N|N\n'
            + 'SPACU|SPAC Inc. - Units|G|N|N|100|N|N\n'
            + 'SPACW|SPAC Inc. - Warrants|G|N|N|100|N|N\n'
            + 'FUND|Investment Trust - Closed End Fund|Q|N|N|100|N|N\n'
            + 'BOND|Issuer - 8% Senior Notes Due 2030|Q|N|N|100|N|N\n'
            + 'TEST|Test Common Stock|Q|Y|N|100|N|N\n'
            + 'ETF|Index Fund|Q|N|N|100|Y|N\n'
            + 'File Creation Time: 1113202621:31|||||||\n').encode(),
        'otherlisted.txt': (OTHER_HEADER
            + 'BRK.B|Berkshire Hathaway Class B Common Stock|N|BRK.B|N|100|N|BRK.B\n'
            + 'BAC$A|Bank Series A Preferred Stock|N|BACpA|N|100|N|BACpA\n'
            + 'LP|Energy LP Common Units|F|LP|N|100|N|LP\n'
            + 'ORD|Foreign Corporation Ordinary Shares|A|ORD|N|100|N|ORD\n'
            + 'PACK.U|Acquisition Units, each consisting of a share and a warrant|N|PACK.U|N|100|N|PACK.U\n'
            + 'RIGHT.R|Issuer Rights|N|RIGHT.R|N|100|N|RIGHT.R\n'
            + 'ETN|Index ETN|P|ETN|N|100|N|ETN\n'
            + 'File Creation Time: 1113202621:31|||||||\n').encode(),
    }


def decoded(raw):
    return list(csv.DictReader(io.StringIO(raw.decode())))


def test_official_flags_type_filters_aliases_venues_and_unknown_currency():
    raw, metadata = refresh.build_snapshot(sample_sources(), '2026-11-15T06:00:00+00:00')
    rows = {row['symbol']: row for row in decoded(raw)}
    assert set(rows) == {'AAA', 'ADR', 'BRK-B', 'BAC-PA', 'LP', 'ORD'}
    assert rows['BRK-B']['listing_symbol'] == 'BRK.B'
    assert rows['BAC-PA']['security_type'] == 'Preferred equity'
    assert rows['LP']['exchange'] == 'TXSE'
    assert rows['ORD']['exchange'] == 'NYSE American'
    assert rows['ADR']['security_type'] == 'Depositary / registry shares'
    assert {row['currency'] for row in rows.values()} == {'Unknown'}
    assert metadata['rows'] == 6
    assert metadata['directory_sha256'] == hashlib.sha256(raw).hexdigest()
    assert metadata['sources'][0]['file_creation_time'] == '2026-11-13T21:31'
    assert metadata['sources'][0]['sha256'] == hashlib.sha256(sample_sources()['nasdaqlisted.txt']).hexdigest()
    assert metadata['sources'][0]['excluded']['test_issue'] == 1


@pytest.mark.parametrize('damage', ['missing_footer', 'bad_footer', 'bad_flag', 'missing_columns', 'unknown_venue', 'alias_collision'])
def test_partial_or_ambiguous_directory_does_not_publish(damage):
    sources = sample_sources()
    text = sources['otherlisted.txt'].decode()
    if damage == 'missing_footer':
        text = text.split('File Creation Time:')[0]
    elif damage == 'bad_footer':
        text = text.replace('1113202621:31', 'not-a-date')
    elif damage == 'bad_flag':
        text = text.replace('BRK.B|N|100|N|BRK.B', 'BRK.B|?|100|N|BRK.B')
    elif damage == 'missing_columns':
        text = text.replace('ACT Symbol', 'Unknown field')
    elif damage == 'unknown_venue':
        text = text.replace('|F|LP|', '|?|LP|')
    else:
        text = text.replace('File Creation Time:', 'BRK-B|Another Company Common Stock|N|BRK-B|N|100|N|BRK-B\nFile Creation Time:')
    sources['otherlisted.txt'] = text.encode()
    with pytest.raises(market_data.MarketDataError):
        refresh.build_snapshot(sources)


def use_fixture(tmp_path, monkeypatch):
    raw, metadata = refresh.build_snapshot(sample_sources())
    refresh.publish_snapshot(raw, metadata, tmp_path)
    monkeypatch.setattr(us_markets, 'DIRECTORY_PATH', tmp_path / 'us_listings.csv')
    monkeypatch.setattr(us_markets, 'DIRECTORY_PROVENANCE_PATH', tmp_path / 'us_listings_provenance.json')
    return raw, metadata


def test_catalog_and_unlimited_offline_search_use_verified_snapshot(tmp_path, monkeypatch):
    use_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(market_data, '_provider', lambda: pytest.fail('Directory browsing must be offline'))
    assert len(us_markets.us_catalog()) == 6
    assert {r['symbol'] for r in us_markets.us_catalog('NYSE')} == {'BRK-B', 'BAC-PA'}
    assert [r['symbol'] for r in us_markets.search_us_symbols('Berkshire', online=False)] == ['BRK-B']
    assert len(us_markets.search_us_symbols('', online=False)) == 6
    assert us_markets.directory_provenance()['rows'] == 6


@pytest.mark.parametrize('damage', ['csv', 'missing_csv', 'missing_metadata', 'count', 'invalid_row'])
def test_snapshot_integrity_fails_closed_without_small_catalog_fallback(tmp_path, monkeypatch, damage):
    raw, metadata = use_fixture(tmp_path, monkeypatch)
    csv_path = tmp_path / 'us_listings.csv'
    meta_path = tmp_path / 'us_listings_provenance.json'
    if damage == 'csv':
        csv_path.write_bytes(raw + b'changed')
    elif damage == 'missing_csv':
        csv_path.unlink()
    elif damage == 'missing_metadata':
        meta_path.unlink()
    elif damage == 'count':
        metadata['rows'] += 1
        meta_path.write_text(json.dumps(metadata))
    else:
        changed = raw.replace(b',NASDAQ,', b',London,')
        csv_path.write_bytes(changed)
        metadata['directory_sha256'] = hashlib.sha256(changed).hexdigest()
        meta_path.write_text(json.dumps(metadata))
    with pytest.raises(market_data.MarketDataError):
        us_markets.us_catalog()


def test_first_download_of_directory_only_symbol_verifies_instrument_type(tmp_path, monkeypatch):
    use_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path / 'prices')
    monkeypatch.setattr(market_data, '_provider', lambda: SimpleNamespace(Ticker=lambda symbol:
        SimpleNamespace(get_info=lambda: {'quoteType': 'ETF', 'exchange': 'NMS', 'currency': 'USD'})))
    with pytest.raises(market_data.MarketDataError, match='only equities'):
        us_markets.resolve_us_listing('AAA')


def test_failed_pair_publication_restores_prior_csv_and_provenance(tmp_path, monkeypatch):
    raw, meta = refresh.build_snapshot(sample_sources())
    refresh.publish_snapshot(raw, meta, tmp_path)
    paths = [tmp_path / 'us_listings.csv', tmp_path / 'us_listings_provenance.json']
    original = {p: p.read_bytes() for p in paths}
    updated = raw.replace(b'AAA Inc.', b'Revised issuer')
    changed_meta = dict(meta, directory_sha256=hashlib.sha256(updated).hexdigest())
    real_replace = Path.replace
    def reject_metadata_once(self, target):
        if str(self).endswith('.tmp') and 'provenance.json.' in self.name:
            raise OSError('Simulated second-file publication failure')
        return real_replace(self, target)
    monkeypatch.setattr(Path, 'replace', reject_metadata_once)
    with pytest.raises(market_data.MarketDataError, match='previous files were restored'):
        refresh.publish_snapshot(updated, changed_meta, tmp_path)
    assert all(p.read_bytes() == original[p] for p in paths)
    assert not list(tmp_path.glob('*.tmp'))


def test_network_failure_preserves_previous_directory(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    raw, meta = refresh.build_snapshot(sample_sources())
    refresh.publish_snapshot(raw, meta, data)
    original = {p: p.read_bytes() for p in data.iterdir()}
    monkeypatch.setattr(refresh, 'ROOT', tmp_path)
    monkeypatch.setattr(refresh, 'download_sources', lambda: (_ for _ in ()).throw(OSError('Offline')))
    assert refresh.main([]) == 1
    assert all(p.read_bytes() == content for p, content in original.items())


def test_real_bundled_directory_is_broad_current_and_verified():
    rows = us_markets.us_catalog()
    metadata = us_markets.directory_provenance()
    assert len(rows) == metadata['rows'] > 5000
    symbols = {row['symbol'] for row in rows}
    assert {'AAPL', 'JPM', 'TSM', 'ASML', 'BRK-B', 'CECO', 'IBOC'} <= symbols
    assert not {'SPY', 'QQQ', '7203.T', 'SAP.DE'} & symbols
    assert len(us_markets.search_us_symbols('', online=False)) == len(rows)
    assert {row['exchange'] for row in rows} >= {'NASDAQ', 'NYSE', 'NYSE American', 'TXSE'}
