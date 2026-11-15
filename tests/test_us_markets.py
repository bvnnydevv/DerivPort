"""US venue/instrument eligibility, including offline cached scope."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from derivport import market_data, us_markets


@pytest.mark.parametrize('value,expected', [
    ('NasdaqGS', 'NASDAQ'), ('NCM', 'NASDAQ'), ('NGM', 'NASDAQ'),
    ('NYQ', 'NYSE'), ('New York Stock Exchange', 'NYSE'),
    ('ASE', 'NYSE American'), ('NYSE MKT', 'NYSE American'),
    ('PCX', 'NYSE Arca'), ('BATS', 'Cboe'), ('IEXG', 'IEX'),
    ('Nasdaq Stockholm', None), ('London', None), ('OTC Markets', None),
])
def test_exact_exchange_aliases(value, expected):
    assert us_markets.canonical_us_exchange(value) == expected


def test_currency_or_domicile_is_not_a_us_listing():
    rows = [
        {'symbol': 'TSM', 'exchange': 'NYSE', 'quoteType': 'EQUITY', 'currency': 'USD', 'region': 'Taiwan'},
        {'symbol': 'BABA', 'exchange': 'NYQ', 'quoteType': 'EQUITY', 'currency': 'USD', 'region': 'China'},
        {'symbol': 'ABC.L', 'exchange': 'London', 'quoteType': 'EQUITY', 'currency': 'USD'},
        {'symbol': 'SAP.DE', 'exchange': 'NYSE', 'quoteType': 'EQUITY', 'currency': 'USD'},
        {'symbol': 'PINK', 'exchange': 'PNK', 'quoteType': 'EQUITY', 'currency': 'USD'},
        {'symbol': 'SPY', 'exchange': 'PCX', 'quoteType': 'ETF', 'currency': 'USD'},
        {'symbol': 'UNKNOWN', 'exchange': 'NYSE', 'currency': 'USD'},
        {'symbol': 'CAD', 'exchange': 'NYSE', 'quoteType': 'EQUITY', 'currency': 'CAD'},
    ]
    assert [r['symbol'] for r in us_markets.filter_us_listings(rows)] == ['BABA', 'TSM']


def test_known_us_venue_does_not_invent_missing_currency():
    row = us_markets.filter_us_listings([{'symbol': 'NEW', 'exchange': 'NYSE', 'quoteType': 'EQUITY'}])[0]
    assert row['exchange'] == 'NYSE' and row['currency'] == 'Unknown'


def test_offline_catalog_groups_stocks_and_us_listed_adrs(monkeypatch):
    monkeypatch.setattr(market_data, '_provider', lambda: pytest.fail('Offline catalog used network'))
    catalog = us_markets.us_catalog()
    assert {'AAPL', 'JPM', 'TSM', 'ASML', 'BRK-B'} <= {r['symbol'] for r in catalog}
    assert not {'7203.T', 'SAP.DE', 'RY.TO'} & {r['symbol'] for r in catalog}
    assert all(row['exchange'] == 'NASDAQ' for row in us_markets.us_catalog('NASDAQ'))
    assert all(row['exchange'] == 'NYSE' for row in us_markets.us_catalog('NYSE'))
    assert us_markets.resolve_us_listing('brk-b')['exchange'] == 'NYSE'


def test_search_filters_existing_equity_search_and_exchange(monkeypatch):
    monkeypatch.setattr(market_data, 'search_symbols', lambda query, online: [
        {'symbol': 'AAA', 'exchange': 'NMS', 'currency': 'USD'},
        {'symbol': 'BBB', 'exchange': 'NYSE', 'currency': 'USD'},
        {'symbol': 'AAA.TO', 'exchange': 'Toronto', 'currency': 'CAD'},
    ])
    assert [r['symbol'] for r in us_markets.search_us_symbols('a', exchange='NASDAQ')] == ['AAA']


def test_cached_scope_requires_curated_or_verified_equity_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(market_data, '_provider', lambda: pytest.fail('Cached listing used network'))
    samples = [
        {'symbol': 'AAPL', 'exchange': 'NasdaqGS', 'currency': 'USD'},
        {'symbol': 'NEW', 'exchange': 'NYSE', 'currency': 'USD', 'quoteType': 'EQUITY'},
        {'symbol': 'FUND', 'exchange': 'NYSE Arca', 'currency': 'USD', 'quoteType': 'ETF'},
        {'symbol': 'MYSTERY', 'exchange': 'NYSE', 'currency': 'USD'},
        {'symbol': 'RY.TO', 'exchange': 'Toronto', 'currency': 'CAD', 'quoteType': 'EQUITY'},
    ]
    monkeypatch.setattr(market_data, 'read_cached_symbols', lambda: [m['symbol'] for m in samples])
    for meta in samples:
        csv_path, meta_path = market_data._cache_paths(meta['symbol'])
        csv_path.write_text('date,adjusted_close\n2024-01-02,100\n')
        meta_path.write_text(json.dumps(meta))
    assert {r['symbol'] for r in us_markets.cached_us_listings()} == {'AAPL', 'NEW'}


@pytest.mark.parametrize('info', [
    {'quoteType': 'ETF', 'exchange': 'PCX', 'currency': 'USD'},
    {'quoteType': 'EQUITY', 'exchange': 'LSE', 'currency': 'USD'},
    {'exchange': 'NYQ', 'currency': 'USD'},
])
def test_new_symbol_verification_rejects_unknown_or_nonstock(info, monkeypatch):
    monkeypatch.setattr(market_data, '_provider', lambda: SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(get_info=lambda: info)))
    with pytest.raises(market_data.MarketDataError, match='only equities'):
        us_markets.resolve_us_listing('NEWXYZ')


def test_validated_addition_retains_type_evidence_without_price_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path)
    info = {'quoteType': 'EQUITY', 'exchange': 'NYQ', 'currency': 'USD', 'longName': 'New stock'}
    monkeypatch.setattr(market_data, '_provider', lambda: SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(get_info=lambda: info)))
    csv_path, meta_path = market_data._cache_paths('NEWXYZ')
    original = b'date,adjusted_close\n2024-01-02,100\n'
    meta = {'symbol': 'NEWXYZ', 'exchange': 'NYSE', 'currency': 'USD', 'cache_sha256': 'unchanged'}
    csv_path.write_bytes(original)
    meta_path.write_text(json.dumps(meta))
    values = pd.Series([100.0], index=pd.to_datetime(['2024-01-02']))
    monkeypatch.setattr(market_data, 'load_symbol_prices', lambda symbol, refresh=False: (values, meta))
    _, result = us_markets.load_us_symbol_prices('NEWXYZ')
    assert result['quoteType'] == 'EQUITY' and result['exchange_group'] == 'NYSE'
    assert csv_path.read_bytes() == original
    assert json.loads(meta_path.read_text())['cache_sha256'] == 'unchanged'
    monkeypatch.setattr(market_data, '_provider', lambda: pytest.fail('Verified cache used network'))
    assert us_markets.resolve_us_listing('NEWXYZ')['exchange'] == 'NYSE'


def test_explicit_nonstock_metadata_cannot_fall_back_to_known_catalog():
    with pytest.raises(market_data.MarketDataError):
        us_markets.resolve_us_listing('AAPL', {'exchange': 'NYSE', 'currency': 'USD', 'quoteType': 'ETF'})
