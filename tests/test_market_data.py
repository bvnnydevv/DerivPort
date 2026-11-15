"""Global search, adjusted-price persistence, and clear offline fallback behavior."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from derivport import market_data


def test_starter_catalog_search_includes_us_and_international_listings_offline(monkeypatch):
    def no_provider():
        raise AssertionError("Offline lookup must not contact the provider.")

    monkeypatch.setattr(market_data, "_provider", no_provider)
    apple = market_data.search_symbols("Apple", online=False)
    toyota = market_data.search_symbols("Toyota", online=False)
    assert any(row["symbol"] == "AAPL" for row in apple)
    assert any(row["symbol"] == "7203.T" and row["currency"] == "JPY" for row in toyota)
    assert all(row["search_source"] == "starter catalog" for row in apple + toyota)
    assert not market_data.get_search_status()["online_attempted"]


def test_provider_failure_preserves_search_matches_and_explains_offline_state(monkeypatch):
    def no_provider():
        raise ConnectionError("network connection unavailable")

    monkeypatch.setattr(market_data, "_provider", no_provider)
    matches = market_data.search_symbols("Apple")
    assert any(row["symbol"] == "AAPL" for row in matches)
    status = market_data.get_search_status()
    assert status["online_attempted"] and not status["online_ok"]
    assert "cached" in status["message"].lower() or "catalog" in status["message"].lower()


def test_provider_search_uses_equity_listings_and_preserves_exchange_metadata(monkeypatch):
    provider = SimpleNamespace(Search=lambda *args, **kwargs: SimpleNamespace(quotes=[
        {"symbol": "TEST.L", "quoteType": "EQUITY", "longname": "Test Listing", "exchDisp": "London", "currency": "GBP"},
        {"symbol": "TEST-ETF", "quoteType": "ETF", "longname": "Excluded fund"},
        {"symbol": "../BAD", "quoteType": "EQUITY"},
    ]))
    monkeypatch.setattr(market_data, "_provider", lambda: provider)
    matches = market_data.search_symbols("Test Listing")
    found = next(row for row in matches if row["symbol"] == "TEST.L")
    assert found["exchange"] == "London"
    assert found["currency"] == "GBP"
    assert found["search_source"] == "Yahoo Finance"
    assert not any(row["symbol"] in {"TEST-ETF", "../BAD"} for row in matches)
    assert market_data.get_search_status()["online_ok"]


@pytest.mark.parametrize("symbol", ["../secrets", "AAPL/../../x", "AAPL;rm", "$(whoami)", "", "a" * 40])
def test_unsafe_symbol_paths_are_rejected(symbol):
    with pytest.raises(market_data.MarketDataError):
        market_data.normalize_symbol(symbol)


def test_original_bundled_stock_load_never_needs_internet(monkeypatch):
    def no_provider():
        raise AssertionError("Bundled prices must load without a provider request.")

    monkeypatch.setattr(market_data, "_provider", no_provider)
    series, meta = market_data.load_symbol_prices(" aapl ")
    assert len(series) > 1000
    assert meta["symbol"] == "AAPL"
    assert meta["adjusted"] is True
    assert meta["cache_status"] in {"bundled", "cached"}
    assert np.isfinite(series).all() and (series > 0).all()


@pytest.mark.parametrize("symbol,currency", [("7203.T", "JPY"), ("SAP.DE", "EUR"), ("RELIANCE.NS", "INR")])
def test_real_international_examples_are_available_from_the_offline_cache(monkeypatch, symbol, currency):
    def no_provider():
        raise AssertionError("The bundled international examples must remain available offline.")

    monkeypatch.setattr(market_data, "_provider", no_provider)
    assert symbol in market_data.read_cached_symbols()
    series, metadata = market_data.load_symbol_prices(symbol)
    assert metadata["currency"] == currency
    assert metadata["adjusted"] is True
    assert metadata["cache_status"] == "cached"
    assert metadata["source_url"].startswith("https://")
    assert series.index.is_unique and series.index.is_monotonic_increasing
    assert len(series) > 1000
    assert np.isfinite(series).all() and (series > 0).all()


def test_new_listing_downloads_adjusted_only_and_caches_for_offline_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    completed = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=80)
    dates = completed.append(pd.DatetimeIndex([today, today + pd.Timedelta(days=1)]))
    adjusted = np.arange(len(dates), dtype=float) + 100
    frame = pd.DataFrame({"Adj Close": adjusted, "Close": adjusted * 2}, index=dates.tz_localize("Asia/Tokyo"))

    def history(**kwargs):
        assert kwargs["auto_adjust"] is False
        assert kwargs["period"] == "max"
        return frame

    ticker = SimpleNamespace(history=history, history_metadata={"currency": "JPY", "fullExchangeName": "Tokyo"})
    monkeypatch.setattr(market_data, "_provider", lambda: SimpleNamespace(Ticker=lambda symbol: ticker))
    downloaded, meta = market_data.load_symbol_prices("7203.T")
    assert len(downloaded) == 80
    assert downloaded.index.max() < today
    np.testing.assert_array_equal(downloaded.to_numpy(), adjusted[:80])
    assert meta["currency"] == "JPY"
    assert meta["price_field"] == "Adj Close"
    assert meta["observations_excluded"] == 2
    assert meta["cache_status"] == "downloaded"
    saved_bytes = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def no_provider():
        raise ConnectionError("network unavailable")

    monkeypatch.setattr(market_data, "_provider", no_provider)
    cached, metadata = market_data.load_symbol_prices("7203.T")
    pd.testing.assert_series_equal(cached, downloaded, check_freq=False)
    assert metadata["cache_status"] == "cached"
    refreshed, metadata = market_data.load_symbol_prices("7203.T", refresh=True)
    pd.testing.assert_series_equal(refreshed, downloaded, check_freq=False)
    assert metadata["refresh_failed"] is True
    assert "preserved" in metadata["warning"].lower()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == saved_bytes

    # Corrupting the disk cache must never silently feed modified prices to the
    # model; an explicit refresh can repair it using newly verified data.
    csv_path = next(tmp_path.glob("*.csv"))
    csv_path.write_text("date,adjusted_close\n2020-01-01,999999\n")
    with pytest.raises(market_data.MarketDataError, match="checksum"):
        market_data.load_symbol_prices("7203.T")
    monkeypatch.setattr(market_data, "_provider", lambda: SimpleNamespace(Ticker=lambda symbol: ticker))
    repaired, metadata = market_data.load_symbol_prices("7203.T", refresh=True)
    pd.testing.assert_series_equal(repaired, downloaded, check_freq=False)
    assert metadata["cache_status"] == "downloaded"


def test_provider_unadjusted_close_is_never_substituted(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, "CACHE_DIR", tmp_path)
    frame = pd.DataFrame({"Close": np.arange(100) + 1.0}, index=pd.bdate_range("2023-01-01", periods=100))
    ticker = SimpleNamespace(history=lambda **kwargs: frame, history_metadata={})
    monkeypatch.setattr(market_data, "_provider", lambda: SimpleNamespace(Ticker=lambda symbol: ticker))
    with pytest.raises(market_data.MarketDataError, match="adjusted|unadjusted"):
        market_data.load_symbol_prices("TEST.L")
    assert not list(tmp_path.iterdir())


def test_cache_without_a_recorded_checksum_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path)
    csv_path, meta_path = market_data._cache_paths('TEST')
    csv_path.write_text('date,adjusted_close\n2020-01-02,100\n')
    meta_path.write_text(json.dumps({'symbol': 'TEST'}))
    monkeypatch.setattr(market_data, '_provider', lambda: pytest.fail('Unverified cache reads must not silently download.'))
    with pytest.raises(market_data.MarketDataError, match='checksum'):
        market_data.load_symbol_prices('TEST')


def test_frozen_fallback_verifies_provenance_hash_before_loading(tmp_path, monkeypatch):
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    path = tmp_path / 'prices.csv'
    provenance = tmp_path / 'provenance.json'
    payload = b'date,ticker,adjusted_close\n2020-01-02,AAPL,100\n2020-01-03,AAPL,101\n'
    path.write_bytes(payload)
    provenance.write_text(json.dumps({'cache_sha256': hashlib.sha256(payload).hexdigest()}))
    monkeypatch.setattr(market_data, 'CACHE_DIR', cache_dir)
    monkeypatch.setattr(market_data, 'BUNDLED_PATH', path)
    monkeypatch.setattr(market_data, 'BUNDLED_PROVENANCE_PATH', provenance)
    prices, _ = market_data.load_symbol_prices('AAPL')
    assert list(prices) == [100., 101.]
    path.write_text('date,ticker,adjusted_close\n2020-01-02,AAPL,99999\n')
    with pytest.raises(market_data.MarketDataError, match='checksum'):
        market_data.load_symbol_prices('AAPL')


def test_second_cache_replace_failure_restores_previous_files(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path)
    dates = pd.bdate_range('2022-01-03', periods=80)
    current = {'frame': pd.DataFrame({'Adj Close': np.arange(80) + 100.}, index=dates)}
    ticker = SimpleNamespace(history=lambda **kwargs: current['frame'], history_metadata={'currency': 'USD'})
    monkeypatch.setattr(market_data, '_provider', lambda: SimpleNamespace(Ticker=lambda symbol: ticker))
    original, _ = market_data.load_symbol_prices('TEST')
    csv_path, meta_path = market_data._cache_paths('TEST')
    previous_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    current['frame'] = pd.DataFrame({'Adj Close': np.arange(80) + 200.}, index=dates)
    real_replace = Path.replace

    def fail_json_replace(path, target):
        if Path(target) == meta_path:
            raise OSError('Injected second replacement failure')
        return real_replace(path, target)

    monkeypatch.setattr(Path, 'replace', fail_json_replace)
    returned, meta = market_data.load_symbol_prices('TEST', refresh=True)
    pd.testing.assert_series_equal(returned, original)
    assert meta['refresh_failed'] and 'preserved' in meta['warning']
    assert previous_files == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    cached, _ = market_data.load_symbol_prices('TEST')
    pd.testing.assert_series_equal(cached, original)


def test_failed_new_cache_publication_leaves_no_partial_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, 'CACHE_DIR', tmp_path)
    frame = pd.DataFrame({'Adj Close': np.arange(80) + 100.}, index=pd.bdate_range('2022-01-03', periods=80))
    ticker = SimpleNamespace(history=lambda **kwargs: frame, history_metadata={})
    monkeypatch.setattr(market_data, '_provider', lambda: SimpleNamespace(Ticker=lambda symbol: ticker))
    _, meta_path = market_data._cache_paths('TEST')
    real_replace = Path.replace

    def fail_json_replace(path, target):
        if Path(target) == meta_path:
            raise OSError('Injected second replacement failure')
        return real_replace(path, target)

    monkeypatch.setattr(Path, 'replace', fail_json_replace)
    with pytest.raises(market_data.MarketDataError):
        market_data.load_symbol_prices('TEST')
    assert not list(tmp_path.iterdir())
