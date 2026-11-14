"""Global search, adjusted-price persistence, and clear offline fallback behavior."""

from datetime import datetime, timezone
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
