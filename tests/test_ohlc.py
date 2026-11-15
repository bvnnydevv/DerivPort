"""Authentic candle validation, offline use, provenance and refresh boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import derivport.ohlc as candles
from derivport.market_data import MarketDataError


def sample_frame():
    return pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']),
        'open': [10., 11.], 'high': [12., 13.], 'low': [9., 10.],
        'close': [11., 12.], 'volume': [100., 200.]})


def cache_sample(monkeypatch, tmp_path, symbol='AAPL'):
    monkeypatch.setattr(candles, 'CANDLE_DIR', tmp_path)
    bars = candles._validate_frame(sample_frame(), symbol)
    meta = {'symbol': symbol, 'schema_version': 1, 'columns': list(candles.COLUMNS),
        'auto_adjust': False, 'back_adjust': False, 'repair': False,
        'source': 'test fixture', 'price_basis': candles.PRICE_BASIS}
    candles._publish(*candles._paths(symbol), bars, meta)
    return bars


def fake_provider(history, metadata=None, error=None):
    def get_history(**kwargs):
        assert kwargs['auto_adjust'] is False
        assert kwargs['back_adjust'] is False
        assert kwargs['repair'] is False
        assert kwargs['keepna'] is True
        assert kwargs['interval'] == '1d'
        if error:
            raise error
        return history.copy()
    ticker = SimpleNamespace(history=get_history, history_metadata=metadata or {'currency': 'USD', 'exchangeName': 'NMS', 'instrumentType': 'EQUITY'})
    return SimpleNamespace(Ticker=lambda symbol: ticker)


def provider_frame():
    f = sample_frame().set_index('date')
    f.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    f.index = f.index.tz_localize('America/New_York')
    return f


@pytest.mark.parametrize('field,value', [
    ('open', 0), ('close', -1), ('high', 9), ('low', 12),
    ('volume', -1), ('volume', np.inf), ('close', np.nan)])
def test_invalid_ohlcv_is_rejected(field, value):
    f = sample_frame()
    f.loc[0, field] = value
    with pytest.raises(MarketDataError, match='invalid'):
        candles._validate_frame(f, 'AAPL')


@pytest.mark.parametrize('bad_dates', [['2024-01-02', '2024-01-02'], ['bad', '2024-01-03'], ['2024-01-02 12:00:00', '2024-01-03']])
def test_invalid_session_dates_are_rejected(bad_dates):
    f = sample_frame()
    f['date'] = bad_dates
    with pytest.raises(MarketDataError, match='invalid'):
        candles._validate_frame(f, 'AAPL')


def test_incomplete_current_or_future_bars_rejected():
    f = sample_frame()
    f.loc[0, 'date'] = pd.Timestamp(datetime.now(timezone.utc).date())
    with pytest.raises(MarketDataError, match='invalid'):
        candles._validate_frame(f, 'AAPL')


def test_missing_ohlc_is_never_synthesized_from_close():
    with pytest.raises(MarketDataError, match='open, high, low, close'):
        candles._validate_frame(sample_frame().drop(columns='open'), 'AAPL')


def test_valid_cache_reads_offline_exactly(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    monkeypatch.setattr(candles, '_provider', lambda: pytest.fail('Valid cache must not use the network'))
    actual, meta = candles.load_candles(' aapl ')
    assert_frame_equal(actual, expected)
    assert meta['cache_status'] == 'cached'
    assert meta['rows'] == 2
    assert meta['end'] == '2024-01-03'


def test_missing_cache_requires_explicit_download(monkeypatch, tmp_path):
    monkeypatch.setattr(candles, 'CANDLE_DIR', tmp_path)
    monkeypatch.setattr(candles, '_provider', lambda: pytest.fail('No automatic network calls'))
    with pytest.raises(MarketDataError, match='Download candles'):
        candles.load_candles('AAPL')


def test_checksum_mismatch_and_missing_checksum_rejected(monkeypatch, tmp_path):
    cache_sample(monkeypatch, tmp_path)
    csv_path, meta_path = candles._paths('AAPL')
    csv_path.write_text(csv_path.read_text().replace('100', '101'))
    with pytest.raises(MarketDataError, match='checksum'):
        candles.load_candles('AAPL')
    cache_sample(monkeypatch, tmp_path)
    meta = json.loads(meta_path.read_text())
    meta.pop('cache_sha256')
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(MarketDataError, match='checksum'):
        candles.load_candles('AAPL')


def test_incomplete_cache_pair_and_wrong_symbol_rejected(monkeypatch, tmp_path):
    cache_sample(monkeypatch, tmp_path)
    csv_path, meta_path = candles._paths('AAPL')
    meta = json.loads(meta_path.read_text())
    meta['symbol'] = 'MSFT'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(MarketDataError, match='metadata'):
        candles.load_candles('AAPL')
    meta_path.unlink()
    with pytest.raises(MarketDataError, match='incomplete'):
        candles.load_candles('AAPL')


def test_unverified_basis_rejected(monkeypatch, tmp_path):
    cache_sample(monkeypatch, tmp_path)
    _, meta_path = candles._paths('AAPL')
    meta = json.loads(meta_path.read_text())
    meta['auto_adjust'] = True
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(MarketDataError, match='price basis'):
        candles.load_candles('AAPL')


def test_download_real_fields_excludes_current_date_and_keeps_ohlc(monkeypatch, tmp_path):
    monkeypatch.setattr(candles, 'CANDLE_DIR', tmp_path)
    h = provider_frame()
    today = pd.Timestamp(datetime.now(timezone.utc).date(), tz='America/New_York')
    h.loc[today] = [500, 510, 490, 505, 1000]
    monkeypatch.setattr(candles, '_provider', lambda: fake_provider(h))
    bars, meta = candles.load_candles('AAPL', allow_download=True)
    assert bars.loc['2024-01-02', ['open', 'high', 'low', 'close']].tolist() == [10, 12, 9, 11]
    assert len(bars) == 2
    assert meta['observations_excluded'] == 1
    assert meta['auto_adjust'] is False
    assert meta['currency'] == 'USD'
    assert meta['instrument_type'] == 'EQUITY'
    assert meta['point_in_time_archive'] is False
    assert hashlib.sha256(candles._paths('AAPL')[0].read_bytes()).hexdigest() == meta['cache_sha256']


def test_bad_download_does_not_replace_valid_cache(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    before = {p: p.read_bytes() for p in candles._paths('AAPL')}
    h = provider_frame()
    h.iloc[0, h.columns.get_loc('High')] = 1
    monkeypatch.setattr(candles, '_provider', lambda: fake_provider(h))
    bars, meta = candles.load_candles('AAPL', refresh=True)
    assert_frame_equal(bars, expected)
    assert meta['refresh_failed'] is True
    assert 'preserved' in meta['warning']
    assert all(p.read_bytes() == old for p, old in before.items())


def test_failed_refresh_returns_prior_cache_without_fabricated_data(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    monkeypatch.setattr(candles, '_provider', lambda: fake_provider(provider_frame(), error=ConnectionError('offline')))
    bars, meta = candles.load_candles('AAPL', refresh=True)
    assert_frame_equal(bars, expected)
    assert meta['refresh_failed'] is True


def test_failed_second_publish_restores_both_old_files(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    csv_path, meta_path = candles._paths('AAPL')
    before = {p: p.read_bytes() for p in (csv_path, meta_path)}
    original_replace = Path.replace
    def fail_metadata(source, target):
        if str(source).endswith('.tmp') and '.json.' in str(source) and target == meta_path:
            raise OSError('simulated disk failure')
        return original_replace(source, target)
    monkeypatch.setattr(Path, 'replace', fail_metadata)
    with pytest.raises(MarketDataError, match='restored'):
        candles._publish(csv_path, meta_path, expected * 2, {'symbol': 'AAPL'})
    assert all(p.read_bytes() == old for p, old in before.items())
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_first_publish_retains_old_pair(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    before = {p: p.read_bytes() for p in candles._paths('AAPL')}
    original_replace = Path.replace
    def fail_csv(source, target):
        if '.csv.' in str(source) and str(source).endswith('.tmp'):
            raise OSError('simulated disk failure')
        return original_replace(source, target)
    monkeypatch.setattr(Path, 'replace', fail_csv)
    with pytest.raises(MarketDataError, match='restored'):
        candles._publish(*candles._paths('AAPL'), expected, {'symbol': 'AAPL'})
    assert all(p.read_bytes() == old for p, old in before.items())


def test_bad_cache_can_be_repaired_only_by_explicit_refresh(monkeypatch, tmp_path):
    cache_sample(monkeypatch, tmp_path)
    candles._paths('AAPL')[0].write_text('invalid')
    monkeypatch.setattr(candles, '_provider', lambda: fake_provider(provider_frame()))
    bars, meta = candles.load_candles('AAPL', refresh=True)
    assert len(bars) == 2
    assert meta['cache_status'] == 'downloaded'


def test_reader_waits_for_pair_publication_and_never_reads_intermediate_csv(monkeypatch, tmp_path):
    expected = cache_sample(monkeypatch, tmp_path)
    csv_path, meta_path = candles._paths('AAPL')
    meta = json.loads(meta_path.read_text())
    csv_replaced, finish_publish, reader_started, reader_done = (Event() for _ in range(4))
    original_replace = Path.replace
    failures, observed = [], []
    def pause_after_csv(source, target):
        result = original_replace(source, target)
        if target == csv_path and str(source).endswith('.tmp'):
            csv_replaced.set()
            assert finish_publish.wait(5)
        return result
    monkeypatch.setattr(Path, 'replace', pause_after_csv)
    def publish():
        try:
            candles._publish(csv_path, meta_path, expected * 2, meta)
        except Exception as exc:
            failures.append(exc)
    def read():
        reader_started.set()
        try:
            observed.append(candles.load_candles('AAPL')[0])
        except Exception as exc:
            failures.append(exc)
        finally:
            reader_done.set()
    writer = Thread(target=publish)
    reader = Thread(target=read)
    writer.start()
    assert csv_replaced.wait(5)
    reader.start()
    assert reader_started.wait(5)
    assert not reader_done.wait(0.05)
    finish_publish.set()
    writer.join(5)
    reader.join(5)
    assert not writer.is_alive() and not reader.is_alive()
    assert not failures
    assert_frame_equal(observed[0], expected * 2)


def test_bundled_real_candles_are_complete_verified_offline(monkeypatch):
    symbols = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA', 'AMD', 'META', 'NFLX',
        'TSLA', 'AVGO', 'ORCL', 'JPM', 'LLY', 'XOM', 'WMT']
    monkeypatch.setattr(candles, '_provider', lambda: pytest.fail('Bundled candles must work offline'))
    count = 0
    for symbol in symbols:
        bars, meta = candles.load_candles(symbol)
        assert bars.index[0] == pd.Timestamp('2018-01-02')
        assert bars.index[-1] == pd.Timestamp('2026-11-13')
        assert meta['source'] == 'Yahoo Finance via yfinance'
        assert meta['instrument_type'] == 'EQUITY'
        assert meta['currency'] == 'USD'
        assert meta['auto_adjust'] is False
        count += len(bars)
    assert count == 33450
