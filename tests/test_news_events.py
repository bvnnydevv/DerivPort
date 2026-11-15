"""News-window arithmetic never manufactures timing, outcomes or causality."""
import numpy as np
import pandas as pd
import pytest

from derivport.news_events import analyze_news_event


@pytest.fixture
def prices():
    dates = pd.bdate_range('2024-01-01', periods=70)
    return pd.Series(100 + np.arange(70, dtype=float), index=dates, name='AAPL')


def test_publication_day_is_excluded_and_horizons_count_observed_sessions(prices):
    event = prices.index[25]
    original = prices.copy(deep=True)
    result = analyze_news_event(prices, event)
    assert result.event_session == prices.index[26]
    assert result.baseline_date == event
    assert result.baseline_price == 125
    for horizon in (1, 5, 21):
        position = 25 + horizon
        assert result.returns[horizon] == pytest.approx(100 * (prices.iloc[position] / 125 - 1))
        assert result.outcome_dates[horizon] == prices.index[position]
    assert result.window.session_offset.iloc[0] == -19
    assert result.window.session_offset.iloc[-1] == 21
    assert result.window.loc[event, 'return_from_baseline_pct'] == 0
    assert result.metadata['status'] == 'complete'
    assert 'does not establish' in result.metadata['interpretation']
    pd.testing.assert_series_equal(prices, original)


def test_weekend_and_missing_session_use_next_observation_without_filling(prices):
    observed = prices.drop(pd.Timestamp('2024-01-08'))
    result = analyze_news_event(observed, '2024-01-06')
    assert result.baseline_date == pd.Timestamp('2024-01-05')
    assert result.event_session == pd.Timestamp('2024-01-09')
    assert pd.Timestamp('2024-01-08') not in result.window.index


def test_cutoff_drops_future_values_before_validation_or_arithmetic(prices):
    cutoff = prices.index[35]
    original = analyze_news_event(prices, prices.index[25], cutoff=cutoff)
    poisoned = prices.copy()
    poisoned.loc[poisoned.index > cutoff] = np.nan
    changed = analyze_news_event(poisoned, prices.index[25], cutoff=cutoff)
    assert changed.returns == original.returns
    assert changed.outcome_dates == original.outcome_dates
    pd.testing.assert_frame_equal(changed.window, original.window)
    assert changed.returns[21] is None
    assert changed.outcome_dates[21] is None
    assert changed.available_sessions == 10
    assert changed.window.index.max() == cutoff
    assert changed.metadata['status'] == 'partial'


def test_publication_at_last_observation_or_future_remains_pending(prices):
    for publication in (prices.index[-1], prices.index[-1] + pd.Timedelta(days=5)):
        result = analyze_news_event(prices, publication)
        assert result.event_session is None
        assert result.baseline_date == prices.index[-1]
        assert result.available_sessions == 0
        assert all(value is None for value in result.returns.values())
        assert all(value is None for value in result.outcome_dates.values())
        assert result.window.index.max() == prices.index[-1]
        assert result.window.session_offset.isna().all()
        assert result.metadata['status'] == 'pending_event_session'


def test_prelisting_event_has_no_invented_baseline_or_return(prices):
    result = analyze_news_event(prices, '2023-12-25')
    assert result.event_session == prices.index[0]
    assert result.baseline_price is None
    assert result.baseline_date is None
    assert all(value is None for value in result.returns.values())
    assert result.window.return_from_baseline_pct.isna().all()
    assert result.metadata['status'] == 'insufficient_baseline'


def test_time_of_day_is_explicitly_not_used(prices):
    date_only = analyze_news_event(prices, '2024-01-15')
    stamped = analyze_news_event(prices, '2024-01-15 09:00:00-05:00')
    assert stamped.event_session == date_only.event_session
    assert stamped.returns == date_only.returns
    assert stamped.metadata['publication_time_used'] is False


@pytest.mark.parametrize('change', ['duplicate', 'reverse', 'invalid_date', 'zero', 'negative', 'nan', 'inf', 'text', 'boolean'])
def test_invalid_observed_inputs_are_rejected(prices, change):
    broken = prices.copy()
    if change == 'duplicate':
        broken.index = pd.DatetimeIndex([*prices.index[:-1], prices.index[-2]])
    elif change == 'reverse':
        broken = broken.iloc[::-1]
    elif change == 'invalid_date':
        broken.index = pd.DatetimeIndex([pd.NaT, *prices.index[1:]])
    elif change == 'boolean':
        broken = broken.astype(bool)
    elif change == 'text':
        broken = broken.astype(object)
        broken.iloc[5] = 'bad'
    else:
        broken.iloc[5] = {'zero': 0, 'negative': -1, 'nan': np.nan, 'inf': np.inf}[change]
    with pytest.raises(ValueError):
        analyze_news_event(broken, '2024-01-15')


@pytest.mark.parametrize('value', [None, True, 7, float('nan'), 'bad-date'])
def test_invalid_publication_date_is_rejected(prices, value):
    with pytest.raises(ValueError, match='Publication date'):
        analyze_news_event(prices, value)


def test_empty_history_and_invalid_window_sizes_fail_cleanly(prices):
    for source in (pd.Series(dtype=float), prices.to_frame()):
        with pytest.raises(ValueError):
            analyze_news_event(source, '2024-01-15')
    with pytest.raises(ValueError, match='No observed'):
        analyze_news_event(prices, '2024-01-15', cutoff='2023-12-01')
    for field, value in (('before', 0), ('after', True), ('after', 253)):
        with pytest.raises(ValueError, match='window'):
            analyze_news_event(prices, '2024-01-15', **{field: value})


def test_numeric_positions_and_overflowing_price_ratios_are_rejected():
    with pytest.raises(ValueError, match='session dates'):
        analyze_news_event(pd.Series([100.]), '2024-01-01')
    prices = pd.Series([1e-300, 1e300], index=pd.date_range('2024-01-01', periods=2))
    with pytest.raises(ValueError, match='numerical range'):
        analyze_news_event(prices, '2024-01-01')
