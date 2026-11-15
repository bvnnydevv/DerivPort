"""Coach drawings are descriptive, bounded and historical-cutoff safe."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from derivport.chart_guides import apply_chart_guide, guide_intent, queue_chart_guide


@pytest.fixture
def bars():
    index = pd.bdate_range('2024-01-02', periods=50, name='date')
    close = np.arange(50, dtype=float) + 100
    return pd.DataFrame({'close': close, 'low': close - 2, 'high': close + 3}, index=index)


@pytest.mark.parametrize('question,expected', [
    ('Draw a trend line at 99999999', 'trend'),
    ('Mark support and resistance', 'range'),
    ('Show the moving average', 'moving_average'),
    ('Add SMA 20', 'moving_average'),
    ('Clear all guides', 'clear'),
    ('Explain the trend', None),
    ("Don't draw a trend", None),
    ('Do not clear guides', None),
    ('Run __import__("os").system("bad")', None),
])
def test_user_requests_map_only_to_bounded_guide_kinds(question, expected):
    assert guide_intent(question) == expected


def test_queue_has_no_model_coordinates_and_clear_preserves_existing_user_indicators():
    state = {'chart_overlays': ['SMA 50']}
    queue_chart_guide(state, 'trend', 'AAPL|cutoff|prices')
    assert state['coach_chart_guide'] == {'kind': 'trend', 'context': 'AAPL|cutoff|prices'}
    queue_chart_guide(state, 'moving_average', 'AAPL|cutoff|prices')
    assert state['chart_overlays'] == ['SMA 50', 'SMA 20']
    queue_chart_guide(state, 'clear', 'AAPL|cutoff|prices')
    assert 'coach_chart_guide' not in state
    assert state['chart_overlays'] == ['SMA 50']
    state['chart_overlays'] = ['SMA 20']
    queue_chart_guide(state, 'moving_average', 'new')
    queue_chart_guide(state, 'clear', 'new')
    assert state['chart_overlays'] == ['SMA 20']
    with pytest.raises(ValueError):
        queue_chart_guide(state, 'execute Python', 'new')


def test_trend_uses_first_and_last_observed_close_and_no_future_endpoint(bars):
    cutoff = bars.index[24]
    fig = apply_chart_guide(go.Figure(), bars,
        {'kind': 'trend', 'context': 'current'}, 'current', cutoff)
    shape = fig.layout.shapes[0]
    assert pd.Timestamp(shape.x0) == bars.index[0]
    assert pd.Timestamp(shape.x1) == cutoff
    assert shape.y0 == bars.close.iloc[0]
    assert shape.y1 == bars.loc[cutoff, 'close']
    changed = bars.copy()
    changed.loc[changed.index > cutoff, :] = np.nan
    stable = apply_chart_guide(go.Figure(), changed,
        {'kind': 'trend', 'context': 'current'}, 'current', cutoff)
    assert fig.to_json() == stable.to_json()


def test_range_ignores_future_extremes_and_labels_only_observed_window_levels(bars):
    cutoff = bars.index[24]
    changed = bars.copy()
    changed.loc[changed.index > cutoff, 'high'] = 1e12
    changed.loc[changed.index > cutoff, 'low'] = 1e-9
    fig = apply_chart_guide(go.Figure(), changed,
        {'kind': 'range', 'context': 'current'}, 'current', cutoff)
    assert [shape.y0 for shape in fig.layout.shapes] == [98., 127.]
    assert all(shape.y0 == shape.y1 for shape in fig.layout.shapes)
    assert all(pd.Timestamp(shape.x1) == cutoff for shape in fig.layout.shapes)
    labels = [annotation.text for annotation in fig.layout.annotations]
    assert labels == ['Window low 98.00', 'Window high 127.00']
    assert not any('support' in label.lower() or 'resistance' in label.lower() for label in labels)


def test_close_only_fallback_uses_honestly_labeled_close_extremes(bars):
    frame = bars[['close']].rename(columns={'close': 'adjusted_close'})
    cutoff = frame.index[24]
    fig = apply_chart_guide(go.Figure(), frame,
        {'kind': 'range', 'context': 'current'}, 'current', cutoff)
    assert [shape.y0 for shape in fig.layout.shapes] == [100., 124.]
    assert [annotation.text for annotation in fig.layout.annotations] == [
        'Close window low 100.00', 'Close window high 124.00']


def test_moving_average_uses_only_known_closes_and_plain_arrays(bars):
    cutoff = bars.index[24]
    fig = apply_chart_guide(go.Figure(), bars,
        {'kind': 'moving_average', 'context': 'current'}, 'current', cutoff)
    expected = bars.loc[:cutoff, 'close'].rolling(20).mean().dropna()
    assert list(pd.to_datetime(fig.data[0].x)) == list(expected.index)
    np.testing.assert_allclose(fig.data[0].y, expected)
    assert max(pd.to_datetime(fig.data[0].x)) == cutoff
    assert 'bdata' not in fig.to_json()
    assert isinstance(json.loads(fig.to_json())['data'][0]['y'], list)
    short = apply_chart_guide(go.Figure(), bars.iloc[:10],
        {'kind': 'moving_average', 'context': 'current'}, 'current', cutoff)
    assert not short.data
    assert '20-bar guide' in short.layout.annotations[0].text


def test_log_axis_annotation_coordinates_and_shape_prices_have_correct_units(bars):
    fig = go.Figure()
    fig.update_yaxes(type='log')
    cutoff = bars.index[24]
    apply_chart_guide(fig, bars, {'kind': 'trend', 'context': 'current'}, 'current', cutoff)
    assert fig.layout.shapes[0].y1 == 124.
    assert fig.layout.annotations[0].y == pytest.approx(np.log10(124.))


def test_idempotent_coach_replacement_and_clear_do_not_remove_user_drawings(bars):
    fig = go.Figure(go.Scatter(x=[bars.index[0]], y=[100.], name='User series'))
    fig.add_shape(type='line', x0=bars.index[0], x1=bars.index[2], y0=99, y1=101, name='manual')
    fig.add_annotation(x=bars.index[0], y=100., text='User note', name='manual-note')
    guide = {'kind': 'range', 'context': 'current'}
    for _ in range(2):
        apply_chart_guide(fig, bars, guide, 'current', bars.index[24])
    assert len(fig.layout.shapes) == 3
    assert len(fig.layout.annotations) == 3
    apply_chart_guide(fig, bars, None, 'current', bars.index[24])
    assert [shape.name for shape in fig.layout.shapes] == ['manual']
    assert [annotation.text for annotation in fig.layout.annotations] == ['User note']
    assert [trace.name for trace in fig.data] == ['User series']


def test_stale_context_never_draws_into_another_stock_or_cutoff(bars):
    fig = apply_chart_guide(go.Figure(), bars,
        {'kind': 'trend', 'context': 'AAPL|later'}, 'MSFT|earlier', bars.index[20])
    assert not fig.layout.shapes
    assert not fig.layout.annotations
    assert not fig.data


def test_existing_sma_is_not_duplicated_and_invalid_known_data_is_not_repaired(bars):
    fig = go.Figure(go.Scatter(x=bars.index[:25].to_pydatetime().tolist(),
        y=bars.close.iloc[:25].tolist(), name='SMA 20'))
    apply_chart_guide(fig, bars, {'kind': 'moving_average', 'context': 'current'}, 'current', bars.index[24])
    assert len(fig.data) == 1
    damaged = deepcopy(bars)
    damaged.loc[damaged.index[3], 'close'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        apply_chart_guide(go.Figure(), damaged,
            {'kind': 'trend', 'context': 'current'}, 'current', bars.index[24])


def test_warmed_causal_sma_is_reused_before_visible_crop_without_a_second_curve(bars):
    prepared = bars.copy()
    prepared['SMA 20'] = prepared.close.rolling(20).mean()
    window = prepared.iloc[30:45]
    cutoff = window.index[-1]
    fig = go.Figure(go.Scatter(x=window.index.to_pydatetime().tolist(),
        y=window['SMA 20'].tolist(), name='SMA 20'))
    apply_chart_guide(fig, window, {'kind': 'moving_average', 'context': 'current'}, 'current', cutoff)
    assert len(fig.data) == 1
    assert fig.layout.annotations[0].text == 'Observed SMA 20'
    assert fig.layout.annotations[0].y == window['SMA 20'].iloc[-1]
    # A figure without the existing overlay uses exactly those same warmed
    # past values, even though fewer than 20 bars remain in the visible crop.
    fresh = apply_chart_guide(go.Figure(), window,
        {'kind': 'moving_average', 'context': 'current'}, 'current', cutoff)
    np.testing.assert_allclose(fresh.data[0].y, window['SMA 20'])
