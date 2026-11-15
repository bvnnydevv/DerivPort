"""Chart bars and overlays remain observations available at the chosen close."""
import numpy as np
import pandas as pd
import pytest
import json
import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools

from derivport.stock_chart import make_adjusted_line_chart, make_stock_chart, prepare_chart_data


def bars(n=160):
    index = pd.bdate_range('2023-01-02', periods=n, name='date')
    close = np.arange(n, dtype=float) + 100
    return pd.DataFrame({'open': close - 1, 'high': close + 3, 'low': close - 4,
                         'close': close, 'volume': np.arange(n, dtype=float) + 1000}, index=index)


@pytest.mark.parametrize('interval', ['Weekly', 'Monthly'])
def test_aggregated_prices_and_partial_bar_use_only_observed_cutoff(interval):
    source = bars(80)
    cutoff = pd.Timestamp('2023-02-08')
    actual = prepare_chart_data(source, cutoff, 'All', interval)
    rule = 'W-FRI' if interval == 'Weekly' else 'ME'
    known = source.loc[:cutoff]
    groups = list(known.groupby(pd.Grouper(freq=rule)))
    groups = [g for _, g in groups if len(g)]
    assert list(actual.index) == [g.index[-1] for g in groups]
    assert actual.index[-1] == cutoff
    for (_, row), group in zip(actual.iterrows(), groups):
        assert row.open == group.open.iloc[0]
        assert row.high == group.high.max()
        assert row.low == group.low.min()
        assert row.close == group.close.iloc[-1]
        assert row.volume == group.volume.sum()
    revised = source.copy()
    revised.loc[revised.index > cutoff, ['open', 'high', 'low', 'close']] = np.nan
    pd.testing.assert_frame_equal(actual, prepare_chart_data(revised, cutoff, 'All', interval))


def test_indicators_warm_up_before_visible_window_and_do_not_mutate_input():
    source = bars()
    original = source.copy(deep=True)
    cutoff = source.index[-1]
    actual = prepare_chart_data(source, cutoff, '1M', overlays=('SMA 20', 'SMA 50', 'EMA 20'))
    expected = source.close.rolling(50).mean().loc[actual.index]
    pd.testing.assert_series_equal(actual['SMA 50'], expected, check_names=False)
    assert actual['SMA 50'].notna().all()
    assert actual['SMA 20'].iloc[-1] == pytest.approx(source.close.iloc[-20:].mean())
    assert actual['EMA 20'].iloc[-1] == pytest.approx(source.close.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1])
    pd.testing.assert_frame_equal(source, original)


def test_weekly_indicator_counts_bars_not_daily_sessions():
    source = bars(320)
    all_weekly = prepare_chart_data(source, source.index[-1], 'All', 'Weekly')
    view = prepare_chart_data(source, source.index[-1], '3M', 'Weekly', ('SMA 20',))
    pd.testing.assert_series_equal(view['SMA 20'], all_weekly.close.rolling(20).mean().loc[view.index], check_names=False)


@pytest.mark.parametrize('style,kind', [('Candles', 'candlestick'), ('OHLC', 'ohlc'), ('Line', 'scatter')])
def test_chart_styles_volume_customization_and_historical_cutoff(style, kind):
    source = bars(100)
    cutoff = source.index[-11]
    fig = make_stock_chart(source, 'AAPL', cutoff, style=style, overlays=('SMA 20',),
                           up_color='#00ff00', down_color='#ff0000', log_scale=True, show_grid=False)
    assert fig.data[0].type == kind
    assert fig.data[-1].type == 'bar'
    assert fig.data[-1].name == 'Observed volume'
    assert fig.layout.yaxis.type == 'log'
    assert fig.layout.yaxis.side == 'right'
    assert fig.layout.yaxis.showgrid is False
    assert fig.layout.xaxis.showspikes is True
    assert fig.layout.xaxis.rangeslider.visible is False
    for trace in fig.data:
        assert max(pd.to_datetime(trace.x)) <= cutoff
    if kind in ('candlestick', 'ohlc'):
        assert fig.data[0].increasing.line.color == '#00ff00'
        assert fig.data[0].decreasing.line.color == '#ff0000'
    assert fig.layout.annotations[-1].y == pytest.approx(np.log10(source.loc[cutoff, 'close']))
    assert 'drawline' in fig.layout.modebar.add
    # Full serialization exercises Plotly property validation as well as values.
    assert 'Observed OHLCV' in fig.to_json()


def test_close_only_fallback_never_creates_candles_or_volume():
    source = bars()
    cutoff = source.index[-21]
    fig = make_adjusted_line_chart(source.close, 'AAPL', cutoff, '1M', overlays=('SMA 50',))
    assert all(trace.type == 'scatter' for trace in fig.data)
    assert [trace.name for trace in fig.data] == ['Adjusted close', 'SMA 50']
    assert max(pd.to_datetime(fig.data[0].x)) <= cutoff
    assert np.isfinite(fig.data[1].y).all()
    assert 'no OHLCV available' in fig.layout.meta['basis']


def test_no_observations_gives_honest_empty_chart():
    source = bars(20)
    fig = make_stock_chart(source, 'AAPL', '2020-01-01')
    assert not len(fig.data[0].x)
    assert any('No observed prices' in annotation.text for annotation in fig.layout.annotations)


@pytest.mark.parametrize('damage', ['missing', 'inverted', 'negative_volume', 'duplicate'])
def test_invalid_observations_are_rejected(damage):
    source = bars(30)
    if damage == 'missing':
        source = source.drop(columns='open')
    elif damage == 'inverted':
        source.iloc[0, source.columns.get_loc('high')] = 1
    elif damage == 'negative_volume':
        source.iloc[0, source.columns.get_loc('volume')] = -1
    else:
        source = pd.concat([source, source.iloc[[0]]])
    with pytest.raises(ValueError):
        prepare_chart_data(source, '2023-12-31')


def test_unsupported_controls_raise_clear_errors_and_absent_volume_is_explicit():
    source = bars(30)
    for kwargs in ({'interval': 'Hourly'}, {'period': '8Y'}, {'overlays': ('Unknown',)}, {'style': 'Fake candles'}):
        with pytest.raises(ValueError):
            make_stock_chart(source, 'AAPL', source.index[-1], **kwargs)
    with pytest.raises(ValueError, match='volume'):
        make_stock_chart(source.drop(columns='volume'), 'AAPL', source.index[-1])
    assert make_stock_chart(source.drop(columns='volume'), 'AAPL', source.index[-1], show_volume=False).data[0].type == 'candlestick'


@pytest.mark.parametrize('style', ['Candles', 'OHLC', 'Line', 'Adjusted close'])
def test_stock_chart_serializes_plain_arrays_for_embedded_plotly_renderer(style):
    source = bars()
    if style == 'Adjusted close':
        fig = make_adjusted_line_chart(source.close, 'AAPL', source.index[-1], overlays=('SMA 20',))
    else:
        fig = make_stock_chart(source, 'AAPL', source.index[-1], style=style, overlays=('SMA 20',))
    # Exercise the actual public conversion path Streamlit uses, plus a fresh
    # Figure clone as used by the UI capture test. No private patching involved.
    for rendered in (fig, go.Figure(fig)):
        converted = plotly.tools.return_figure_from_figure_or_data(rendered, validate_figure=True)
        serialized = pio.to_json(converted, validate=False)
        assert '"bdata"' not in serialized
        payload = json.loads(serialized)
        for trace in payload['data']:
            assert isinstance(trace['x'], list)
            assert all(isinstance(stamp, str) and stamp.startswith('2023-') for stamp in trace['x'])
            for field in ('open', 'high', 'low', 'close', 'y'):
                if field in trace:
                    assert isinstance(trace[field], list)
                    assert len(trace[field]) == len(trace['x'])
