"""Real chart controls keep displayed bars separate from research decisions."""
from datetime import date
from io import BytesIO
from pathlib import Path
import socket
import json

import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from derivport.market_data import MarketDataError
from derivport.ohlc import load_candles
import derivport.stock_chart_ui as chart_ui


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def captured_chart(monkeypatch):
    """Observe the actual figure/download arguments while retaining widgets."""
    def deny_network(*args, **kwargs):
        raise AssertionError('Cached stock charts must work offline.')
    monkeypatch.setattr(socket, 'create_connection', deny_network)
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    observed = {}
    real_chart = st.plotly_chart
    real_download = st.download_button

    def capture_chart(*args, **kwargs):
        if kwargs.get('key') == 'known_price_hero':
            observed['figure'] = go.Figure(args[0])
            observed['config'] = dict(kwargs['config'])
        return real_chart(*args, **kwargs)

    def capture_download(*args, **kwargs):
        if kwargs.get('key') == 'export_chart_data':
            observed['download'] = kwargs.get('data', args[1] if len(args) > 1 else None)
            observed['filename'] = kwargs['file_name']
        return real_download(*args, **kwargs)

    monkeypatch.setattr(st, 'plotly_chart', capture_chart)
    monkeypatch.setattr(st, 'download_button', capture_download)
    return observed


def frozen_app():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not app.exception
    app.selectbox(key='history_mode').set_value('Frozen demo (2018–2024)').run()
    assert not app.exception
    app.date_input(key='decision_date').set_value(date(2024, 5, 3)).run()
    assert not app.exception
    return app


def assert_unrevealed(app):
    assert not app.session_state['reveals']
    assert 'actual_return' not in app.session_state['current_scores']
    assert not any('Actual 21d return' in table.value.columns for table in app.dataframe)


def test_real_chart_controls_and_export_preserve_scores_and_historical_boundary(captured_chart, monkeypatch):
    app = frozen_app()
    source, source_meta = load_candles('AAPL')
    assert app.session_state['chart_basis'] == 'observed_ohlcv'
    figure = captured_chart['figure']
    assert figure.data[0].type == 'candlestick'
    assert figure.data[-1].type == 'bar'
    # The real embedded runtime requires ordinary arrays rather than Plotly's
    # binary NumPy transport; otherwise a valid figure can render blank.
    serialized = json.loads(figure.to_json())
    for trace in serialized['data']:
        for field in ('x', 'y', 'open', 'high', 'low', 'close'):
            if field in trace:
                assert isinstance(trace[field], list)
    assert captured_chart['config']['scrollZoom'] is True
    toolbar_additions = [*figure.layout.modebar.add, *captured_chart['config']['modeBarButtonsToAdd']]
    assert {'drawline', 'drawrect', 'eraseshape', 'drawopenpath'}.issubset(toolbar_additions)
    # Plotly aborts the complete render if layout and config register a button
    # twice, even though the figure and its data otherwise validate correctly.
    assert len(toolbar_additions) == len(set(toolbar_additions))
    baseline_scores = app.session_state['current_scores'].copy(deep=True)
    selected = app.session_state['chart_data']
    pd.testing.assert_frame_equal(selected[['open', 'high', 'low', 'close', 'volume']],
        source.loc[selected.index], check_freq=False)
    assert_unrevealed(app)

    # These are normal user-facing widgets, not assignments to rendering state.
    app.selectbox(key='chart_period').set_value('1Y')
    app.selectbox(key='chart_interval').set_value('Weekly')
    app.multiselect(key='chart_overlays').set_value(['SMA 50', 'EMA 20'])
    app.checkbox(key='chart_volume').uncheck()
    app.checkbox(key='chart_log').check()
    app.checkbox(key='chart_grid').uncheck()
    app.color_picker(key='chart_up').set_value('#00ff88')
    app.color_picker(key='chart_down').set_value('#ff3355').run()
    assert not app.exception
    figure = captured_chart['figure']
    assert [trace.type for trace in figure.data] == ['candlestick', 'scatter', 'scatter']
    assert [trace.name for trace in figure.data[1:]] == ['SMA 50', 'EMA 20']
    assert figure.data[0].increasing.line.color == '#00ff88'
    assert figure.data[0].decreasing.line.color == '#ff3355'
    assert figure.layout.yaxis.type == 'log'
    assert figure.layout.yaxis.showgrid is False
    assert pd.to_datetime(figure.data[0].x).max() <= pd.Timestamp('2024-05-03')
    pd.testing.assert_frame_equal(app.session_state['current_scores'], baseline_scores)
    assert_unrevealed(app)

    app.selectbox(key='chart_interval').set_value('Monthly')
    app.selectbox(key='chart_style').set_value('OHLC')
    app.checkbox(key='chart_volume').check().run()
    assert not app.exception
    assert captured_chart['figure'].data[0].type == 'ohlc'
    assert captured_chart['figure'].data[-1].type == 'bar'
    displayed = app.session_state['chart_data'].copy(deep=True)
    assert len(displayed) <= 13
    assert displayed.index.max() <= pd.Timestamp(app.session_state['cutoff'])
    exported = pd.read_csv(BytesIO(captured_chart['download']), index_col='date', parse_dates=['date'])
    pd.testing.assert_frame_equal(exported, displayed, check_exact=False, rtol=1e-12, atol=1e-12,
                                  check_dtype=False, check_freq=False)
    assert captured_chart['filename'] == 'AAPL_monthly_2024-05-03.csv'
    pd.testing.assert_frame_equal(app.session_state['current_scores'], baseline_scores)

    app.selectbox(key='chart_style').set_value('Line').run()
    assert not app.exception
    assert captured_chart['figure'].data[0].type == 'scatter'
    app.selectbox(key='chart_interval').set_value('Daily')
    app.selectbox(key='chart_period').set_value('All')
    app.date_input(key='decision_date').set_value(date(2023, 11, 1)).run()
    assert not app.exception
    earlier = app.session_state['chart_data'].copy(deep=True)
    earlier_scores = app.session_state['current_scores'].copy(deep=True)
    assert earlier.index.max() <= pd.Timestamp('2023-11-01')
    # Mutating cached observations after the earlier close must not change the
    # historical chart, overlays, export or current analysis inputs.
    changed_future = source.copy(deep=True)
    changed_future.loc[changed_future.index > '2023-11-01', :] *= 20
    monkeypatch.setattr(chart_ui, 'cached_candles', lambda ticker: (changed_future, source_meta))
    app.run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state['chart_data'], earlier)
    pd.testing.assert_frame_equal(app.session_state['current_scores'], earlier_scores)
    assert_unrevealed(app)


def test_missing_candles_uses_explicit_adjusted_close_fallback(captured_chart, monkeypatch):
    def unavailable(symbol):
        raise MarketDataError('No genuine OHLCV cache is available for this stock.')
    monkeypatch.setattr(chart_ui, 'cached_candles', unavailable)
    app = frozen_app()
    assert app.session_state['chart_basis'] == 'adjusted_close_only'
    assert all(trace.type == 'scatter' for trace in captured_chart['figure'].data)
    assert captured_chart['figure'].data[0].name == 'Adjusted close'
    assert app.selectbox(key='chart_interval').disabled
    assert app.selectbox(key='chart_style').disabled
    assert app.checkbox(key='chart_volume').disabled
    assert any('never turned into invented bars' in item.value for item in app.caption)
    export = pd.read_csv(BytesIO(captured_chart['download']), parse_dates=['date'])
    assert list(export.columns) == ['date', 'adjusted_close']
    assert export.date.max() <= pd.Timestamp(app.session_state['cutoff'])
    assert_unrevealed(app)
