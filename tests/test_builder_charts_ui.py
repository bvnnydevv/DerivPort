"""Builder price mirrors and draft score replays share the Research experiment."""
from pathlib import Path
import json
import socket

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from derivport import market_data, ohlc
from derivport.blocks import compile_blocks
from derivport.formula import evaluate_formula


ROOT = Path(__file__).resolve().parents[1]
CHART_KEYS = ('known_price_hero', 'builder_stock_chart', 'builder_algorithm_chart')


@pytest.fixture
def builder_figures(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError('The builder preview must use bundled observations only.')
    monkeypatch.setattr(socket, 'create_connection', deny_network)
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    monkeypatch.setattr(market_data, '_provider', deny_network)
    monkeypatch.setattr(ohlc, '_provider', deny_network)
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    figures = {}
    real_plotly = st.plotly_chart
    def capture_chart(*args, **kwargs):
        if kwargs.get('key') in (*CHART_KEYS, 'backtest_chart'):
            figures[kwargs['key']] = go.Figure(args[0])
        return real_plotly(*args, **kwargs)
    monkeypatch.setattr(st, 'plotly_chart', capture_chart)
    return figures


def dashboard():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not app.exception
    return app


def chart_data(fig):
    return json.loads(fig.to_json())['data']


def curve_values(fig):
    return np.array([np.nan if value is None else value for value in fig.data[0].y], dtype=float)


def assert_known_dates(fig, cutoff):
    for trace in fig.data:
        if trace.x is not None and len(trace.x):
            assert pd.DatetimeIndex(trace.x).max() <= pd.Timestamp(cutoff)


def test_builder_mirrors_research_candles_colors_and_indicators_without_duplicate_widgets(builder_figures):
    app = dashboard()
    assert set(CHART_KEYS) <= builder_figures.keys()
    app.color_picker(key='chart_up').set_value('#3355aa')
    app.color_picker(key='chart_down').set_value('#cc3344')
    app.multiselect(key='chart_overlays').set_value(['SMA 50', 'EMA 20']).run()
    assert not app.exception
    research = builder_figures['known_price_hero']
    mirror = builder_figures['builder_stock_chart']
    assert research.data[0].type == mirror.data[0].type == 'candlestick'
    assert chart_data(mirror) == chart_data(research)
    assert mirror.layout.paper_bgcolor == research.layout.paper_bgcolor
    assert mirror.data[0].increasing.line.color == '#3355aa'
    assert mirror.data[0].decreasing.line.color == '#cc3344'
    assert {'SMA 50', 'EMA 20'} <= {trace.name for trace in mirror.data}
    assert len([widget for widget in app.selectbox if widget.key == 'chart_period']) == 1
    assert not app.session_state['reveals']
    for key in CHART_KEYS:
        assert_known_dates(builder_figures[key], app.session_state['cutoff'])
    score_chart = builder_figures['builder_algorithm_chart']
    assert score_chart.layout.meta['kind'] == 'algorithm_score_history'
    assert 'not a price forecast' in score_chart.layout.meta['status']
    assert score_chart.data[0].line.color != mirror.data[0].increasing.line.color


def test_draft_changes_score_curve_but_not_prices_or_active_rule_until_apply(builder_figures):
    app = dashboard()
    active = app.session_state['current_scores'].copy(deep=True)
    active_formula = app.session_state['valid_formula']
    cutoff = app.session_state['cutoff']
    ticker = app.selectbox(key='stock').value
    original_prices = chart_data(builder_figures['known_price_hero'])
    original_curve = curve_values(builder_figures['builder_algorithm_chart'])
    app.slider(key='guide_model_share').set_value(90).run()
    assert not app.exception
    changed = builder_figures['builder_algorithm_chart']
    assert not np.allclose(curve_values(changed), original_curve, equal_nan=True)
    assert chart_data(builder_figures['known_price_hero']) == original_prices
    assert chart_data(builder_figures['builder_stock_chart']) == original_prices
    assert app.session_state['valid_formula'] == active_formula
    pd.testing.assert_frame_equal(app.session_state['current_scores'], active)
    assert app.session_state['cutoff'] == cutoff
    formula = compile_blocks(app.session_state['block_items'])
    inputs = {name: float(active.loc[ticker, name]) for name in
        ('predicted_return', 'momentum', 'uncertainty', 'volatility')}
    expected = evaluate_formula(formula, inputs, app.session_state['valid_k'])
    assert pd.Timestamp(changed.data[0].x[-1]) == pd.Timestamp(cutoff)
    assert changed.data[0].y[-1] == pytest.approx(expected)
    assert not app.session_state['reveals']
    app.button(key='block_apply').click().run()
    assert not app.exception
    assert app.session_state['valid_formula'] == formula
    assert app.session_state['current_scores'].loc[ticker, 'score'] == pytest.approx(expected)
    assert chart_data(builder_figures['known_price_hero']) == original_prices
    assert chart_data(builder_figures['builder_stock_chart']) == original_prices
    assert not app.session_state['reveals']


def test_research_range_updates_both_builder_charts_without_changing_scores(builder_figures):
    app = dashboard()
    active = app.session_state['current_scores'].copy(deep=True)
    before = builder_figures['builder_stock_chart']
    previous_count = len(before.data[0].x)
    previous_start = pd.Timestamp(before.data[0].x[0])
    app.selectbox(key='chart_period').set_value('1M').run()
    assert not app.exception
    research = builder_figures['known_price_hero']
    mirror = builder_figures['builder_stock_chart']
    score_chart = builder_figures['builder_algorithm_chart']
    assert chart_data(research) == chart_data(mirror)
    assert len(mirror.data[0].x) < previous_count
    start = pd.Timestamp(mirror.data[0].x[0])
    end = pd.Timestamp(mirror.data[0].x[-1])
    assert start > previous_start
    dates = pd.DatetimeIndex(score_chart.data[0].x)
    assert len(dates) > 0
    assert dates.min() >= start
    assert dates.max() <= end
    assert_known_dates(score_chart, app.session_state['cutoff'])
    pd.testing.assert_frame_equal(app.session_state['current_scores'], active)
    assert not app.session_state['reveals']


def test_backtest_compares_algorithm_and_selected_stock_on_the_same_dates(builder_figures):
    app = dashboard()
    ticker = app.selectbox(key='stock').value
    app.button(key='run_backtest').click().run()
    assert not app.exception
    result = app.session_state['backtest_result']
    figure = builder_figures['backtest_chart']
    assert {trace.name for trace in figure.data} == {'Momentum', 'AI model', 'Your algorithm', f'{ticker} buy & hold'}
    stock = next(trace for trace in figure.data if trace.name == f'{ticker} buy & hold')
    assert list(pd.to_datetime(stock.x)) == list(result['linked_returns'].index)
    assert stock.y[0] == 0
    observed, _ = market_data.load_symbol_prices(ticker)
    expected_end = (observed.loc[stock.x[-1]] / observed.loc[stock.x[0]] - 1) * 100
    assert stock.y[-1] == pytest.approx(expected_end)
    algorithm = next(trace for trace in figure.data if trace.name == 'Your algorithm')
    assert algorithm.y[-1] == pytest.approx(result['summary'].loc['student', 'linked_return'])
    assert not app.session_state['reveals']
