"""Dashboard interactions preserve the historical experiment and data scope."""
from pathlib import Path
import socket

import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from derivport import market_data, ohlc, us_markets
from derivport.theme import theme_from_state


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_dashboard(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError('Bundled dashboard interactions must remain offline.')
    monkeypatch.setattr(socket, 'create_connection', deny_network)
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    monkeypatch.setattr(market_data, '_provider', deny_network)
    monkeypatch.setattr(ohlc, '_provider', deny_network)
    monkeypatch.setenv('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    figures = {}
    real_chart = st.plotly_chart
    def capture_chart(*args, **kwargs):
        if kwargs.get('key') in ('known_price_hero', 'fullscreen_price_chart',
                'builder_stock_chart', 'builder_algorithm_chart',
                'future_scenario_chart', 'future_distribution_chart'):
            figures[kwargs['key']] = go.Figure(args[0])
        return real_chart(*args, **kwargs)
    monkeypatch.setattr(st, 'plotly_chart', capture_chart)
    return figures


def dashboard():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    assert not app.exception
    return app


def assert_unrevealed(app):
    assert not app.session_state['reveals']
    assert 'actual_return' not in app.session_state['current_scores']
    assert not any('Actual 21d return' in table.value.columns or 'Realized rank' in table.value.columns
        for table in app.dataframe)


def test_full_stock_directory_and_exchange_navigation_use_cached_defaults_offline(offline_dashboard):
    app = dashboard()
    options = app.selectbox(key='stock').options
    assert len(options) > 5000
    assert any(value.startswith('BAC ·') for value in options)
    for excluded in ('SPY ·', 'QQQ ·', '7203.T ·', 'SAP.DE ·'):
        assert not any(value.startswith(excluded) for value in options)
    # The visible directory is broad; the research comparison menu still
    # contains only prices that can be used without a provider request.
    assert not any('· BAC' in value for value in app.multiselect(key='universe_selection').options)
    before = list(app.multiselect(key='universe_selection').value)
    app.selectbox(key='exchange_filter').set_value('NYSE').run()
    assert not app.exception
    assert app.selectbox(key='stock').value == 'JPM'
    assert any(value.startswith('JPM ·') for value in app.selectbox(key='stock').options)
    assert not any(value.startswith('AAPL ·') for value in app.selectbox(key='stock').options)
    assert set(before) <= set(app.multiselect(key='universe_selection').value)
    assert 'JPM' in app.session_state['current_scores'].index
    assert_unrevealed(app)


def test_display_modes_update_stock_builder_and_fullscreen_without_changing_research(offline_dashboard):
    app = dashboard()
    scores = app.session_state['current_scores'].copy(deep=True)
    bars = app.session_state['chart_data'].copy(deep=True)
    cutoff = app.session_state['cutoff']
    stock_options = app.selectbox(key='stock').options
    chart_options = app.selectbox(key='chart_style').options
    assert app.radio(key='display_mode').value == 'Day'
    assert app.radio(key='display_mode').options == ['Day', 'Night', 'Pause']
    assert not any(element.proto.popover.label == 'Theme' for element in app.get('popover'))
    backgrounds = set()
    for mode in ('Night', 'Pause', 'Day'):
        app.radio(key='display_mode').set_value(mode).run()
        assert not app.exception
        palette = theme_from_state({'theme_name': mode})
        rendered = '\n'.join(element.value for element in app.html)
        assert 'Helvetica Neue' in rendered
        assert f"--workspace:{palette['background']}" in rendered
        assert app.session_state['theme_name'] == mode
        backgrounds.add(palette['background'])
        current = offline_dashboard['known_price_hero']
        for key in ('known_price_hero', 'builder_stock_chart', 'builder_algorithm_chart'):
            figure = offline_dashboard[key]
            assert figure.layout.paper_bgcolor == palette['panel']
            assert figure.layout.plot_bgcolor == palette['panel']
            assert 'Helvetica Neue' in figure.layout.font.family
        assert current.data[0].type == 'candlestick'
        mirror = offline_dashboard['builder_stock_chart']
        assert list(mirror.data[0].x) == list(current.data[0].x)
        assert list(mirror.data[0].close) == list(current.data[0].close)
        assert app.selectbox(key='stock').options == stock_options
        assert app.selectbox(key='chart_style').options == chart_options
        assert app.session_state['cutoff'] == cutoff
        pd.testing.assert_frame_equal(app.session_state['chart_data'], bars)
        pd.testing.assert_frame_equal(app.session_state['current_scores'], scores)
        assert_unrevealed(app)

        # An ordinary rerun must retain the selected mode, and opening the
        # separate chart dialog must use that mode without altering OHLCV.
        app.run()
        assert not app.exception
        assert app.radio(key='display_mode').value == mode
        app.button(key='open_fullscreen_chart').click().run()
        assert not app.exception
        fullscreen = offline_dashboard['fullscreen_price_chart']
        assert fullscreen.layout.paper_bgcolor == palette['panel']
        assert fullscreen.layout.plot_bgcolor == palette['panel']
        assert fullscreen.data[0].type == 'candlestick'
        for field in ('x', 'open', 'high', 'low', 'close'):
            assert list(getattr(fullscreen.data[0], field)) == list(getattr(current.data[0], field))
        assert app.session_state['cutoff'] == cutoff
        pd.testing.assert_frame_equal(app.session_state['chart_data'], bars)
        pd.testing.assert_frame_equal(app.session_state['current_scores'], scores)
        assert_unrevealed(app)
    assert len(backgrounds) == 3


def test_coach_chart_actions_work_in_every_display_mode_without_changing_research(offline_dashboard):
    app = dashboard()
    # Keep a user-selected indicator while the coach temporarily adds its own.
    app.multiselect(key='chart_overlays').set_value(['SMA 50']).run()
    assert not app.exception
    scores = app.session_state['current_scores'].copy(deep=True)
    fields = ['open', 'high', 'low', 'close', 'volume']
    bars = app.session_state['chart_data'][fields].copy(deep=True)
    cutoff = app.session_state['cutoff']

    def assert_research_unchanged(mode):
        assert not app.exception
        assert app.radio(key='display_mode').value == mode
        assert app.session_state['cutoff'] == cutoff
        pd.testing.assert_frame_equal(app.session_state['current_scores'], scores)
        pd.testing.assert_frame_equal(app.session_state['chart_data'][fields], bars)
        assert_unrevealed(app)

    def coach_shapes(figure):
        return [shape for shape in figure.layout.shapes if str(shape.name or '').startswith('coach-')]

    for mode in ('Day', 'Night', 'Pause'):
        app.radio(key='display_mode').set_value(mode).run()
        assert_research_unchanged(mode)
        palette = theme_from_state({'theme_name': mode})

        app.button(key='coach_mark_trend').click().run()
        assert_research_unchanged(mode)
        figure = offline_dashboard['known_price_hero']
        trend, = coach_shapes(figure)
        assert trend.name == 'coach-trend'
        assert pd.Timestamp(trend.x0) == bars.index[0]
        assert pd.Timestamp(trend.x1) == bars.index[-1] <= pd.Timestamp(cutoff)
        assert trend.y0 == pytest.approx(bars.close.iloc[0])
        assert trend.y1 == pytest.approx(bars.close.iloc[-1])
        assert trend.line.color == palette['accent']

        app.button(key='open_fullscreen_chart').click().run()
        assert_research_unchanged(mode)
        fullscreen = offline_dashboard['fullscreen_price_chart']
        assert fullscreen.layout.paper_bgcolor == palette['panel']
        fullscreen_trend, = coach_shapes(fullscreen)
        assert fullscreen_trend.to_plotly_json() == trend.to_plotly_json()

        app.button(key='coach_mark_range').click().run()
        assert_research_unchanged(mode)
        low, high = coach_shapes(offline_dashboard['known_price_hero'])
        assert (low.name, high.name) == ('coach-window-low', 'coach-window-high')
        assert low.y0 == low.y1 == pytest.approx(bars.low.min())
        assert high.y0 == high.y1 == pytest.approx(bars.high.max())
        for shape in (low, high):
            assert pd.Timestamp(shape.x0) == bars.index[0]
            assert pd.Timestamp(shape.x1) == bars.index[-1] <= pd.Timestamp(cutoff)

        app.button(key='coach_mark_average').click().run()
        assert_research_unchanged(mode)
        assert app.multiselect(key='chart_overlays').value == ['SMA 50', 'SMA 20']
        figure = offline_dashboard['known_price_hero']
        average = next(trace for trace in figure.data if trace.name == 'SMA 20')
        pd.testing.assert_series_equal(pd.Series(average.y),
            app.session_state['chart_data']['SMA 20'].reset_index(drop=True), check_names=False)
        assert any(annotation.name == 'coach-moving-average-label' for annotation in figure.layout.annotations)

        app.button(key='coach_clear_guides').click().run()
        assert_research_unchanged(mode)
        assert app.multiselect(key='chart_overlays').value == ['SMA 50']
        assert not coach_shapes(offline_dashboard['known_price_hero'])
        assert not any(str(annotation.name or '').startswith('coach-')
            for annotation in offline_dashboard['known_price_hero'].layout.annotations)
        assert 'coach_chart_guide' not in app.session_state


def test_projection_risk_controls_and_distinct_simulation_charts_preserve_history(offline_dashboard):
    app = dashboard()
    scores = app.session_state['current_scores'].copy(deep=True)
    cutoff = app.session_state['cutoff']
    app.slider(key='projection_drift').set_value(8.5).run()
    assert not app.exception
    for preset, multiplier in (('High', 1.5), ('Low', .75), ('Medium', 1.0), ('High', 1.5)):
        app.radio(key='projection_risk').set_value(preset).run()
        assert not app.exception
        assert app.slider(key='projection_volatility').value == multiplier
        assert app.slider(key='projection_drift').value == 8.5
    app.button(key='project_future').click().run()
    assert not app.exception
    fan = offline_dashboard['future_scenario_chart']
    distribution = offline_dashboard['future_distribution_chart']
    median = next(trace for trace in fan.data if trace.name == 'Simulated median')
    assert distribution.data[0].type == 'histogram'
    assert median.line.color != distribution.data[0].marker.color
    assert median.line.color != offline_dashboard['known_price_hero'].data[0].increasing.line.color
    assert fan.layout.meta['kind'] == 'simulated_percentile_fan'
    assert distribution.layout.meta['kind'] == 'simulated_terminal_distribution'
    assert 'Hypothetical simulation' in fan.layout.meta['status']
    assert all(pd.Timestamp(stamp) >= pd.Timestamp(cutoff) for stamp in median.x)
    assert app.session_state['projection_result'].assumptions['volatility_multiplier'] == 1.5
    assert app.session_state['cutoff'] == cutoff
    pd.testing.assert_frame_equal(app.session_state['current_scores'], scores)
    assert_unrevealed(app)


def test_unavailable_directory_selection_recovers_without_poisoning_research(offline_dashboard, monkeypatch):
    original_cached = us_markets.cached_us_listings
    monkeypatch.setattr(us_markets, 'cached_us_listings', lambda exchange=None:
        [row for row in original_cached(exchange) if row['symbol'] != 'BAC'])
    original_load = us_markets.load_us_symbol_prices
    original_resolve = us_markets.resolve_us_listing
    def unavailable_load(symbol, refresh=False):
        if symbol.upper() == 'BAC':
            raise market_data.MarketDataError('BAC: simulated provider history unavailable.')
        return original_load(symbol, refresh=refresh)
    def unavailable_identity(symbol, metadata=None):
        if symbol.upper() == 'BAC':
            raise market_data.MarketDataError('BAC: simulated listing provider unavailable.')
        return original_resolve(symbol, metadata)
    monkeypatch.setattr(us_markets, 'load_us_symbol_prices', unavailable_load)
    monkeypatch.setattr(us_markets, 'resolve_us_listing', unavailable_identity)
    app = dashboard()
    before = app.session_state['current_scores'].copy(deep=True)
    group = list(app.multiselect(key='universe_selection').value)
    app.selectbox(key='stock').set_value('BAC').run()
    assert not app.exception
    assert any('unavailable' in value.value.lower() for value in [*app.warning, *app.info])
    assert 'BAC' not in app.session_state['universe_selection']
    assert not app.session_state['reveals']
    app.selectbox(key='stock').set_value('AAPL').run()
    assert not app.exception
    assert app.multiselect(key='universe_selection').value == group
    assert 'BAC' not in app.session_state['current_scores'].index
    pd.testing.assert_frame_equal(app.session_state['current_scores'], before)
    assert_unrevealed(app)
