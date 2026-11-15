"""DerivPort: historical research and a transparent algorithm builder."""
from __future__ import annotations
from datetime import date
import hashlib
import html
import json
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from derivport import (DEFAULT_FORMULA, FormulaError, analyze_cutoff, apply_rule,
    backtest_rule, default_crossovers, evaluate_history, formula_curve,
    historical_forecasts, load_prices, rank_correlation,
    reveal_outcomes, simulate_paths, simulation_summary, SCENARIOS)
from derivport.market_data import (MarketDataError, get_search_status,
    load_symbol_prices, symbol_info, verify_frozen_cache)
from derivport.us_markets import (US_EXCHANGES, cached_us_listings,
    load_us_symbol_prices, search_us_symbols, us_catalog, directory_provenance, resolve_us_listing)
from derivport.stock_chart_ui import render_stock_chart
from derivport.company_news_ui import render_company_news
from derivport.ohlc import load_candles
from derivport.theme import current_theme, theme_css, style_plot
from derivport.blocks_ui import render_blocks
from derivport.projections import project_scenarios
from derivport.projection_views import (RISK_PRESETS, apply_risk_preset,
    scenario_fan, scenario_distribution)
from derivport.calculus import calculus_experiment, differential, formula_to_latex
from derivport.calculus_ui import render_calculus_lab
from derivport.research_ui import render_benchmarking
from algorithms.derivport_signal import NAME as SIGNAL_NAME, FORMULA as SIGNAL_FORMULA, DEFAULT_K as SIGNAL_K

ROOT = Path(__file__).parent
DEFAULT_STOCKS = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA']
COLORS = {'Momentum': '#c8b82e', 'AI model': '#b8a8d0', 'Your algorithm': '#d9e5d2'}
st.set_page_config(page_title='DerivPort | Equity research', layout='wide', page_icon='D')


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return f'{float(value):+.2f}%'


def heading(title, description=''):
    st.html(f'<div class="section-title"><h2>{esc(title)}</h2><span>{esc(description)}</span></div>')


def asset_uri(name, decorative=True):
    path = ROOT / 'assets' / name
    if not path.exists() or (decorative and not st.session_state.get('theme_images', True)):
        return ''
    mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()


def visual_heading(title, description, image):
    uri = asset_uri(image)
    picture = f'<img src="{uri}" alt="{esc(title)} illustration">' if uri else ''
    st.html(f'<div class="visual-header"><div><h1>{esc(title)}</h1><p>{esc(description)}</p></div>{picture}</div>')


def chart(fig, height=230):
    margins = dict(l=10, r=16, t=38, b=20) | fig.layout.margin.to_plotly_json()
    legend = dict(orientation='h', y=1.03, x=0, yanchor='bottom') | fig.layout.legend.to_plotly_json()
    fig.update_layout(height=height, margin=margins,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif', size=12),
        legend=legend)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='#e6e4dd')
    fig.update_yaxes(gridcolor='#e6e4dd', zerolinecolor='#b9bdb5')
    return style_plot(fig, PALETTE)


@st.cache_data(show_spinner=False)
def frozen_prices():
    verify_frozen_cache()
    return load_prices(ROOT / 'data/prices.csv')


@st.cache_data(show_spinner=False)
def cached_symbol(symbol):
    if any(item['symbol'] == symbol for item in cached_us_listings()):
        return load_symbol_prices(symbol)
    result = load_us_symbol_prices(symbol)
    try:
        load_candles(symbol, allow_download=True)
    except MarketDataError:
        pass  # The chart honestly falls back to verified adjusted closes.
    return result


@st.cache_data(show_spinner='Preparing historical forecasts…')
def cached_history(prices):
    return historical_forecasts(prices)


@st.cache_data(show_spinner=False)
def cached_analysis(prices, cutoff, history):
    return analyze_cutoff(prices, cutoff, history)


@st.cache_data(show_spinner=False)
def cached_simulation(stocks, scenario):
    return simulate_paths(stocks, scenario, n_paths=600, seed=2024)


@st.cache_data(show_spinner=False)
def cached_projection(prices, target, drift, multiplier, correlation, paths):
    return project_scenarios(prices, target, annual_drift=drift / 100,
        volatility_multiplier=multiplier, correlation=correlation, n_paths=paths, seed=2037)


st.session_state.setdefault('valid_formula', DEFAULT_FORMULA)
st.session_state.setdefault('valid_k', 2.0)
st.session_state.setdefault('reveals', {})
st.session_state.setdefault('seen_backtests', [])
if 'history_mode' not in st.session_state:
    st.session_state.history_mode = 'Full cached history'
    st.session_state.pop('decision_date', None)
if 'pending_stock' in st.session_state:
    pending = st.session_state.pop('pending_stock')
    current = list(st.session_state.get('universe_selection', DEFAULT_STOCKS))
    if pending not in current:
        current = (current + [pending])[-10:]
    st.session_state.universe_selection = current
    st.session_state.stock = pending
    st.session_state.history_mode = 'Full cached history'

# One fixed editorial palette, including sessions opened before theme removal.
st.session_state.theme_name = 'Editorial'
st.session_state.theme_accent = '#28543a'
st.session_state.theme_density = 'Comfortable'
PALETTE = current_theme()
st.html('<style>' + (ROOT / 'assets/style.css').read_text() + '\n' + theme_css(PALETTE) + '</style>')
COLORS = {'Momentum': '#90a4bf', 'AI model': PALETTE['accent'], 'Your algorithm': '#ffbd69'}
mark = asset_uri('derivport-mark.svg', decorative=False)
mark_html = f'<img class="brand-mark" src="{mark}" alt="DerivPort logo">' if mark else ''
st.html(f'<div class="masthead"><div class="brand">{mark_html}DerivPort</div></div>')

def clear_search():
    st.session_state.pop('search_results', None)
    st.session_state.pop('search_status', None)


def change_history_mode():
    for key in ('decision_date', 'test_start', 'test_end'):
        st.session_state.pop(key, None)


def change_comparison_group():
    st.session_state.pop('chart_auto_member', None)
    group = st.session_state.universe_selection
    if group and st.session_state.get('stock') not in group:
        st.session_state.stock = group[0]
        st.session_state.exchange_filter = 'All US exchanges'


cached_listings = cached_us_listings()
cached_lookup = {item['symbol']: item for item in cached_listings}
us_listings = us_catalog()
us_lookup = {row['symbol']: dict(cached_lookup.get(row['symbol'], {}), **row) for row in us_listings}
us_lookup.update({key:value for key,value in cached_lookup.items() if key not in us_lookup})
# Migrate a session opened before the US-only scope change.
prior_group = st.session_state.get('universe_selection', DEFAULT_STOCKS)
clean_group = [symbol for symbol in prior_group if symbol in us_lookup]
if clean_group != prior_group:
    st.session_state.universe_selection = clean_group or list(DEFAULT_STOCKS)
exchange_choices = ['All US exchanges'] + [venue for venue in US_EXCHANGES
    if any(row['exchange'] == venue for row in us_listings)]
if st.session_state.get('exchange_filter') not in exchange_choices:
    st.session_state.exchange_filter = 'All US exchanges'
stock_col, date_col, settings_col, status_col = st.columns([2.4, 1.1, .8, 1], gap='medium', vertical_alignment='bottom')
with settings_col:
    market_settings = st.popover('Settings', width='stretch')
with market_settings:
    exchange_col, mode_col = st.columns(2)
with exchange_col:
    exchange_filter = st.selectbox('Exchange', exchange_choices, key='exchange_filter')
chart_listings = [row for row in us_listings if exchange_filter == 'All US exchanges' or row['exchange'] == exchange_filter]
stock_options = [row['symbol'] for row in chart_listings]
if st.session_state.get('stock') not in stock_options:
    group = st.session_state.get('universe_selection', DEFAULT_STOCKS)
    st.session_state.stock = next((symbol for symbol in group if symbol in stock_options),
        next((symbol for symbol in cached_lookup if symbol in stock_options), stock_options[0]))
with stock_col:
    ticker = st.selectbox('Stock', stock_options,
        format_func=lambda s: f"{s} · {us_lookup[s].get('name', s)}", key='stock')
# New selections download only that stock. Failed additions never poison the
# comparison group; a recent IPO can still show an observed chart alone.
if ticker not in cached_lookup:
    try:
        with st.spinner(f'Loading {ticker} price history…'):
            cached_symbol(ticker)
        cached_listings = cached_us_listings()
        cached_lookup = {item['symbol']: item for item in cached_listings}
    except MarketDataError as exc:
        st.warning(str(exc))
        try:
            listing = resolve_us_listing(ticker)
            bars, candle_meta = load_candles(ticker, allow_download=True)
            st.info('Chart only: this stock does not yet have the verified adjusted history needed for the research models. Select another stock to continue research.')
            render_stock_chart(bars.close, ticker, str(bars.index[-1].date()), dict(candle_meta, exchange=listing['exchange']))
        except MarketDataError:
            st.info('This listing is in the directory, but its price history is currently unavailable. Select a cached stock or try another listing.')
        st.stop()
group = list(st.session_state.get('universe_selection', DEFAULT_STOCKS))
previous_auto = st.session_state.get('chart_auto_member')
if previous_auto and previous_auto != ticker:
    group = [symbol for symbol in group if symbol != previous_auto]
    st.session_state.universe_selection = group
    st.session_state.pop('chart_auto_member', None)
if ticker not in group:
    st.session_state.universe_selection = (group + [ticker])[-10:]
    st.session_state.chart_auto_member = ticker
with mode_col:
    history_mode = st.selectbox('Price history', ['Full cached history', 'Frozen demo (2018–2024)'],
        key='history_mode', on_change=change_history_mode)
extended = history_mode == 'Full cached history'

with market_settings:
    st.caption(f"{len(us_listings):,} current US equity candidates from the NasdaqTrader directory. Choose any listing above; only the selected stock is downloaded. Cached histories work offline. Source snapshot: 2026-11-13; provider coverage varies.")
    directory_col, query_col, search_col = st.columns([1.4, 3, 1], vertical_alignment='bottom')
    with directory_col:
        directory_exchange = st.selectbox('Directory exchange', ['All US exchanges', *US_EXCHANGES], key='directory_exchange', on_change=clear_search)
    venue = None if directory_exchange == 'All US exchanges' else directory_exchange
    with query_col:
        query = st.text_input('Company or symbol', placeholder='Apple, JPMorgan, NVDA, BRK-B…', key='stock_query', on_change=clear_search)
    with search_col:
        if st.button('Search listings', key='search_listings', width='stretch'):
            with st.spinner('Searching equity listings…'):
                st.session_state.search_results = search_us_symbols(query, online=True, exchange=venue)
                st.session_state.search_status = get_search_status()
    matches = st.session_state.get('search_results', us_catalog(venue) if not query else search_us_symbols(query, online=False, exchange=venue))
    if matches:
        lookup = {item['symbol']: item for item in matches}
        if st.session_state.get('listing_match') not in lookup:
            st.session_state.pop('listing_match', None)
        match = st.selectbox('Matching listings', list(lookup),
            format_func=lambda s: f"{s} · {lookup[s]['name']} · {lookup[s]['exchange']}", key='listing_match')
    else:
        match = query.strip().upper()
    if st.session_state.get('search_status'):
        st.caption(st.session_state.search_status['message'])
        st.caption('Only supported US-listed equities are shown. This starter directory and provider search do not guarantee coverage of every US stock.')
    load_col, refresh_col = st.columns(2)
    with load_col:
        if st.button('Cache and add stock', disabled=not bool(match), key='add_stock'):
            try:
                with st.spinner(f'Loading adjusted history for {match}…'):
                    _, added_meta = load_us_symbol_prices(match)
                st.session_state.pending_stock = added_meta['symbol']
                cached_symbol.clear()
                st.rerun()
            except MarketDataError as exc:
                st.error(str(exc))
    with refresh_col:
        if st.button('Refresh selected listing', disabled=not bool(match), key='refresh_stock'):
            try:
                with st.spinner('Refreshing adjusted price history…'):
                    _, refreshed = load_us_symbol_prices(match, refresh=True)
                cached_symbol.clear()
                if refreshed.get('warning'):
                    st.warning(refreshed['warning'])
                else:
                    st.success(f"{refreshed['symbol']} cached through {refreshed['end']}.")
            except MarketDataError as exc:
                st.error(str(exc))
    st.multiselect('Comparison group · 2–10 stocks', list(cached_lookup), default=DEFAULT_STOCKS,
        format_func=lambda s: f"{us_lookup[s]['exchange']} · {s}",
        max_selections=10, key='universe_selection', on_change=change_comparison_group)
    st.caption('Full cached history includes the downloaded 2026 data. Frozen demo preserves the original five-stock 2018–2024 example. Choose the range in the visible Price history control.')
    st.caption('Rankings and backtests use this US stock group across exchanges. Only shared observed sessions are used; no missing dates are filled.')

universe = sorted(st.session_state.get('universe_selection', DEFAULT_STOCKS))
if len(universe) < 2:
    st.info('Choose at least two stocks in Stocks & comparison group.')
    st.stop()
series, metadata = [], {}
try:
    for symbol in universe:
        if not extended and symbol in DEFAULT_STOCKS:
            values = frozen_prices()[symbol]
            info = symbol_info(symbol)
            info.update(source='Yahoo Finance via pinned Akma86 mirror', source_url='https://github.com/Akma86/us-tech-giants-market-analytics',
                start='2018-01-02', end='2024-12-31', fetched_at='2026-11-14', cache_status='frozen demo')
        else:
            values, info = cached_symbol(symbol)
        info = dict(info, exchange=us_lookup[symbol]['exchange'])
        series.append(values.rename(symbol))
        metadata[symbol] = info
    prices = pd.concat(series, axis=1, sort=True).dropna().sort_index()
except MarketDataError as exc:
    st.error(str(exc))
    st.stop()
if len(prices) < 295:
    st.info('Chart only: this comparison needs 295 shared observations for training and replay. Remove a recently listed stock from the group, or browse an established stock to restore research.')
    own_history = next(item for item in series if item.name == ticker)
    if not own_history.empty:
        render_stock_chart(own_history, ticker, str(own_history.index[-1].date()), metadata[ticker])
    st.stop()
earliest = max(prices.index[252], pd.Timestamp('2020-04-01'))
latest = prices.index[-1]
latest_replay = prices.index[-22]
st.session_state.latest_historical_date = latest.date().isoformat()
st.session_state.latest_replay_date = latest_replay.date().isoformat()
if earliest > latest:
    st.info('This group needs more historical data. Choose an earlier-listed stock.')
    st.stop()
initial_date = latest.date() if extended else date(2024, 5, 3)
existing_date = pd.Timestamp(st.session_state.get('decision_date', initial_date))
if not earliest <= existing_date <= latest:
    st.session_state.decision_date = min(max(existing_date, earliest), latest).date()
with date_col:
    default_date = min(max(pd.Timestamp(initial_date), earliest), latest).date()
    chosen_date = st.date_input('Historical cutoff', value=default_date, min_value=earliest.date(),
        max_value=latest.date(), format='YYYY-MM-DD', key='decision_date')
cutoff = prices.index[prices.index <= pd.Timestamp(chosen_date)][-1].date().isoformat()
st.session_state.cutoff = cutoff
data_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest()[:12]
context = ','.join(universe) + '|' + cutoff + '|' + data_fingerprint
st.session_state.active_replay_key = context
subsequent_sessions = len(prices) - prices.index.get_loc(pd.Timestamp(cutoff)) - 1
outcome_complete = subsequent_sessions >= 21
with status_col:
    st.html(f'<div class="data-status">{len(prices):,} shared sessions · {len(universe)} stocks<br>Cache through {prices.index[-1]:%d %b %Y}</div>')
if cutoff != chosen_date.isoformat():
    st.caption(f'{chosen_date} is not a shared trading date. Using the preceding observed session, {cutoff}.')
with market_settings:
    st.caption(f"Historical data ends {latest:%Y-%m-%d}. Completed replays through {latest_replay:%Y-%m-%d}. Future dates through 2037 are simulations.")
history = cached_history(prices)
try:
    analysis = cached_analysis(prices, cutoff, history)
except ValueError as exc:
    st.info(f'{exc} Choose a later date or a group with longer shared history.')
    st.stop()
base, meta = analysis.stocks, metadata[ticker]
notes = json.loads((ROOT / 'data/notes.json').read_text())
available_notes = [n for n in notes if n['ticker'] == ticker and pd.Timestamp(n['published']) < pd.Timestamp(cutoff)]
note = max(available_notes, key=lambda n: n['published']) if available_notes else None
research, builder, future, benchmarking = st.tabs(['Research', 'Algorithm builder', 'Future scenarios', 'Benchmarks'], key='workspace')
with research:
    render_stock_chart(prices[ticker], ticker, cutoff, meta,
        coach_snapshot=dict(row=base.loc[ticker], note=note, context=context))
    if ticker == 'AAPL' and cutoff == '2024-05-03' and not extended:
        st.caption('Curated example: Apple after its May 2 earnings release. Choose any available date or change the stock group.')
    render_company_news(meta.get('name', ticker), ticker, prices[ticker], cutoff, plotstyle=lambda figure: chart(figure, 280))
    with st.expander('Research tools', expanded=False):
        panels = st.expander('Company & model details', expanded=False)
        ranks_area = st.expander('Compare stock rankings', expanded=False)
        replay_area = st.expander('Historical replay', expanded=False)
        extras_area = st.expander('More research tools', expanded=False)
with builder:
    visual_heading('Build your own signal.', 'Choose what matters. See why stocks rank. Test your idea on the past.', 'rule-lab.svg')
    render_blocks(base, ticker)
    with st.expander('Example recipe & Python download'):
        with st.container(border=True, key='signal_card'):
            st.html(f'<div class="summary-signal"><div><h2>{esc(SIGNAL_NAME)}</h2><p>A balanced model and momentum score with explicit error and volatility penalties.</p></div></div>')
            signal_info, signal_actions = st.columns([2, 1], gap='large')
            with signal_info:
                st.latex(r'S = ' + formula_to_latex(SIGNAL_FORMULA))
                st.caption('60% model estimate + 40% trailing momentum. k controls a penalty of 50% forecast error + 25% volatility. Fixed starting weights; no claim that they are optimal.')
            with signal_actions:
                def load_balanced_algorithm():
                    st.session_state.formula = SIGNAL_FORMULA
                    st.session_state.k = float(SIGNAL_K)
                st.button('Use my balanced algorithm', key='load_balanced_algorithm',
                    on_click=load_balanced_algorithm, type='primary', width='stretch')
                st.download_button('Download Python algorithm', (ROOT / 'algorithms/derivport_signal.py').read_bytes(),
                    file_name='derivport_signal.py', mime='text/x-python', key='download_algorithm', width='stretch')
    with st.expander('Explore the formula & Calculus Lab', expanded=False):
        editor, sensitivity = st.columns(2, gap='large')
        with editor, st.container(border=True, key='algorithm_editor'):
            formula = st.text_input('Ranking formula', value=st.session_state.valid_formula, key='formula', max_chars=400)
            st.caption('Variables: predicted_return, uncertainty, momentum, volatility, k. Use numbers, + − * / and parentheses. Return inputs are percentage points.')
            k = st.slider('Risk penalty weight · k', 0.0, 10.0, value=float(st.session_state.valid_k), step=.05, key='k')
            old_formula, old_k = st.session_state.valid_formula, st.session_state.valid_k
            try:
                scores = apply_rule(base, formula, k)
                if (formula, k) != (old_formula, old_k):
                    try:
                        old_scores = apply_rule(base, old_formula, old_k)
                        change = float(scores.loc[ticker, 'score'] - old_scores.loc[ticker, 'score'])
                        cause = f'k ({old_k:g} → {k:g})' if formula == old_formula else 'the formula'
                        st.session_state.rule_change = {'context': context, 'text': f"Changing {cause} changed {ticker}'s score by {change:+.2f}; stock inputs stayed fixed."}
                    except FormulaError:
                        st.session_state.pop('rule_change', None)
                st.session_state.valid_formula, st.session_state.valid_k = formula, k
            except FormulaError as exc:
                st.error(f'{exc} Showing the last valid formula and k.')
                try:
                    scores = apply_rule(base, old_formula, old_k)
                except FormulaError:
                    st.warning('The previous rule is undefined for this new group or date. Using the default rule until you enter a valid expression.')
                    scores = apply_rule(base, DEFAULT_FORMULA, old_k)
                    st.session_state.valid_formula = DEFAULT_FORMULA
            st.session_state.current_scores = scores.copy()
            active_formula, active_k = st.session_state.valid_formula, st.session_state.valid_k
            st.session_state.active_algorithm_name = SIGNAL_NAME if active_formula.strip() == SIGNAL_FORMULA else 'Custom expression'
            st.caption(f'Active algorithm: {st.session_state.active_algorithm_name}')
            row = scores.loc[ticker]
            values = {name: float(row[name]) for name in ('predicted_return', 'uncertainty', 'momentum', 'volatility')}
            values['k'] = active_k
            st.latex(r'S = ' + formula_to_latex(active_formula))
            st.latex(r'S \approx ' + formula_to_latex(active_formula, values) + rf' \approx {row.score:.3f}')
            st.markdown(r'$\widehat r_{21}$ is the model estimate, $M_{21}$ momentum, $\varepsilon$ past forecast error, and $\sigma_{21}$ volatility. Return-like inputs are percentage points.')
            st.caption('Substituted inputs are rounded; scores use full precision. A ranking score is not a return forecast.')
            if active_formula.strip() == DEFAULT_FORMULA:
                st.latex(rf'k\varepsilon \approx ({active_k:g})({row.uncertainty:.3f}) \approx {active_k * row.uncertainty:.3f}\;\mathrm{{pp}}')
            elif active_formula.strip() == SIGNAL_FORMULA:
                st.latex(rf'k(0.5\varepsilon+0.25\sigma_{{21}}) \approx {active_k * (.5 * row.uncertainty + .25 * row.volatility):.3f}\;\mathrm{{pp}}')
            else:
                st.caption('Custom arithmetic can change score units; return × return differs from a return. Normalize ratios when a dimensionless term is intended.')
            change_note = st.session_state.get('rule_change')
            if change_note and change_note['context'] == context:
                st.caption(change_note['text'])
            crossings = default_crossovers(base, max_k=10)
            if len(crossings):
                crossing = crossings.iloc[(crossings.k - active_k).abs().argsort()[:1]].iloc[0]
                low = max(0.0, round(float(np.floor(crossing.k / .05) * .05), 2))
                target = low if active_k > crossing.k else min(10.0, round(low + .05, 2))
                st.caption(f"Default-rule crossover: {crossing['first']} / {crossing['second']} at k ≈ {crossing.k:.3f}. Scores change continuously; ranks change only when two scores cross.")
                def try_crossover():
                    st.session_state.formula = DEFAULT_FORMULA
                    st.session_state.k = target
                st.button(f'Try default rule at k = {target:.2f}', key='try_crossover', on_click=try_crossover)
        with sensitivity, st.container(border=True, key='algorithm_sensitivity'):
            st.html('<div class="small-title">Score sensitivity</div>')
            curve = formula_curve(active_formula, values, active_k)
            fig = go.Figure(go.Scatter(x=curve.uncertainty, y=curve.score, mode='lines', line=dict(color='#2454ff', width=2.5), showlegend=False))
            fig.add_trace(go.Scatter(x=[row.uncertainty], y=[row.score], mode='markers', marker=dict(color='#101820', size=9), showlegend=False))
            fig.add_vline(x=row.uncertainty, line_dash='dot', line_color='#bbc3ce')
            fig.update_xaxes(title='Uncertainty (percentage points)')
            fig.update_yaxes(title='Rule score')
            try:
                derivatives = differential(active_formula, values, active_k)
                slope = derivatives.gradient['uncertainty']
                curvature = derivatives.hessian['uncertainty']['uncertainty']
                fig.add_trace(go.Scatter(x=curve.uncertainty,
                    y=row.score + slope * (curve.uncertainty - row.uncertainty), name='Tangent',
                    line=dict(color='#a1a7b0', width=1.5, dash='dot'), showlegend=False))
                st.latex(rf'\left.\frac{{\partial S}}{{\partial\varepsilon}}\right|_{{\mathbf{{x}}_0}}={slope:.4g},\qquad \left.\frac{{\partial^2 S}}{{\partial\varepsilon^2}}\right|_{{\mathbf{{x}}_0}}={curvature:.4g}')
                st.caption('Automatic differentiation of your rule gives the local slope and curvature. The dotted line is its tangent; all other inputs stay fixed.')
            except FormulaError:
                st.caption('Derivatives are undefined or outside numerical range here; the current score remains valid.')
            try:
                # A sampled curve must not draw a line across an unsampled pole.
                for endpoint in (curve.uncertainty.iloc[0], curve.uncertainty.iloc[-1]):
                    calculus_experiment(active_formula, values, 'uncertainty',
                        float(endpoint - row.uncertainty), active_k, points=3)
                st.plotly_chart(chart(fig, 250), width='stretch', config={'displayModeBar': False}, key='calculus')
            except FormulaError:
                st.info('This preview range includes an undefined point. Choose a valid interval in Calculus lab; your current score is unchanged.')
            if active_formula.strip() == DEFAULT_FORMULA and active_k == 2:
                st.latex(r'\Delta S=-k\,\Delta\varepsilon=-2(1)=-2\;\mathrm{pp}')
        render_calculus_lab(active_formula, values, active_k, plotstyle=chart)
        st.caption('Calculus charts are deterministic experiments on your mathematical rule, not market-return forecasts or simulated future prices.')
        feature_export = scores[['predicted_return', 'uncertainty', 'momentum', 'volatility']].rename_axis('ticker').reset_index()
        feature_export.insert(1, 'cutoff', cutoff)
        st.download_button('Download this cutoff’s algorithm inputs', feature_export.to_csv(index=False).encode(),
            file_name=f'derivport_features_{cutoff}.csv', mime='text/csv', key='export_algorithm_inputs')
        st.caption('The standalone file implements DerivPort Signal and matches that preset with the same k. Custom formula edits are not included in the download. It uses Python’s standard library and does not place trades.')
        if active_formula.strip() == SIGNAL_FORMULA:
            st.code(f'python3 derivport_signal.py --input derivport_features_{cutoff}.csv --output ranked.csv --k {active_k:g}', language='bash')
    heading('4. Test on past data', 'Compare your rule with the model and momentum on the same periods')
    for field in ('test_start', 'test_end'):
        if field in st.session_state:
            prior_date = pd.Timestamp(st.session_state[field])
            if not prices.index[0] <= prior_date <= prices.index[-1]:
                st.session_state[field] = min(max(prior_date, prices.index[0]), prices.index[-1]).date()
    start_col, end_col, action_col = st.columns(3, vertical_alignment='bottom')
    with start_col:
        test_start = st.date_input('Test from', value=max(earliest.date(), date(2022, 1, 1)), min_value=prices.index[0].date(), max_value=prices.index[-1].date(), key='test_start')
    with end_col:
        test_end = st.date_input('Test through', value=prices.index[-1].date(), min_value=prices.index[0].date(), max_value=prices.index[-1].date(), key='test_end')
    with action_col:
        run_test = st.button('Run backtest', type='primary', width='stretch', key='run_backtest')
    st.caption('Running a test reveals outcomes for that range. Only complete 21-shared-session periods count. Rules chosen or tuned now are retrospective experiments, not independent validation.')
    test_config = (tuple(universe), data_fingerprint, extended, active_formula, active_k, str(test_start), str(test_end))
    if run_test:
        try:
            result = backtest_rule(prices, test_start, test_end, active_formula, active_k, history)
            st.session_state.backtest_result = result
            st.session_state.backtest_config = test_config
            st.session_state.seen_backtests.append((tuple(universe), str(test_start), str(test_end)))
        except (ValueError, FormulaError) as exc:
            st.error(str(exc))
    result = st.session_state.get('backtest_result')
    if result is not None:
        if test_config != st.session_state.get('backtest_config'):
            st.info('Settings changed. These are the previous test results; run again to update them.')
        summary = result['summary'].rename(index={'model': 'AI model', 'momentum': 'Momentum', 'student': 'Your algorithm'})
        st.dataframe(summary.rename(columns={'n_dates': 'Periods', 'mean_spearman': 'Mean Spearman ρ', 'mean_return': 'Mean top-stock return (%)', 'linked_return': 'Linked top-stock return (%)', 'first_fraction': 'Finished first'}), width='stretch', column_config={'Finished first': st.column_config.NumberColumn(format='percent')})
        linked = result['linked_returns']
        fig = go.Figure()
        for method, label in (('momentum', 'Momentum'), ('model', 'AI model'), ('student', 'Your algorithm')):
            fig.add_trace(go.Scatter(x=linked.index, y=linked[method], name=label, line=dict(color=COLORS[label], width=2)))
        fig.update_yaxes(title='Linked top-stock return (%)')
        st.plotly_chart(chart(fig, 280), width='stretch', config={'displayModeBar': False}, key='backtest_chart')
        m = result['metrics']
        st.caption(f"{m['n_dates']} completed periods / {m['n_forecasts']} forecasts. Forecast RMSE: AI {m['model_rmse']:.2f} pp; momentum {m['baseline_rmse']:.2f} pp. Linked returns compound successive top selections without execution, costs, taxes, FX conversion, or sizing.")
        st.latex(r'S = ' + formula_to_latex(result['metadata']['formula']) + rf',\qquad k={result["metadata"]["k"]:g}')
        st.caption(f"Fixed group: {', '.join(result['metadata']['universe'])}.")
        with st.expander('Per-period backtest results'):
            st.dataframe(result['per_date'], width='stretch', hide_index=True)
            st.caption(result['metadata']['assumptions'])
        st.download_button('Export test results', result['per_date'].to_csv(index=False).encode(), file_name='derivport_backtest.csv', mime='text/csv')

with future:
    visual_heading('Future scenarios', 'Choose a risk assumption and explore hypothetical paths through 2037.', 'scenario-lab.svg')
    st.info(f'SIMULATED SCENARIOS · Actual historical prices stop at {latest:%Y-%m-%d}. Values after this anchor are hypothetical, not known prices or validated forecasts.')
    def select_risk_preset():
        apply_risk_preset(st.session_state)
    def custom_projection_risk():
        st.session_state.projection_risk = 'Custom'
    st.radio('Projection risk assumption', ['Low','Medium','High','Custom'], index=1,
        horizontal=True, key='projection_risk', on_change=select_risk_preset)
    st.caption('Low = 0.75× historical volatility · Medium = 1× · High = 1.5×. Presets change volatility, not your log-drift assumption. Higher risk means a wider range of outcomes; it does not promise better returns.')
    target_col, drift_col, run_col = st.columns([1, 1, 1], vertical_alignment='bottom')
    with target_col:
        projection_target = st.date_input('Scenario end date', value=date(2037, 12, 31),
            min_value=(latest + pd.Timedelta(days=1)).date(), max_value=date(2037, 12, 31),
            format='YYYY-MM-DD', key='projection_target')
    with drift_col:
        projection_drift = st.slider('Annual log-price drift assumption (%)', -20.0, 20.0, 6.0, .5, key='projection_drift')
    with st.expander('Scenario assumptions'):
        v_col, c_col, p_col = st.columns(3)
        with v_col:
            projection_volatility = st.slider('Historical volatility multiplier', .25, 3.0, 1.0, .25, key='projection_volatility', on_change=custom_projection_risk)
        with c_col:
            projection_correlation = st.slider('Correlation between stocks', 0.0, 1.0, .6, .05, key='projection_correlation')
        with p_col:
            projection_paths = st.number_input('Simulated paths', min_value=200, max_value=2000, value=600, step=200, key='projection_paths')
        st.caption('Monthly correlated lognormal scenarios. Volatility uses the last three years of observed log returns, annualized with 252 sessions/year, then multiplied by your setting. Drift and correlation are assumptions, not model estimates. Seed 2037 keeps the same assumptions reproducible; editing your ranking rule does not change the paths.')
        st.latex(r'\Delta\log P_i=\mu\Delta t+\sigma_i\sqrt{\Delta t}\left(\sqrt{\rho}\,Z_0+\sqrt{1-\rho}\,Z_i\right)')
        st.latex(r'\operatorname{Median}(P_T)=P_0e^{\mu T},\qquad \frac{\partial\operatorname{Median}(P_T)}{\partial\mu}=T P_0e^{\mu T}')
        st.caption('This derivative describes the assumed lognormal scenario median. It is not an explanation of the regression or a forecast of actual future prices.')
    with run_col:
        run_projection = st.button('Simulate through target date', type='primary', width='stretch', key='project_future')
    projection_config = (tuple(universe), data_fingerprint, str(projection_target), projection_drift,
        projection_volatility, projection_correlation, int(projection_paths))
    if run_projection:
        try:
            with st.spinner('Generating hypothetical monthly paths…'):
                projected = cached_projection(prices, projection_target, projection_drift,
                    projection_volatility, projection_correlation, int(projection_paths))
            st.session_state.projection_result = projected
            st.session_state.projection_config = projection_config
        except ValueError as exc:
            st.error(str(exc))
    projected = st.session_state.get('projection_result')
    if projected is not None:
        if projection_config != st.session_state.get('projection_config'):
            st.info('Scenario settings or data changed. The results below belong to your previous simulation; run again to update them.')
        symbols = list(projected.latest_prices.index)
        plotted_symbol = ticker if ticker in symbols else symbols[0]
        band = projected.quantiles.loc[projected.quantiles['ticker'] == plotted_symbol]
        heading(f'{plotted_symbol} · hypothetical price paths', f'{projected.dates[0]:%Y-%m-%d} to {projected.dates[-1]:%Y-%m-%d}')
        fan_col, distribution_col = st.columns([1.6, 1])
        with fan_col:
            st.plotly_chart(chart(scenario_fan(projected, plotted_symbol), 340), theme=None,
                width='stretch', config={'displayModeBar': True, 'displaylogo': False}, key='future_scenario_chart')
        with distribution_col:
            st.plotly_chart(chart(scenario_distribution(projected, plotted_symbol), 340), theme=None,
                width='stretch', config={'displayModeBar': True, 'displaylogo': False}, key='future_distribution_chart')
        terminal = projected.terminal_returns
        summary_projection = pd.DataFrame({
            'Last actual close': projected.latest_prices,
            'Assumed annual volatility (%)': projected.annual_volatility * 100,
            'Simulated return · 10th (%)': terminal.quantile(.10),
            'Simulated return · median (%)': terminal.quantile(.50),
            'Simulated return · 90th (%)': terminal.quantile(.90),
        })
        st.dataframe(summary_projection.round(2), width='stretch')
        st.caption('SIMULATED, NOT REALIZED · Bands are empirical percentiles of hypothetical paths, not confidence intervals or guaranteed probabilities. The 21-session regression is not extrapolated to 2037. No future company statements, actual returns, or prediction-accuracy scores are invented.')
        with st.expander('Simulation provenance and assumptions'):
            st.json(projected.assumptions)
        st.download_button('Export simulated monthly percentiles', projected.quantiles.to_csv(index=False).encode(),
            file_name='derivport_hypothetical_scenarios.csv', mime='text/csv')

with benchmarking:
    render_benchmarking(prices, metadata, {item.name: item for item in series}, cutoff, chart)

snapshot = st.session_state.reveals.get(context)
edited = bool(snapshot and (formula != snapshot['formula'] or k != snapshot['k']))
with panels:
    left, right = st.columns(2, gap='large')
    with left, st.container(border=True, key='analyst_company'):
        st.html('<h3>The business story</h3>')
        if note:
            st.caption('Date-only releases become eligible on a later observed date; same-day releases are embargoed because publication times are unavailable.')
            source_age = (pd.Timestamp(cutoff) - pd.Timestamp(note['published'])).days
            if source_age > 180:
                st.caption(f"Historical context only: this source is {source_age} days before the cutoff. More recent company evidence is not cached.")
            st.markdown(f"**Reported fact.** {note['fact']}")
            st.markdown(f"**Interpretation.** {note['interpretation']}")
            st.markdown(f"**Open question.** {note['limitation']}")
            st.caption(f"Published {note['published']} · [{note['source_title']}]({note['url']})")
        else:
            st.info('No suitable dated company source is cached for this stock and cutoff. There is insufficient evidence for a company interpretation.')
    with right, st.container(border=True, key='analyst_quant'):
        st.html('<h3>The model’s view</h3>')
        a, b, c = st.columns(3)
        a.metric('Estimated next 21 sessions', pct(row.predicted_return))
        b.metric('Typical forecast error', f'{row.uncertainty:.2f} pp')
        c.metric('AI model rank', f'{row.model_rank:g} / {len(universe)}')
        st.markdown(f'The regression places **{ticker} at rank {row.model_rank:g}** by estimated return. It uses past momentum and volatility. Feature influence is not measured, so no individual feature is presented as the cause of the prediction.')
        st.caption(f'Error is RMSE from {int(row.uncertainty_n)} completed past out-of-sample forecasts, not a confidence interval.')
        st.latex(r'S = ' + formula_to_latex(active_formula))
        st.latex(r'S \approx ' + formula_to_latex(active_formula, values) + rf' \approx {row.score:.3f}')
        st.caption(f'Your rule ranks this score {row.student_rank:g} out of {len(universe)}.')
    st.caption('The company view interprets reported business evidence; the model compares price-based forecasts. A strong business result can coexist with a low relative stock rank.' if note else 'The model view is available; comparison with a company interpretation is unsupported without a suitable dated source.')
with ranks_area:
    heading('Rankings', 'Same group · same cutoff · descending scores')
    table = scores.sort_values('student_rank', kind='stable').reset_index()
    table = table.rename(columns={table.columns[0]: 'Stock', 'momentum_rank': 'Momentum rank', 'model_rank': 'AI rank', 'student_rank': 'Algorithm rank', 'predicted_return': 'Estimated next 21 (%)', 'uncertainty': 'Past error (pp)', 'score': 'Algorithm score'})
    st.dataframe(table[['Stock', 'Momentum rank', 'AI rank', 'Algorithm rank', 'Estimated next 21 (%)', 'Past error (pp)', 'Algorithm score']], hide_index=True, width='stretch', height=38+35*len(universe),
        column_config={name: st.column_config.NumberColumn(format='%.2f') for name in ('Estimated next 21 (%)', 'Past error (pp)', 'Algorithm score')}, key='rankings')
    st.caption('1 = highest. Momentum ranks trailing 21-session returns; AI ranks estimated next 21-session returns; your algorithm ranks its score. Ties share an average rank. Edit the rule in Algorithm builder.')
    if edited:
        st.warning('Edited after seeing the outcome. The ranking at reveal is frozen below. Use another unrevealed date for a new hypothesis.')
    if any(group == tuple(universe) and start <= cutoff <= end for group, start, end in st.session_state.seen_backtests):
        st.info('This replay overlaps a backtest you already viewed. Further tuning is retrospective.')
with replay_area:
    st.divider()
    copy_col, action = st.columns([2, 1], vertical_alignment='center')
    with copy_col:
        st.html('<h3>Historical replay</h3>')
        if outcome_complete:
            st.caption('Future returns and realized ranks stay hidden until reveal. Replays use 21 shared observed trading sessions.')
        else:
            st.caption(f'Outcome pending: only {subsequent_sessions} of 21 subsequent shared sessions are cached. Research is available, but actual future returns and realized ranks do not exist in this cache yet.')
    with action:
        reveal_label = '21-session outcome pending' if not outcome_complete else ('Outcome revealed' if snapshot else 'Reveal next 21 trading days')
        if st.button(reveal_label, key='reveal', type='primary', disabled=bool(snapshot) or not outcome_complete, width='stretch'):
            st.session_state.reveals[context] = {'scores': scores.copy(), 'formula': active_formula, 'k': active_k}
            st.rerun()
    if snapshot and outcome_complete:
        outcomes = reveal_outcomes(prices, cutoff)
        original = snapshot['scores']
        display = outcomes.join(original[['momentum_rank', 'model_rank', 'student_rank']]).sort_values('realized_rank').rename(columns={'actual_return': 'Actual 21d return', 'realized_rank': 'Realized rank', 'momentum_rank': 'Momentum rank', 'model_rank': 'AI rank', 'student_rank': 'Algorithm rank at reveal'})
        st.dataframe(display, width='stretch', column_config={'Actual 21d return': st.column_config.NumberColumn(format='%+.2f%%')})
        columns = st.columns(3 if not edited else 4)
        methods = [('Momentum', original.momentum_rank), ('AI model', original.model_rank), ('Algorithm at reveal', original.student_rank)]
        if edited:
            methods.append(('Revised · outcome seen', scores.student_rank))
        for col, (label, order) in zip(columns, methods):
            rho = rank_correlation(order, outcomes.realized_rank)
            col.metric(f'{label} · Spearman ρ', 'Undefined' if rho is None else f'{rho:+.2f}')
        st.caption(f"Single-cutoff result through {pd.Timestamp(outcomes.attrs['end_date']):%Y-%m-%d}. Spearman compares ranked orders: +1 the same, −1 the reverse, 0 no rank association; ties use average ranks and constant rankings are undefined. Company notes made no numerical return forecast and receive no accuracy score.")
        st.latex(r'\rho_s=\operatorname{Corr}\!\left(\operatorname{rank}_{\rm avg}(s),\operatorname{rank}_{\rm avg}(R^{\rm actual}_{21})\right)')
        path = prices.loc[pd.Timestamp(cutoff):pd.Timestamp(outcomes.attrs['end_date'])]
        path = (path / path.iloc[0] - 1) * 100
        fig = go.Figure()
        for name in path:
            fig.add_trace(go.Scatter(x=path.index, y=path[name], name=name, line=dict(width=2.5 if name == ticker else 1.2)))
        fig.update_yaxes(title='Return from cutoff (%)')
        st.plotly_chart(chart(fig, 260), width='stretch', config={'displayModeBar': False}, key='historical_outcomes')
with extras_area:
    with st.expander('Full price history available at the cutoff'):
        st.caption(f'Historical observed adjusted closes through {cutoff}; no later prices are plotted.')
        known = prices.loc[:cutoff, ticker]
        fig = go.Figure(go.Scatter(x=known.index, y=known, line=dict(color='#2454ff', width=1.7)))
        fig.update_yaxes(title=f"Adjusted close ({meta.get('currency', 'local currency')})")
        st.plotly_chart(chart(fig, 220), width='stretch', config={'displayModeBar': False})
    with st.expander('Simulated stress test'):
        st.caption('SIMULATED PROJECTIONS · Each algorithm freezes its initial ranking. All use identical correlated paths. Editing a rule never regenerates paths.')
        scenario = st.selectbox('Scenario', list(SCENARIOS), format_func=lambda s: SCENARIOS[s]['label'], key='scenario')
        simulation = cached_simulation(base, scenario)
        st.write(SCENARIOS[scenario]['description'])
        st.caption(simulation.assumptions)
        distributions, sim_summary = simulation_summary(simulation, {'Momentum': scores.momentum_rank, 'AI model': scores.model_rank, 'Your algorithm': scores.student_rank})
        st.dataframe(sim_summary.rename(columns={'method': 'Algorithm', 'ticker': 'Top stock', 'mean_return': 'Mean (%)', 'p10': '10th percentile (%)', 'median_return': 'Median (%)', 'p90': '90th percentile (%)', 'first_fraction': 'Finished first'}), hide_index=True, width='stretch', column_config={'Finished first': st.column_config.NumberColumn(format='percent')})
        fig = go.Figure()
        for method in COLORS:
            fig.add_trace(go.Box(y=distributions.loc[distributions.method == method, 'return'], name=method, marker_color=COLORS[method], boxpoints=False))
        fig.update_yaxes(title='Simulated next 21-session return (%)')
        st.plotly_chart(chart(fig, 240), width='stretch', config={'displayModeBar': False})
        st.caption('600 paths per scenario; fixed seed 2024. Initial top-score ties use ticker order; terminal first-place ties split credit. Simulated projections are separate from historical evidence.')
    with st.expander('Methodology, data & past evaluation'):
        st.markdown('**Research scope.** Finance education only: no real brokerage accounts, balances, orders, or buy/sell recommendations. Benchmark portfolio weights and fills are hypothetical research allocations. Finance theme confirmed; a detailed official scoring rubric was not supplied.')
        st.markdown(f"**US stock universe.** Browse {len(us_listings):,} current equity candidates from NasdaqTrader, including common/ordinary shares, ADRs, preferred shares and partnership equity units after disclosed filters. {len(cached_listings)} histories are cached locally. Exchanges include {', '.join(US_EXCHANGES)} when available. The snapshot is not a historical/delisted security master; name-based classification and provider history availability have limits. New downloads verify equity identity. Dates require sufficient shared observed history; no missing dates are filled.")
        st.markdown('**Price charts.** Candles use genuine daily open, high, low, close and volume from a separate chart cache, with no dividend adjustment. Weekly/monthly bars aggregate only observations known by the cutoff and can include a partial final period. SMA and EMA windows count the selected bar interval. Research features, forecasts and returns retain the original dividend-adjusted close inputs.')
        st.markdown('**Model.** Standardized Ridge regression uses momentum and volatility. Training labels complete by each cutoff. Return inputs are percentage points; k is dimensionless under the default rule.')
        st.latex(r'M_{21,t}=100\left(\frac{P_t^{\rm adj}}{P_{t-21}^{\rm adj}}-1\right),\qquad \sigma_{21,t}=100\sqrt{21}\;\operatorname{SD}(r_{t-20},\ldots,r_t)')
        st.latex(r'\widehat{\boldsymbol\beta}=\underset{\beta_0,\boldsymbol\beta}{\operatorname{argmin}}\;\sum_{i\in\mathcal{T}_t}\left(y_i-\beta_0-\boldsymbol\beta^\top\widetilde{\mathbf{x}}_i\right)^2+25\lVert\boldsymbol\beta\rVert_2^2')
        st.caption(f"Current fit: {analysis.training['n_rows']:,} rows; last completed training label {pd.Timestamp(analysis.training['last_label_end']):%Y-%m-%d}.")
        st.markdown('**Uncertainty.** Root mean squared error uses up to 24 recent completed out-of-sample errors for that stock, with outcomes already known at cutoff. Fewer than six stock errors uses a disclosed pooled last-120-error fallback. This is an error scale, not a confidence interval.')
        st.latex(r'\varepsilon_t=\sqrt{\frac{1}{n_t}\sum_{i\in\mathcal{E}_t}\left(\widehat r_{i,21}-r_{i,21}^{\rm actual}\right)^2}')
        st.dataframe(base[['uncertainty', 'uncertainty_n', 'uncertainty_source']], width='stretch')
        past = analysis.error_summary
        a, b, c = st.columns(3)
        a.metric('Past AI forecast RMSE', 'Not available' if past['model_rmse'] is None else f"{past['model_rmse']:.2f} pp")
        b.metric('Past momentum RMSE', 'Not available' if past['baseline_rmse'] is None else f"{past['baseline_rmse']:.2f} pp")
        c.metric('Completed forecasts', str(past['n_forecasts']))
        st.caption('Only held-out outcomes since 2022 completed by this cutoff appear here. The momentum baseline assumes the next 21-session return repeats the preceding return. Lower RMSE does not guarantee better rankings.')
        if past['n_forecasts'] == 0:
            st.caption('No evaluation outcomes since 2022 had completed by this cutoff. Earlier completed forecasts can still estimate uncertainty.')
        elif past['baseline_rmse'] < past['model_rmse']:
            st.caption('The momentum baseline has lower forecast error here.')
        try:
            evaluation = evaluate_history(history, active_formula, active_k, cutoff)
            st.dataframe(evaluation['per_date'], width='stretch', hide_index=True)
        except FormulaError as exc:
            st.info(f'This rule cannot be evaluated on all past dates: {exc}')
        st.caption('Across-date results are separate from this single replay. Student rules are reapplied retrospectively; repeated tuning can overfit.')
        st.markdown('**Data limits.** Adjusted data are retrospective snapshots, not vintage vendor archives; later corporate actions or revisions can change historical levels. Default and custom groups can have selection and survivorship bias. Twenty curated official company notes are not exhaustive; missing sources are explicitly shown. Reveal state is local to the session.')
        sources = [{'symbol': s, 'source': item['source'], 'currency': item.get('currency'), 'from': item['start'], 'through': item['end'], 'retrieved': item.get('fetched_at'), 'source_url': item['source_url']} for s, item in metadata.items()]
        st.dataframe(pd.DataFrame(sources), hide_index=True, width='stretch')
        st.caption("Base cache: Yahoo Finance via Akmal Yaasir Fauzaan's pinned mirror. Extended caches: Yahoo Finance via yfinance. Provenance and hashes are in data/CREDITS.md, data/provenance.json, and each extended-cache metadata file. Online search, new downloads and external source links need internet. Existing cached research runs offline.")
        export = prices.loc[:cutoff].rename_axis('date').reset_index().melt(id_vars='date', var_name='ticker', value_name='adjusted_close')
        st.download_button('Export prices known by cutoff', export.to_csv(index=False).encode(), file_name=f'derivport_prices_through_{cutoff}.csv', mime='text/csv')
st.html('<div class="footer"><span>DerivPort</span><span>Historical research. Transparent assumptions. No investment recommendations.</span></div>')
with st.expander('Image credits'):
    st.markdown('Rule and scenario illustrations are original DerivPort artwork. Logo designed and exported in Figma; see docs/BRAND.md.')
