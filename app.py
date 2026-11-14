"""DerivPort: historical research and a transparent algorithm builder."""
from __future__ import annotations
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from derivport import (DEFAULT_FORMULA, FormulaError, analyze_cutoff, apply_rule,
    backtest_rule, default_crossovers, evaluate_history, formula_curve,
    historical_forecasts, load_prices, local_slope, rank_correlation,
    reveal_outcomes, simulate_paths, simulation_summary, SCENARIOS)
from derivport.market_data import (MarketDataError, get_search_status,
    load_symbol_prices, read_cached_symbols, search_symbols, symbol_info)

ROOT = Path(__file__).parent
DEFAULT_STOCKS = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA']
COLORS = {'Momentum': '#a1a7b0', 'AI model': '#3977bc', 'Your algorithm': '#20252c'}
st.set_page_config(page_title='DerivPort | Equity research', layout='wide', page_icon='D')
st.html(f"<style>{(ROOT / 'assets/style.css').read_text()}</style>")


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return f'{float(value):+.2f}%'


def heading(title, description=''):
    st.html(f'<div class="section-title"><h2>{esc(title)}</h2><span>{esc(description)}</span></div>')


def chart(fig, height=230):
    fig.update_layout(height=height, margin=dict(l=5, r=10, t=15, b=5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif', size=11, color='#717782'),
        legend=dict(orientation='h', y=1.15, x=0), hoverlabel=dict(bgcolor='white'))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='#e5e7eb')
    fig.update_yaxes(gridcolor='#eff0f2', zerolinecolor='#d4d7dd')
    return fig


@st.cache_data(show_spinner=False)
def frozen_prices():
    return load_prices(ROOT / 'data/prices.csv')


@st.cache_data(show_spinner=False)
def cached_symbol(symbol):
    return load_symbol_prices(symbol)


@st.cache_data(show_spinner='Preparing historical forecasts…')
def cached_history(prices):
    return historical_forecasts(prices)


@st.cache_data(show_spinner=False)
def cached_analysis(prices, cutoff, history):
    return analyze_cutoff(prices, cutoff, history)


@st.cache_data(show_spinner=False)
def cached_simulation(stocks, scenario):
    return simulate_paths(stocks, scenario, n_paths=600, seed=2024)


st.session_state.setdefault('valid_formula', DEFAULT_FORMULA)
st.session_state.setdefault('valid_k', 2.0)
st.session_state.setdefault('reveals', {})
st.session_state.setdefault('seen_backtests', [])
if 'pending_stock' in st.session_state:
    pending = st.session_state.pop('pending_stock')
    current = list(st.session_state.get('universe_selection', DEFAULT_STOCKS))
    if pending not in current:
        current = (current + [pending])[-10:]
    st.session_state.universe_selection = current
    st.session_state.stock = pending
    st.session_state.extended_history = True

st.html('''<div class="masthead"><div class="brand">DerivPort<span>Equity research</span></div>
<div class="mast-right">RESEARCH & EDUCATION <span class="connection">Local cache</span></div></div>''')
def clear_search():
    st.session_state.pop('search_results', None)
    st.session_state.pop('search_status', None)

with st.expander('Stocks & comparison group', expanded=False):
    st.caption('Search US and international equities or enter a Yahoo Finance symbol directly. Provider coverage and historical availability vary. Cached stocks work offline.')
    query_col, search_col = st.columns([4, 1], vertical_alignment='bottom')
    with query_col:
        query = st.text_input('Company or symbol', placeholder='Apple, ASML, 0700.HK, SAP.DE, 7203.T…', key='stock_query', on_change=clear_search)
    with search_col:
        if st.button('Search listings', key='search_listings', width='stretch'):
            with st.spinner('Searching equity listings…'):
                st.session_state.search_results = search_symbols(query, online=True)
                st.session_state.search_status = get_search_status()
    matches = st.session_state.get('search_results', [])
    if matches:
        lookup = {item['symbol']: item for item in matches}
        match = st.selectbox('Matching listings', list(lookup),
            format_func=lambda s: f"{s} · {lookup[s]['name']} · {lookup[s]['exchange']}", key='listing_match')
    else:
        match = query.strip().upper()
    if st.session_state.get('search_status'):
        st.caption(st.session_state.search_status['message'])
    load_col, refresh_col = st.columns(2)
    with load_col:
        if st.button('Cache and add stock', disabled=not bool(match), key='add_stock'):
            try:
                with st.spinner(f'Loading adjusted history for {match}…'):
                    _, added_meta = load_symbol_prices(match)
                st.session_state.pending_stock = added_meta['symbol']
                cached_symbol.clear()
                st.rerun()
            except MarketDataError as exc:
                st.error(str(exc))
    with refresh_col:
        if st.button('Refresh selected listing', disabled=not bool(match), key='refresh_stock'):
            try:
                with st.spinner('Refreshing adjusted price history…'):
                    _, refreshed = load_symbol_prices(match, refresh=True)
                cached_symbol.clear()
                if refreshed.get('warning'):
                    st.warning(refreshed['warning'])
                else:
                    st.success(f"{refreshed['symbol']} cached through {refreshed['end']}.")
            except MarketDataError as exc:
                st.error(str(exc))
    st.multiselect('Comparison group · 2–10 stocks', read_cached_symbols(), default=DEFAULT_STOCKS,
        max_selections=10, key='universe_selection')
    extended = st.toggle('Use extended downloaded histories', value=False, key='extended_history',
        help='Off keeps the original five stocks on their reproducible 2018–2024 demo cache. On uses full cached histories through the most recent completed download.')
    st.caption('Rankings and backtests use this fixed group. International comparisons use shared observed dates and local-currency returns, without FX conversion or filling missing days.')

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
        series.append(values.rename(symbol))
        metadata[symbol] = info
    prices = pd.concat(series, axis=1, sort=True).dropna().sort_index()
except MarketDataError as exc:
    st.error(str(exc))
    st.stop()
if len(prices) < 295:
    st.info('This group needs at least 295 shared observations for training, past forecast errors, and replay. Choose stocks with a longer shared history.')
    st.stop()
earliest = max(prices.index[252], pd.Timestamp('2020-04-01'))
latest = prices.index[-22]
if earliest > latest:
    st.info('This group needs more historical data. Choose an earlier-listed stock.')
    st.stop()
if st.session_state.get('stock') not in universe:
    st.session_state.stock = 'AAPL' if 'AAPL' in universe else universe[0]
existing_date = pd.Timestamp(st.session_state.get('decision_date', date(2024, 5, 3)))
if not earliest <= existing_date <= latest:
    st.session_state.decision_date = min(max(existing_date, earliest), latest).date()
stock_col, date_col, status_col = st.columns([1.2, 1.2, 1.6], gap='large', vertical_alignment='bottom')
with stock_col:
    ticker = st.selectbox('Stock', universe, format_func=lambda s: f"{s} · {metadata[s].get('name', s)}", key='stock')
with date_col:
    default_date = min(max(pd.Timestamp('2024-05-03'), earliest), latest).date()
    chosen_date = st.date_input('Historical cutoff', value=default_date, min_value=earliest.date(),
        max_value=latest.date(), format='YYYY-MM-DD', key='decision_date')
cutoff = prices.index[prices.index <= pd.Timestamp(chosen_date)][-1].date().isoformat()
st.session_state.cutoff = cutoff
data_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest()[:12]
context = ','.join(universe) + '|' + cutoff + '|' + data_fingerprint
st.session_state.active_replay_key = context
with status_col:
    st.html(f'<div class="data-status">{len(prices):,} shared sessions · {len(universe)} stocks<br>Cache through {prices.index[-1]:%d %b %Y}</div>')
if cutoff != chosen_date.isoformat():
    st.caption(f'{chosen_date} is not a shared trading date. Using the preceding observed session, {cutoff}.')
history = cached_history(prices)
try:
    analysis = cached_analysis(prices, cutoff, history)
except ValueError as exc:
    st.info(f'{exc} Choose a later date or a group with longer shared history.')
    st.stop()
base, meta = analysis.stocks, metadata[ticker]
notes = json.loads((ROOT / 'data/notes.json').read_text())
available_notes = [n for n in notes if n['ticker'] == ticker and pd.Timestamp(n['published']) <= pd.Timestamp(cutoff)]
note = max(available_notes, key=lambda n: n['published']) if available_notes else None
research, builder = st.tabs(['Research', 'Algorithm builder'], key='workspace')
with research:
    st.html(f'''<div class="stock-header"><div><h1>{esc(meta.get('name', ticker))} <span>{esc(ticker)}</span></h1>
    <p>{esc(meta.get('exchange', ''))} · {esc(meta.get('currency', 'Unknown'))} · Historical adjusted prices</p></div>
    <div class="price">{prices.loc[pd.Timestamp(cutoff), ticker]:,.2f}<small>Adjusted close · {cutoff}</small></div></div>''')
    if ticker == 'AAPL' and cutoff == '2024-05-03' and not extended:
        st.caption('Curated example: Apple after its May 2 earnings release. Choose any available date or change the stock group.')
    panels, ranks_area, replay_area, extras_area = [st.container() for _ in range(4)]
with builder:
    st.html('<div class="page-title"><h1>Algorithm builder</h1><p>Write a ranking rule. Inspect the math. Test it on historical outcomes.</p></div>')
    editor, sensitivity = st.columns(2, gap='large')
    with editor:
        formula = st.text_input('Ranking formula', value=st.session_state.valid_formula, key='formula', max_chars=400)
        st.caption('Variables: predicted_return, uncertainty, momentum, volatility, k. Use numbers, + − * / and parentheses. Return inputs are percentage points.')
        k = st.slider('Uncertainty weight · k', 0.0, 10.0, value=float(st.session_state.valid_k), step=.05, key='k')
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
        row = scores.loc[ticker]
        values = {name: float(row[name]) for name in ('predicted_return', 'uncertainty', 'momentum', 'volatility')}
        values['k'] = active_k
        arithmetic = re.sub(r'\b(predicted_return|uncertainty|momentum|volatility|k)\b', lambda m: f"({values[m.group()]:.2f})", active_formula)
        st.code(f'{arithmetic} ≈ {row.score:+.2f}', language=None)
        st.caption('Substituted inputs are rounded; scores use full precision. A ranking score is not a return forecast.')
        if active_formula.strip() == DEFAULT_FORMULA:
            st.caption(f'Uncertainty penalty: {active_k:g} × {row.uncertainty:.2f} ≈ {active_k * row.uncertainty:.2f} pp subtracted from the prediction.')
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
    with sensitivity:
        st.html('<div class="small-title">Score sensitivity</div>')
        curve = formula_curve(active_formula, values, active_k)
        fig = go.Figure(go.Scatter(x=curve.uncertainty, y=curve.score, mode='lines', line=dict(color='#20252c', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[row.uncertainty], y=[row.score], mode='markers', marker=dict(color='#3977bc', size=8), showlegend=False))
        fig.add_vline(x=row.uncertainty, line_dash='dot', line_color='#bbc3ce')
        fig.update_xaxes(title='Uncertainty (percentage points)')
        fig.update_yaxes(title='Rule score')
        st.plotly_chart(chart(fig, 250), width='stretch', config={'displayModeBar': False}, key='calculus')
        try:
            slope = local_slope(active_formula, values, active_k)
            label = 'Exact slope = −k' if active_formula.strip() == DEFAULT_FORMULA else 'Numerical local slope'
            st.caption(f'{label}: {slope:+.2f} score units per 1 pp uncertainty; other inputs held fixed.')
        except FormulaError:
            st.caption('The numerical local slope is undefined here; the current score remains valid.')
        if active_formula.strip() == DEFAULT_FORMULA and active_k == 2:
            st.caption('At k = 2, one extra percentage point of uncertainty lowers the score by two percentage points.')
    heading('Historical backtest', 'Same group and periods for every algorithm')
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
        st.caption(f"Tested rule: {result['metadata']['formula']} · k = {result['metadata']['k']:g}. Fixed group: {', '.join(result['metadata']['universe'])}.")
        with st.expander('Per-period backtest results'):
            st.dataframe(result['per_date'], width='stretch', hide_index=True)
            st.caption(result['metadata']['assumptions'])
        st.download_button('Export test results', result['per_date'].to_csv(index=False).encode(), file_name='derivport_backtest.csv', mime='text/csv')

snapshot = st.session_state.reveals.get(context)
edited = bool(snapshot and (formula != snapshot['formula'] or k != snapshot['k']))
with panels:
    left, right = st.columns(2, gap='large')
    with left:
        st.html('<div class="analyst-label">COMPANY & BUSINESS</div><h3>Stock analyst</h3>')
        if note:
            st.markdown(f"**Reported fact.** {note['fact']}")
            st.markdown(f"**Interpretation.** {note['interpretation']}")
            st.markdown(f"**Open question.** {note['limitation']}")
            st.caption(f"Published {note['published']} · [{note['source_title']}]({note['url']})")
        else:
            st.info('No suitable dated company source is cached for this stock and cutoff. There is insufficient evidence for a company interpretation.')
    with right:
        st.html('<div class="analyst-label">MODEL & RANKING</div><h3>Quant analyst — model and ranking</h3>')
        a, b, c = st.columns(3)
        a.metric('Estimated next 21 sessions', pct(row.predicted_return))
        b.metric('Typical forecast error', f'{row.uncertainty:.2f} pp')
        c.metric('AI model rank', f'{row.model_rank:g} / {len(universe)}')
        st.markdown(f'The regression places **{ticker} at rank {row.model_rank:g}** by estimated return. It uses past momentum and volatility. Feature influence is not measured, so no individual feature is presented as the cause of the prediction.')
        st.caption(f'Error is RMSE from {int(row.uncertainty_n)} completed past out-of-sample forecasts, not a confidence interval.')
        st.code(f'{active_formula}\n{arithmetic} ≈ {row.score:+.2f} → rank {row.student_rank:g}', language=None)
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
        st.caption('Future returns and realized ranks stay hidden until reveal. International experiments use 21 shared trading sessions.')
    with action:
        if st.button('Reveal next 21 trading days' if not snapshot else 'Outcome revealed', key='reveal', type='primary', disabled=bool(snapshot), width='stretch'):
            st.session_state.reveals[context] = {'scores': scores.copy(), 'formula': active_formula, 'k': active_k}
            st.rerun()
    if snapshot:
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
        path = prices.loc[pd.Timestamp(cutoff):pd.Timestamp(outcomes.attrs['end_date'])]
        path = (path / path.iloc[0] - 1) * 100
        fig = go.Figure()
        for name in path:
            fig.add_trace(go.Scatter(x=path.index, y=path[name], name=name, line=dict(width=2.5 if name == ticker else 1.2)))
        fig.update_yaxes(title='Return from cutoff (%)')
        st.plotly_chart(chart(fig, 260), width='stretch', config={'displayModeBar': False}, key='historical_outcomes')
with extras_area:
    with st.expander('Price history available at the cutoff'):
        known = prices.loc[:cutoff, ticker].tail(252)
        fig = go.Figure(go.Scatter(x=known.index, y=known, line=dict(color='#20252c', width=1.7)))
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
        st.markdown('**Research scope.** Finance education only: no brokerage, balances, sizing, trades, or buy/sell recommendations. Finance theme confirmed; a detailed official scoring rubric was not supplied.')
        st.markdown(f'**Global dates.** Any available date can be selected after sufficient shared history. Provider-wide US and international equity search is available online; a 109-symbol starter catalog and {len(read_cached_symbols())} price histories are cached offline. Coverage varies, so availability of every stock is not guaranteed. No missing-date filling is used. International experiments use 21 shared observed sessions, which can span more than 21 sessions on an individual exchange, using exchange-local daily closes. Local-currency returns exclude FX conversion.')
        st.markdown('**Model.** StandardScaler + Ridge(alpha=25) uses momentum and volatility. Training labels complete by each cutoff. Momentum is trailing 21-session return. Volatility is the trailing daily-return sample standard deviation × √21 × 100. Return inputs are percentage points; k is dimensionless under the default rule.')
        st.caption(f"Current fit: {analysis.training['n_rows']:,} rows; last completed training label {pd.Timestamp(analysis.training['last_label_end']):%Y-%m-%d}.")
        st.markdown('**Uncertainty.** RMSE = √mean((prediction − actual)²), using up to 24 recent completed out-of-sample errors for that stock, with outcomes already known at cutoff. Fewer than six stock errors uses a disclosed pooled last-120-error fallback. This is an error scale, not a confidence interval.')
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
