"""The additive research workspace; expensive experiments run only on request."""
from dataclasses import asdict
from datetime import date
from io import BytesIO
import hashlib
import json
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .analysis import price_fingerprint
from .research import ExperimentConfig, json_safe, run_experiment


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_experiment(prices, config, metadata, sources):
    return run_experiment(prices, ExperimentConfig(**config), metadata, sources)


def _archive(result):
    buffer, hashes = BytesIO(), {}
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in result.items():
            if isinstance(value, pd.DataFrame):
                payload = value.to_csv(index=name in {'model_summary', 'ranking_summary', 'portfolio_summary', 'regimes', 'aligned_prices'}).encode()
                archive.writestr(name + '.csv', payload)
                hashes[name + '.csv'] = hashlib.sha256(payload).hexdigest()
        for number, (strategy, portfolio) in enumerate(sorted(result['portfolios'].items())):
            for name in ('daily', 'trades', 'weights'):
                filename = f'portfolio_{number:02d}_{name}.csv'
                payload = portfolio[name].to_csv(index=name != 'trades').encode()
                archive.writestr(filename, payload)
                hashes[filename] = hashlib.sha256(payload).hexdigest()
        manifest = dict(result['manifest'], output_sha256=hashes, portfolio_file_order=sorted(result['portfolios']))
        archive.writestr('manifest.json', json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return buffer.getvalue()


def render_benchmarking(prices, metadata, sources, cutoff, plotstyle):
    st.html('<div class="page-title"><h1>Research &amp; Benchmarking</h1><p>Compare forecasts, ranking rules, and delayed portfolios on the same historical evidence.</p></div>')
    visible = prices.loc[:pd.Timestamp(cutoff)]
    for key in ('benchmark_validation_start', 'benchmark_test_start'):
        if key in st.session_state:
            value = pd.Timestamp(st.session_state[key])
            st.session_state[key] = min(max(value, visible.index[0]), visible.index[-1]).date()
    if st.session_state.get('benchmark_top_n', 1) > len(visible.columns):
        st.session_state.benchmark_top_n = len(visible.columns)
    st.caption(f'Historical research through {cutoff}. This workspace never evaluates outcomes after the selected main cutoff. No experiment runs until requested.')
    st.info('Retrospective fixed-universe research. Model parameters and constrained signal weights are selected on development data, then frozen for the final test. This is not prospective proof of profitability.')
    a, b, c = st.columns(3)
    with a:
        validation = st.date_input('Development forecasts from',
            value=min(max(date(2020, 1, 1), visible.index[0].date()), visible.index[-1].date()),
            min_value=visible.index[0].date(), max_value=visible.index[-1].date(), key='benchmark_validation_start')
    with b:
        test_start = st.date_input('Final test starts',
            value=min(max(date(2023, 1, 1), visible.index[0].date()), visible.index[-1].date()),
            min_value=visible.index[0].date(), max_value=visible.index[-1].date(), key='benchmark_test_start')
    with c:
        rebalance = st.selectbox('Forecast / rebalance spacing (shared sessions)', [21, 42, 63], key='benchmark_rebalance')
    with st.expander('Portfolio and inference assumptions'):
        a, b, c = st.columns(3)
        with a:
            top_n = st.number_input('Maximum stocks held', min_value=1, max_value=len(visible.columns),
                value=min(3, len(visible.columns)), step=1, key='benchmark_top_n')
            cap = st.slider('Target position cap', .05, 1., .4, .05, key='benchmark_cap')
            exposure = st.slider('Target stock exposure', .1, 1., 1., .05, key='benchmark_exposure')
        with b:
            commission = st.number_input('Commission (basis points per traded notional)', 0., 200., 10., 1., key='benchmark_commission')
            slippage = st.number_input('Slippage (basis points per traded notional)', 0., 200., 5., 1., key='benchmark_slippage')
            lag = st.number_input('Execution lag (shared closes)', 1, 5, 1, key='benchmark_lag')
        with c:
            k = st.slider('Signal risk penalty k', 0., 5., 1., .1, key='benchmark_k')
            seed = st.number_input('Research seed', 0, 2**31-1, 2026, key='benchmark_seed')
            block = st.number_input('Bootstrap block length (forecast origins)', 1, 12, 3, key='benchmark_block')
        st.caption('Forecast horizon stays 21 shared sessions. Costs apply to full buy-plus-sell notional, including entry and final liquidation. Caps apply to targets; weights drift between rebalances. Cash earns zero. Sharpe assumes 252 shared sessions/year and a zero risk-free benchmark. The equal-weight comparator uses the entire selected group with the same cap and costs.')
    cfg = ExperimentConfig(validation_start=str(validation), test_start=str(test_start), end=str(cutoff),
        seed=int(seed), rebalance_sessions=int(rebalance), top_n=int(top_n), max_weight=float(cap),
        gross_exposure=float(exposure), execution_lag=int(lag), commission_bps=float(commission),
        slippage_bps=float(slippage), k=float(k), block_length=int(block))
    signature = (price_fingerprint(visible), json.dumps(asdict(cfg), sort_keys=True),
                 json.dumps(json_safe(metadata), sort_keys=True))
    if st.button('Run research experiment', key='benchmark_run', type='primary'):
        try:
            with st.spinner('Fitting chronological models and evaluating historical research…'):
                result = _cached_experiment(visible, asdict(cfg), metadata, sources)
            st.session_state.research_experiment = result
            st.session_state.research_experiment_config = signature
            st.session_state.seen_backtests.append((tuple(sorted(visible.columns)), str(validation), str(cutoff)))
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    result = st.session_state.get('research_experiment')
    if result is None:
        st.caption('The existing Ridge model and balanced algorithm remain available in the other tabs. Run this experiment for the additional model and signal comparisons.')
        return
    if st.session_state.get('research_experiment_config') != signature:
        st.info('Experiment settings, prices, or historical cutoff changed. Previous results are hidden; run again to produce matching evidence.')
        return
    manifest = result['manifest']
    final = manifest['final_test']
    st.caption(f"Run {manifest['run_id']} · {final['n_dates']} completed final-test dates · {final['n_assets']} assets · last observed outcome {final['last_outcome'][:10]}")
    if final['n_dates'] < 40:
        st.warning('This experiment has few final-test dates. Bootstrap intervals and adjusted p-values are exploratory; more independent periods and broader point-in-time data are required to establish reliable superiority.')
    comparisons, algorithms, portfolios, regimes, statistics, provenance = st.tabs([
        'Forecast accuracy', 'Signal comparisons', 'Portfolio performance', 'Market regimes', 'Statistical evidence', 'Reproducibility'])
    with comparisons:
        st.subheader('Historical out-of-sample forecast accuracy')
        st.caption('Returns and errors are percentage points. RMSE/MAE/R² evaluate predicted returns, not ranking scores. IC is the mean cross-sectional correlation across forecast dates. Constant forecasts have undefined correlation; zero-return direction is counted as zero, not a positive guess.')
        observed = result['forecasts'].loc[result['forecasts'].model.eq('Ridge'), 'actual_return']
        st.caption(f'Observed positive-return fraction: {(observed > 0).mean():.1%}. Always predicting a positive direction would achieve that accuracy; directional accuracy alone does not establish predictive skill. Negative R² means higher squared error than the ex-post test-sample mean reference, which is not a tradable forecast.')
        st.dataframe(result['model_summary'], width='stretch')
        model = st.selectbox('Forecast model', list(result['model_summary'].index), key='benchmark_model')
        if st.session_state.get('benchmark_stock') not in visible.columns:
            st.session_state.pop('benchmark_stock', None)
        ticker = st.selectbox('Forecast stock', list(visible.columns), key='benchmark_stock')
        panel = result['forecasts'].loc[result['forecasts'].model.eq(model) & result['forecasts'].ticker.eq(ticker)].sort_values('cutoff')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=panel.cutoff, y=panel.interval_upper, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=panel.cutoff, y=panel.interval_lower, line=dict(width=0), fill='tonexty',
            fillcolor='rgba(36,84,255,.10)', name='Past-residual 80% band'))
        fig.add_trace(go.Scatter(x=panel.cutoff, y=panel.actual_return, name='Historical observed 21-session return', line=dict(color='#101820')))
        fig.add_trace(go.Scatter(x=panel.cutoff, y=panel.predicted_return, name='Historical OOS forecast', line=dict(color='#2454ff', dash='dash')))
        fig.update_yaxes(title='21-shared-session return (%)')
        st.plotly_chart(plotstyle(fig, 320), width='stretch', config={'displayModeBar': False}, key='benchmark_forecasts_chart')
        st.caption('Bands use only past completed residuals available at each origin. They are empirical prediction-error bands with measured coverage, not guaranteed confidence intervals. RMSE confidence intervals resample forecast dates in blocks; they answer a different question.')
        st.dataframe(panel[['cutoff', 'label_end', 'predicted_return', 'actual_return', 'uncertainty', 'uncertainty_n', 'uncertainty_source', 'interval_lower', 'interval_upper']], hide_index=True, width='stretch')
    with algorithms:
        st.subheader('Ranking rules on identical final-test dates')
        st.dataframe(result['ranking_summary'], width='stretch')
        choices = sorted(result['scores'].strategy.unique())
        if st.session_state.get('benchmark_signal') not in choices:
            st.session_state.pop('benchmark_signal', None)
        selected = st.selectbox('Research ranking rule', choices, index=choices.index('Balanced original'), key='benchmark_signal')
        equations = {
            'Balanced original': r'S=.6\widehat r+.4M-k(.5\varepsilon+.25\sigma)',
            'Standardized weighted': r'S=.6z_t(\widehat r)+.4z_t(M)-k[.5z_t(\varepsilon)+.25z_t(\sigma)]',
            'Risk-adjusted return': r'S=\frac{\widehat r}{\max(\sigma,1\;\mathrm{pp})}',
            'Uncertainty-penalized': r'S=\widehat r-k\varepsilon',
            'Constrained weights': r'S=\mathbf w^\top[z_t(\widehat r),z_t(M),-z_t(\varepsilon),-z_t(\sigma)],\quad \mathbf1^\top\mathbf w=1,\;0\leq w_j\leq .6',
        }
        if selected in equations:
            st.latex(equations[selected])
        st.caption('The original formula is preserved. Standardized and constrained scores are dimensionless, and the risk-adjusted ratio uses a 1 percentage-point volatility floor. None of these ranking scores is itself a return forecast. Cross-sectional normalization uses only contemporaneously known inputs; constant factors contribute zero.')
        sub = result['scores'].loc[result['scores'].strategy.eq(selected)]
        if st.session_state.get('benchmark_ranking_date') not in set(sub.cutoff):
            st.session_state.pop('benchmark_ranking_date', None)
        origin = st.selectbox('Ranking comparison date', sorted(sub.cutoff.unique()), format_func=lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'), key='benchmark_ranking_date')
        snapshot = sub.loc[sub.cutoff.eq(origin)].sort_values(['score', 'ticker'], ascending=[False, True]).copy()
        snapshot['rank'] = snapshot.score.rank(ascending=False, method='average')
        st.dataframe(snapshot[['ticker', 'rank', 'score', 'predicted_return', 'momentum', 'uncertainty', 'volatility', 'actual_return']], hide_index=True, width='stretch')
        st.caption('This is an explicitly evaluated historical OOS comparison: actual returns are shown. The original replay tab keeps its separate reveal controls.')
        st.json(manifest['signals']['weight_fit'], expanded=False)
    with portfolios:
        st.subheader('Historical hypothetical portfolio performance')
        st.caption(manifest['portfolio_status'])
        if not result['portfolio_summary'].empty:
            if st.session_state.get('benchmark_portfolio') not in result['portfolios']:
                st.session_state.pop('benchmark_portfolio', None)
            st.dataframe(result['portfolio_summary'], width='stretch')
            fig = go.Figure()
            for name, portfolio in result['portfolios'].items():
                fig.add_trace(go.Scatter(x=portfolio['daily'].index, y=portfolio['daily'].net_equity, name=name))
            fig.update_yaxes(title='Net wealth / initial wealth')
            st.plotly_chart(plotstyle(fig, 350), width='stretch', config={'displayModeBar': False}, key='benchmark_portfolio_chart')
            selected = st.selectbox('Portfolio details', list(result['portfolios']), key='benchmark_portfolio')
            detail = result['portfolios'][selected]
            fig = go.Figure(go.Scatter(x=detail['daily'].index, y=detail['daily'].drawdown * 100,
                name='Historical hypothetical daily drawdown', line=dict(color='#2454ff'), fill='tozeroy'))
            fig.update_yaxes(title='Drawdown (%)')
            st.plotly_chart(plotstyle(fig, 220), width='stretch', config={'displayModeBar': False}, key='benchmark_drawdown_chart')
            st.caption('Summary returns/drawdowns are decimals; multiply by 100 for percent. Costs are units of initial wealth. Turnover is half-L1 risky-weight turnover; fees use full traded notional. Gross performance uses the same delayed targets without costs. These are assumptions applied to observed adjusted prices, not realized trades or future scenarios.')
            with st.expander('Trade and holding records'):
                st.dataframe(detail['trades'], hide_index=True, width='stretch')
                st.dataframe(detail['weights'], width='stretch')
    with regimes:
        st.subheader('Historical OOS performance by trailing regime')
        st.caption('Bull/bear/sideways use a trailing 126-shared-session return of the selected-group equal-weight proxy and ±5% thresholds. High/low volatility compares trailing 21-session volatility with a strictly past expanding median. These are descriptive proxy regimes, not an independent market index. Regime subsets can be noncontiguous; no regime-specific inference is reported.')
        st.dataframe(result['regime_models'], hide_index=True, width='stretch')
        if not result['regime_portfolios'].empty:
            st.caption('Portfolio daily returns use the previous observed close’s regime. Compounded subset returns combine noncontiguous regime observations and do not describe a continuously invested regime-switching strategy.')
            st.dataframe(result['regime_portfolios'], hide_index=True, width='stretch')
    with statistics:
        st.subheader('Paired date-block statistical diagnostics')
        st.caption(f"Circular moving blocks of {cfg.block_length} forecast origins; {cfg.bootstrap_samples:,} replicates; seed {cfg.seed}. At least {max(12, 2 * cfg.block_length)} paired dates are required. Blocks follow paired valid-date order. Intervals are unadjusted percentile approximations; p-values are centered-null, two-sided and Holm-adjusted separately within each displayed family. Simultaneous stock rows are never treated as independent samples.")
        st.markdown('**Prediction loss:** candidate minus Ridge MSE; negative differences favor the candidate.')
        st.dataframe(result['model_tests'], hide_index=True, width='stretch')
        st.markdown('**Ranking IC:** candidate minus the original balanced signal; positive differences favor the candidate.')
        st.dataframe(result['signal_tests'], hide_index=True, width='stretch')
        if not result['portfolio_tests'].empty:
            st.markdown('**Net portfolio return:** candidate minus the original balanced signal per decision interval; positive differences favor the candidate.')
            st.caption('Accounting intervals run after one decision close through the next decision close, or through the final common outcome mark. They include delayed fills, prior holdings at the next fill, and costs; they are not individual forecast holding-period returns.')
            st.dataframe(result['portfolio_tests'], hide_index=True, width='stretch')
        st.caption('Conditional p-values do not remove survivorship, data-vintage, repeated-research or regime-selection bias. A small p-value alone does not establish a generally superior model. Broader point-in-time data and additional independent observations are needed.')
    with provenance:
        st.subheader('Experiment record and limitations')
        st.dataframe(result['selection'], hide_index=True, width='stretch')
        st.json(manifest, expanded=False)
        st.markdown('\n'.join('- ' + item for item in manifest['limitations']))
        st.caption('Data hashes, source hashes, package versions, selected hyperparameters, fold bounds, frozen signal weights, seed, costs and constraints are included in the download. The command-line runner reproduces the same protocol without a network request.')
    st.download_button('Download research evidence (ZIP)', _archive(result),
        file_name=f"derivport_research_{manifest['run_id']}.zip", mime='application/zip', key='benchmark_download')
