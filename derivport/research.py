"""Reproducible experiments that keep prediction, ranking and portfolios separate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .analysis import historical_forecasts, price_fingerprint, rank_correlation, validate_prices
from .portfolio import backtest_portfolio
from .research_models import walk_forward_models
from .research_signals import build_signal_panel, market_regimes

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    'Fixed, retrospectively selected survivor universe; no historical constituents or delisting returns. Market-wide generalization is unsupported.',
    'Retrospectively adjusted vendor snapshots, not point-in-time archives. Adjusted closes are not executable quotes; revisions and corporate-action effects remain possible.',
    'Only two price features. Small predefined model grids and constrained weights are research hypotheses, not optimal parameters.',
    'Final-test dates are withheld from parameter and weight selection by code, but this is a retrospective study of data that may previously have been inspected.',
    'Model fitting expands during the test using only outcomes that have completed. Frozen hyperparameters and signal weights never use test outcomes for selection.',
    'Portfolio execution is hypothetical at a later adjusted close. Liquidity, spread variation, market impact and taxes are not estimated from these data.',
    'Position and exposure targets apply at rebalancing; holdings drift between fills. Cash earns zero. Sharpe uses an assumed 252 shared sessions/year.',
    'Regimes describe a trailing equal-weight proxy for the selected stocks, not an independent market index. Shared sessions can span different exchange calendars.',
    'Date-block bootstrap intervals and Holm-adjusted tests are approximate conditional diagnostics. Small samples and selected survivors cannot establish universal superiority; broader vintage data and more independent periods are required.',
]


@dataclass(frozen=True)
class ExperimentConfig:
    validation_start: str = '2020-01-01'
    test_start: str = '2023-01-01'
    end: str | None = None
    seed: int = 2026
    rebalance_sessions: int = 21
    top_n: int = 3
    max_weight: float = .4
    gross_exposure: float = 1.
    execution_lag: int = 1
    commission_bps: float = 10.
    slippage_bps: float = 5.
    annual_rf: float = 0.
    k: float = 1.
    bootstrap_samples: int = 1000
    block_length: int = 3


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _code_identity():
    paths = [ROOT / 'app.py', ROOT / 'requirements.txt', ROOT / '.streamlit/config.toml']
    for directory in ('derivport', 'algorithms', 'scripts'):
        paths.extend(sorted((ROOT / directory).glob('*.py')))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.exists()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=3).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {'repository_commit': commit, 'working_tree_dirty': dirty, 'source_sha256': hashes}


def data_manifest(prices, source_metadata=None, source_series=None):
    metadata, raw = source_metadata or {}, source_series or {}
    symbols = []
    for ticker in prices.columns:
        series = raw.get(ticker, prices[ticker]).sort_index()
        observed = series.loc[:prices.index[-1]]
        retained = observed.reindex(prices.index)
        if retained.isna().any() or not np.allclose(retained, prices[ticker], rtol=1e-12, atol=1e-12):
            raise ValueError(f'Source observations for {ticker} do not match aligned experiment prices.')
        changes = observed.pct_change(fill_method=None)
        flags = changes.loc[changes.abs() > .5]
        info = metadata.get(ticker, {})
        symbols.append({
            'ticker': ticker, 'currency': info.get('currency'), 'source_url': info.get('source_url'),
            'retrieved': info.get('fetched_at', info.get('retrieved')),
            'point_in_time_archive': bool(info.get('point_in_time_archive', False)),
            'vendor_cache_sha256': info.get('cache_sha256', info.get('snapshot_sha256')),
            'observed_series_sha256': hashlib.sha256(pd.util.hash_pandas_object(observed, index=True).to_numpy().tobytes()).hexdigest(),
            'raw_rows_through_end': len(observed), 'retained_rows': len(prices),
            'excluded_nonshared_dates': int((~observed.index.isin(prices.index)).sum()),
            'first_raw_date': observed.index.min(), 'last_raw_date': observed.index.max(),
            'large_change_flags': [{'date': date, 'return_pct': float(change * 100)} for date, change in flags.iloc[:20].items()],
            'large_change_count': len(flags),
        })
    return json_safe({'aligned_sha256': price_fingerprint(prices), 'universe': list(prices.columns),
        'first_date': prices.index[0], 'last_date': prices.index[-1], 'shared_rows': len(prices),
        'calendar_policy': 'Complete common observed sessions; no forward filling or interpolation.',
        'quality_policy': 'Absolute observed one-session changes above 50% flagged for reconciliation, never clipped or silently removed.',
        'symbols': symbols})


def _cash_portfolio(index):
    daily = pd.DataFrame({'net_equity': 1., 'gross_equity': 1., 'net_return': 0., 'gross_return': 0.,
        'drawdown': 0., 'turnover': 0., 'notional_turnover': 0., 'cost': 0., 'cash_weight': 1.,
        'max_weight': 0., 'signal_cutoff': pd.NaT}, index=index)
    metrics = dict(total_return=0., gross_total_return=0., max_drawdown=0., sharpe=None,
        annualized_return=0., turnover=0., notional_turnover=0., costs=0., n_sessions=len(index)-1,
        n_rebalances=0, max_observed_weight=0.)
    return dict(daily=daily, trades=pd.DataFrame(), weights=pd.DataFrame(index=index), metrics=metrics,
        metadata={'policy': 'Non-interest-bearing cash; no trades. Zero-return prediction and cash portfolio are distinct baselines.'})


def run_experiment(prices: pd.DataFrame, config: ExperimentConfig | None = None,
                   source_metadata=None, source_series=None) -> dict:
    """Run a bounded, deterministic historical experiment; never fetch data."""
    from .research_metrics import prediction_metrics, per_date_metrics, rank_metrics, paired_comparisons
    config = config or ExperimentConfig()
    if not isinstance(config, ExperimentConfig):
        raise ValueError('Pass an ExperimentConfig instance.')
    for name, minimum in (('top_n', 1), ('execution_lag', 1)):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer of at least {minimum}.')
    for name in ('max_weight', 'gross_exposure', 'commission_bps', 'slippage_bps', 'annual_rf', 'k'):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'{name} must be finite numeric input.')
    if not 0 <= config.max_weight <= config.gross_exposure <= 1:
        raise ValueError('Use nonnegative target position cap <= target gross exposure <= 1.')
    if min(config.commission_bps, config.slippage_bps, config.k) < 0 or config.commission_bps + config.slippage_bps >= 10000 or config.annual_rf <= -1:
        raise ValueError('Use nonnegative costs and k, total costs below 10,000 bps, and annual risk-free rate greater than -1.')
    if isinstance(config.bootstrap_samples, bool) or not isinstance(config.bootstrap_samples, int) or not 200 <= config.bootstrap_samples <= 10000:
        raise ValueError('Use 200–10,000 bootstrap samples.')
    if isinstance(config.block_length, bool) or not isinstance(config.block_length, int) or config.block_length < 1:
        raise ValueError('Block length must be a positive integer number of forecast dates.')
    clean = validate_prices(prices).sort_index(axis=1)
    if config.end is not None:
        clean = clean.loc[:pd.Timestamp(config.end)]
    if clean.empty:
        raise ValueError('No prices are available through the experiment end.')
    fit = walk_forward_models(clean, validation_start=config.validation_start,
        test_start=config.test_start, end=clean.index[-1], seed=config.seed,
        rebalance_sessions=config.rebalance_sessions)
    signals = build_signal_panel(fit['forecasts'], test_start=config.test_start, k=config.k)
    scores = signals['scores'].loc[signals['scores']['split'].eq('test')].copy()
    common_dates = pd.DatetimeIndex(sorted(scores['cutoff'].unique()))
    if len(common_dates) < 2:
        raise ValueError('At least two completed final-test origins with common model support are required. Choose a longer observed period.')
    forecasts = fit['forecasts'].loc[fit['forecasts']['split'].eq('test') & fit['forecasts']['cutoff'].isin(common_dates)].copy()
    inference_args = dict(seed=config.seed, samples=config.bootstrap_samples, block_length=config.block_length)
    model_summary = prediction_metrics(forecasts, **inference_args)
    per_date = per_date_metrics(forecasts)
    ranking_summary = rank_metrics(scores)
    model_tests = paired_comparisons(per_date, baseline='Ridge', **inference_args)
    rank_rows = []
    for (strategy, cutoff), group in scores.groupby(['strategy', 'cutoff'], sort=True):
        rank_rows.append({'strategy': strategy, 'cutoff': cutoff, 'spearman_ic': rank_correlation(group.score, group.actual_return)})
    rank_dates = pd.DataFrame(rank_rows)
    signal_names = ['Balanced original', 'Standardized weighted', 'Risk-adjusted return', 'Uncertainty-penalized', 'Constrained weights']
    signal_tests = paired_comparisons(rank_dates.loc[rank_dates.strategy.isin(signal_names)],
        method_col='strategy', value_col='spearman_ic', baseline='Balanced original', **inference_args)
    regimes = market_regimes(clean)
    regime_rows = []
    tagged = forecasts.merge(regimes[['trend', 'volatility_regime']], left_on='cutoff', right_index=True, how='left')
    for dimension in ('trend', 'volatility_regime'):
        for regime, group in tagged.groupby(dimension, sort=True):
            summary = prediction_metrics(group, **inference_args).drop(
                columns=['rmse_ci_low', 'rmse_ci_high', 'inference_status']).reset_index()
            summary['dimension'], summary['regime'] = dimension, regime
            regime_rows.extend(summary.to_dict('records'))
    regime_models = pd.DataFrame(regime_rows)
    manifest_data = data_manifest(clean, source_metadata, source_series)
    raw_rows = []
    for ticker in clean.columns:
        series = (source_series or {}).get(ticker, clean[ticker]).loc[:clean.index[-1]]
        raw_rows.append(series.rename('adjusted_close').rename_axis('date').reset_index().assign(ticker=ticker))
    source_observations = pd.concat(raw_rows, ignore_index=True)[['date', 'ticker', 'adjusted_close']]
    currencies = [str(item['currency'] or '').strip().upper() for item in manifest_data['symbols']]
    compatible = all(len(currency) == 3 and currency.isalpha() and currency not in {'XXX', 'NAN'}
                     for currency in currencies) and len(set(currencies)) == 1
    portfolio_status = ('Same-currency hypothetical portfolios in ' + currencies[0] if compatible else
        'Portfolio results unavailable: all assets need one known common currency. Global local-return predictions and rankings remain available; no FX conversion is invented.')
    portfolio_end = forecasts.label_end.max()
    portfolios, portfolio_rows, period_rows, regime_portfolio_rows = {}, [], [], []
    if compatible:
        for strategy, group in scores.groupby('strategy', sort=True):
            if strategy == 'Zero return':
                continue  # A zero prediction is not an instruction to buy alphabetically tied stocks.
            kwargs = dict(top_n=len(clean.columns) if strategy == 'Equal weight' else config.top_n,
                max_weight=config.max_weight, gross_exposure=config.gross_exposure, execution_lag=config.execution_lag,
                rebalance_sessions=config.rebalance_sessions, commission_bps=config.commission_bps,
                slippage_bps=config.slippage_bps, annual_rf=config.annual_rf, liquidate=True)
            result = backtest_portfolio(clean, group, start=config.test_start, end=portfolio_end, **kwargs)
            portfolios[strategy] = result
        first = next(iter(portfolios.values()))['daily'].index
        portfolios['Cash'] = _cash_portfolio(first)
        for strategy, result in portfolios.items():
            portfolio_rows.append({'strategy': strategy, **result['metrics']})
            observed = result['daily']
            # Partition all daily accounting returns by fixed decision intervals
            # (origin, next origin], not by the fill-day's new holding label.
            # This includes delayed fills, old-position marks, and fees without
            # misattributing a fill-day return to an individual prediction.
            for number, origin in enumerate(common_dates):
                endpoint = common_dates[number + 1] if number + 1 < len(common_dates) else portfolio_end
                group = observed.loc[(observed.index > origin) & (observed.index <= endpoint)]
                if group.empty:
                    continue
                period_rows.append({'strategy': strategy, 'cutoff': origin, 'interval_start': group.index[0],
                    'label_end': group.index[-1], 'net_period_return': float((1 + group.net_return).prod() - 1), 'n_sessions': len(group)})
            # Daily returns receive the PREVIOUS observed close's regime, never today's closing information.
            daily_regimes = observed[['net_return']].join(regimes[['trend', 'volatility_regime']].shift(1))
            daily_regimes = daily_regimes.iloc[1:]
            for dimension in ('trend', 'volatility_regime'):
                for regime, group in daily_regimes.groupby(dimension, sort=True):
                    regime_portfolio_rows.append({'strategy': strategy, 'dimension': dimension, 'regime': regime,
                        'n_sessions': len(group), 'mean_daily_return': float(group.net_return.mean()),
                        'compounded_subset_return': float((1 + group.net_return).prod() - 1)})
    period_returns = pd.DataFrame(period_rows, columns=['strategy', 'cutoff', 'interval_start', 'label_end', 'net_period_return', 'n_sessions'])
    portfolio_tests = (paired_comparisons(period_returns, method_col='strategy', value_col='net_period_return',
        baseline='Balanced original', **inference_args) if not period_returns.empty else pd.DataFrame())
    # Recompute the unchanged baseline on the same dates: no before/after uplift is invented.
    legacy = historical_forecasts(clean, start=config.validation_start)
    ridge = forecasts.loc[forecasts.model.eq('Ridge')]
    matched = ridge.merge(legacy[['cutoff', 'ticker', 'predicted_return']], on=['cutoff', 'ticker'], suffixes=('_new', '_legacy'))
    parity = {'matched_forecasts': len(matched), 'new_forecasts': len(ridge),
        'max_prediction_difference_pp': float((matched.predicted_return_new - matched.predicted_return_legacy).abs().max()) if len(matched) else None,
        'before_rmse_pp': float(np.sqrt(np.mean((matched.predicted_return_legacy - matched.actual_return) ** 2))) if len(matched) else None,
        'after_rmse_pp': float(np.sqrt(np.mean((matched.predicted_return_new - matched.actual_return) ** 2))) if len(matched) else None,
        'interpretation': 'Unchanged Ridge evaluated on identical final-test origins. Differences in other methods are measured comparisons, not promised improvements.'}
    identity = _code_identity()
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'scikit-learn', 'streamlit', 'plotly', 'yfinance'):
        versions[package] = importlib.metadata.version(package)
    manifest = json_safe({'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc),
        'config': asdict(config), 'data': manifest_data, 'source_metadata': source_metadata or {},
        'code': identity, 'software_versions': versions,
        'models': fit['metadata'], 'signals': signals['diagnostics'], 'signal_weights': signals['weights'],
        'portfolio_status': portfolio_status, 'portfolio_assumptions': {key: result['metadata'] for key, result in portfolios.items()},
        'final_test': {'first_forecast': common_dates[0], 'last_forecast': common_dates[-1],
            'last_outcome': forecasts.label_end.max(), 'n_dates': len(common_dates), 'n_assets': len(clean.columns),
            'support': 'All models and signals share complete model/asset forecast dates.'},
        'portfolio_evaluation_end': portfolio_end,
        'portfolio_endpoint_policy': 'Liquidate at the common last predictive-outcome mark. Do not extend the last score into an unscored tail. The final holding interval can be shortened by execution lag.',
        'inference': {'method': 'Paired circular moving-date-block bootstrap; centered-null two-sided tests; Holm correction within each comparison family.',
            'samples': config.bootstrap_samples, 'block_length_origins': config.block_length,
            'min_dates': max(12, 2 * config.block_length), 'minimum_rule': 'max(12, 2 * block_length)',
            'small_sample_warning': len(common_dates) < 40,
            'ci_scope': 'Unadjusted percentile intervals are conditional approximate diagnostics, not simultaneous or distribution-free guarantees.'},
        'limitations': LIMITATIONS, 'baseline_parity': parity})
    stable_identity = {'config': manifest['config'], 'data': manifest['data'], 'sources': identity['source_sha256'], 'versions': versions}
    manifest['run_id'] = hashlib.sha256(json.dumps(stable_identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    return dict(manifest=manifest, model_summary=model_summary, per_date=per_date, ranking_summary=ranking_summary,
        forecasts=forecasts, development_forecasts=fit['forecasts'].loc[fit['forecasts']['split'].eq('development')],
        scores=scores, selection=fit['selection'], folds=fit['folds'], model_tests=model_tests,
        signal_tests=signal_tests, portfolio_tests=portfolio_tests, regime_models=regime_models,
        portfolios=portfolios, portfolio_summary=pd.DataFrame(portfolio_rows).set_index('strategy') if portfolio_rows else pd.DataFrame(),
        period_returns=period_returns, regime_portfolios=pd.DataFrame(regime_portfolio_rows), regimes=regimes,
        aligned_prices=clean, source_observations=source_observations)


def write_experiment(result: dict, directory: str | Path) -> Path:
    """Export full evidence and output checksums; no network or hidden model state."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            path = target / f'{name}.csv'
            value.to_csv(path, index=name in {'model_summary', 'ranking_summary', 'portfolio_summary', 'regimes', 'aligned_prices'})
            exported[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for number, (strategy, portfolio) in enumerate(sorted(result['portfolios'].items())):
        for name in ('daily', 'trades', 'weights'):
            path = target / f'portfolio_{number:02d}_{name}.csv'
            portfolio[name].to_csv(path, index=name != 'trades')
            exported[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(result['manifest'])
    manifest['output_sha256'] = exported
    manifest['portfolio_file_order'] = sorted(result['portfolios'])
    (target / 'manifest.json').write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + '\n')
    return target
