"""Transparent ranking comparisons and strictly trailing market regimes.

Forecast accuracy belongs to the forecasting models. These signals rank assets;
their scores are not interchangeable return predictions. Learned weights are
estimated only from completed development forecasts and frozen for final tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.derivport_signal import FORMULA as BALANCED_FORMULA

SIGNAL_NAMES = (
    'Balanced original', 'Standardized weighted', 'Risk-adjusted return',
    'Uncertainty-penalized', 'Constrained weights',
)
FACTOR_NAMES = ('predicted_return', 'momentum', 'negative_uncertainty', 'negative_volatility')
RISK_FLOOR_PP = 1.0
WEIGHT_CAP = 0.6
WEIGHT_LAMBDAS = (0.1, 1.0, 10.0)


def _zscore(values: np.ndarray) -> np.ndarray:
    """Cross-sectional population z scores; a constant factor has no signal."""
    values = np.asarray(values, dtype=float)
    scale = float(values.std(ddof=0))
    return np.zeros_like(values) if scale <= 1e-12 else (values - values.mean()) / scale


def _capped_simplex(values: np.ndarray, cap: float = WEIGHT_CAP) -> np.ndarray:
    """Euclidean projection onto sum(w)=1 and 0 <= w_i <= cap."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) * cap < 1:
        raise ValueError('A feasible finite capped simplex is required.')
    lower, upper = float(values.min() - cap), float(values.max())
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if np.clip(values - midpoint, 0.0, cap).sum() > 1:
            lower = midpoint
        else:
            upper = midpoint
    return np.clip(values - (lower + upper) / 2, 0.0, cap)


def _fit_weights(x: np.ndarray, target: np.ndarray, penalty: float) -> tuple[np.ndarray, dict]:
    """Minimize mean squared error + penalty*||w||² on the capped simplex."""
    weights = np.full(x.shape[1], 1 / x.shape[1])
    lipschitz = 2 * (float(np.linalg.norm(x, ord=2)) ** 2 / len(x) + penalty)
    step = 1.0 / lipschitz
    converged = False
    for iteration in range(1, 20001):
        gradient = 2 * x.T @ (x @ weights - target) / len(x) + 2 * penalty * weights
        updated = _capped_simplex(weights - step * gradient)
        if float(np.max(np.abs(updated - weights))) <= 1e-10:
            weights = updated
            converged = True
            break
        weights = updated
    return weights, {'converged': converged, 'iterations': iteration,
        'objective': float(np.mean((x @ weights - target) ** 2) + penalty * (weights @ weights))}


def _complete_panel(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {'model', 'cutoff', 'ticker', 'predicted_return', 'momentum',
        'uncertainty', 'volatility', 'actual_return', 'label_end', 'split'}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError('Missing forecast fields: ' + ', '.join(sorted(missing)))
    frame = forecasts.copy()
    if frame.empty:
        raise ValueError('At least one complete historical forecast date is required.')
    frame['cutoff'] = pd.to_datetime(frame.cutoff, errors='raise')
    frame['label_end'] = pd.to_datetime(frame.label_end, errors='raise')
    if frame[['cutoff', 'label_end', 'ticker', 'model', 'split']].isna().any().any():
        raise ValueError('Forecast identifiers and dates must be present.')
    if frame.duplicated(['cutoff', 'ticker', 'model']).any():
        raise ValueError('Duplicate model/stock/forecast-date rows are not allowed.')
    if (frame.label_end <= frame.cutoff).any():
        raise ValueError('Forecast label_end must follow its cutoff.')
    numeric = ['predicted_return', 'momentum', 'uncertainty', 'volatility', 'actual_return']
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
    if ((frame.uncertainty < 0) | (frame.volatility < 0)).any():
        raise ValueError('Forecast error and volatility must be nonnegative.')
    models = sorted(frame.model.unique())
    tickers = sorted(frame.ticker.unique())
    if 'Ridge' not in models or len(tickers) < 2:
        raise ValueError('Ridge forecasts for at least two assets are required.')
    expected = len(models) * len(tickers)
    keep = []
    for cutoff, group in frame.groupby('cutoff', sort=True):
        if len(group) != expected or not np.isfinite(group[numeric].to_numpy(dtype=float)).all():
            continue
        # Every model must evaluate the very same realized target and split.
        if group.label_end.nunique() != 1 or group.split.nunique() != 1:
            raise ValueError('Models must share label dates and splits at each forecast cutoff.')
        if group.groupby('ticker').actual_return.nunique().ne(1).any():
            raise ValueError('Models must share actual returns for every stock/date.')
        keep.append(cutoff)
    result = frame.loc[frame.cutoff.isin(keep)].sort_values(['cutoff', 'model', 'ticker']).reset_index(drop=True)
    if result.empty:
        raise ValueError('No forecast date has complete common model/asset support.')
    return result, {'supplied_dates': int(frame.cutoff.nunique()), 'common_dates': len(keep),
        'excluded_incomplete_dates': int(frame.cutoff.nunique() - len(keep)),
        'models': models, 'tickers': tickers,
        'support_policy': 'Complete common model/asset dates; no missing-value substitution.'}


def build_signal_panel(forecasts: pd.DataFrame, test_start='2023-01-01', k=1.0) -> dict:
    """Return model-score baselines and five explicit ranking hypotheses.

    Standardization uses only the contemporaneous stock cross section. Weight
    selection uses chronological development OOS forecasts whose target has
    completed strictly before test_start; final-test rows never fit weights.
    """
    try:
        k = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError('Risk penalty k must be finite and nonnegative.') from exc
    if not np.isfinite(k) or k < 0:
        raise ValueError('Risk penalty k must be finite and nonnegative.')
    boundary = pd.Timestamp(test_start)
    if pd.isna(boundary):
        raise ValueError('A valid final-test boundary is required.')
    panel, diagnostics = _complete_panel(forecasts)
    ridge = panel.loc[panel.model.eq('Ridge')].copy().sort_values(['cutoff', 'ticker'])
    factors, targets = [], []
    for _, group in ridge.groupby('cutoff', sort=True):
        factors.append(np.column_stack([_zscore(group.predicted_return), _zscore(group.momentum),
            -_zscore(group.uncertainty), -_zscore(group.volatility)]))
        targets.append(_zscore(group.actual_return))
    factor_matrix = np.vstack(factors)
    target_vector = np.concatenate(targets)
    ridge.index = pd.RangeIndex(len(ridge))
    development = ridge.split.eq('development') & (ridge.label_end < boundary) & (ridge.cutoff < boundary)
    development_dates = pd.DatetimeIndex(ridge.loc[development, 'cutoff'].unique()).sort_values()
    weight_diagnostics = {'status': 'unavailable', 'minimum_development_dates': 12,
        'minimum_validation_dates': 4, 'development_dates': len(development_dates),
        'test_start': boundary.isoformat(), 'fit_rule': 'development OOS only; label_end < test_start',
        'weight_cap': WEIGHT_CAP, 'sum_to_one': True,
        'reason': 'At least 12 completed development dates and 4 chronological weight-validation dates are required.'}
    learned = {}
    if len(development_dates) >= 12:
        split_position = int(np.floor(2 * len(development_dates) / 3))
        train_dates, validation_dates = development_dates[:split_position], development_dates[split_position:]
        train_mask = development & ridge.cutoff.isin(train_dates)
        validation_mask = development & ridge.cutoff.isin(validation_dates)
        # An internal weight-training label must be known before validation begins.
        train_mask &= ridge.label_end < validation_dates[0]
        if validation_mask.any() and int(ridge.loc[validation_mask, 'cutoff'].nunique()) >= 4 and train_mask.any():
            trials = []
            for penalty in WEIGHT_LAMBDAS:
                candidate, fit = _fit_weights(factor_matrix[train_mask], target_vector[train_mask], penalty)
                mse = float(np.mean((factor_matrix[validation_mask] @ candidate - target_vector[validation_mask]) ** 2))
                trials.append({'lambda': penalty, 'validation_mse': mse, **fit})
            chosen = min(trials, key=lambda item: (item['validation_mse'], item['lambda']))
            fitted, final_fit = _fit_weights(factor_matrix[development], target_vector[development], chosen['lambda'])
            if chosen['converged'] and final_fit['converged']:
                learned = dict(zip(FACTOR_NAMES, map(float, fitted)))
                weight_diagnostics.update(status='available', reason='', chosen_lambda=chosen['lambda'],
                    weights=learned, train_first=ridge.loc[train_mask, 'cutoff'].min().isoformat(),
                    train_last=ridge.loc[train_mask, 'cutoff'].max().isoformat(),
                    train_last_label_end=ridge.loc[train_mask, 'label_end'].max().isoformat(),
                    validation_first=validation_dates[0].isoformat(), validation_last=validation_dates[-1].isoformat(),
                    final_fit_last_label_end=ridge.loc[development, 'label_end'].max().isoformat(),
                    trials=trials, **final_fit)
            else:
                weight_diagnostics.update(reason='The constrained optimizer did not converge; no constrained scores were published.', trials=trials)
        else:
            weight_diagnostics['reason'] = 'Insufficient purged chronological weight-training/validation support.'
    output = []
    for name, group in panel.groupby('model', sort=True):
        item = group.copy()
        item['strategy'], item['score'] = name, item.predicted_return
        output.append(item)
    scores = {
        'Balanced original': .6 * ridge.predicted_return + .4 * ridge.momentum - k * (.5 * ridge.uncertainty + .25 * ridge.volatility),
        'Standardized weighted': factor_matrix @ np.array([.6, .4, .5 * k, .25 * k]),
        'Risk-adjusted return': ridge.predicted_return / np.maximum(ridge.volatility, RISK_FLOOR_PP),
        'Uncertainty-penalized': ridge.predicted_return - k * ridge.uncertainty,
        'Equal weight': np.zeros(len(ridge)),
    }
    if learned:
        scores['Constrained weights'] = factor_matrix @ np.array(list(learned.values()))
    for name, values in scores.items():
        item = ridge.copy()
        item['strategy'], item['score'] = name, np.asarray(values)
        output.append(item)
    diagnostics.update(k=float(k), risk_floor_pp=RISK_FLOOR_PP, balanced_formula=BALANCED_FORMULA,
        weight_fit=weight_diagnostics,
        score_units={'Balanced original': 'percentage-point score', 'Standardized weighted': 'dimensionless z score',
            'Risk-adjusted return': 'dimensionless return/volatility ratio',
            'Uncertainty-penalized': 'percentage-point score', 'Constrained weights': 'dimensionless z score'},
        development_score_warning='Constrained scores on development dates use development-fitted weights and are not out-of-sample weight-selection results.')
    result = pd.concat(output, ignore_index=True).sort_values(['cutoff', 'strategy', 'ticker']).reset_index(drop=True)
    return {'scores': result, 'weights': learned, 'diagnostics': diagnostics}


def market_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    """Trailing regimes of an equal-weight daily-rebalanced price-return proxy.

    Trend uses 126 trailing shared observations and ±5% thresholds. Volatility
    uses 21 shared-return observations and a strictly past expanding median with
    at least 126 available volatility observations. Annualization assumes 252
    shared sessions/year; this is a fixed-universe proxy, not a market index.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('Regimes require a dated nonempty price DataFrame.')
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.columns.has_duplicates:
        raise ValueError('Regime price dates and symbols must be unique and dates increasing.')
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Regimes require complete finite positive prices; no filling is performed.')
    daily = prices.pct_change(fill_method=None).mean(axis=1, skipna=False)
    wealth = pd.Series(np.r_[1.0, np.cumprod(1 + daily.iloc[1:].to_numpy())], index=prices.index)
    trend_return = wealth.pct_change(126, fill_method=None) * 100
    annual_volatility = daily.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252) * 100
    past_median = annual_volatility.expanding(min_periods=126).median().shift(1)
    trend = pd.Series('Unknown', index=prices.index)
    trend.loc[trend_return.notna()] = 'Sideways'
    trend.loc[trend_return > 5] = 'Bull'
    trend.loc[trend_return < -5] = 'Bear'
    volatility_regime = pd.Series('Unknown', index=prices.index)
    available = past_median.notna() & annual_volatility.notna()
    volatility_regime.loc[available & (annual_volatility > past_median)] = 'High volatility'
    volatility_regime.loc[available & (annual_volatility <= past_median)] = 'Low volatility'
    result = pd.DataFrame({'trend': trend, 'volatility_regime': volatility_regime,
        'trend_return': trend_return, 'annual_volatility': annual_volatility,
        'past_volatility_median': past_median})
    result.index.name = 'date'
    result.attrs.update(proxy='Equal-weight daily-rebalanced fixed-universe local-currency return proxy',
        annualization='252 shared sessions/year assumed; not calendar-day adjusted',
        threshold_policy='Current trailing volatility versus strictly past expanding median; warm-up explicitly Unknown')
    return result
