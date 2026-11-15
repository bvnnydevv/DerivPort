"""Independent arithmetic, chronological weight fitting, and regime invariance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algorithms.derivport_signal import score_row
from derivport.research_signals import build_signal_panel, market_regimes


def small_panel(volatility=(5., 5.)):
    return pd.DataFrame([
        dict(model='Ridge', cutoff='2023-01-03', ticker=ticker, predicted_return=prediction,
            momentum=10., uncertainty=error, volatility=vol, actual_return=actual,
            label_end='2023-02-01', split='test')
        for ticker, prediction, error, vol, actual in zip(
            ['AAA', 'BBB'], [1., 3.], [2., 4.], volatility, [2., -1.])])


def historical_panel():
    rows = []
    models = ['Ridge', 'Elastic Net', 'Random Forest', 'Gradient Boosting', 'Momentum', 'Zero return']
    for i, cutoff in enumerate(pd.date_range('2018-01-01', periods=42, freq='MS')):
        for j, ticker in enumerate(['AAA', 'BBB', 'CCC', 'DDD']):
            prediction = np.sin(i * .4 + j) * 4 + j
            momentum = np.cos(i * .1 + j) * 8
            actual = .6 * prediction + .1 * momentum + np.sin(i + 2 * j)
            for model in models:
                forecast = 0. if model == 'Zero return' else momentum if model == 'Momentum' else prediction + .1 * models.index(model)
                rows.append(dict(model=model, cutoff=cutoff, ticker=ticker,
                    predicted_return=forecast, momentum=momentum, uncertainty=2 + j * .7 + (i % 5) * .1,
                    volatility=3 + j + (i % 3) * .1, actual_return=actual,
                    label_end=cutoff + pd.offsets.BDay(21),
                    split='development' if cutoff < pd.Timestamp('2021-01-01') else 'test'))
    return pd.DataFrame(rows)


def by_strategy(result, strategy):
    return result['scores'].loc[result['scores'].strategy.eq(strategy)].sort_values(['cutoff', 'ticker'])


def test_explicit_signal_math_and_units_are_not_interchangeable_forecasts():
    result = build_signal_panel(small_panel())
    np.testing.assert_allclose(by_strategy(result, 'Balanced original').score, [2.35, 2.55])
    # z(pred) = [-1,1], z(momentum)=0, z(error)=[-1,1], z(vol)=0.
    np.testing.assert_allclose(by_strategy(result, 'Standardized weighted').score, [-.1, .1])
    np.testing.assert_allclose(by_strategy(result, 'Risk-adjusted return').score, [.2, .6])
    np.testing.assert_allclose(by_strategy(result, 'Uncertainty-penalized').score, [-1., -1.])
    np.testing.assert_allclose(by_strategy(result, 'Equal weight').score, [0., 0.])
    assert result['diagnostics']['score_units']['Risk-adjusted return'].startswith('dimensionless')
    assert result['diagnostics']['score_units']['Balanced original'].startswith('percentage-point')
    assert result['weights'] == {}
    assert result['diagnostics']['weight_fit']['status'] == 'unavailable'
    assert 'Constrained weights' not in result['scores'].strategy.unique()


def test_balanced_signal_matches_the_original_download_for_every_row():
    panel = historical_panel()
    result = build_signal_panel(panel, test_start='2021-01-01', k=2.5)
    actual = by_strategy(result, 'Balanced original')
    expected = [score_row(row, k=2.5) for row in actual.to_dict('records')]
    np.testing.assert_allclose(actual.score, expected, rtol=0, atol=1e-12)


def test_constant_factors_and_the_volatility_floor_are_explicit():
    panel = small_panel(volatility=(0., .5))
    np.testing.assert_allclose(by_strategy(build_signal_panel(panel), 'Risk-adjusted return').score, [1., 3.])
    for column in ['predicted_return', 'momentum', 'uncertainty', 'volatility']:
        panel[column] = 2.
    standardized = by_strategy(build_signal_panel(panel), 'Standardized weighted')
    assert standardized.score.eq(0).all()
    assert np.isfinite(standardized.score).all()


def test_constrained_weights_use_purged_development_and_remain_frozen_for_test():
    panel = historical_panel()
    first = build_signal_panel(panel, test_start='2021-01-01')
    fit = first['diagnostics']['weight_fit']
    assert fit['status'] == 'available' and fit['converged']
    weights = np.array(list(first['weights'].values()))
    assert np.all(weights >= 0) and np.all(weights <= .6 + 1e-12)
    assert weights.sum() == pytest.approx(1., abs=1e-12)
    assert pd.Timestamp(fit['train_last_label_end']) < pd.Timestamp(fit['validation_first'])
    assert pd.Timestamp(fit['final_fit_last_label_end']) < pd.Timestamp('2021-01-01')
    assert fit['chosen_lambda'] in [.1, 1., 10.]
    changed = panel.copy()
    test = changed.split.eq('test')
    changed.loc[test, 'actual_return'] = changed.loc[test, 'actual_return'] ** 3 + 900
    changed.loc[test, 'predicted_return'] *= 100
    changed.loc[test, 'uncertainty'] += 77
    changed.loc[test, 'volatility'] += 99
    second = build_signal_panel(changed, test_start='2021-01-01')
    assert first['weights'] == second['weights']
    assert first['diagnostics']['weight_fit'] == second['diagnostics']['weight_fit']
    pd.testing.assert_frame_equal(by_strategy(first, 'Constrained weights').loc[lambda x:x.split.eq('development')],
        by_strategy(second, 'Constrained weights').loc[lambda x:x.split.eq('development')])


def test_incomplete_cross_sections_have_one_common_date_support_without_filling():
    panel = historical_panel()
    absent_date = panel.cutoff.iloc[0]
    panel = panel.drop(panel.index[(panel.cutoff == absent_date) & panel.model.eq('Random Forest') & panel.ticker.eq('AAA')])
    result = build_signal_panel(panel, test_start='2021-01-01')
    assert absent_date not in set(result['scores'].cutoff)
    assert result['diagnostics']['excluded_incomplete_dates'] == 1
    dates = result['scores'].groupby('strategy').cutoff.nunique()
    assert dates.nunique() == 1
    assert np.isfinite(result['scores'].score).all()


def test_nonfinite_complete_date_is_excluded_and_inconsistent_targets_are_rejected():
    panel = historical_panel()
    panel.loc[0, 'predicted_return'] = np.nan
    assert build_signal_panel(panel, test_start='2021-01-01')['diagnostics']['excluded_incomplete_dates'] == 1
    panel = historical_panel()
    panel.loc[0, 'actual_return'] += 5
    with pytest.raises(ValueError, match='actual returns'):
        build_signal_panel(panel, test_start='2021-01-01')


@pytest.mark.parametrize('column', ['uncertainty', 'volatility'])
def test_negative_risk_inputs_are_rejected(column):
    panel = small_panel()
    panel.loc[0, column] = -1.
    with pytest.raises(ValueError, match='nonnegative'):
        build_signal_panel(panel)


def test_regime_classification_never_uses_future_prices_or_current_median():
    generator = np.random.default_rng(731)
    dates = pd.bdate_range('2018-01-01', periods=300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(generator.normal(.0002, .01, (300, 2)), axis=0)), index=dates, columns=['AAA','BBB'])
    first = market_regimes(prices)
    changed = prices.copy()
    changed.iloc[181:] *= np.exp(np.arange(119)[:, None] / 3)
    second = market_regimes(changed)
    pd.testing.assert_frame_equal(first.iloc[:181], second.iloc[:181])
    assert first.trend.iloc[:126].eq('Unknown').all()
    assert first.volatility_regime.iloc[:147].eq('Unknown').all()
    expected = first.annual_volatility.iloc[:180].dropna().median()
    assert first.past_volatility_median.iloc[180] == pytest.approx(expected)
    assert 'fixed-universe' in first.attrs['proxy']


def test_flat_proxy_is_sideways_and_zero_volatility_is_not_called_high():
    prices = pd.DataFrame(100., index=pd.bdate_range('2020-01-01', periods=200), columns=['AAA','BBB'])
    regimes = market_regimes(prices)
    assert regimes.trend.iloc[-1] == 'Sideways'
    assert regimes.volatility_regime.iloc[-1] == 'Low volatility'
    assert regimes.trend_return.iloc[-1] == 0
    assert regimes.annual_volatility.iloc[-1] == 0


def test_missing_price_regimes_do_not_silently_forward_fill():
    prices = pd.DataFrame({'AAA':[100., np.nan, 101.]}, index=pd.bdate_range('2020-01-01', periods=3))
    with pytest.raises(ValueError, match='no filling'):
        market_regimes(prices)
