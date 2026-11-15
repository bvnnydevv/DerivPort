"""Risk presets change dispersion; future charts retain simulated meanings."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from derivport.projections import project_scenarios
from derivport.projection_views import (RISK_PRESETS, apply_risk_preset,
    scenario_distribution, scenario_fan)


@pytest.fixture
def prices():
    dates = pd.bdate_range('2024-01-02', '2026-11-13')
    rng = np.random.default_rng(91)
    values = 100 * np.exp(np.cumsum(rng.normal(.0001, .014, len(dates))))
    return pd.DataFrame({'AAPL': values}, index=dates)


@pytest.mark.parametrize('risk,multiplier', [('Low', .75), ('Medium', 1.0), ('High', 1.5)])
def test_risk_preset_only_updates_volatility_and_never_promises_a_higher_drift(risk, multiplier):
    state = dict(projection_risk=risk, projection_volatility=2.0,
        projection_drift=-3.0, projection_correlation=.25, unrelated='keep')
    apply_risk_preset(state)
    assert state['projection_volatility'] == multiplier
    assert state['projection_drift'] == -3.0
    assert state['projection_correlation'] == .25
    assert state['unrelated'] == 'keep'
    assert RISK_PRESETS[risk]['annual_log_drift_percent'] == 6.0
    assert RISK_PRESETS[risk]['correlation'] == .6


def test_custom_preset_does_not_overwrite_manual_assumptions():
    state = dict(projection_risk='Custom', projection_volatility=2.25,
        projection_drift=2., projection_correlation=.3)
    before = deepcopy(state)
    apply_risk_preset(state)
    assert state == before
    with pytest.raises(ValueError, match='Choose'):
        apply_risk_preset({'projection_risk': 'Unknown'})


def test_higher_risk_widens_seeded_terminal_dispersion_without_raising_log_drift(prices):
    runs = [project_scenarios(prices, '2030-12-31', annual_drift=.06,
        volatility_multiplier=preset['volatility_multiplier'], correlation=.6, seed=81)
        for preset in RISK_PRESETS.values()]
    widths = [np.quantile(run.terminal_returns.AAPL, .9) - np.quantile(run.terminal_returns.AAPL, .1)
        for run in runs]
    assert widths[0] < widths[1] < widths[2]
    assert all(run.assumptions['annual_log_drift'] == .06 for run in runs)
    assert all(run.assumptions['correlation'] == .6 for run in runs)
    # Same analytic median-growth assumption; do not claim equality of mean
    # simple returns, which still depends on lognormal volatility/convexity.
    years = (runs[0].target_date - runs[0].anchor_date).days / 365.25
    expected_medians = [run.latest_prices.AAPL * np.exp(run.assumptions['annual_log_drift'] * years) for run in runs]
    np.testing.assert_allclose(expected_medians, expected_medians[0])


def test_fan_uses_actual_scenario_percentiles_and_only_one_observed_anchor(prices):
    projected = project_scenarios(prices, '2030-12-31')
    original = projected.quantiles.copy(deep=True)
    fig = scenario_fan(projected, 'AAPL')
    selected = projected.quantiles[projected.quantiles.ticker.eq('AAPL')]
    for trace, field in zip(fig.data[:3], ['p90', 'p10', 'p50']):
        np.testing.assert_allclose(trace.y, selected[field])
        assert list(pd.to_datetime(trace.x)) == list(projected.dates)
    assert len(fig.data[3].x) == len(fig.data[3].y) == 1
    assert pd.Timestamp(fig.data[3].x[0]) == projected.anchor_date
    assert fig.data[3].y[0] == projected.latest_prices.AAPL
    assert 'observed future' in fig.layout.meta['status']
    assert fig.layout.meta['kind'] == 'simulated_percentile_fan'
    pd.testing.assert_frame_equal(projected.quantiles, original)


def test_distribution_uses_shared_terminal_samples_and_correct_percentile_markers(prices):
    projected = project_scenarios(prices, '2030-12-31', seed=44)
    fig = scenario_distribution(projected, 'AAPL')
    assert fig.data[0].type == 'histogram'
    np.testing.assert_array_equal(fig.data[0].x, projected.terminal_returns.AAPL)
    expected = np.quantile(projected.terminal_returns.AAPL, [.5, .1, .9])
    np.testing.assert_allclose([shape.x0 for shape in fig.layout.shapes], expected)
    assert all(shape.x0 == shape.x1 for shape in fig.layout.shapes)
    assert 'Hypothetical return' in fig.layout.xaxis.title.text
    assert not fig.layout.annotations, 'Percentile values use a wrapping legend, not overlapping plot labels.'
    assert fig.layout.meta['kind'] == 'simulated_terminal_distribution'


def test_both_views_serialize_as_plain_arrays_and_represent_different_scenario_questions(prices):
    projected = project_scenarios(prices, '2030-12-31')
    fan, histogram = scenario_fan(projected, 'AAPL'), scenario_distribution(projected, 'AAPL')
    assert fan.data[2].line.color != histogram.data[0].marker.color
    for fig in (fan, histogram):
        payload = json.loads(fig.to_json())
        assert 'bdata' not in fig.to_json()
        assert not fig.layout.modebar.add
        for trace in payload['data']:
            assert isinstance(trace['x'], list)
            if 'y' in trace:
                assert isinstance(trace['y'], list)


@pytest.mark.parametrize('render', [scenario_fan, scenario_distribution])
def test_bad_stock_and_nonfinite_scenario_values_are_rejected(prices, render):
    projected = project_scenarios(prices, '2030-12-31')
    with pytest.raises(ValueError, match='chosen stock'):
        render(projected, 'MISSING')
    damaged = deepcopy(projected)
    damaged.terminal_returns.loc[0, 'AAPL'] = np.inf
    with pytest.raises(ValueError, match='finite'):
        render(damaged, 'AAPL')
    damaged = deepcopy(projected)
    damaged.quantiles.loc[1, 'p90'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        render(damaged, 'AAPL')
