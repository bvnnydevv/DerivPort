"""Future dates contain reproducible hypothetical scenarios, never actual data."""

import numpy as np
import pandas as pd
import pytest

from derivport.projections import project_scenarios


@pytest.fixture
def prices():
    dates = pd.bdate_range("2022-01-03", "2026-11-13")
    rng = np.random.default_rng(47)
    returns = rng.normal([0.0002, -0.0001], [0.012, 0.021], (len(dates), 2))
    values = np.exp(np.cumsum(returns, axis=0)) * [80.0, 150.0]
    return pd.DataFrame(values, index=dates, columns=["BETA", "ALPHA"])


def test_scenarios_anchor_to_actual_prices_and_end_exactly_in_2037(prices):
    result = project_scenarios(prices, "2037-12-31")
    assert result.anchor_date == prices.index[-1]
    assert result.target_date == pd.Timestamp("2037-12-31")
    assert result.dates.is_unique and result.dates.is_monotonic_increasing
    assert all(day.is_month_end for day in result.dates[1:-1])
    assert result.terminal_returns.shape == (600, 2)
    for ticker in prices.columns:
        anchor = result.quantiles.loc[
            (result.quantiles["date"] == prices.index[-1]) & (result.quantiles["ticker"] == ticker)
        ].iloc[0]
        assert anchor[["p10", "p50", "p90"]].eq(prices.iloc[-1][ticker]).all()
    assert np.isfinite(result.quantiles[["p10", "p50", "p90"]]).all().all()
    assert (result.quantiles["p10"] > 0).all()
    assert (result.quantiles["p10"] <= result.quantiles["p50"]).all()
    assert (result.quantiles["p50"] <= result.quantiles["p90"]).all()
    assert "Hypothetical" in result.assumptions["status"]
    assert "not historical" in result.assumptions["status"]
    assert "21-session regression is not extrapolated" in result.assumptions["limitations"]
    assert not {"actual_return", "realized_rank", "accuracy"}.intersection(result.quantiles.columns)


def test_terminal_price_bands_match_the_shared_return_samples(prices):
    result = project_scenarios(prices, "2037-06-17")
    endpoint = result.quantiles.loc[result.quantiles["date"] == result.target_date].set_index("ticker")
    for ticker in prices.columns:
        terminal_prices = (1 + result.terminal_returns[ticker] / 100) * prices.iloc[-1][ticker]
        np.testing.assert_allclose(
            endpoint.loc[ticker, ["p10", "p50", "p90"]].to_numpy(dtype=float),
            np.quantile(terminal_prices, [.1, .5, .9]),
        )


def test_same_inputs_and_seed_are_stable_even_after_column_reordering(prices):
    first = project_scenarios(prices, "2037-12-31", seed=88)
    second = project_scenarios(prices[prices.columns[::-1]], "2037-12-31", seed=88)
    pd.testing.assert_frame_equal(first.quantiles, second.quantiles)
    pd.testing.assert_frame_equal(first.terminal_returns, second.terminal_returns)
    third = project_scenarios(prices, "2037-12-31", seed=89)
    assert not first.terminal_returns.equals(third.terminal_returns)


def test_zero_volatility_has_exact_assumed_log_growth(prices):
    result = project_scenarios(prices, "2037-12-31", annual_drift=.06, volatility_multiplier=0)
    years = (result.target_date - result.anchor_date).days / 365.25
    expected_return = np.expm1(.06 * years) * 100
    np.testing.assert_allclose(result.terminal_returns.to_numpy(), expected_return, atol=1e-12)
    assert result.annual_volatility.eq(0).all()
    assert result.quantiles["p10"].equals(result.quantiles["p90"])
    flat = project_scenarios(prices, "2026-11-20", annual_drift=0, volatility_multiplier=0)
    np.testing.assert_array_equal(flat.terminal_returns.to_numpy(), 0)


def test_drift_assumption_changes_log_growth_without_changing_random_samples(prices):
    low = project_scenarios(prices, "2037-12-31", annual_drift=-.03, seed=33)
    high = project_scenarios(prices, "2037-12-31", annual_drift=.07, seed=33)
    years = (high.target_date - high.anchor_date).days / 365.25
    relative_growth = (1 + high.terminal_returns / 100) / (1 + low.terminal_returns / 100)
    np.testing.assert_allclose(relative_growth.to_numpy(), np.exp(.1 * years), rtol=1e-12)


def test_log_innovation_correlation_matches_the_stated_assumption(prices):
    result = project_scenarios(prices, "2027-11-13", correlation=.65, n_paths=6000)
    innovations = np.log1p(result.terminal_returns / 100)
    empirical = innovations.corr().iloc[0, 1]
    assert empirical == pytest.approx(.65, abs=.035)


def test_calibration_uses_only_recent_completed_daily_returns(prices):
    result = project_scenarios(prices, "2037-12-31", volatility_multiplier=1.5)
    window = np.log(prices).diff().loc[prices.index[-1] - pd.DateOffset(years=3):].dropna()
    expected = window.std(ddof=1).sort_index() * np.sqrt(252) * 1.5
    np.testing.assert_allclose(result.annual_volatility.to_numpy(), expected.to_numpy())
    assert result.assumptions["volatility_observations"] == len(window)
    changed = prices.copy()
    changed.loc[changed.index < prices.index[-1] - pd.DateOffset(years=3) - pd.Timedelta(days=7)] *= 5
    stable = project_scenarios(changed, "2037-12-31", volatility_multiplier=1.5)
    pd.testing.assert_frame_equal(result.terminal_returns, stable.terminal_returns)


def test_one_stock_and_zero_historical_volatility_are_supported():
    single = pd.DataFrame({"ONLY": 120.0}, index=pd.bdate_range("2025-01-01", "2026-11-13"))
    result = project_scenarios(single, "2037-12-31", annual_drift=0)
    assert result.annual_volatility.eq(0).all()
    np.testing.assert_array_equal(result.terminal_returns.to_numpy(), 0)
    assert result.quantiles[["p10", "p50", "p90"]].eq(120).all().all()


@pytest.mark.parametrize("target", ["2026-11-13", "2025-01-01", "2038-01-01", "not-a-date", None])
def test_past_missing_and_unsupported_future_dates_are_rejected(prices, target):
    with pytest.raises(ValueError):
        project_scenarios(prices, target)


@pytest.mark.parametrize("kwargs", [
    {"annual_drift": float("nan")}, {"annual_drift": float("inf")},
    {"volatility_multiplier": -1}, {"volatility_multiplier": float("inf")},
    {"correlation": -.1}, {"correlation": 1.1}, {"correlation": float("nan")},
    {"n_paths": 199}, {"n_paths": 20001}, {"n_paths": 600.5},
    {"seed": -1}, {"seed": True}, {"annual_drift": 1e308},
])
def test_invalid_or_extreme_scenario_assumptions_are_readable_errors(prices, kwargs):
    with pytest.raises(ValueError):
        project_scenarios(prices, "2037-12-31", **kwargs)


def test_missing_bad_or_insufficient_price_history_is_rejected(prices):
    for invalid in [prices.iloc[:30], prices.assign(BETA=np.nan), prices.assign(BETA=-1), prices.iloc[0:0]]:
        with pytest.raises(ValueError):
            project_scenarios(invalid, "2037-12-31")
    duplicate = pd.concat([prices, prices.iloc[[-1]]])
    with pytest.raises(ValueError, match="unique"):
        project_scenarios(duplicate, "2037-12-31")
