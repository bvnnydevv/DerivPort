"""Portfolio accounting, delayed execution, and data-boundary invariants."""

import numpy as np
import pandas as pd
import pytest

from derivport.portfolio import backtest_portfolio


def prices(n=64):
    return pd.DataFrame(100.0, index=pd.bdate_range("2020-01-01", periods=n), columns=["A", "B"])


def signals(frame, origins=(0,), leaders=("A",)):
    rows = []
    for position, leader in zip(origins, leaders):
        for ticker in frame.columns:
            rows.append({"cutoff": frame.index[position], "ticker": ticker,
                         "score": 1.0 if ticker == leader else 0.0,
                         "training_last_label_end": frame.index[position]})
    return pd.DataFrame(rows)


def run(frame, forecasts=None, **kwargs):
    arguments = dict(top_n=1, max_weight=1.0, commission_bps=0.0, slippage_bps=0.0)
    arguments.update(kwargs)
    return backtest_portfolio(frame, signals(frame) if forecasts is None else forecasts, **arguments)


def test_cutoff_close_information_cannot_earn_the_return_into_the_next_close():
    frame = prices(6)
    frame["A"] = [100, 200, 300, 300, 300, 300]
    result = run(frame, liquidate=False)
    daily = result["daily"]
    assert daily.iloc[0].net_equity == 1
    assert daily.iloc[1].net_equity == 1, "The 100→200 pre-fill jump is unavailable."
    assert daily.iloc[2].net_equity == pytest.approx(1.5)
    assert result["metrics"]["total_return"] == pytest.approx(0.5)
    assert result["trades"].iloc[0]["date"] == frame.index[1]
    assert daily.iloc[1].signal_cutoff == frame.index[0]


def test_exact_self_financing_entry_and_liquidation_costs():
    frame = prices(6)
    commission, slippage = 10 / 10000, 5 / 10000
    rate = commission + slippage
    result = run(frame, commission_bps=10, slippage_bps=5)
    expected_entry = 1 / (1 + rate)
    expected_exit = expected_entry * (1 - rate)
    assert result["daily"].iloc[1].net_equity == pytest.approx(expected_entry)
    assert result["daily"].iloc[-1].net_equity == pytest.approx(expected_exit)
    assert result["daily"].iloc[1].net_return == pytest.approx(expected_entry - 1)
    assert result["metrics"]["total_return"] == pytest.approx(expected_exit - 1)
    assert result["metrics"]["costs"] == pytest.approx(2 * rate / (1 + rate))
    assert result["metrics"]["gross_total_return"] == 0
    assert result["metrics"]["turnover"] == pytest.approx(1)
    assert result["metrics"]["notional_turnover"] == pytest.approx(expected_entry + 1)
    assert result["trades"]["notional"].sum() == pytest.approx(2 * expected_entry)
    assert result["trades"]["commission"].sum() == pytest.approx(commission * 2 * expected_entry)
    assert result["trades"]["slippage"].sum() == pytest.approx(slippage * 2 * expected_entry)
    assert result["trades"]["cost"].sum() == pytest.approx(result["daily"]["cost"].sum())
    assert result["trades"]["event"].tolist() == ["rebalance", "liquidation"]
    assert result["weights"].iloc[-1].sum() == 0
    assert result["daily"].iloc[-1].cash_weight == 1


def test_switching_assets_charges_both_sale_and_purchase_notionals():
    frame = prices(45)
    rate = 15 / 10000
    result = run(frame, signals(frame, (0, 21), ("A", "B")),
                 commission_bps=10, slippage_bps=5, liquidate=False)
    expected = 1 / (1 + rate) * (1 - rate) / (1 + rate)
    assert result["daily"].iloc[-1].net_equity == pytest.approx(expected)
    switch = result["trades"].loc[result["trades"]["date"] == frame.index[22]]
    assert set(switch["side"]) == {"buy", "sell"}
    assert switch["notional"].sum() > 1.9
    assert switch["cost"].sum() == pytest.approx(rate * switch["notional"].sum())
    assert result["weights"].loc[frame.index[22], "B"] == pytest.approx(1)
    assert result["metrics"]["gross_total_return"] == 0


def test_weights_drift_between_rebalances_and_caps_apply_to_targets():
    frame = prices(45)
    frame.loc[frame.index[3]:, "A"] = 200
    result = run(frame, signals(frame, (0, 21), ("A", "A")), top_n=2, max_weight=.4,
                 gross_exposure=1, liquidate=False)
    assert result["weights"].iloc[1].to_numpy() == pytest.approx([.4, .4])
    assert result["daily"].iloc[1].cash_weight == pytest.approx(.2)
    assert result["weights"].iloc[3]["A"] == pytest.approx(.8 / 1.4)
    assert result["daily"].iloc[3].max_weight > .4
    assert result["metrics"]["max_observed_weight"] > .4
    assert result["weights"].loc[frame.index[22]].to_numpy() == pytest.approx([.4, .4])
    np.testing.assert_allclose(result["weights"].sum(axis=1) + result["daily"].cash_weight, 1)
    assert (result["weights"] >= 0).all().all()


def test_daily_marking_captures_a_drawdown_hidden_by_period_endpoints():
    frame = prices(25)
    frame.loc[frame.index[10]:frame.index[18], "A"] = 50
    result = run(frame)
    assert result["metrics"]["total_return"] == 0
    assert result["metrics"]["max_drawdown"] == pytest.approx(-.5)
    assert result["daily"].loc[frame.index[10], "drawdown"] == pytest.approx(-.5)


def test_future_price_mutations_do_not_change_earlier_nav_or_trades():
    frame = prices(64)
    forecast = signals(frame, (0, 21, 42), ("A", "B", "A"))
    changed = frame.copy()
    changed.loc[changed.index[35]:, "A"] *= 4
    first = run(frame, forecast, commission_bps=10)
    second = run(changed, forecast, commission_bps=10)
    pd.testing.assert_frame_equal(first["daily"].iloc[:35], second["daily"].iloc[:35])
    pd.testing.assert_frame_equal(
        first["trades"].loc[first["trades"]["date"] < frame.index[35]],
        second["trades"].loc[second["trades"]["date"] < frame.index[35]],
    )


def test_flat_observed_returns_have_undefined_sharpe_and_zero_growth():
    result = run(prices(10))
    assert result["metrics"]["sharpe"] is None
    assert result["metrics"]["annualized_return"] == 0
    assert result["metrics"]["max_drawdown"] == 0
    assert result["daily"]["net_return"].eq(0).all()


def test_sharpe_subtracts_compounded_daily_risk_free_return():
    frame = prices(6)
    frame["A"] = [100, 100, 102, 100.98, 104.0094, 104.0094]
    result = run(frame, annual_rf=.05, liquidate=False)
    expected = np.array([0, .02, -.01, .03, 0]) - np.expm1(np.log1p(.05) / 252)
    sharpe = expected.mean() / expected.std(ddof=1) * np.sqrt(252)
    assert result["metrics"]["sharpe"] == pytest.approx(sharpe)
    assert result["metrics"]["annualized_return"] == pytest.approx(1.040094 ** (252 / 5) - 1)


def test_scores_and_prices_are_order_invariant_with_deterministic_ties():
    frame = prices(10)
    forecast = signals(frame).assign(score=0)
    first = run(frame, forecast, liquidate=False)
    second = run(frame[["B", "A"]], forecast.iloc[::-1], liquidate=False)
    pd.testing.assert_frame_equal(first["daily"], second["daily"])
    pd.testing.assert_frame_equal(first["weights"], second["weights"])
    pd.testing.assert_frame_equal(first["trades"], second["trades"])
    assert first["trades"].iloc[0].ticker == "A"


def test_untrusted_actual_return_columns_are_never_used_for_pnl():
    frame = prices(10)
    original = signals(frame)
    fabricated = original.assign(actual_return=999999, label_end=frame.index[-1])
    pd.testing.assert_frame_equal(run(frame, original)["daily"], run(frame, fabricated)["daily"])


def test_minimum_rebalance_spacing_skips_intermediate_scores_and_honors_end():
    frame = prices(65)
    forecast = signals(frame, (0, 21, 42), ("A", "B", "A"))
    result = run(frame, forecast, rebalance_sessions=42)
    assert result["metadata"]["execution_dates"] == [frame.index[1].date().isoformat(), frame.index[43].date().isoformat()]
    assert result["metadata"]["skipped_for_rebalance_interval"] == 1
    shortened = run(frame, forecast, end=frame.index[30])
    assert shortened["daily"].index[-1] == frame.index[30]
    assert shortened["metrics"]["n_rebalances"] == 2


def test_cash_exposure_and_top_count_larger_than_small_universe_are_supported():
    frame = prices(6)
    result = run(frame, top_n=3, max_weight=0, gross_exposure=0)
    assert result["weights"].eq(0).all().all()
    assert result["daily"].cash_weight.eq(1).all()
    assert result["trades"].empty
    assert result["metrics"]["sharpe"] is None


@pytest.mark.parametrize("kwargs", [
    {"top_n": 0}, {"top_n": 1.5}, {"execution_lag": 0}, {"execution_lag": True},
    {"rebalance_sessions": 20}, {"max_weight": .8, "gross_exposure": .5},
    {"gross_exposure": 1.1}, {"commission_bps": -1}, {"slippage_bps": np.nan},
    {"commission_bps": 10000}, {"annual_rf": -1}, {"liquidate": "yes"},
])
def test_invalid_constraints_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run(prices(), **kwargs)


def test_forecast_cutoff_completeness_training_dates_and_prices_are_enforced():
    frame = prices()
    forecast = signals(frame)
    invalid = [
        forecast.iloc[:1], pd.concat([forecast, forecast.iloc[:1]]),
        forecast.assign(score=np.nan), forecast.assign(cutoff="2037-01-01"),
        forecast.assign(training_last_label_end=frame.index[1]),
        forecast.assign(training_last_label_end=pd.NaT),
    ]
    for changed in invalid:
        with pytest.raises(ValueError):
            run(frame, changed)
    missing = frame.copy()
    missing.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="complete|finite"):
        run(missing, forecast)
    with pytest.raises(ValueError, match="No complete"):
        run(frame, forecast, start=frame.index[2])
