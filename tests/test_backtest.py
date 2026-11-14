"""Chronological backtests use completed windows and the same paths for each rule."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivport import analyze_cutoff, backtest_rule, historical_forecasts, load_prices, validate_prices


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prices():
    return load_prices(ROOT / "data" / "prices.csv")


@pytest.fixture(scope="module")
def history(prices):
    return historical_forecasts(prices)


def test_momentum_rule_reproduces_the_momentum_baseline_exactly(prices, history):
    result = backtest_rule(
        prices, "2023-01-01", "2024-12-31", formula="momentum", k=2, history=history
    )
    per_date = result["per_date"]
    assert len(per_date) > 12
    assert (per_date["student_top"] == per_date["momentum_top"]).all()
    np.testing.assert_array_equal(per_date["student_return"], per_date["momentum_return"])
    np.testing.assert_allclose(per_date["student_spearman"], per_date["momentum_spearman"])
    pd.testing.assert_series_equal(
        result["linked_returns"]["student"], result["linked_returns"]["momentum"], check_names=False
    )
    pd.testing.assert_series_equal(
        result["summary"].loc["student"], result["summary"].loc["momentum"], check_names=False
    )


def test_backtest_date_bounds_and_compounding_are_honored(prices, history):
    start, end = pd.Timestamp("2023-03-01"), pd.Timestamp("2023-12-31")
    result = backtest_rule(prices, start, end, history=history)
    per_date = result["per_date"]
    assert not per_date.empty
    assert (pd.to_datetime(per_date["cutoff"]) >= start).all()
    assert (pd.to_datetime(per_date["label_end"]) <= end).all()
    assert (pd.to_datetime(per_date["label_end"]) > pd.to_datetime(per_date["cutoff"])).all()
    for _, row in per_date.iterrows():
        origin = prices.index.get_loc(pd.Timestamp(row["cutoff"]))
        finish = prices.index.get_loc(pd.Timestamp(row["label_end"]))
        assert finish - origin == 21
    assert result["metrics"]["n_dates"] == len(per_date)
    assert result["metrics"]["n_forecasts"] == len(per_date) * prices.shape[1]
    for method in ["model", "momentum", "student"]:
        returns = per_date[f"{method}_return"].to_numpy()
        expected = ((1 + returns / 100).cumprod() - 1) * 100
        curve = result["linked_returns"][method]
        # A curve may explicitly include its zero-return starting point.
        np.testing.assert_allclose(curve.to_numpy()[-len(expected):], expected)
        assert result["summary"].loc[method, "linked_return"] == pytest.approx(expected[-1])


def test_backtest_constant_rule_has_no_defined_rank_correlation(prices, history):
    result = backtest_rule(prices, "2023-01-01", "2023-12-31", formula="1", history=history)
    per_date = result["per_date"]
    assert per_date["student_spearman"].isna().all()
    assert per_date["student_top"].eq("AAPL").all(), "The shared alphabetical tie policy must apply."


def test_non_default_two_stock_universe_is_point_in_time(prices):
    smaller = prices[["AAPL", "MSFT"]]
    history = historical_forecasts(smaller)
    analysis = analyze_cutoff(smaller, "2024-05-03", history)
    assert set(analysis.stocks.index) == {"AAPL", "MSFT"}
    assert analysis.stocks["model_rank"].between(1, 2).all()
    assert (pd.to_datetime(history["training_last_label_end"]) <= pd.to_datetime(history["cutoff"])).all()
    result = backtest_rule(smaller, "2023-01-01", "2023-12-31", formula="momentum", history=history)
    assert result["metrics"]["n_forecasts"] == result["metrics"]["n_dates"] * 2
    np.testing.assert_array_equal(result["per_date"]["student_return"], result["per_date"]["momentum_return"])


def test_custom_symbol_names_load_without_a_fixed_five_stock_gate(tmp_path, prices):
    selected = prices[["AAPL", "MSFT", "GOOGL"]].rename(columns={
        "AAPL": "TEST.US", "MSFT": "TEST.L", "GOOGL": "TEST.T"
    })
    raw = selected.rename_axis(columns="ticker").stack().rename("adjusted_close").reset_index()
    path = tmp_path / "custom_symbols.csv"
    raw.to_csv(path, index=False)
    reloaded = load_prices(path)
    assert set(reloaded.columns) == set(selected.columns)
    pd.testing.assert_frame_equal(reloaded.sort_index(axis=1), selected.sort_index(axis=1), check_freq=False)


def test_history_from_another_universe_cannot_be_reused(prices, history):
    smaller = prices[["AAPL", "MSFT"]]
    with pytest.raises(ValueError, match="universe"):
        analyze_cutoff(smaller, "2024-05-03", history)
    with pytest.raises(ValueError, match="universe"):
        backtest_rule(smaller, "2023-01-01", "2023-12-31", history=history)


def test_invalid_and_unavailable_backtest_ranges_explain_the_problem(prices, history):
    with pytest.raises(ValueError, match="start date"):
        backtest_rule(prices, "2024-01-01", "2023-01-01", history=history)
    with pytest.raises(ValueError, match="complete|range"):
        backtest_rule(prices, "2023-01-01", "2023-01-02", history=history)
    result = backtest_rule(prices, "2023-01-01", "2030-01-01", history=history)
    assert pd.Timestamp(result["metadata"]["effective_end"]) == prices.index.max()
    assert pd.to_datetime(result["per_date"]["label_end"]).max() <= prices.index.max()


def test_incomplete_or_single_stock_universes_are_rejected_without_filling(prices):
    with pytest.raises(ValueError, match="at least two"):
        validate_prices(prices[["AAPL"]])
    missing = prices[["AAPL", "MSFT"]].copy()
    missing.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match="complete|forward"):
        validate_prices(missing)
