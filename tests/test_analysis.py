"""Point-in-time training and consistent ranking on the bundled price history."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from derivport import (
    DEFAULT_FORMULA,
    analyze_cutoff,
    apply_rule,
    evaluate_history,
    historical_forecasts,
    load_prices,
    rank_correlation,
    reveal_outcomes,
)


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2024-05-03")


@pytest.fixture(scope="module")
def prices():
    return load_prices(ROOT / "data" / "prices.csv")


@pytest.fixture(scope="module")
def history(prices):
    return historical_forecasts(prices)


def test_analysis_is_unchanged_when_unavailable_future_prices_change(prices, history):
    original = analyze_cutoff(prices, CUTOFF, history)
    altered = prices.copy()
    future = altered.index > CUTOFF
    # Deliberately destroy future labels without touching any observable price.
    multipliers = np.linspace(3.0, 12.0, int(future.sum()))
    altered.loc[future] = altered.loc[future].mul(multipliers, axis=0)
    altered_history = historical_forecasts(altered)
    recomputed = analyze_cutoff(altered, CUTOFF, altered_history)
    pd.testing.assert_frame_equal(original.stocks, recomputed.stocks)
    assert original.training == recomputed.training
    assert original.error_summary == recomputed.error_summary
    assert pd.Timestamp(original.training["last_label_end"]) <= CUTOFF


def test_heldout_error_labels_finish_after_forecast(prices, history):
    assert not history.empty
    assert (pd.to_datetime(history["label_end"]) > pd.to_datetime(history["cutoff"])).all()
    assert (
        pd.to_datetime(history["training_last_label_end"])
        <= pd.to_datetime(history["cutoff"])
    ).all()
    for forecast_date, rows in history.groupby("cutoff"):
        assert set(rows["ticker"]) == set(prices.columns)
        assert len(rows) == 5
        start = prices.index.get_loc(pd.Timestamp(forecast_date))
        for label_end in rows["label_end"]:
            assert prices.index.get_loc(pd.Timestamp(label_end)) - start == 21


def test_uncertainty_is_rmse_of_only_completed_out_of_sample_errors(prices, history):
    analysis = analyze_cutoff(prices, CUTOFF, history)
    known = history.loc[pd.to_datetime(history["label_end"]) <= CUTOFF]
    for ticker, row in analysis.stocks.iterrows():
        errors = known.loc[known["ticker"] == ticker].sort_values("cutoff").tail(24)
        assert len(errors) >= 6
        expected = np.sqrt(np.mean((errors["predicted_return"] - errors["actual_return"]) ** 2))
        assert row["uncertainty"] == pytest.approx(expected)
        assert row["uncertainty_n"] == len(errors)


def test_heldout_model_and_baseline_errors_match_completed_labels(history):
    report = evaluate_history(history, cutoff=CUTOFF)
    known = history.loc[
        (pd.to_datetime(history["cutoff"]) >= pd.Timestamp("2022-01-01"))
        & (pd.to_datetime(history["label_end"]) <= CUTOFF)
    ].dropna(subset=["actual_return", "uncertainty"])
    expected_model = np.sqrt(np.mean((known["predicted_return"] - known["actual_return"]) ** 2))
    expected_baseline = np.sqrt(np.mean((known["momentum"] - known["actual_return"]) ** 2))
    assert report["model_rmse"] == pytest.approx(expected_model)
    assert report["baseline_rmse"] == pytest.approx(expected_baseline)
    assert report["n_forecasts"] == len(known)
    assert (pd.to_datetime(report["per_date"]["label_end"]) <= CUTOFF).all()


def test_spearman_uses_average_ties_and_handles_constant_rankings():
    # Average ranks are [1.5, 1.5, 3, 4], correlated with [1, 2, 3, 4].
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1]
    assert rank_correlation([1, 1, 3, 4], [1, 2, 3, 4]) == pytest.approx(expected)
    assert rank_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert rank_correlation([1, 1, 1], [1, 2, 3]) is None
    assert rank_correlation([1, 2, 3], [1, 1, 1]) is None


def test_spearman_aligns_stock_labels_before_comparing_orders():
    predicted = pd.Series([3., 2., 1.], index=["AAPL", "MSFT", "7203.T"])
    realized = pd.Series([1., 3., 2.], index=["7203.T", "AAPL", "MSFT"])
    assert rank_correlation(predicted, realized) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same stocks"):
        rank_correlation(predicted, realized.rename(index={"7203.T": "SAP.DE"}))


def test_displayed_rule_score_and_rank_are_the_same_calculation(prices, history):
    analysis = analyze_cutoff(prices, CUTOFF, history)
    ranked = apply_rule(analysis.stocks, DEFAULT_FORMULA, 2.0)
    expected = ranked["predicted_return"] - 2 * ranked["uncertainty"]
    np.testing.assert_allclose(ranked["score"], expected)
    pd.testing.assert_series_equal(
        ranked["student_rank"],
        expected.rank(ascending=False, method="average").rename("student_rank"),
        check_dtype=False,
    )


def test_all_algorithms_use_average_ranks_for_ties():
    stocks = pd.DataFrame({
        "predicted_return": [6., 6., 2., 1., 0.],
        "uncertainty": [2., 2., 2., 2., 2.],
        "momentum": [3., 3., 2., 1., 0.],
        "volatility": [1., 1., 1., 1., 1.],
    }, index=["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"])
    ranked = apply_rule(stocks, DEFAULT_FORMULA, 2.0)
    assert ranked.loc["AAPL", "student_rank"] == 1.5
    assert ranked.loc["AMZN", "student_rank"] == 1.5
    assert ranked.loc["GOOGL", "student_rank"] == 3


def test_reveal_returns_exactly_twenty_one_trading_days(prices):
    outcomes = reveal_outcomes(prices, CUTOFF)
    position = prices.index.get_loc(CUTOFF)
    expected_end = prices.index[position + 21]
    expected = (prices.loc[expected_end] / prices.loc[CUTOFF] - 1) * 100
    np.testing.assert_allclose(outcomes.loc[expected.index, "actual_return"], expected)
    assert pd.Timestamp(outcomes.attrs["end_date"]) == expected_end
    expected_ranks = expected.rank(ascending=False, method="average")
    np.testing.assert_allclose(outcomes.loc[expected.index, "realized_rank"], expected_ranks)
