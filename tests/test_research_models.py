"""Point-in-time model selection, aligned support and observable calibration."""
import json

import numpy as np
import pandas as pd
import pytest

from derivport.analysis import FEATURES, HORIZON, _fit, _uncertainty, build_features, historical_forecasts
from derivport.research_models import MODEL_NAMES, _calibration, walk_forward_models


@pytest.fixture(scope="module")
def prices():
    # Explicitly synthetic, correlated prices with changing drift and volatility.
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2018-01-02", periods=850)
    common = rng.normal(0, .008, len(dates))
    innovations = rng.normal(0, .009, (len(dates), 3))
    drift = np.where(np.arange(len(dates)) < 450, .0006, -.0002)
    returns = common[:, None] + innovations + drift[:, None]
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
        columns=["B", "A", "C"])


def settings(prices):
    return dict(validation_start=prices.index[280], test_start=prices.index[610], seed=2026)


@pytest.fixture(scope="module")
def result(prices):
    return walk_forward_models(prices, **settings(prices))


def test_hand_calculated_features_and_target_use_exactly_twenty_one_sessions():
    dates = pd.bdate_range("2020-01-01", periods=70)
    daily_b = np.resize(np.array([.02, -.01, .005]), len(dates) - 1)
    prices = pd.DataFrame({
        "A": 100 * 1.01 ** np.arange(len(dates)),
        "B": 100 * np.r_[1., np.cumprod(1 + daily_b)],
    }, index=dates)
    features = build_features(prices).set_index(["cutoff", "ticker"])
    row = features.loc[(dates[35], "A")]
    assert row.momentum == pytest.approx((1.01 ** 21 - 1) * 100)
    assert row.actual_return == pytest.approx((1.01 ** 21 - 1) * 100)
    assert row.volatility == pytest.approx(0, abs=1e-12)
    assert row.label_end == dates[56]
    b = features.loc[(dates[35], "B")]
    assert b.volatility == pytest.approx(np.std(daily_b[14:35], ddof=1) * np.sqrt(21) * 100)
    assert b.momentum == pytest.approx((np.prod(1 + daily_b[14:35]) - 1) * 100)


def test_models_share_support_and_every_training_outcome_is_already_observed(prices, result):
    forecasts, folds, selection = result["forecasts"], result["folds"], result["selection"]
    test_start = settings(prices)["test_start"]
    assert set(forecasts.model) == set(MODEL_NAMES)
    assert set(forecasts.split) == {"development", "test"}
    for _, group in forecasts.groupby("cutoff"):
        assert len(group) == len(MODEL_NAMES) * len(prices.columns)
        assert all(set(rows.ticker) == set(prices.columns) for _, rows in group.groupby("model"))
        assert group.training_n.nunique() == 1
        assert group.label_end.nunique() == 1
    assert (forecasts.training_last_label_end <= forecasts.cutoff).all()
    assert (folds.training_last_label_end <= folds.cutoff).all()
    assert (folds.training_last_feature_date < folds.cutoff).all()
    assert (folds.label_end < test_start).all()
    assert (selection.last_validation_label_end < test_start).all()
    assert (forecasts.loc[forecasts.split == "development", "label_end"] < test_start).all()
    assert (forecasts.loc[forecasts.split == "test", "cutoff"] >= test_start).all()
    positions = [prices.index.get_loc(date) for date in forecasts.cutoff.unique()]
    assert np.all(np.diff(positions) >= HORIZON)
    assert all(prices.index.get_loc(end) - prices.index.get_loc(start) == HORIZON
        for start, end in forecasts[["cutoff", "label_end"]].drop_duplicates().itertuples(index=False, name=None))
    for model, rows in forecasts.loc[forecasts.split == "test"].groupby("model"):
        assert rows.selected_params.nunique() == 1
        assert json.loads(rows.selected_params.iloc[0]) == result["metadata"]["selected_params"][model]


def test_ridge_predictions_preserve_the_existing_baseline_and_naive_definitions(prices, result):
    features = build_features(prices)
    origin = result["forecasts"].loc[result["forecasts"].split == "test", "cutoff"].min()
    old_model, train = _fit(features, origin)
    today = features.loc[features.cutoff == origin].sort_values("ticker")
    rows = result["forecasts"].loc[result["forecasts"].cutoff == origin]
    ridge = rows.loc[rows.model == "Ridge"].sort_values("ticker")
    np.testing.assert_allclose(ridge.predicted_return, old_model.predict(today[FEATURES]), rtol=0, atol=1e-12)
    assert ridge.training_n.eq(len(train)).all()
    momentum = rows.loc[rows.model == "Momentum"]
    np.testing.assert_allclose(momentum.predicted_return, momentum.momentum, rtol=0, atol=0)
    assert rows.loc[rows.model == "Zero return", "predicted_return"].eq(0).all()


def test_unavailable_future_prices_cannot_change_any_model_at_a_development_origin(prices, result):
    origin = sorted(result["forecasts"].cutoff.unique())[8]
    changed = prices.copy()
    future = changed.index > origin
    changed.loc[future] = changed.loc[future].mul(np.linspace(1.5, 4, int(future.sum())), axis=0)
    recomputed = walk_forward_models(changed, **settings(prices))
    fields = ["model", "ticker", "predicted_return", "uncertainty", "uncertainty_n", "uncertainty_bias",
        "interval_lower", "interval_upper", "training_last_label_end", "training_n", "selected_params"]
    before = result["forecasts"].loc[result["forecasts"].cutoff == origin, fields].reset_index(drop=True)
    after = recomputed["forecasts"].loc[recomputed["forecasts"].cutoff == origin, fields].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_final_test_labels_never_select_parameters_or_modify_development(prices, result):
    changed = prices.copy()
    final = changed.index >= settings(prices)["test_start"]
    changed.loc[final] = changed.loc[final].mul(np.linspace(3., .25, int(final.sum())), axis=0)
    recomputed = walk_forward_models(changed, **settings(prices))
    assert recomputed["metadata"]["selected_params"] == result["metadata"]["selected_params"]
    pd.testing.assert_frame_equal(recomputed["selection"], result["selection"])
    pd.testing.assert_frame_equal(recomputed["folds"], result["folds"])
    pd.testing.assert_frame_equal(
        recomputed["forecasts"].loc[lambda rows: rows.split == "development"].reset_index(drop=True),
        result["forecasts"].loc[lambda rows: rows.split == "development"].reset_index(drop=True))


def test_appending_future_data_does_not_rewrite_earlier_development_predictions(prices, result):
    prefix_end = prices.index[550]
    prefix = walk_forward_models(prices.loc[:prefix_end], **settings(prices))
    columns = ["model", "split", "cutoff", "ticker", "predicted_return", "uncertainty",
        "uncertainty_n", "interval_lower", "interval_upper", "selected_params"]
    full = result["forecasts"].loc[lambda rows: rows.cutoff.isin(prefix["forecasts"].cutoff.unique()), columns]
    pd.testing.assert_frame_equal(full.reset_index(drop=True), prefix["forecasts"][columns].reset_index(drop=True))


def test_fixed_seed_reproduces_tree_models_and_all_other_forecasts(prices, result):
    repeated = walk_forward_models(prices, **settings(prices))
    for name in ("forecasts", "selection", "folds"):
        pd.testing.assert_frame_equal(result[name], repeated[name])
    assert result["metadata"] == repeated["metadata"]


def test_purged_bridge_predictions_can_calibrate_only_after_their_outcomes_complete(prices, result):
    metadata = result["metadata"]
    bridge = pd.to_datetime(metadata["calibration_bridge_origins"])
    assert len(bridge) == 1
    assert not result["forecasts"].cutoff.isin(bridge).any()
    assert not result["folds"].cutoff.isin(bridge).any()
    history = historical_forecasts(prices, start=settings(prices)["validation_start"])
    test = result["forecasts"].loc[lambda rows: (rows.split == "test") & (rows.model == "Ridge")]
    first_origin = test.cutoff.min()
    for row in test.loc[test.cutoff == first_origin].itertuples():
        error, count, _ = _uncertainty(history, first_origin, row.ticker)
        assert row.uncertainty == pytest.approx(error)
        assert row.uncertainty_n == count


def test_calibration_uses_only_completed_model_residuals_and_exposes_small_support():
    cutoff = pd.Timestamp("2023-07-01")
    history = pd.DataFrame({
        "ticker": ["A", "A", "A", "A"],
        "cutoff": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-06-20"]),
        "label_end": pd.to_datetime(["2023-02-01", "2023-03-01", "2023-04-01", "2023-07-20"]),
        "actual_return": [1., -1., 3., 1000.], "predicted_return": [0., 0., 0., 0.],
    })
    calibration = _calibration(history, cutoff, "A", 100.)
    assert calibration["uncertainty"] == pytest.approx(np.sqrt(11 / 3))
    assert calibration["uncertainty_bias"] == 1
    assert calibration["uncertainty_n"] == calibration["interval_n"] == 3
    assert calibration["interval_lower"] == pytest.approx(99.4)
    assert calibration["interval_upper"] == pytest.approx(102.6)
    assert "pooled" in calibration["uncertainty_source"]
    empty = _calibration(history, pd.Timestamp("2023-01-01"), "A", 100.)
    assert empty["uncertainty_n"] == empty["interval_n"] == 0
    assert np.isnan(empty["uncertainty"]) and np.isnan(empty["interval_lower"])


def test_first_origin_has_no_future_filled_uncertainty_and_end_is_enforced(prices):
    end = prices.index[530]
    output = walk_forward_models(prices, end=end, models=["Ridge", "Momentum", "Zero return"], **settings(prices))
    forecasts = output["forecasts"]
    first = forecasts.loc[forecasts.cutoff == forecasts.cutoff.min()]
    assert first.uncertainty.isna().all() and first.interval_lower.isna().all()
    assert first.uncertainty_n.eq(0).all()
    assert (forecasts.label_end <= end).all()
    assert (output["folds"].label_end <= end).all()
    assert set(forecasts.split) == {"development"}


@pytest.mark.parametrize("kwargs", [
    {"rebalance_sessions": 20}, {"rebalance_sessions": True}, {"seed": -1},
    {"models": ["Unknown"]}, {"models": []}, {"models": ["Ridge", "Ridge"]},
    {"models": "Ridge"}, {"min_training_rows": 0},
    {"validation_start": "2024-01-01", "test_start": "2020-01-01"},
])
def test_invalid_experiment_settings_fail_before_fitting(prices, kwargs):
    with pytest.raises(ValueError):
        walk_forward_models(prices, **kwargs)
