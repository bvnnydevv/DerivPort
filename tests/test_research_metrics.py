"""Hand-calculated metrics and dependence-aware, date-level inference contracts."""

import numpy as np
import pandas as pd
import pytest

from derivport.research_metrics import (
    bootstrap_mean, holm_adjust, paired_comparisons, per_date_metrics,
    prediction_metrics, rank_metrics,
)


def one_date():
    return pd.DataFrame({
        "model": "Ridge", "cutoff": pd.Timestamp("2023-01-03"),
        "label_end": pd.Timestamp("2023-02-01"), "ticker": list("ABCD"),
        "predicted_return": [-1., 0., 3., 2.], "actual_return": [-2., 0., 2., 4.],
        "uncertainty": 2., "interval_lower": [-2., -1., 2., 1.],
        "interval_upper": [0., 1., 4., 3.],
    })


def many_dates(n=20, stocks=2):
    records = []
    for index, date in enumerate(pd.date_range("2022-01-31", periods=n, freq="ME")):
        error = 0.5 + np.sin(index * .25) + index / 10
        for stock in range(stocks):
            actual = -2.0 if stock % 2 == 0 else 2.0
            records.append({"model": "Ridge", "cutoff": date, "label_end": date + pd.Timedelta(days=29),
                "ticker": f"S{stock:03}", "predicted_return": actual + error, "actual_return": actual})
    return pd.DataFrame(records)


def test_prediction_metrics_match_hand_calculations_and_interval_denominators():
    forecasts = one_date()
    result = prediction_metrics(forecasts).loc["Ridge"]
    assert result.rmse == pytest.approx(np.sqrt(1.5))
    assert result.mae == 1
    assert result.r2 == pytest.approx(.7)
    assert result.pearson == pytest.approx(12 / np.sqrt(200))
    assert result.spearman == pytest.approx(.8)
    assert result.mean_spearman_ic == pytest.approx(.8)
    assert result.directional_accuracy == 1
    assert result.coverage80 == pytest.approx(.75)
    assert result.n_intervals == 4
    assert result.standardized_sq_error_mean == pytest.approx(.375)
    assert result.n_uncertainty == 4
    assert np.isnan(result.rmse_ci_low)
    assert "more_data_required" in result.inference_status
    dated = per_date_metrics(forecasts).iloc[0]
    assert dated.mse == 1.5
    assert dated.mae == 1
    assert dated.spearman_ic == pytest.approx(.8)
    assert dated.n_forecasts == 4


def test_zero_forecast_is_not_credited_as_a_directional_up_or_down_call():
    forecasts = one_date().assign(predicted_return=0.)
    result = prediction_metrics(forecasts).iloc[0]
    assert result.directional_accuracy == .25
    assert np.isnan(result.pearson)
    assert np.isnan(result.spearman)
    assert np.isnan(result.mean_spearman_ic)
    assert result.r2 == pytest.approx(-.2)


def test_constant_actual_returns_have_undefined_r_squared_and_correlations():
    forecasts = one_date().assign(actual_return=1.)
    result = prediction_metrics(forecasts).iloc[0]
    assert np.isnan(result.r2)
    assert np.isnan(result.pearson)
    assert np.isnan(result.mean_pearson_ic)
    assert np.isnan(result.spearman)


def test_rank_scores_are_not_treated_as_numerical_return_predictions():
    scores = one_date().rename(columns={"model": "strategy", "predicted_return": "score"})
    result = rank_metrics(scores)
    assert result.loc["Ridge", "mean_spearman_ic"] == pytest.approx(.8)
    assert result.loc["Ridge", "mean_pearson_ic"] == pytest.approx(12 / np.sqrt(200))
    assert not {"rmse", "mae", "r2", "directional_accuracy"}.intersection(result.columns)
    changed_scale = rank_metrics(scores.assign(score=scores.score * 1000 + 500))
    pd.testing.assert_frame_equal(result, changed_scale)


def test_available_prediction_intervals_and_uncertainty_use_explicit_support():
    forecasts = one_date()
    forecasts.loc[0, ["interval_lower", "interval_upper"]] = np.nan
    forecasts.loc[0, "uncertainty"] = 0
    forecasts.loc[1, "uncertainty"] = np.nan
    result = prediction_metrics(forecasts).iloc[0]
    assert result.coverage80 == pytest.approx(2 / 3)
    assert result.n_intervals == 3
    assert result.n_uncertainty == 2
    assert result.standardized_sq_error_mean == pytest.approx((.25 + 1) / 2)


def test_rmse_intervals_resample_dates_so_copying_cross_sections_does_not_narrow_them():
    # Each extra stock repeats the SAME date shock. It adds no independent date evidence.
    small = prediction_metrics(many_dates(stocks=2))
    expanded = prediction_metrics(many_dates(stocks=30))
    for column in ["rmse", "rmse_ci_low", "rmse_ci_high"]:
        assert small.loc["Ridge", column] == pytest.approx(expanded.loc["Ridge", column])
    assert small.loc["Ridge", "n_dates"] == expanded.loc["Ridge", "n_dates"] == 20
    assert expanded.loc["Ridge", "n_forecasts"] == 600


def test_prediction_intervals_forward_custom_bootstrap_configuration():
    forecasts = many_dates()
    arguments = dict(seed=71, samples=501, block_length=2)
    first = prediction_metrics(forecasts, **arguments)
    second = prediction_metrics(forecasts, **arguments)
    pd.testing.assert_frame_equal(first, second)
    dated = per_date_metrics(forecasts).set_index("cutoff")
    expected = bootstrap_mean(dated["mse"], **arguments)
    assert first.loc["Ridge", "rmse_ci_low"] == pytest.approx(np.sqrt(expected["ci_low"]))
    assert first.loc["Ridge", "rmse_ci_high"] == pytest.approx(np.sqrt(expected["ci_high"]))
    assert first.attrs["inference"]["seed"] == 71
    assert first.attrs["inference"]["samples"] == 501
    assert first.attrs["inference"]["block_length"] == 2


def test_bootstrap_is_seeded_and_zero_sequence_has_exact_zero_interval_and_p_one():
    values = pd.Series(np.sin(np.arange(36) / 3) + .15,
                       index=pd.date_range("2021-01-31", periods=36, freq="ME"))
    first = bootstrap_mean(values, seed=37, samples=500)
    second = bootstrap_mean(values, seed=37, samples=500)
    assert first == second
    third = bootstrap_mean(values, seed=38, samples=500)
    assert (first["ci_low"], first["ci_high"]) != (third["ci_low"], third["ci_high"])
    zero = bootstrap_mean(np.zeros(12))
    assert zero["mean"] == zero["ci_low"] == zero["ci_high"] == 0
    assert zero["p_value"] == 1
    assert zero["n_dates"] == 12


def test_bootstrap_refuses_independent_stock_matrix_and_reports_insufficient_dates():
    with pytest.raises(ValueError, match="aggregate per date"):
        bootstrap_mean(np.ones((12, 5)))
    with pytest.raises(ValueError, match="unique chronological"):
        bootstrap_mean(pd.Series([1., 2.], index=pd.to_datetime(["2021-01-01", "2021-01-01"])))
    insufficient = bootstrap_mean([*range(11), np.nan, np.inf])
    assert insufficient["n_dates"] == 11
    assert insufficient["n_excluded"] == 2
    assert insufficient["mean"] == 5
    assert np.isnan(insufficient["p_value"])
    assert np.isnan(insufficient["ci_low"])
    assert "more_data_required" in insufficient["status"]


def test_holm_adjustment_matches_known_step_down_values_and_keeps_family_size():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    np.testing.assert_allclose(holm_adjust([.01, np.nan, .04]), [.03, np.nan, .08], equal_nan=True)
    np.testing.assert_array_equal(holm_adjust([0, 1]), [0, 1])


def test_paired_comparison_aligns_exact_dates_and_ignores_unpaired_extremes():
    dates = pd.date_range("2022-01-31", periods=15, freq="ME")
    baseline = pd.DataFrame({"model": "Ridge", "cutoff": dates,
        "label_end": dates + pd.Timedelta(days=29), "mse": np.arange(15.)})
    candidate = baseline.iloc[3:].assign(model="Other", mse=lambda x: x.mse + 2)
    extra = pd.DataFrame({"model": ["Other"], "cutoff": [pd.Timestamp("2037-01-01")],
        "label_end": [pd.Timestamp("2037-02-01")], "mse": [999999.]})
    panel = pd.concat([candidate.iloc[::-1], extra, baseline], ignore_index=True)
    result = paired_comparisons(panel, samples=500).iloc[0]
    assert result.n_dates == 12
    assert result.mean_difference == 2
    assert result.ci_low == result.ci_high == 2
    assert result.p_value == pytest.approx(1 / 501)
    assert result.p_adjusted == result.p_value
    # Same origin with a different horizon is NOT a matched observation.
    panel.loc[(panel.model == "Other") & (panel.cutoff == dates[3]), "label_end"] += pd.Timedelta(days=1)
    mismatch = paired_comparisons(panel, samples=500).iloc[0]
    assert mismatch.n_dates == 11
    assert np.isnan(mismatch.p_value)


def test_pairing_rejects_stock_rows_and_supports_strategy_net_return_comparisons():
    with pytest.raises(ValueError, match="one nonmissing dated aggregate"):
        paired_comparisons(one_date().rename(columns={"predicted_return": "mse"}))
    dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    baseline = pd.DataFrame({"strategy": "Balanced original", "cutoff": dates, "net_return": .01})
    candidate = baseline.assign(strategy="Experiment", net_return=.02)
    result = paired_comparisons(pd.concat([baseline, candidate]), method_col="strategy",
        value_col="net_return", baseline="Balanced original").iloc[0]
    assert result.mean_difference == pytest.approx(.01)
    assert result.n_dates == 12


def test_invalid_evaluation_support_is_never_silently_repaired():
    forecasts = one_date()
    for changed in [forecasts.iloc[:1], pd.concat([forecasts, forecasts.iloc[:1]]),
                    forecasts.assign(actual_return=np.nan), forecasts.assign(predicted_return=np.inf),
                    forecasts.assign(label_end=forecasts.cutoff),
                    forecasts.assign(interval_lower=100., interval_upper=0.)]:
        with pytest.raises(ValueError):
            prediction_metrics(changed)
    different_target = forecasts.assign(model="Other", actual_return=999.)
    with pytest.raises(ValueError, match="same observed"):
        prediction_metrics(pd.concat([forecasts, different_target]))
