"""All algorithms face identical, correlated, reproducible hypothetical paths."""

import numpy as np
import pandas as pd
import pytest

from derivport import SCENARIOS, simulate_paths, simulation_summary


@pytest.fixture
def stocks():
    return pd.DataFrame({
        "momentum": [5., 3., -1., 2., 8.],
        "volatility": [5., 6., 7., 8., 9.],
        "predicted_return": [2., 3., 1., 4., 5.],
        "uncertainty": [4., 5., 6., 7., 8.],
    }, index=["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"])


@pytest.mark.parametrize("scenario", ["rally", "reversal", "sideways"])
def test_each_scenario_has_at_least_200_reproducible_paths(stocks, scenario):
    first = simulate_paths(stocks, scenario, n_paths=600, seed=2024)
    second = simulate_paths(stocks, scenario, n_paths=600, seed=2024)
    assert first.paths.shape == (600, 22, 5)
    assert first.terminal_returns.shape == (600, 5)
    assert np.isfinite(first.paths).all()
    assert (first.paths > 0).all()
    np.testing.assert_array_equal(first.paths, second.paths)
    np.testing.assert_array_equal(first.paths[:, 0, :], np.full((600, 5), 100.))
    assert first.assumptions
    # Estimate innovation correlation across both paths and sessions.
    increments = np.diff(np.log(first.paths), axis=1).reshape(-1, 5)
    measured = np.corrcoef(increments, rowvar=False)[np.triu_indices(5, k=1)].mean()
    assert measured == pytest.approx(SCENARIOS[scenario]["correlation"], abs=0.05)


def test_identical_rankings_on_identical_paths_have_identical_results(stocks):
    simulation = simulate_paths(stocks, "reversal", n_paths=600)
    ranking = pd.Series([1., 2., 3., 4., 5.], index=stocks.index)
    distributions, summary = simulation_summary(simulation, {
        "Momentum": ranking, "AI model": ranking.copy(), "Student rule": ranking.copy()
    })
    reference = summary.iloc[0].drop("method")
    for index in [1, 2]:
        pd.testing.assert_series_equal(reference, summary.iloc[index].drop("method"), check_names=False)
    returns = distributions.pivot(index="path", columns="method", values="return")
    np.testing.assert_array_equal(returns["Momentum"], returns["AI model"])
    np.testing.assert_array_equal(returns["Momentum"], returns["Student rule"])


def test_rule_edits_cannot_change_paths(stocks):
    original = simulate_paths(stocks, "rally")
    edited = stocks.copy()
    edited["score"] = [-100., 20., 200., 10., 0.]
    edited["student_rank"] = edited["score"].rank(ascending=False)
    edited["predicted_return"] = 999.0
    edited["uncertainty"] = 999.0
    rerun = simulate_paths(edited, "rally")
    np.testing.assert_array_equal(original.paths, rerun.paths)


def test_tied_top_stocks_use_same_representative_and_split_realized_ties(stocks):
    flat = stocks.assign(momentum=0., volatility=0.)
    simulation = simulate_paths(flat, "sideways")
    all_tied = pd.Series(3., index=flat.index)
    _, summary = simulation_summary(simulation, {"A": all_tied, "B": all_tied.copy()})
    assert summary["ticker"].tolist() == ["AAPL", "AAPL"]
    np.testing.assert_allclose(summary["first_fraction"], 0.2)


def test_too_few_paths_are_rejected(stocks):
    with pytest.raises(ValueError, match="200"):
        simulate_paths(stocks, n_paths=199)


def test_simulation_supports_a_custom_two_stock_comparison(stocks):
    smaller = stocks.loc[["AAPL", "MSFT"]]
    simulation = simulate_paths(smaller, "sideways")
    assert simulation.paths.shape == (600, 22, 2)
    ranking = smaller["momentum"].rank(ascending=False, method="average")
    distributions, summary = simulation_summary(simulation, {"Momentum": ranking, "Same rule": ranking.copy()})
    assert len(distributions) == 1200
    assert summary["ticker"].eq("AAPL").all()
    np.testing.assert_array_equal(
        distributions.loc[distributions["method"] == "Momentum", "return"].to_numpy(),
        distributions.loc[distributions["method"] == "Same rule", "return"].to_numpy(),
    )


@pytest.mark.parametrize("n_paths", [600.5, True, "600"])
def test_simulation_rejects_noninteger_path_counts(stocks, n_paths):
    with pytest.raises(ValueError, match="integer"):
        simulate_paths(stocks, n_paths=n_paths)


@pytest.mark.parametrize("seed", [-1, True, 2.5])
def test_simulation_rejects_invalid_seeds(stocks, seed):
    with pytest.raises(ValueError, match="seed"):
        simulate_paths(stocks, seed=seed)


def test_simulation_rejects_incomplete_nonfinite_or_duplicate_stock_inputs(stocks):
    invalid = [stocks.iloc[:1], stocks.assign(momentum=np.nan),
               stocks.assign(volatility=np.inf), stocks.assign(volatility=-1),
               stocks.assign(volatility=1e308), stocks.drop(columns="momentum"),
               pd.concat([stocks, stocks.iloc[:1]])]
    for frame in invalid:
        with pytest.raises(ValueError):
            simulate_paths(frame)
