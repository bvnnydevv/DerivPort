"""Shared, reproducible hypothetical paths. These are never historical evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

SCENARIOS = {
    "rally": {
        "label": "Rally",
        "description": "A broad climb with mild continuation of the initial momentum signal.",
        "assumptions": "21 sessions; market log drift +0.14% per session; stock momentum tilt capped at ±0.10% per session; correlation 0.65; trailing daily volatility × 1.0.",
        "correlation": 0.65,
        "volatility_multiplier": 1.0,
    },
    "reversal": {
        "label": "Sudden reversal",
        "description": "An early rise gives way to a decline, with recent leaders more exposed to the reversal.",
        "assumptions": "First 5 sessions: market log drift +0.15% per session. Next 16: −0.35% per session; momentum tilt reverses and is capped at ±0.10% per session; correlation 0.80; trailing daily volatility × 1.3.",
        "correlation": 0.80,
        "volatility_multiplier": 1.3,
    },
    "sideways": {
        "label": "Sideways / volatile",
        "description": "No expected log-price drift, with larger random swings.",
        "assumptions": "21 sessions; zero log drift; no momentum tilt; correlation 0.45; trailing daily volatility × 1.6. Zero log drift does not imply zero expected simple return.",
        "correlation": 0.45,
        "volatility_multiplier": 1.6,
    },
}


@dataclass
class Simulation:
    scenario: str
    tickers: tuple[str, ...]
    paths: np.ndarray  # n_paths x 22 sessions (including initial index 100) x n_stocks
    terminal_returns: pd.DataFrame  # all stocks share the same numbered paths
    assumptions: str
    seed: int


def simulate_paths(stocks: pd.DataFrame, scenario: str = "rally", n_paths: int = 600,
                   seed: int = 2024) -> Simulation:
    """Generate common-factor-correlated lognormal paths from cutoff-known inputs.

    The random generator never consumes the student's formula, scores, or ranks.
    Ranking changes only choose a different column of these same frozen paths.
    """
    if scenario not in SCENARIOS:
        raise ValueError("Choose rally, reversal, or sideways.")
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)):
        raise ValueError("Use an integer number of simulated paths.")
    if n_paths < 200:
        raise ValueError("Use at least 200 paths for each scenario.")
    if n_paths > 20000:
        raise ValueError("Use at most 20,000 paths to keep the local demo responsive.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The simulation seed must be a nonnegative integer.")
    if not isinstance(stocks, pd.DataFrame) or len(stocks) < 2:
        raise ValueError("Simulation requires at least two stocks.")
    if not stocks.index.is_unique or not all(isinstance(name, str) and name.strip() for name in stocks.index):
        raise ValueError("Simulation requires unique nonempty stock symbols.")
    if not {"momentum", "volatility"}.issubset(stocks.columns):
        raise ValueError("Simulation requires momentum and volatility for each stock.")
    tickers = tuple(sorted(stocks.index))
    try:
        inputs = stocks.loc[list(tickers), ["momentum", "volatility"]].astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Simulation requires numeric finite momentum and nonnegative volatility.") from exc
    if not np.isfinite(inputs.to_numpy()).all() or (inputs["volatility"] < 0).any():
        raise ValueError("Simulation requires finite momentum and nonnegative finite volatility.")
    spec = SCENARIOS[scenario]
    rng = np.random.default_rng(int(seed))
    correlation = spec["correlation"]
    common = rng.standard_normal((n_paths, 21, 1))
    individual = rng.standard_normal((n_paths, 21, len(tickers)))
    noise = np.sqrt(correlation) * common + np.sqrt(1 - correlation) * individual
    sigma = inputs["volatility"].to_numpy(dtype=float) / 100 / np.sqrt(21)
    sigma *= spec["volatility_multiplier"]
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("Simulation requires finite nonnegative trailing volatility.")
    tilt = np.clip(inputs["momentum"].to_numpy(dtype=float) / 100 / 21, -0.001, 0.001)
    if scenario == "rally":
        drift = np.broadcast_to(0.0014 + tilt, (21, len(tickers))).copy()
    elif scenario == "reversal":
        drift = np.empty((21, len(tickers)))
        drift[:5] = 0.0015 + tilt
        drift[5:] = -0.0035 - tilt
    else:
        drift = np.zeros((21, len(tickers)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        increments = drift[None, :, :] + sigma[None, None, :] * noise
        growth = np.exp(np.cumsum(increments, axis=1))
        paths = np.concatenate([np.full((n_paths, 1, len(tickers)), 100.0), growth * 100], axis=1)
        terminal_values = (growth[:, -1, :] - 1) * 100
    if not np.isfinite(paths).all() or (paths <= 0).any() or not np.isfinite(terminal_values).all():
        raise ValueError("These inputs produce extreme or non-finite simulated paths; reduce volatility or momentum.")
    terminal = pd.DataFrame(terminal_values, columns=tickers)
    terminal.index.name = "path"
    return Simulation(scenario, tickers, paths, terminal, spec["assumptions"], seed)


def simulation_summary(simulation: Simulation, rankings: Mapping[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each method's frozen top stock on exactly the same shared paths.

    Ranks are numeric, lower is better. Tied first-ranked stocks use the same
    alphabetical representative for every method. Terminal first-place ties split
    one unit of credit equally across all tied winners on a path.
    """
    terminal = simulation.terminal_returns
    maxima = terminal.max(axis=1).to_numpy()
    tied_winners = np.isclose(terminal.to_numpy(), maxima[:, None], rtol=0, atol=1e-10)
    win_credit = tied_winners / tied_winners.sum(axis=1, keepdims=True)
    distributions, summaries = [], []
    for method, ranking in rankings.items():
        aligned = pd.Series(ranking, dtype=float).reindex(simulation.tickers)
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all():
            raise ValueError("Each simulation ranking must cover every stock in this universe with finite ranks.")
        best = aligned.min()
        ticker = sorted(aligned.index[np.isclose(aligned.to_numpy(), best, rtol=0, atol=1e-10)])[0]
        samples = terminal[ticker]
        distributions.append(pd.DataFrame({"method": method, "ticker": ticker,
                                           "path": terminal.index, "return": samples.to_numpy()}))
        summaries.append({"method": method, "ticker": ticker, "mean_return": float(samples.mean()),
                          "p10": float(samples.quantile(.10)), "median_return": float(samples.median()),
                          "p90": float(samples.quantile(.90)),
                          "first_fraction": float(win_credit[:, list(terminal.columns).index(ticker)].mean())})
    return pd.concat(distributions, ignore_index=True), pd.DataFrame(summaries)
