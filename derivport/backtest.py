"""Research backtests on a fixed universe with chronological, completed labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import (
    DEFAULT_FORMULA, HORIZON, apply_rule, historical_forecasts,
    rank_correlation, validate_prices, validate_forecast_history,
)
from .formula import FormulaError


def backtest_rule(prices: pd.DataFrame, start: str | pd.Timestamp,
                  end: str | pd.Timestamp, formula: str = DEFAULT_FORMULA,
                  k: float = 1.0, history: pd.DataFrame | None = None) -> dict:
    """Compare three algorithms on identical nonoverlapping historical origins.

    Only origins at/after start whose entire 21-common-session labels finish by
    end are scored. Every prediction and uncertainty was computed with evidence
    known at that origin. A rule chosen now is evaluated retrospectively; repeated
    tuning is research, not untouched prospective evidence. Linked top-selection
    returns describe successive hypothetical observations, without execution,
    costs, taxes, currency conversion, or position sizing.
    """
    prices = validate_prices(prices)
    try:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError) as exc:
        raise ValueError("Choose valid start and end dates for the historical test.") from exc
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("The backtest start date must be on or before its end date.")
    effective_end = min(end, prices.index[-1])
    if history is None:
        history = historical_forecasts(prices)
    universe = set(prices.columns)
    validate_forecast_history(prices, history)
    if history.empty:
        raise ValueError("There is not enough completed historical data to build and evaluate an out-of-sample forecast.")
    known = history.loc[(history["cutoff"] >= start) & (history["label_end"] <= effective_end)].copy()
    known = known.dropna(subset=["actual_return", "uncertainty", "predicted_return", "momentum", "volatility"])
    known = known.groupby("cutoff", group_keys=False).filter(
        lambda group: len(group) == len(universe) and set(group["ticker"]) == universe
    )
    if known.empty:
        raise ValueError("No complete 21-session test periods fit this range with sufficient training and completed forecast errors. Widen the range or choose stocks with longer shared history.")
    rows = []
    for cutoff, group in known.groupby("cutoff", sort=True):
        stocks = group.set_index("ticker").sort_index()
        try:
            stocks = apply_rule(stocks, formula, k)
        except FormulaError as exc:
            raise FormulaError(f"The rule is invalid at historical cutoff {cutoff:%Y-%m-%d}: {exc}") from exc
        ranks = {
            "model": stocks["predicted_return"].rank(ascending=False, method="average"),
            "momentum": stocks["momentum"].rank(ascending=False, method="average"),
            "student": stocks["student_rank"],
        }
        actual = stocks["actual_return"]
        realized_ranks = actual.rank(ascending=False, method="average")
        best_actual = realized_ranks.min()
        actual_winners = realized_ranks.index[realized_ranks == best_actual]
        row = {"cutoff": cutoff, "label_end": stocks["label_end"].max(),
               "model_rmse": float(np.sqrt(np.mean((stocks["predicted_return"] - actual) ** 2))),
               "baseline_rmse": float(np.sqrt(np.mean((stocks["momentum"] - actual) ** 2)))}
        for method, ranking in ranks.items():
            top = sorted(ranking.index[ranking == ranking.min()])[0]
            row[f"{method}_spearman"] = rank_correlation(ranking, realized_ranks)
            row[f"{method}_top"] = top
            row[f"{method}_return"] = float(actual.loc[top])
            row[f"{method}_first_credit"] = 1 / len(actual_winners) if top in actual_winners else 0.0
        rows.append(row)
    per_date = pd.DataFrame(rows)
    if len(per_date) > 1 and (per_date["cutoff"].iloc[1:].to_numpy() < per_date["label_end"].iloc[:-1].to_numpy()).any():
        raise ValueError("Backtest forecast origins overlap. Recompute the standard 21-session historical forecast grid.")
    summary_rows = []
    linked = pd.DataFrame({method: (1 + per_date[f"{method}_return"].to_numpy() / 100).cumprod() * 100 - 100
                           for method in ("model", "momentum", "student")},
                          index=pd.DatetimeIndex(per_date["label_end"], name="date"))
    # A zero point locates the start of the first complete observed period.
    initial = pd.DataFrame(0.0, index=pd.DatetimeIndex([per_date.iloc[0]["cutoff"]], name="date"), columns=linked.columns)
    linked = pd.concat([initial, linked])
    for method in ("model", "momentum", "student"):
        correlations = per_date[f"{method}_spearman"].dropna()
        summary_rows.append({"method": method, "n_dates": len(per_date),
                             "mean_spearman": float(correlations.mean()) if not correlations.empty else None,
                             "mean_return": float(per_date[f"{method}_return"].mean()),
                             "linked_return": float(linked[method].iloc[-1]),
                             "first_fraction": float(per_date[f"{method}_first_credit"].mean())})
    summary = pd.DataFrame(summary_rows).set_index("method")
    metrics = {
        "n_dates": len(per_date), "n_forecasts": len(known),
        "model_rmse": float(np.sqrt(np.mean((known["predicted_return"] - known["actual_return"]) ** 2))),
        "baseline_rmse": float(np.sqrt(np.mean((known["momentum"] - known["actual_return"]) ** 2))),
    }
    metadata = {
        "requested_start": start, "requested_end": end, "effective_end": effective_end,
        "first_cutoff": per_date["cutoff"].min(), "last_label_end": per_date["label_end"].max(),
        "universe": sorted(universe), "formula": formula, "k": float(k), "horizon_sessions": HORIZON,
        "assumptions": "Fixed universe; common observed sessions; nonoverlapping 21-session forecasts; point-in-time training and uncertainty. Average tied ranks; alphabetical representative for tied top ranks. Local-currency adjusted returns; no currency conversion, execution, costs, taxes, or position sizing. Rule selected retrospectively; repeated tuning can overfit.",
    }
    return {"per_date": per_date, "summary": summary, "linked_returns": linked,
            "metrics": metrics, "metadata": metadata}
