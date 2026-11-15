"""Delayed, self-financing, long-only portfolio experiments on observed prices.

This module consumes dated scores, not model objects. Scores observed after a
cutoff close execute at a later common-session adjusted close. Adjusted prices
make this a hypothetical total-return experiment, not an order-fill simulator.
All returns are decimal returns; equity and costs use initial wealth of one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _date(value, name):
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError("missing date")
        return result.tz_localize(None).normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid date.") from exc


def _prices(prices):
    if not isinstance(prices, pd.DataFrame) or prices.empty or len(prices.columns) < 2:
        raise ValueError("Portfolio research needs at least two stocks with shared observed prices.")
    if not prices.columns.is_unique or not all(isinstance(s, str) and s.strip() for s in prices.columns):
        raise ValueError("Prices must have unique nonempty stock symbols.")
    try:
        clean = prices.astype(float).copy()
        clean.index = pd.DatetimeIndex(clean.index).tz_localize(None).normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Prices need numeric values and valid daily session dates.") from exc
    if clean.index.isna().any() or not clean.index.is_unique:
        raise ValueError("Prices need unique valid daily session dates.")
    if not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any().any():
        raise ValueError("Prices must be positive, finite, and complete; missing prices are never filled.")
    return clean.sort_index().reindex(columns=sorted(clean.columns))


def _forecasts(forecasts, score_column, prices):
    if not isinstance(score_column, str) or not score_column.strip():
        raise ValueError("Choose a nonempty score column name.")
    required = {"cutoff", "ticker", score_column}
    if not isinstance(forecasts, pd.DataFrame) or forecasts.empty or not required.issubset(forecasts.columns):
        raise ValueError("Scored forecasts need cutoff, ticker, and the selected score column.")
    rows = forecasts.copy()
    rows["cutoff"] = rows["cutoff"].map(lambda value: _date(value, "Forecast cutoff"))
    if rows.duplicated(["cutoff", "ticker"]).any():
        raise ValueError("Each cutoff must contain each stock exactly once; duplicate scored forecasts were found.")
    if not rows["cutoff"].isin(prices.index).all():
        raise ValueError("Every forecast cutoff must be a shared observed price session.")
    try:
        rows[score_column] = rows[score_column].astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Every selected score must be numeric and finite.") from exc
    if not np.isfinite(rows[score_column].to_numpy()).all():
        raise ValueError("Every selected score must be numeric and finite.")
    universe = set(prices.columns)
    for _, group in rows.groupby("cutoff"):
        if len(group) != len(universe) or set(group["ticker"]) != universe:
            raise ValueError("Every scored cutoff must cover the same complete fixed stock universe as prices.")
    if "training_last_label_end" in rows:
        training_end = rows["training_last_label_end"].map(lambda value: _date(value, "Last training label date"))
        if (training_end > rows["cutoff"]).any():
            raise ValueError("Training labels must finish by the forecast cutoff; future training evidence was found.")
    return rows.sort_values(["cutoff", "ticker"], kind="stable")


def _rebalance(asset_values, cash, target_weights, rate):
    """Solve post-fee wealth against trades sized from that same wealth."""
    before = float(asset_values.sum() + cash)
    if rate == 0:
        after = before
    else:
        low, high = 0.0, before
        for _ in range(80):
            candidate = (low + high) / 2
            residual = candidate + rate * np.abs(target_weights * candidate - asset_values).sum() - before
            if residual > 0:
                high = candidate
            else:
                low = candidate
        after = (low + high) / 2
    desired = target_weights * after
    delta = desired - asset_values
    notional = float(np.abs(delta).sum())
    fee = rate * notional
    # Holding all assets in cash and entering a full position are explicit trades.
    half_l1 = 0.5 * float(np.abs(target_weights - asset_values / before).sum())
    remaining_cash = max(0.0, (1 - float(target_weights.sum())) * after)
    return desired, remaining_cash, {
        "before": before, "after": after, "delta": delta,
        "notional": notional, "cost": fee, "turnover": half_l1,
        "notional_turnover": notional / before,
    }


def backtest_portfolio(
    prices: pd.DataFrame,
    scored_forecasts: pd.DataFrame,
    score_column: str = "score",
    start=None,
    end=None,
    top_n: int = 3,
    max_weight: float = 0.4,
    gross_exposure: float = 1.0,
    execution_lag: int = 1,
    rebalance_sessions: int = 21,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    annual_rf: float = 0.0,
    liquidate: bool = True,
) -> dict:
    """Run one score strategy with delayed fills, cash, drift, and trading costs.

    At most ``top_n`` stocks receive equal target weights, each bounded by
    ``max_weight``; unused exposure remains non-interest-bearing cash. Exact score
    ties are broken alphabetically. Caps constrain targets AT REBALANCE, not
    weights that subsequently drift with observed prices. Forecasts may be more
    frequent than the selected rebalance interval; those inside the interval are
    skipped. Forecast groups themselves must always be complete and on-grid.

    ``daily.turnover`` is half-L1 risky-weight turnover (cash excluded).
    ``daily.notional_turnover`` is full buy-plus-sell notional / pre-trade NAV;
    fees use the latter's dollar notional, never half-L1 turnover. Returns include
    the initial fee and, when requested, final liquidation. A fee-free shadow
    portfolio uses identical targets. Sharpe and annualized return use 252 shared
    observed sessions/year, with the supplied annual risk-free rate used only as
    a Sharpe benchmark. They are descriptive, not significance tests.
    """
    clean = _prices(prices)
    rows = _forecasts(scored_forecasts, score_column, clean)
    top_n = _integer(top_n, "Top-stock count", 1)
    lag = _integer(execution_lag, "Execution lag", 1)
    rebalance = _integer(rebalance_sessions, "Rebalance interval", 21)
    exposure = _number(gross_exposure, "Gross exposure", minimum=0, maximum=1)
    cap = _number(max_weight, "Position cap", minimum=0, maximum=exposure)
    commission = _number(commission_bps, "Commission basis points", minimum=0) / 10000
    slippage = _number(slippage_bps, "Slippage basis points", minimum=0) / 10000
    rate = commission + slippage
    if rate >= 1:
        raise ValueError("Combined commission and slippage must be less than 10,000 basis points.")
    rf = _number(annual_rf, "Annual risk-free rate", minimum=-0.999999)
    if not isinstance(liquidate, (bool, np.bool_)):
        raise ValueError("Liquidate must be true or false.")

    first = _date(start, "Portfolio start") if start is not None else rows["cutoff"].min()
    last = _date(end, "Portfolio end") if end is not None else clean.index[-1]
    if first > last:
        raise ValueError("Portfolio start must be on or before its end.")
    dates = clean.index[(clean.index >= first) & (clean.index <= last)]
    if len(dates) < 3:
        raise ValueError("At least three shared observed sessions are needed for a delayed portfolio experiment.")
    first_position, final_position = clean.index.get_loc(dates[0]), clean.index.get_loc(dates[-1])
    selected_count = min(top_n, len(clean.columns))
    schedules = {}
    prior_origin = None
    skipped = 0
    for cutoff, group in rows.groupby("cutoff", sort=True):
        position = clean.index.get_loc(cutoff)
        execution = position + lag
        # Do not open a new position at the terminal mark with no observed holding return.
        if position < first_position or execution >= final_position:
            continue
        if prior_origin is not None and position - prior_origin < rebalance:
            skipped += 1
            continue
        ranked = group.sort_values([score_column, "ticker"], ascending=[False, True], kind="stable")
        chosen = ranked["ticker"].iloc[:selected_count]
        target = np.zeros(len(clean.columns), dtype=float)
        target[clean.columns.get_indexer(chosen)] = min(cap, exposure / selected_count)
        schedules[execution] = (cutoff, target)
        prior_origin = position
    if not schedules:
        raise ValueError("No complete scored cutoff can execute with a subsequent observed return inside this range.")

    assets = np.zeros(len(clean.columns), dtype=float)
    gross_assets = assets.copy()
    cash = gross_cash = 1.0
    previous_net = previous_gross = peak = 1.0
    records, weight_rows, trades = [], [], []
    active_cutoff = pd.NaT
    last_execution = pd.NaT

    def execute(date, cutoff, target, event):
        nonlocal assets, cash, gross_assets, gross_cash
        pre_values = assets.copy()
        assets, cash, info = _rebalance(assets, cash, target, rate)
        gross_assets, gross_cash, _ = _rebalance(gross_assets, gross_cash, target, 0.0)
        for index, ticker in enumerate(clean.columns):
            delta = float(info["delta"][index])
            if abs(delta) <= max(1e-14, info["before"] * 1e-13):
                continue
            notional = abs(delta)
            trades.append({
                "date": date, "cutoff": cutoff, "ticker": ticker, "event": event,
                "side": "buy" if delta > 0 else "sell", "notional": notional,
                "trade_fraction": notional / info["before"], "target_weight": float(target[index]),
                "pretrade_value": float(pre_values[index]), "posttrade_value": float(assets[index]),
                "commission": commission * notional, "slippage": slippage * notional,
                "cost": rate * notional, "pretrade_equity": info["before"],
                "posttrade_equity": info["after"],
            })
        return info

    for number, date in enumerate(dates):
        position = clean.index.get_loc(date)
        if number:
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                change = (clean.loc[date] / clean.loc[dates[number - 1]]).to_numpy()
                assets *= change
                gross_assets *= change
            if not np.isfinite(change).all() or not np.isfinite(assets).all() or not np.isfinite(gross_assets).all():
                raise ValueError("Observed price changes produce non-finite portfolio values.")
        before = float(assets.sum() + cash)
        if not np.isfinite(before) or before <= 0:
            raise ValueError("Observed price changes exhausted the portfolio's finite positive equity.")
        pretrade_max_weight = float((assets / before).max())
        fee = turnover = notional_turnover = traded_notional = 0.0
        if position in schedules:
            active_cutoff, target = schedules[position]
            last_execution = date
            info = execute(date, active_cutoff, target, "rebalance")
            fee += info["cost"]
            turnover += info["turnover"]
            notional_turnover += info["notional_turnover"]
            traded_notional += info["notional"]
        if date == dates[-1] and liquidate:
            info = execute(date, active_cutoff, np.zeros(len(clean.columns)), "liquidation")
            fee += info["cost"]
            turnover += info["turnover"]
            notional_turnover += info["notional_turnover"]
            traded_notional += info["notional"]
        net = float(assets.sum() + cash)
        gross = float(gross_assets.sum() + gross_cash)
        peak = max(peak, net)
        records.append({
            "date": date, "net_equity": net, "gross_equity": gross,
            "net_return": net / previous_net - 1 if number else 0.0,
            "gross_return": gross / previous_gross - 1 if number else 0.0,
            "drawdown": net / peak - 1, "turnover": turnover,
            "notional_turnover": notional_turnover, "traded_notional": traded_notional,
            "cost": fee, "cash_weight": cash / net,
            "max_weight": float((assets / net).max()),
            "pretrade_max_weight": pretrade_max_weight,
            "signal_cutoff": active_cutoff, "execution_date": last_execution,
        })
        weight_rows.append(assets / net)
        previous_net, previous_gross = net, gross

    daily = pd.DataFrame(records).set_index("date")
    weights = pd.DataFrame(weight_rows, index=daily.index, columns=clean.columns)
    weights.columns.name = "ticker"
    trade_columns = ["date", "cutoff", "ticker", "event", "side", "notional", "trade_fraction",
                     "target_weight", "pretrade_value", "posttrade_value", "commission", "slippage",
                     "cost", "pretrade_equity", "posttrade_equity"]
    trade_frame = pd.DataFrame(trades, columns=trade_columns)
    returns = daily["net_return"].iloc[1:]
    excess = returns - np.expm1(np.log1p(rf) / 252)
    deviation = float(excess.std(ddof=1))
    sharpe = float(excess.mean() / deviation * np.sqrt(252)) if np.isfinite(deviation) and deviation > 1e-12 else None
    with np.errstate(over="ignore", invalid="ignore"):
        annualized = float(np.expm1(np.log(daily["net_equity"].iloc[-1]) * 252 / len(returns)))
    metrics = {
        "total_return": float(daily["net_equity"].iloc[-1] - 1),
        "gross_total_return": float(daily["gross_equity"].iloc[-1] - 1),
        "max_drawdown": float(daily["drawdown"].min()), "sharpe": sharpe,
        "annualized_return": annualized if np.isfinite(annualized) else None,
        "turnover": float(daily["turnover"].sum()),
        "notional_turnover": float(daily["notional_turnover"].sum()),
        "costs": float(daily["cost"].sum()), "traded_notional": float(daily["traded_notional"].sum()),
        "n_sessions": len(returns), "n_rebalances": len(schedules),
        "max_observed_weight": float(daily[["max_weight", "pretrade_max_weight"]].to_numpy().max()),
    }
    metadata = {
        "initial_equity": 1.0, "equity_units": "Units of initial wealth; returns are decimals",
        "first_date": dates[0].date().isoformat(), "last_date": dates[-1].date().isoformat(),
        "universe": list(clean.columns), "score_column": score_column,
        "top_n": top_n, "selected_count": selected_count, "max_weight": cap,
        "gross_exposure": exposure, "execution_lag": lag, "rebalance_sessions": rebalance,
        "commission_bps": commission * 10000, "slippage_bps": slippage * 10000,
        "annual_rf": rf, "annualization_sessions": 252, "liquidated": bool(liquidate),
        "skipped_for_rebalance_interval": skipped,
        "forecast_cutoffs": [cutoff.date().isoformat() for cutoff, _ in schedules.values()],
        "execution_dates": [clean.index[position].date().isoformat() for position in schedules],
        "tie_policy": "Descending scores; exact ties choose symbols alphabetically.",
        "cost_method": "Exact self-financing post-cost target sizing; commission and slippage on full buy-plus-sell notional, including entry and optional liquidation.",
        "turnover_method": "turnover is half-L1 risky-weight change (cash excluded); notional_turnover is full traded asset notional divided by pre-trade NAV.",
        "constraints": "Long only, no leverage; equal selected target weights capped at rebalance, residual cash earns zero. Weights drift between rebalances and can subsequently exceed target caps.",
        "timing": "Scores known after cutoff close; execute at least one later shared observed adjusted close. No pre-fill return is earned; final fills without a subsequent observed return are excluded.",
        "limitations": "Hypothetical adjusted-close execution, not quoted bid/ask fills. Fixed universe and current adjusted-price vintage; survivorship, liquidity, taxes, borrow and market impact are not modeled. No FX conversion; callers must restrict comparisons to compatible currencies. Sharpe assumes 252 common sessions/year and is descriptive, not proof of significance.",
    }
    return {"daily": daily, "trades": trade_frame, "weights": weights,
            "metrics": metrics, "metadata": metadata}
