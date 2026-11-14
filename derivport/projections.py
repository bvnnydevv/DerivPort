"""Hypothetical long-horizon scenarios anchored to observed adjusted prices.

These scenarios are an assumption-driven simulation, not historical data or an
extension of the 21-session regression forecast. Nothing in this module fetches
future prices or assigns predictive accuracy to a simulated outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MAX_TARGET_DATE = pd.Timestamp("2037-12-31")
MIN_CALIBRATION_RETURNS = 63


@dataclass
class Projection:
    """Monthly hypothetical price bands and shared terminal-return samples."""

    dates: pd.DatetimeIndex
    quantiles: pd.DataFrame  # date, ticker, p10, p50, p90; local-currency price units
    terminal_returns: pd.DataFrame  # path x ticker; percentage returns from anchor
    latest_prices: pd.Series
    annual_volatility: pd.Series  # decimal log-return standard deviation / year
    assumptions: dict
    seed: int

    @property
    def anchor_date(self) -> pd.Timestamp:
        return self.dates[0]

    @property
    def target_date(self) -> pd.Timestamp:
        return self.dates[-1]


def _finite_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty or len(prices.columns) < 1:
        raise ValueError("Provide at least one stock with observed adjusted-price history.")
    if not prices.columns.is_unique or not all(isinstance(name, str) and name.strip() for name in prices.columns):
        raise ValueError("Every stock needs a unique nonempty symbol.")
    try:
        clean = prices.astype(float).copy()
        clean.index = pd.DatetimeIndex(clean.index).tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Observed prices need numeric values and valid daily session dates.") from exc
    if clean.index.isna().any() or not clean.index.is_unique:
        raise ValueError("Observed history must have unique valid daily session dates.")
    if not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any().any():
        raise ValueError("Observed prices must be finite, positive and complete; missing prices are never filled.")
    return clean.sort_index().reindex(columns=sorted(clean.columns))


def project_scenarios(
    prices: pd.DataFrame,
    target_date: str | pd.Timestamp,
    annual_drift: float = 0.06,
    volatility_multiplier: float = 1.0,
    correlation: float = 0.6,
    n_paths: int = 600,
    seed: int = 2037,
) -> Projection:
    """Simulate monthly local-currency price scenarios through an exact target.

    ``annual_drift`` is an assumed annual *log-price* drift in decimal units,
    not a fitted return prediction. Each stock's log-return volatility is
    estimated from the latest three calendar years of observed data, annualized
    with sqrt(252), and then held fixed. Independent monthly normal increments
    share a common factor with the requested correlation. Actual calendar year
    fractions (days / 365.25) set each step's scale.

    Bands are empirical path percentiles, not calibrated confidence intervals.
    A fixed seed preserves the samples; formula/ranking edits are not inputs.
    Only one month's path states are held in memory at a time.
    """
    clean = _clean_prices(prices)
    anchor = clean.index[-1]
    try:
        target = pd.Timestamp(target_date)
        if pd.isna(target):
            raise ValueError("missing target")
        target = target.tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Choose a valid future scenario end date.") from exc
    if target > MAX_TARGET_DATE:
        raise ValueError("Future scenarios are supported through December 31, 2037.")
    if target <= anchor:
        raise ValueError("The scenario end date must be after the latest observed price date.")
    drift = _finite_number(annual_drift, "Annual log drift")
    multiplier = _finite_number(volatility_multiplier, "Volatility multiplier")
    rho = _finite_number(correlation, "Correlation")
    if multiplier < 0:
        raise ValueError("The volatility multiplier must be nonnegative.")
    if not 0 <= rho <= 1:
        raise ValueError("The common-factor correlation must be between 0 and 1.")
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)) or not 200 <= n_paths <= 20000:
        raise ValueError("Use between 200 and 20,000 simulated paths.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The random seed must be a nonnegative integer.")

    lookback_start = anchor - pd.DateOffset(years=3)
    log_returns = np.log(clean).diff().loc[lookback_start:].dropna()
    if len(log_returns) < MIN_CALIBRATION_RETURNS:
        raise ValueError("At least 63 completed daily returns within the latest three years are needed to estimate scenario volatility.")
    measured_volatility = log_returns.std(ddof=1) * np.sqrt(252)
    annual_volatility = measured_volatility * multiplier
    if not np.isfinite(annual_volatility.to_numpy()).all():
        raise ValueError("These assumptions produce non-finite volatility. Reduce the volatility multiplier.")
    annual_volatility.name = "annual_volatility"
    latest_prices = clean.iloc[-1].copy()
    latest_prices.name = "latest_adjusted_close"
    latest_prices.index.name = "ticker"

    month_ends = pd.date_range(anchor + pd.offsets.MonthEnd(1), target, freq="ME")
    intermediate = month_ends[month_ends < target]
    dates = pd.DatetimeIndex([anchor, *intermediate, target], name="date")
    # This guard limits extreme historical-anchor inputs while normal 2026–2037
    # experiments use about 134 steps. It is not an invented trading calendar.
    if len(dates) > 1201:
        raise ValueError("Choose observed history with a more recent last price; scenarios are limited to 100 years of monthly checkpoints.")

    sigma = annual_volatility.to_numpy(dtype=float)
    initial = latest_prices.to_numpy(dtype=float)
    log_growth = np.zeros((int(n_paths), len(clean.columns)), dtype=float)
    rng = np.random.default_rng(int(seed))
    rows = []

    def record(date, bands):
        for column, ticker in enumerate(clean.columns):
            rows.append({"date": date, "ticker": ticker,
                         "p10": float(bands[0, column]),
                         "p50": float(bands[1, column]),
                         "p90": float(bands[2, column])})

    record(anchor, np.broadcast_to(initial, (3, len(initial))))
    growth = np.ones_like(log_growth)
    for previous, current in zip(dates[:-1], dates[1:]):
        years = (current - previous).days / 365.25
        common = rng.standard_normal((int(n_paths), 1))
        individual = rng.standard_normal(log_growth.shape)
        shocks = np.sqrt(rho) * common + np.sqrt(1 - rho) * individual
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            log_growth += drift * years + sigma[None, :] * np.sqrt(years) * shocks
            growth = np.exp(log_growth)
            simulated_prices = growth * initial[None, :]
        if not np.isfinite(simulated_prices).all() or (simulated_prices <= 0).any():
            raise ValueError("These assumptions produce extreme or non-finite simulated prices. Reduce drift, volatility, or the horizon.")
        record(current, np.quantile(simulated_prices, [.1, .5, .9], axis=0))

    with np.errstate(over="ignore", invalid="ignore"):
        terminal_values = (growth - 1) * 100
    if not np.isfinite(terminal_values).all():
        raise ValueError("These assumptions produce non-finite simulated returns. Reduce drift, volatility, or the horizon.")
    terminal_returns = pd.DataFrame(terminal_values, columns=clean.columns)
    terminal_returns.index.name = "path"
    assumptions = {
        "status": "Hypothetical simulation; not historical prices or a validated forecast",
        "anchor_date": anchor.date().isoformat(),
        "target_date": target.date().isoformat(),
        "annual_log_drift": drift,
        "drift_meaning": "User-assumed annual log-price drift; median-path growth is exp(drift × years), before finite-sample variation. Mean simple return also depends on volatility.",
        "volatility_multiplier": multiplier,
        "annual_volatility": {ticker: float(value) for ticker, value in annual_volatility.items()},
        "volatility_lookback_start": log_returns.index[0].date().isoformat(),
        "volatility_lookback_end": log_returns.index[-1].date().isoformat(),
        "volatility_observations": len(log_returns),
        "volatility_method": "Latest three calendar years of completed log daily returns; sample standard deviation × sqrt(252), multiplied by the chosen factor and held fixed.",
        "correlation": rho,
        "correlation_method": "User-assumed common-factor correlation of normal log-price innovations; constant across stocks and time.",
        "time_steps": "Calendar month-end checkpoints plus the exact chosen end date; year fractions use days / 365.25, not a forecast of future exchange holidays.",
        "bands": "Empirical 10th, 50th, and 90th path percentiles; assumption-dependent scenario ranges, not calibrated confidence intervals.",
        "n_paths": int(n_paths),
        "seed": int(seed),
        "currency": "Each stock stays in its observed local price currency; no FX conversion.",
        "limitations": "Fixed drift, volatility and correlation; lognormal continuous paths omit regime changes, jumps, delistings and future corporate actions. No future filings, business facts, actual outcomes or forecast accuracy are invented. The 21-session regression is not extrapolated to 2037.",
    }
    return Projection(dates, pd.DataFrame(rows), terminal_returns, latest_prices,
                      annual_volatility, assumptions, int(seed))
