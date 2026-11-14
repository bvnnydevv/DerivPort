"""Chronological, completed-label-only forecasts and honest held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .formula import evaluate_formula

TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
DEFAULT_FORMULA = "predicted_return - k * uncertainty"
FEATURES = ["momentum", "volatility"]
HORIZON = 21
EVALUATION_START = pd.Timestamp("2022-01-01")
HISTORY_COLUMNS = [
    "cutoff", "ticker", "momentum", "volatility", "actual_return", "label_end",
    "predicted_return", "uncertainty", "uncertainty_n", "uncertainty_source",
    "training_last_label_end", "training_n",
]


@dataclass
class Analysis:
    cutoff: pd.Timestamp
    stocks: pd.DataFrame
    error_summary: dict
    training: dict


def load_prices(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "adjusted_close"}
    if not required.issubset(raw.columns):
        raise ValueError("Price cache needs date, ticker, and adjusted_close columns.")
    if raw.duplicated(["date", "ticker"]).any():
        raise ValueError("The price cache contains duplicate stock/date observations.")
    if raw["ticker"].isna().any() or not raw["ticker"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every price observation needs a nonempty stock symbol.")
    present = set(raw["ticker"])
    order = [ticker for ticker in TICKERS if ticker in present] + sorted(present - set(TICKERS))
    prices = raw.pivot(index="date", columns="ticker", values="adjusted_close").sort_index().reindex(columns=order)
    return validate_prices(prices)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete, common-session fixed universe without filling prices."""
    if not isinstance(prices, pd.DataFrame) or len(prices.columns) < 2:
        raise ValueError("Choose at least two stocks with overlapping historical data for a ranking experiment.")
    if prices.empty:
        raise ValueError("These stocks have no shared historical trading sessions.")
    if not prices.columns.is_unique:
        raise ValueError("Choose each stock symbol only once.")
    try:
        clean = prices.astype(float).copy()
        clean.index = pd.DatetimeIndex(clean.index).tz_localize(None)
    except (ValueError, TypeError) as exc:
        raise ValueError("Prices need numeric adjusted closes and valid session dates.") from exc
    if clean.index.isna().any() or not clean.index.is_unique:
        raise ValueError("The price cache contains missing or duplicate session dates.")
    if clean.isna().any().any() or not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any().any():
        raise ValueError("Prices must be positive, finite, and complete for every stock on each shared session; prices are never forward-filled.")
    clean.index.name = "date"
    return clean.sort_index()


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return long-form daily features and labels, all return units percentage points."""
    momentum = prices.pct_change(HORIZON, fill_method=None) * 100
    volatility = prices.pct_change(fill_method=None).rolling(HORIZON).std(ddof=1) * np.sqrt(HORIZON) * 100
    target = (prices.shift(-HORIZON) / prices - 1) * 100
    label_end = pd.Series(prices.index, index=prices.index).shift(-HORIZON)
    frames = []
    for ticker in prices.columns:
        frame = pd.DataFrame({
            "cutoff": prices.index, "ticker": ticker,
            "momentum": momentum[ticker].to_numpy(), "volatility": volatility[ticker].to_numpy(),
            "actual_return": target[ticker].to_numpy(), "label_end": label_end.to_numpy(),
        })
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).dropna(subset=FEATURES)
    return result.sort_values(["cutoff", "ticker"], ignore_index=True)


def _fit(features: pd.DataFrame, cutoff: pd.Timestamp):
    train = features.loc[(features["label_end"] <= cutoff) & features["actual_return"].notna()]
    if len(train) < 250:
        raise ValueError("This cutoff does not have enough completed historical training observations.")
    # Standardization is fitted on the training rows, never on validation or future data.
    model = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
    model.fit(train[FEATURES], train["actual_return"])
    return model, train


def _uncertainty(history: pd.DataFrame, cutoff: pd.Timestamp, ticker: str):
    if history.empty:
        return np.nan, 0, "No completed out-of-sample forecasts yet"
    known = history.loc[(history["label_end"] <= cutoff) & history["actual_return"].notna()]
    own = known.loc[known["ticker"] == ticker].sort_values("cutoff").tail(24)
    if len(own) >= 6:
        sample, source = own, "stock: last 24 completed OOS forecasts (or all available)"
    else:
        sample, source = known.sort_values(["cutoff", "ticker"]).tail(120), "pooled: last 120 completed OOS forecasts; fewer than 6 stock errors"
    if sample.empty:
        return np.nan, 0, source
    errors = sample["predicted_return"] - sample["actual_return"]
    return float(np.sqrt(np.mean(errors ** 2))), len(sample), source


def historical_forecasts(prices: pd.DataFrame, start: str = "2020-01-01") -> pd.DataFrame:
    """Expanding-window forecast origins 21 sessions apart, with OOS-error warm-up.

    Each model trains only on targets completed by that origin. Uncertainty for
    each historical prediction uses only errors completed at that same origin.
    No target or feature is shuffled into the past.
    """
    features = build_features(prices)
    dates = prices.index[prices.index >= pd.Timestamp(start)]
    if len(prices) <= HORIZON:
        empty = pd.DataFrame(columns=HISTORY_COLUMNS)
        empty.attrs["universe"] = tuple(prices.columns)
        return empty
    dates = dates[dates <= prices.index[-HORIZON - 1]][::HORIZON]
    records = []
    for cutoff in dates:
        today = features.loc[features["cutoff"] == cutoff].copy()
        if len(today) != len(prices.columns):
            continue
        try:
            model, train = _fit(features, cutoff)
        except ValueError:
            continue
        today["predicted_return"] = model.predict(today[FEATURES])
        prior = pd.DataFrame(records)
        today["uncertainty"] = np.nan
        today["uncertainty_n"] = 0
        today["uncertainty_source"] = ""
        for idx, row in today.iterrows():
            u, n, source = _uncertainty(prior, cutoff, row["ticker"])
            today.loc[idx, ["uncertainty", "uncertainty_n", "uncertainty_source"]] = [u, n, source]
        today["training_last_label_end"] = train["label_end"].max()
        today["training_n"] = len(train)
        records.extend(today.to_dict("records"))
    result = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    result.attrs["universe"] = tuple(prices.columns)
    return result


def rank_correlation(first, second) -> float | None:
    """Spearman = Pearson correlation of average tied ranks, including tied input."""
    if isinstance(first, pd.Series) and isinstance(second, pd.Series) and not first.index.equals(second.index):
        if (not first.index.is_unique or not second.index.is_unique
                or set(first.index) != set(second.index)):
            raise ValueError("Rankings must describe the same stocks.")
        second = second.reindex(first.index)
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Rankings must describe the same stocks.")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return None
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_history(history: pd.DataFrame, formula: str = DEFAULT_FORMULA, k: float = 1.0,
                     cutoff: str | pd.Timestamp | None = None) -> dict:
    """Evaluate only completed OOS outcomes since 2022, never future cutoff labels.

    A submitted rule is applied retrospectively with each origin's then-known
    inputs. This is a research comparison, not an untouched prospective test.
    """
    universe = set(history.attrs.get("universe", history["ticker"].unique()))
    known = history.loc[history["cutoff"] >= EVALUATION_START].copy()
    if cutoff is not None:
        known = known.loc[known["label_end"] <= pd.Timestamp(cutoff)]
    known = known.dropna(subset=["actual_return", "uncertainty"])
    if not known.empty:
        known = known.groupby("cutoff", group_keys=False).filter(
            lambda group: len(group) == len(universe) and set(group["ticker"]) == universe
        )
    rows = []
    for date, group in known.groupby("cutoff", sort=True):
        score = evaluate_formula(formula, group, k)
        if np.isscalar(score):
            score = np.full(len(group), score)
        rows.append({"cutoff": date, "label_end": group["label_end"].max(),
                     "model_rmse": float(np.sqrt(np.mean((group["predicted_return"] - group["actual_return"]) ** 2))),
                     "baseline_rmse": float(np.sqrt(np.mean((group["momentum"] - group["actual_return"]) ** 2))),
                     "model_spearman": rank_correlation(group["predicted_return"], group["actual_return"]),
                     "baseline_spearman": rank_correlation(group["momentum"], group["actual_return"]),
                     "student_spearman": rank_correlation(score, group["actual_return"])})
    per_date = pd.DataFrame(rows, columns=["cutoff", "label_end", "model_rmse", "baseline_rmse", "model_spearman", "baseline_spearman", "student_spearman"])
    if known.empty:
        return {"model_rmse": None, "baseline_rmse": None, "n_forecasts": 0, "n_dates": 0,
                "model_spearman": None, "baseline_spearman": None, "student_spearman": None,
                "per_date": per_date}
    def mean_or_none(column):
        return None if per_date.empty or per_date[column].dropna().empty else float(per_date[column].mean())
    return {"model_rmse": float(np.sqrt(np.mean((known["predicted_return"] - known["actual_return"]) ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean((known["momentum"] - known["actual_return"]) ** 2))),
            "n_forecasts": len(known), "n_dates": len(per_date),
            "model_spearman": mean_or_none("model_spearman"),
            "baseline_spearman": mean_or_none("baseline_spearman"),
            "student_spearman": mean_or_none("student_spearman"), "per_date": per_date}


def analyze_cutoff(prices: pd.DataFrame, cutoff: str | pd.Timestamp,
                   history: pd.DataFrame | None = None) -> Analysis:
    cutoff = pd.Timestamp(cutoff)
    if cutoff not in prices.index:
        raise ValueError("Choose a trading session available in the local cache.")
    if history is None:
        history = historical_forecasts(prices)
    if set(history.attrs.get("universe", history["ticker"].unique())) != set(prices.columns):
        raise ValueError("Forecast history must be recomputed for the current comparison universe.")
    features = build_features(prices)
    today = features.loc[features["cutoff"] == cutoff].copy().set_index("ticker")
    if len(today) != len(prices.columns):
        raise ValueError("There is not enough trailing price history for this cutoff.")
    model, train = _fit(features, cutoff)
    stocks = today[FEATURES].copy()
    stocks["predicted_return"] = model.predict(stocks[FEATURES])
    for ticker in stocks.index:
        u, n, source = _uncertainty(history, cutoff, ticker)
        if not np.isfinite(u):
            raise ValueError("No completed out-of-sample forecasts are available to estimate uncertainty.")
        stocks.loc[ticker, "uncertainty"] = u
        stocks.loc[ticker, "uncertainty_n"] = n
        stocks.loc[ticker, "uncertainty_source"] = source
    stocks["uncertainty_n"] = stocks["uncertainty_n"].astype(int)
    stocks["model_rank"] = stocks["predicted_return"].rank(ascending=False, method="average")
    stocks["momentum_rank"] = stocks["momentum"].rank(ascending=False, method="average")
    summary = evaluate_history(history, cutoff=cutoff)
    summary.pop("per_date")
    training = {"n_rows": len(train), "first_feature_date": train["cutoff"].min(),
                "last_feature_date": train["cutoff"].max(), "last_label_end": train["label_end"].max(),
                "feature_columns": FEATURES.copy(), "ridge_alpha": 25.0}
    return Analysis(cutoff, stocks.sort_index(), summary, training)


def apply_rule(stocks: pd.DataFrame, formula: str = DEFAULT_FORMULA, k: float = 1.0) -> pd.DataFrame:
    result = stocks.copy()
    result["score"] = evaluate_formula(formula, result, k)
    result["student_rank"] = result["score"].rank(ascending=False, method="average")
    return result


def reveal_outcomes(prices: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    if cutoff not in prices.index:
        raise ValueError("This cutoff is not a cached trading session.")
    position = prices.index.get_loc(cutoff)
    if position + HORIZON >= len(prices):
        raise ValueError("The cache does not yet include 21 subsequent trading sessions.")
    end = prices.index[position + HORIZON]
    actual = (prices.loc[end] / prices.loc[cutoff] - 1) * 100
    result = pd.DataFrame({"actual_return": actual})
    result["realized_rank"] = result["actual_return"].rank(ascending=False, method="average")
    result.index.name = "ticker"
    result.attrs["end_date"] = end
    return result.sort_index()
