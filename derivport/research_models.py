"""Chronological model comparisons with a frozen final-test selection boundary.

The existing two-feature, 21-session Ridge model is retained exactly. Alternative
models share its training rows, targets, forecast origins and evaluation support.
Development predictions select parameters using only validation outcomes already
known at that origin; parameters then freeze before the final test starts.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import FEATURES, HORIZON, build_features, validate_prices


MODEL_NAMES = ("Ridge", "Elastic Net", "Random Forest", "Gradient Boosting", "Momentum", "Zero return")
PARAMETER_GRIDS = {
    "Ridge": ({"alpha": 25.0},),
    "Elastic Net": ({"alpha": .1, "l1_ratio": .25}, {"alpha": 1.0, "l1_ratio": .75}),
    "Random Forest": ({"max_depth": 3}, {"max_depth": 5}),
    "Gradient Boosting": ({"max_depth": 1}, {"max_depth": 2}),
    "Momentum": ({},),
    "Zero return": ({},),
}
FORECAST_COLUMNS = [
    "model", "split", "cutoff", "ticker", "momentum", "volatility", "actual_return", "label_end",
    "predicted_return", "uncertainty", "uncertainty_n", "uncertainty_source", "uncertainty_bias",
    "interval_lower", "interval_upper", "interval_n", "training_last_label_end", "training_n",
    "selected_params",
]
FOLD_COLUMNS = [
    "model", "candidate_params", "cutoff", "label_end", "training_last_label_end",
    "training_last_feature_date", "training_n", "n_validation_forecasts", "rmse",
]


def _params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _estimator(name: str, params: dict, seed: int):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
    if name == "Elastic Net":
        return make_pipeline(StandardScaler(), ElasticNet(**params, max_iter=10000, tol=1e-6,
            selection="cyclic", random_state=seed))
    if name == "Random Forest":
        return RandomForestRegressor(**params, n_estimators=32, min_samples_leaf=20,
            max_features=1.0, random_state=seed, n_jobs=1)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(**params, n_estimators=32, learning_rate=.05,
            min_samples_leaf=20, subsample=1.0, loss="squared_error", random_state=seed,
            n_iter_no_change=None)
    raise ValueError(f"{name} is a prediction baseline, not a fitted estimator.")


def _calibration(history: pd.DataFrame, cutoff: pd.Timestamp, ticker: str, prediction: float) -> dict:
    """Model-specific past residual diagnostics, without a distribution guarantee.

    Bias and interval residuals use actual minus predicted return. RMSE is in
    percentage points, not a posterior standard deviation. The 80% empirical
    residual interval is descriptive and its support is always exposed.
    """
    empty = dict(uncertainty=np.nan, uncertainty_n=0,
        uncertainty_source="No completed out-of-sample forecasts yet", uncertainty_bias=np.nan,
        interval_lower=np.nan, interval_upper=np.nan, interval_n=0)
    if history.empty:
        return empty
    known = history.loc[(history["label_end"] <= cutoff) & history["actual_return"].notna()]
    own = known.loc[known["ticker"] == ticker].sort_values("cutoff").tail(24)
    if len(own) >= 6:
        sample, source = own, "stock: last 24 completed OOS forecasts (or all available)"
    else:
        sample = known.sort_values(["cutoff", "ticker"]).tail(120)
        source = "pooled: last 120 completed OOS forecasts; fewer than 6 stock errors"
    if sample.empty:
        return empty
    residual = (sample["actual_return"] - sample["predicted_return"]).to_numpy(dtype=float)
    lower, upper = np.quantile(residual, [.1, .9])
    return dict(uncertainty=float(np.sqrt(np.mean(residual ** 2))), uncertainty_n=len(sample),
        uncertainty_source=source, uncertainty_bias=float(np.mean(residual)),
        interval_lower=float(prediction + lower), interval_upper=float(prediction + upper),
        interval_n=len(sample))


def walk_forward_models(prices: pd.DataFrame, validation_start="2020-01-01", test_start="2023-01-01",
                        end=None, seed=2026, rebalance_sessions=21, min_training_rows=250,
                        models=None) -> dict:
    """Return common-support development/test forecasts and selection audit data.

    Six fixed quarterly calendar thresholds in the 18 months before the test
    boundary identify inner validation origins. Each fold fits only completed labels. At a development
    origin, only folds completed by that origin can influence its parameter
    choice. Final choices use folds whose outcomes finish strictly before
    ``test_start`` and remain fixed throughout the test. Training data itself
    expands during the test as older test outcomes become observable.

    Targets always span 21 shared sessions. Origins are at least 21 sessions
    apart, so evaluation targets do not overlap. Crossing-boundary development
    outcomes are purged from development fitting and returned rows. Their
    forecasts are retained internally and can calibrate later origins once
    their outcomes have actually completed. Returned rows all have observed outcomes by ``end``.
    This is a retrospective, fixed-universe research comparison, not evidence
    that the supplied stocks or final-test dates were prospectively selected.
    """
    if isinstance(rebalance_sessions, bool) or not isinstance(rebalance_sessions, Integral) or rebalance_sessions < HORIZON:
        raise ValueError("Forecast origins must be at least 21 shared sessions apart.")
    if isinstance(min_training_rows, bool) or not isinstance(min_training_rows, Integral) or min_training_rows < 2:
        raise ValueError("min_training_rows must be an integer of at least two.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer.")
    if models is None:
        names = list(MODEL_NAMES)
    else:
        if isinstance(models, str):
            raise ValueError("models must be a nonempty sequence of model names.")
        names = list(models)
        if not names or len(set(names)) != len(names) or any(name not in MODEL_NAMES for name in names):
            raise ValueError(f"Choose unique model names from {', '.join(MODEL_NAMES)}.")
    try:
        validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
        if pd.isna(validation_start) or pd.isna(test_start) or validation_start.tz is not None or test_start.tz is not None:
            raise ValueError
        if validation_start >= test_start:
            raise ValueError
        requested_end = pd.Timestamp(end) if end is not None else None
        if requested_end is not None and (pd.isna(requested_end) or requested_end.tz is not None):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError("Use valid timezone-naive dates with validation_start before test_start.") from exc
    clean = validate_prices(prices).sort_index(axis=1)
    if requested_end is not None:
        clean = clean.loc[:requested_end]
    if clean.empty or clean.index[-1] < validation_start:
        raise ValueError("The selected end must include observed validation dates.")
    effective_end = clean.index[-1]
    features = build_features(clean)
    observed = features["actual_return"].notna()
    if (not np.isfinite(features[FEATURES].to_numpy()).all()
            or not np.isfinite(features.loc[observed, "actual_return"].to_numpy()).all()):
        raise ValueError("Price changes produced non-finite features or returns.")
    n_stocks = len(clean.columns)
    # Eligible means features exist, the target is observable, and the common
    # expanding training set is large enough; anchor the fixed schedule here.
    candidates = []
    by_date = {date: rows.copy() for date, rows in features.groupby("cutoff", sort=True)}
    for date in clean.index[clean.index >= validation_start]:
        today = by_date.get(date)
        if today is None or len(today) != n_stocks or today["actual_return"].isna().any():
            continue
        train_count = int(((features["label_end"] <= date) & observed).sum())
        if train_count >= min_training_rows:
            candidates.append(date)
    scheduled = candidates[::int(rebalance_sessions)]
    development = [date for date in scheduled if by_date[date]["label_end"].max() < test_start]
    origins = [date for date in scheduled if date >= test_start or date in development]
    if not origins:
        raise ValueError("Not enough completed history for the selected walk-forward experiment.")
    # Define the fold calendar from the declared split boundary, not the end of
    # the supplied data. Selecting the last six *available* folds globally would
    # change old development predictions when unseen future data is appended.
    validation_origins = []
    for months in (18, 15, 12, 9, 6, 3):
        threshold = test_start - pd.DateOffset(months=months)
        eligible = [date for date in development if date >= threshold]
        if eligible and eligible[0] not in validation_origins:
            validation_origins.append(eligible[0])
    prediction_cache = {}
    fold_records = []

    def training_at(origin):
        return features.loc[(features["label_end"] <= origin) & observed]

    def predict(name, params, origin, train, today):
        key = (name, _params_json(params), origin)
        if key not in prediction_cache:
            if name == "Momentum":
                values = today["momentum"].to_numpy(dtype=float)
            elif name == "Zero return":
                values = np.zeros(len(today), dtype=float)
            else:
                estimator = _estimator(name, params, int(seed))
                estimator.fit(train[FEATURES], train["actual_return"])
                values = estimator.predict(today[FEATURES])
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} produced a non-finite forecast.")
            prediction_cache[key] = values
        return prediction_cache[key]

    for origin in validation_origins:
        train, today = training_at(origin), by_date[origin]
        for name in names:
            for params in PARAMETER_GRIDS[name]:
                prediction = predict(name, params, origin, train, today)
                fold_records.append(dict(model=name, candidate_params=_params_json(params), cutoff=origin,
                    label_end=today["label_end"].max(), training_last_label_end=train["label_end"].max(),
                    training_last_feature_date=train["cutoff"].max(), training_n=len(train),
                    n_validation_forecasts=len(today),
                    rmse=float(np.sqrt(np.mean((prediction - today["actual_return"].to_numpy()) ** 2)))))
    folds = pd.DataFrame(fold_records, columns=FOLD_COLUMNS)

    def choose(name, as_of=None):
        choices = PARAMETER_GRIDS[name]
        if folds.empty:
            return dict(choices[0]), folds
        available = folds.loc[folds["model"] == name]
        if as_of is not None:
            available = available.loc[available["label_end"] <= as_of]
        if available.empty:
            return dict(choices[0]), available
        errors = {}
        for params in choices:
            key = _params_json(params)
            sample = available.loc[available["candidate_params"] == key]
            errors[key] = float(np.average(sample["rmse"] ** 2, weights=sample["n_validation_forecasts"]))
        # Grid order is a deterministic tie breaker, not a final-test choice.
        winner = min(choices, key=lambda params: errors[_params_json(params)])
        return dict(winner), available.loc[available["candidate_params"] == _params_json(winner)]

    frozen, selection_records = {}, []
    for name in names:
        params, selected_folds = choose(name)
        frozen[name] = params
        validation_rmse = (float(np.sqrt(np.average(selected_folds["rmse"] ** 2,
            weights=selected_folds["n_validation_forecasts"]))) if not selected_folds.empty else np.nan)
        selection_records.append(dict(model=name, selected_params=_params_json(params),
            validation_rmse=validation_rmse, validation_n_origins=len(selected_folds),
            validation_n_forecasts=int(selected_folds["n_validation_forecasts"].sum()) if not selected_folds.empty else 0,
            last_validation_label_end=selected_folds["label_end"].max() if not selected_folds.empty else pd.NaT,
            selection_frozen_at=test_start, tuned=len(PARAMETER_GRIDS[name]) > 1 and not selected_folds.empty))
    selection = pd.DataFrame(selection_records)
    records, model_history = [], {name: [] for name in names}
    for origin in scheduled:
        train, today = training_at(origin), by_date[origin]
        split = ("test" if origin >= test_start else
            "development" if today["label_end"].max() < test_start else "purged")
        for name in names:
            params = frozen[name] if split == "test" else choose(name, origin)[0]
            predictions = predict(name, params, origin, train, today)
            prior = pd.DataFrame(model_history[name])
            current = []
            for (_, row), prediction in zip(today.iterrows(), predictions):
                current.append(dict(model=name, split=split, cutoff=origin, ticker=row["ticker"],
                    momentum=float(row["momentum"]), volatility=float(row["volatility"]),
                    actual_return=float(row["actual_return"]), label_end=row["label_end"],
                    predicted_return=float(prediction),
                    **_calibration(prior, origin, row["ticker"], float(prediction)),
                    training_last_label_end=train["label_end"].max(), training_n=len(train),
                    selected_params=_params_json(params)))
            # The entire contemporaneous cross-section is appended together;
            # today's unknown outcome cannot calibrate another stock today.
            model_history[name].extend(current)
            if split != "purged":
                records.extend(current)
    forecasts = pd.DataFrame(records, columns=FORECAST_COLUMNS).sort_values(
        ["cutoff", "model", "ticker"], ignore_index=True)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(clean, index=True).to_numpy().tobytes()
        + json.dumps(list(clean.columns)).encode()).hexdigest()
    last_selection_end = folds["label_end"].max() if not folds.empty else None
    metadata = dict(seed=int(seed), horizon_sessions=HORIZON, rebalance_sessions=int(rebalance_sessions),
        min_training_rows=int(min_training_rows), universe=list(clean.columns), features=list(FEATURES),
        models=names, validation_start=validation_start.isoformat(), test_start=test_start.isoformat(),
        end=effective_end.isoformat(), price_fingerprint=fingerprint, selected_params=frozen,
        parameter_selection_last_label_end=last_selection_end.isoformat() if last_selection_end is not None else None,
        validation_fold_origins=[date.isoformat() for date in validation_origins],
        validation_fold_calendar="First scheduled origin at/after test_start minus18,15,12,9,6,3 calendar months; outcomes strictly before test_start.",
        calibration_bridge_origins=[date.isoformat() for date in scheduled
            if date < test_start and by_date[date]["label_end"].max() >= test_start],
        selection_rule="Development uses only completed validation folds; final-test parameters freeze before test_start.",
        selection_metric="Pooled inner-validation RMSE; maximum two predefined candidates per alternative model.",
        calibration="Model-specific completed OOS residuals: last24 stock errors when at least6 exist, otherwise last120 pooled errors. Empirical10th/90th residual quantiles; no coverage guarantee.",
        bias_definition="Mean actual_return minus predicted_return in percentage points on the calibration sample.",
        training_rule="All observed feature rows whose 21-session labels complete by the forecast close. Forecasts are after-close, not same-close executable orders.",
        baseline_rule="Momentum and Zero return have no fitted coefficients; training audit fields describe the shared available training set.",
        uncertainty_units="Percentage-point historical forecast RMSE, not a posterior standard deviation.",
        software_versions=dict(sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
    return dict(forecasts=forecasts, selection=selection, folds=folds, metadata=metadata)
