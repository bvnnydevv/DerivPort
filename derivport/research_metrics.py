"""Separate forecast accuracy, ranking quality, and date-level inference.

Information coefficient means a cross-sectional prediction/score versus realized
return correlation at one forecast date; its time mean is distinct from a pooled
correlation. Bootstrap intervals and centered-null p-values are approximate,
conditional on the supplied date sequence and fixed survivor universe. They do
not correct selection of dates, assets, models, or experiments after inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INFERENCE_NOTE = (
    "Approximate 95% circular moving-block bootstrap on ordered date aggregates, "
    "not independent stock rows. Centered-null two-sided p-values test mean zero. "
    "Small samples, selected survivors, regime changes and repeated experimentation "
    "limit inference; adjusted significance is not proof of prospective superiority."
)


def _constant(values):
    values = np.asarray(values, dtype=float)
    return float(np.ptp(values)) <= 1e-12 * max(1.0, float(np.max(np.abs(values))))


def _correlation(first, second, rank=False):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(a) < 2 or _constant(a) or _constant(b):
        return np.nan
    if rank:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    a, b = a - a.mean(), b - b.mean()
    a, b = a / np.max(np.abs(a)), b / np.max(np.abs(b))
    return float(np.clip((a @ b) / np.sqrt((a @ a) * (b @ b)), -1, 1))


def _panel(frame, method, value):
    required = {method, "cutoff", "ticker", value, "actual_return"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("Evaluation rows need method, cutoff, ticker, score/prediction, and actual return fields.")
    rows = frame.copy()
    if rows.empty:
        return rows
    if not all(isinstance(name, str) and name.strip() for name in rows[method]):
        raise ValueError("Every evaluated method needs a nonempty name.")
    if not all(isinstance(name, str) and name.strip() for name in rows["ticker"]):
        raise ValueError("Every forecast needs a nonempty stock symbol.")
    try:
        rows["cutoff"] = pd.to_datetime(rows["cutoff"], errors="raise")
        rows[[value, "actual_return"]] = rows[[value, "actual_return"]].astype(float)
        if "label_end" in rows:
            rows["label_end"] = pd.to_datetime(rows["label_end"], errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Evaluation needs valid dates and numeric finite predictions/scores and outcomes.") from exc
    if rows["cutoff"].isna().any() or not np.isfinite(rows[[value, "actual_return"]].to_numpy()).all():
        raise ValueError("Evaluate only dated finite predictions/scores and observed actual returns.")
    if rows.duplicated([method, "cutoff", "ticker"]).any():
        raise ValueError("Duplicate method/date/stock evaluation rows are not allowed.")
    if "label_end" in rows and (rows["label_end"].isna().any() or (rows["label_end"] <= rows["cutoff"]).any()):
        raise ValueError("Every observed label end must follow its forecast cutoff.")
    for _, group in rows.groupby([method, "cutoff"]):
        if len(group) < 2:
            raise ValueError("Cross-sectional evaluation needs at least two stocks at every method/date.")
        if "label_end" in group and group["label_end"].nunique() != 1:
            raise ValueError("Stocks at one forecast date must share a label end.")
    if rows.groupby(["cutoff", "ticker"])["actual_return"].nunique().gt(1).any():
        raise ValueError("Methods must share the same observed stock/date outcomes.")
    return rows.sort_values([method, "cutoff", "ticker"], kind="stable")


def _interval_coverage(rows):
    if not {"interval_lower", "interval_upper"}.issubset(rows.columns):
        return np.nan, 0
    try:
        lower = rows["interval_lower"].to_numpy(dtype=float)
        upper = rows["interval_upper"].to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Prediction interval bounds must be numeric or missing.") from exc
    valid = np.isfinite(lower) & np.isfinite(upper)
    if (lower[valid] > upper[valid]).any():
        raise ValueError("Prediction interval lower bounds must not exceed upper bounds.")
    if not valid.any():
        return np.nan, 0
    actual = rows["actual_return"].to_numpy(dtype=float)
    return float(((actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid])).mean()), int(valid.sum())


def per_date_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """One observed cross-sectional forecast evaluation per model and date.

    RMSE/MAE inputs use percentage-point return units as supplied. Sign agreement
    uses np.sign, so an exactly zero forecast is not credited as an up/down call.
    The nominal central-80% residual interval coverage uses available finite
    interval_lower/interval_upper bounds and exposes its smaller denominator.
    """
    rows = _panel(forecasts, "model", "predicted_return")
    if "label_end" not in rows:
        raise ValueError("Prediction evaluation needs observed label_end dates.")
    records = []
    for (model, cutoff), group in rows.groupby(["model", "cutoff"], sort=True):
        predicted, actual = group["predicted_return"].to_numpy(), group["actual_return"].to_numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            error = predicted - actual
            mse = float(np.mean(error ** 2))
        if not np.isfinite(mse):
            raise ValueError("Prediction errors are too large for finite squared-error evaluation.")
        coverage, count = _interval_coverage(group)
        records.append({"model": model, "cutoff": cutoff, "label_end": group["label_end"].iloc[0],
            "n_forecasts": len(group), "mse": mse, "mae": float(np.mean(np.abs(error))),
            "spearman_ic": _correlation(predicted, actual, rank=True),
            "pearson_ic": _correlation(predicted, actual),
            "directional_accuracy": float((np.sign(predicted) == np.sign(actual)).mean()),
            "coverage80": coverage, "n_intervals": count})
    return pd.DataFrame(records, columns=["model", "cutoff", "label_end", "n_forecasts", "mse", "mae",
        "spearman_ic", "pearson_ic", "directional_accuracy", "coverage80", "n_intervals"])


def prediction_metrics(forecasts: pd.DataFrame, *, seed=2026, samples=1000, block_length=3) -> pd.DataFrame:
    """Pooled prediction metrics plus date-mean IC and date-block RMSE intervals."""
    rows = _panel(forecasts, "model", "predicted_return")
    dated = per_date_metrics(rows)
    records = []
    for model, group in rows.groupby("model", sort=True):
        per_date = dated.loc[dated["model"] == model]
        predicted, actual = group["predicted_return"].to_numpy(), group["actual_return"].to_numpy()
        errors = predicted - actual
        mse = float(np.mean(errors ** 2))
        total_variation = float(np.sum((actual - actual.mean()) ** 2))
        r_squared = np.nan if _constant(actual) else 1 - float(np.sum(errors ** 2)) / total_variation
        coverage, interval_count = _interval_coverage(group)
        calibration, uncertainty_count = np.nan, 0
        if "uncertainty" in group:
            uncertainty = group["uncertainty"].to_numpy(dtype=float)
            if (uncertainty[np.isfinite(uncertainty)] < 0).any():
                raise ValueError("Historical forecast uncertainty must be nonnegative.")
            valid = np.isfinite(uncertainty) & (uncertainty > 0)
            uncertainty_count = int(valid.sum())
            if valid.any():
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    calibration = float(np.mean((errors[valid] / uncertainty[valid]) ** 2))
        # A mean of per-date MSEs matches pooled MSE only on equal-size cross sections.
        if per_date["n_forecasts"].nunique() == 1:
            inference = bootstrap_mean(per_date.set_index("cutoff")["mse"], seed=seed,
                                       samples=samples, block_length=block_length)
            lower = float(np.sqrt(max(0, inference["ci_low"]))) if np.isfinite(inference["ci_low"]) else np.nan
            upper = float(np.sqrt(max(0, inference["ci_high"]))) if np.isfinite(inference["ci_high"]) else np.nan
        else:
            inference = {"status": "unavailable: equal-size date cross sections required for this RMSE interval"}
            lower = upper = np.nan
        records.append({"model": model, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(errors))),
            "r2": r_squared, "pearson": _correlation(predicted, actual),
            "spearman": _correlation(predicted, actual, rank=True),
            "mean_spearman_ic": float(per_date["spearman_ic"].mean()),
            "mean_pearson_ic": float(per_date["pearson_ic"].mean()),
            "directional_accuracy": float((np.sign(predicted) == np.sign(actual)).mean()),
            "n_dates": len(per_date), "n_forecasts": len(group),
            "n_spearman_dates": int(per_date["spearman_ic"].notna().sum()),
            "n_pearson_dates": int(per_date["pearson_ic"].notna().sum()),
            "coverage80": coverage, "n_intervals": interval_count,
            "standardized_sq_error_mean": calibration, "n_uncertainty": uncertainty_count,
            "rmse_ci_low": lower, "rmse_ci_high": upper, "inference_status": inference["status"]})
    columns = ["model", "rmse", "mae", "r2", "pearson", "spearman", "mean_spearman_ic", "mean_pearson_ic",
        "directional_accuracy", "n_dates", "n_forecasts", "n_spearman_dates", "n_pearson_dates", "coverage80",
        "n_intervals", "standardized_sq_error_mean", "n_uncertainty", "rmse_ci_low", "rmse_ci_high", "inference_status"]
    result = pd.DataFrame(records, columns=columns).set_index("model")
    result.attrs["inference"] = {"seed": seed, "samples": samples, "block_length": block_length,
                                 "min_dates": 12, "confidence": .95, "note": INFERENCE_NOTE}
    return result


def rank_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    """Evaluate scores only as rankings; dimensionless scores get no RMSE/MAE."""
    rows = _panel(scores, "strategy", "score")
    records = []
    for strategy, group in rows.groupby("strategy", sort=True):
        dated = [(len(date_rows), _correlation(date_rows["score"], date_rows["actual_return"], rank=True),
                  _correlation(date_rows["score"], date_rows["actual_return"]))
                 for _, date_rows in group.groupby("cutoff", sort=True)]
        values = pd.DataFrame(dated, columns=["count", "spearman", "pearson"])
        records.append({"strategy": strategy, "mean_spearman_ic": float(values["spearman"].mean()),
            "mean_pearson_ic": float(values["pearson"].mean()), "n_dates": len(values), "n_forecasts": len(group),
            "n_spearman_dates": int(values["spearman"].notna().sum()),
            "n_pearson_dates": int(values["pearson"].notna().sum())})
    return pd.DataFrame(records, columns=["strategy", "mean_spearman_ic", "mean_pearson_ic", "n_dates",
        "n_forecasts", "n_spearman_dates", "n_pearson_dates"]).set_index("strategy")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def bootstrap_mean(values, seed=2026, samples=1000, block_length=3, min_dates=12) -> dict:
    """Approximate mean-zero inference by circular blocks of ordered date aggregates.

    Missing/nonfinite aggregates are excluded and counted. Datetime-indexed
    Series must have unique chronological dates. Every bootstrap draw consists
    of consecutive circular blocks, truncated to the original finite-date count.
    Percentile intervals are 95%; p-values use a separately centered null sample
    and the finite-resampling +1 correction. No independent-stock resampling is
    performed. The caller supplies one aggregate per forecast/holding period.
    """
    seed = _integer(seed, "Seed", minimum=0)
    samples = _integer(samples, "Bootstrap samples")
    length = _integer(block_length, "Block length")
    minimum = _integer(min_dates, "Minimum dates", minimum=2)
    if isinstance(values, pd.Series) and isinstance(values.index, pd.DatetimeIndex):
        if not values.index.is_unique or not values.index.is_monotonic_increasing:
            raise ValueError("Bootstrap inputs require one aggregate per unique chronological date.")
    try:
        array = np.asarray(values, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Bootstrap values must be a one-dimensional numeric sequence of date aggregates.") from exc
    if array.ndim != 1:
        raise ValueError("Bootstrap one aggregate per date, not a stock-by-date matrix.")
    total = len(array)
    array = array[np.isfinite(array)]
    n = len(array)
    result = {"mean": float(array.mean()) if n else np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "p_value": np.nan, "n_dates": n, "n_excluded": total - n, "block_length": length,
        "samples": samples, "seed": seed, "status": "more_data_required", "inference_note": INFERENCE_NOTE}
    if n < minimum or n < 2 * length:
        result["status"] = f"more_data_required: need at least {max(minimum, 2 * length)} finite date aggregates"
        return result
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(samples, int(np.ceil(n / length))))
    indices = ((starts[:, :, None] + np.arange(length)) % n).reshape(samples, -1)[:, :n]
    draws = array[indices]
    boot_means = draws.mean(axis=1)
    null_means = (draws - result["mean"]).mean(axis=1)
    result["ci_low"], result["ci_high"] = [float(value) for value in np.quantile(boot_means, [.025, .975])]
    result["p_value"] = float((1 + np.count_nonzero(np.abs(null_means) >= abs(result["mean"]))) / (samples + 1))
    result["status"] = "approximate_date_block_inference"
    return result


def holm_adjust(pvalues):
    """Holm family-wise adjustment, counting untestable planned comparisons too.

    NaNs remain NaN but count in the family size as conservative untestable
    comparisons. Finite p-values must lie in [0,1].
    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
        raise ValueError("P-values must be a one-dimensional sequence in [0,1], with NaN for unavailable tests.")
    result = np.full(len(values), np.nan)
    order = np.flatnonzero(np.isfinite(values))
    order = order[np.argsort(values[order], kind="stable")]
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(order))) * values[order])
    result[order] = np.minimum(1, adjusted)
    return result


def paired_comparisons(panel: pd.DataFrame, method_col="model", value_col="mse", baseline="Ridge",
                       seed=2026, samples=1000, block_length=3) -> pd.DataFrame:
    """Pair EXACT date/horizon support, test method-minus-baseline mean, then Holm.

    The input already has one dated aggregate per method, never one row per
    stock. If label_end exists it is part of the pairing key, preventing mixing
    different horizons at a shared cutoff. Lower MSE differences favor a model;
    higher net-return differences favor a strategy. No superiority is inferred
    by this function from the sign alone.
    """
    required = {method_col, "cutoff", value_col}
    if not isinstance(panel, pd.DataFrame) or not required.issubset(panel.columns):
        raise ValueError("Paired comparisons need method, cutoff, and dated metric columns.")
    rows = panel.copy()
    if rows.empty:
        return pd.DataFrame(columns=["method", "baseline", "n_dates", "mean_difference", "ci_low", "ci_high", "p_value", "p_adjusted", "status"])
    if baseline not in set(rows[method_col]):
        raise ValueError("The requested comparison baseline is absent.")
    try:
        rows["cutoff"] = pd.to_datetime(rows["cutoff"], errors="raise")
        rows[value_col] = pd.to_numeric(rows[value_col], errors="raise")
        if "label_end" in rows:
            rows["label_end"] = pd.to_datetime(rows["label_end"], errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Paired comparisons require valid dates and numeric dated aggregates.") from exc
    if rows["cutoff"].isna().any() or rows[method_col].isna().any() or rows.duplicated([method_col, "cutoff"]).any():
        raise ValueError("Pair exactly one nonmissing dated aggregate per method and cutoff.")
    keys = ["cutoff"] + (["label_end"] if "label_end" in rows else [])
    if rows[keys].isna().any().any():
        raise ValueError("Every comparison needs a valid cutoff and, when supplied, label end.")
    reference = rows.loc[rows[method_col] == baseline, keys + [value_col]].rename(columns={value_col: "baseline_value"})
    records = []
    for method in sorted(set(rows[method_col]) - {baseline}):
        candidate = rows.loc[rows[method_col] == method, keys + [value_col]]
        paired = candidate.merge(reference, on=keys, how="inner", validate="one_to_one").sort_values("cutoff")
        finite = np.isfinite(paired[value_col]) & np.isfinite(paired["baseline_value"])
        paired = paired.loc[finite]
        difference = pd.Series((paired[value_col] - paired["baseline_value"]).to_numpy(),
                               index=pd.DatetimeIndex(paired["cutoff"]))
        inference = bootstrap_mean(difference, seed=seed, samples=samples, block_length=block_length)
        records.append({"method": method, "baseline": baseline, "n_dates": inference["n_dates"],
            "mean_difference": inference["mean"], "ci_low": inference["ci_low"], "ci_high": inference["ci_high"],
            "p_value": inference["p_value"], "status": inference["status"]})
    result = pd.DataFrame(records, columns=["method", "baseline", "n_dates", "mean_difference", "ci_low", "ci_high", "p_value", "status"])
    result["p_adjusted"] = holm_adjust(result["p_value"].to_numpy())
    return result[["method", "baseline", "n_dates", "mean_difference", "ci_low", "ci_high", "p_value", "p_adjusted", "status"]]
