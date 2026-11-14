#!/usr/bin/env python3
"""DerivPort Signal: a transparent, balanced stock-ranking research rule.

Inputs must already be computed from evidence available at the decision cutoff:
  predicted_return: estimated next 21-session adjusted-price return, in % points
  uncertainty: RMSE of completed past out-of-sample forecasts, in % points
  momentum: trailing 21-session adjusted-price return, in % points
  volatility: trailing daily-return sample std * sqrt(21), in % points

This file ranks those inputs. It does not download prices, fit a forecast, invent
future observations, or place trades. A score is not a predicted return or a
buy/sell recommendation. The coefficients are an explicit research hypothesis,
not optimized parameters or evidence of profitable performance.

Run with Python 3.10+; no third-party dependencies are needed:
  python derivport_signal.py --input features.csv --output ranked.csv --k 1
  python derivport_signal.py --demo

CSV columns: ticker,predicted_return,uncertainty,momentum,volatility
Here 3.5 means 3.5%, not 0.035. k is dimensionless and must be nonnegative.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys


NAME = "DerivPort Signal"
FORMULA = "0.6 * predicted_return + 0.4 * momentum - k * (0.5 * uncertainty + 0.25 * volatility)"
DEFAULT_K = 1.0
INPUT_FIELDS = ("predicted_return", "uncertainty", "momentum", "volatility")
OUTPUT_FIELDS = ("ticker", *INPUT_FIELDS, "score", "rank")


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _inputs(row):
    try:
        values = {field: _number(row[field], field) for field in INPUT_FIELDS}
    except KeyError as exc:
        raise ValueError(f"Missing input column: {exc.args[0]}.") from exc
    for field in ("uncertainty", "volatility"):
        if values[field] < 0:
            raise ValueError(f"{field} must be nonnegative.")
    return values


def _weight(k):
    k = _number(k, "k")
    if k < 0:
        raise ValueError("k must be nonnegative.")
    return k


def score_row(row, k=DEFAULT_K):
    """Return one full-precision research score; do not mutate the input.

    A 60/40 blend of the model estimate and trailing momentum supplies the
    signal. k controls penalties for measured forecast error and volatility.
    Every term has percentage-point units. Negative model estimates or momentum
    are valid; error/volatility and k must be nonnegative.
    """
    values, k = _inputs(row), _weight(k)
    score = (
        0.6 * values["predicted_return"]
        + 0.4 * values["momentum"]
        - k * (0.5 * values["uncertainty"] + 0.25 * values["volatility"])
    )
    if not math.isfinite(score):
        raise ValueError("The score is non-finite; reduce the input magnitudes or k.")
    return score


def rank_stocks(rows, k=DEFAULT_K):
    """Return scored copies, sorted best score first with average tied ranks.

    Ticker order is only a display tie breaker. Exact score ties share their
    average rank, matching the DerivPort app. Duplicate symbols are rejected.
    """
    k = _weight(k)
    result, seen = [], set()
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            raise ValueError("Each stock needs a nonempty ticker.")
        if ticker in seen:
            raise ValueError(f"Duplicate ticker: {ticker}.")
        seen.add(ticker)
        values = _inputs(row)
        result.append({"ticker": ticker, **values, "score": score_row(values, k)})
    if not result:
        raise ValueError("Provide at least one row of stock inputs.")
    result.sort(key=lambda row: (-row["score"], row["ticker"]))
    start = 0
    while start < len(result):
        end = start + 1
        while end < len(result) and result[end]["score"] == result[start]["score"]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for row in result[start:end]:
            row["rank"] = average_rank
        start = end
    return result


def _write_csv(rows, handle):
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=f"{NAME}: balanced model + momentum ranking for research. Scores are not return forecasts.",
        epilog="Input columns: ticker,predicted_return,uncertainty,momentum,volatility. All return-like inputs are percentage points; 3.5 means 3.5%. Use cutoff-known inputs only.",
    )
    parser.add_argument("--input", help="CSV containing precomputed stock inputs")
    parser.add_argument("--output", help="Write ranked CSV here; otherwise write to stdout")
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="Nonnegative uncertainty/volatility weight (default: 1)")
    parser.add_argument("--demo", action="store_true", help="Use explicitly synthetic arithmetic examples, not historical market data")
    args = parser.parse_args(argv)
    if args.demo and args.input:
        parser.error("Choose --demo or --input, not both.")
    if not args.demo and not args.input:
        parser.error("Provide --input features.csv or use --demo.")
    try:
        if args.demo:
            print("SYNTHETIC INPUT DEMO: made-up arithmetic examples, not historical prices or forecasts.", file=sys.stderr)
            rows = [
                dict(ticker="SAMPLE_A", predicted_return=4, uncertainty=3, momentum=6, volatility=4),
                dict(ticker="SAMPLE_B", predicted_return=3, uncertainty=2, momentum=2, volatility=3),
                dict(ticker="SAMPLE_C", predicted_return=-1, uncertainty=4, momentum=-2, volatility=6),
            ]
        else:
            with open(args.input, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        ranked = rank_stocks(rows, args.k)
        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as handle:
                _write_csv(ranked, handle)
        else:
            _write_csv(ranked, sys.stdout)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
