#!/usr/bin/env python3
"""Run and export a reproducible historical research experiment, entirely offline.

From the repository root:
  .venv/bin/python scripts/run_research.py --dataset frozen
  .venv/bin/python scripts/run_research.py --dataset cache --symbols AAPL,MSFT,ASML

This command never downloads or refreshes market data. The cache loader refuses
an unavailable or unauthenticated snapshot instead of contacting a provider.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from derivport.analysis import load_prices, validate_prices
from derivport.market_data import (
    BUNDLED_PATH, BUNDLED_PROVENANCE_PATH, _read_cached, normalize_symbol,
    read_cached_symbols, symbol_info, verify_frozen_cache,
)
from derivport.research import ExperimentConfig, run_experiment, write_experiment

DEFAULT_SYMBOLS = "AAPL,AMZN,GOOGL,MSFT,NVDA"


def _symbols(value: str) -> list[str]:
    symbols = [normalize_symbol(item.strip()) for item in value.split(",")]
    if len(symbols) < 2:
        raise ValueError("Choose at least two comma-separated symbols.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Choose each symbol only once.")
    return symbols


def load_offline_dataset(dataset: str, symbols: list[str]):
    """Return aligned prices, provenance and raw series without a network path."""
    frozen_digest = verify_frozen_cache()
    source_metadata, source_series = {}, {}
    if dataset == "frozen":
        prices = load_prices(BUNDLED_PATH)
        unavailable = sorted(set(symbols).difference(prices.columns))
        if unavailable:
            raise ValueError("Symbols absent from the frozen dataset: " + ", ".join(unavailable))
        provenance = json.loads(BUNDLED_PROVENANCE_PATH.read_text())
        source_files = {item["ticker"]: item for item in provenance.get("files", [])}
        for symbol in symbols:
            source_series[symbol] = prices[symbol].rename(symbol)
            source_metadata[symbol] = dict(symbol_info(symbol), currency="USD",
                source="Yahoo Finance via pinned Akma86 public mirror",
                source_url=source_files.get(symbol, {}).get("url", provenance.get("mirror_repository")),
                retrieved=provenance.get("retrieved"), snapshot_sha256=frozen_digest,
                point_in_time_archive=False, cache_status="frozen")
    elif dataset == "cache":
        cached = set(read_cached_symbols())
        unavailable = sorted(set(symbols).difference(cached))
        if unavailable:
            raise ValueError("No offline snapshot is available for: " + ", ".join(unavailable)
                + ". Cache it explicitly in the app before running this offline command.")
        for symbol in symbols:
            # Use the disk-only reader, never the provider-fallback loader. This
            # also prevents a deletion race from triggering an implicit request.
            loaded = _read_cached(symbol)
            if loaded is None:
                raise ValueError(f"{symbol}: its offline snapshot disappeared; no network request was made.")
            series, info = loaded
            source_series[symbol] = series.rename(symbol)
            source_metadata[symbol] = dict(info)
            if info.get("cache_status") == "bundled":
                source_metadata[symbol]["snapshot_sha256"] = frozen_digest
    else:
        raise ValueError("Choose the frozen or cache dataset.")
    aligned = pd.concat([source_series[symbol] for symbol in symbols], axis=1, sort=True).sort_index().dropna(how="any")
    return validate_prices(aligned).sort_index(axis=1), source_metadata, source_series


def _parser():
    parser = argparse.ArgumentParser(description="DerivPort offline walk-forward models, signals and cost-adjusted research portfolios.")
    parser.add_argument("--dataset", choices=("frozen", "cache"), default="frozen")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS, help="Comma-separated fixed comparison universe; every symbol must already be cached.")
    parser.add_argument("--validation-start", default="2020-01-01")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--end", default=None, help="Last observed date permitted in the experiment, YYYY-MM-DD.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebalance-sessions", type=int, default=21, help="Shared sessions between forecast/rebalance decisions; at least 21.")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-weight", type=float, default=.4)
    parser.add_argument("--gross-exposure", type=float, default=1.)
    parser.add_argument("--execution-lag", type=int, default=1, help="Execute at a later observed close, never the signal close.")
    parser.add_argument("--commission-bps", type=float, default=10.)
    parser.add_argument("--slippage-bps", type=float, default=5.)
    parser.add_argument("--annual-rf", type=float, default=0., help="Annual risk-free assumption as a decimal, e.g. 0.02 for 2%%.")
    parser.add_argument("--k", type=float, default=1.)
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Paired date-block bootstrap samples, 200–10000.")
    parser.add_argument("--block-length", type=int, default=3, help="Forecast origins in each resampled date block.")
    parser.add_argument("--output", type=Path, default=None, help="Export directory; default artifacts/research_runs/<run_id>.")
    return parser


def _reproduction_command(args, config, result, target):
    # Pin the actual observed end, even when the original invocation omitted it.
    # This prevents a later cache extension from silently extending the study.
    values = asdict(config)
    values["end"] = pd.Timestamp(result["manifest"]["data"]["last_date"]).date().isoformat()
    command = [".venv/bin/python", "scripts/run_research.py", "--dataset", args.dataset,
        "--symbols", ",".join(_symbols(args.symbols))]
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    try:
        portable_target = str(target.resolve().relative_to(ROOT))
    except ValueError:
        portable_target = str(target.resolve())
    command.extend(["--output", portable_target])
    return shlex.join(command)


def _table(frame, preferred):
    if frame.empty:
        return "Unavailable for this experiment."
    columns = [name for name in preferred if name in frame.columns]
    shown = frame[columns] if columns else frame
    return shown.to_string(float_format=lambda value: f"{value:.4f}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        symbols = _symbols(args.symbols)
        fields = {name: getattr(args, name) for name in ExperimentConfig.__dataclass_fields__}
        config = ExperimentConfig(**fields)
        prices, metadata, raw = load_offline_dataset(args.dataset, symbols)
        result = run_experiment(prices, config, source_metadata=metadata, source_series=raw)
        target = args.output or ROOT / "artifacts" / "research_runs" / result["manifest"]["run_id"]
        if (target / "manifest.json").exists():
            previous = json.loads((target / "manifest.json").read_text())
            if previous.get("run_id") != result["manifest"]["run_id"]:
                raise ValueError("The output directory contains a different experiment. Choose a new --output directory.")
        result["manifest"]["reproduction_command"] = _reproduction_command(args, config, result, target)
        result["manifest"]["reproduction_cwd"] = "Repository root"
        result["manifest"]["dataset"] = args.dataset
        exported = write_experiment(result, target)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Run {result['manifest']['run_id']} | offline {args.dataset} | {len(prices.columns)} stocks")
    print("\nOut-of-sample prediction accuracy (return inputs in percentage points):")
    print(_table(result["model_summary"], ["rmse", "rmse_pp", "mae", "mae_pp", "r2", "pearson", "directional_accuracy", "mean_spearman_ic", "n_dates"]))
    print("\nRanking diagnostics (scores are not return forecasts):")
    print(_table(result["ranking_summary"], ["spearman_ic", "mean_spearman_ic", "mean_pearson_ic", "n_dates"]))
    print("\nHypothetical cost-adjusted portfolios (returns are decimals):")
    print(_table(result["portfolio_summary"], ["total_return", "max_drawdown", "sharpe", "turnover", "costs", "n_rebalances"]))
    print(result["manifest"]["portfolio_status"])
    print("\nLimitations:")
    for limitation in result["manifest"]["limitations"]:
        print("- " + limitation)
    print(f"\nExported evidence: {exported.resolve()}")
    print("Reproduce from the repository root:\n" + result["manifest"]["reproduction_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
