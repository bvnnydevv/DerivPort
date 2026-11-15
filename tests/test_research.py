"""End-to-end research evidence, currency gates, offline CLI and exports."""
from copy import deepcopy
import hashlib
import json
import shlex
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from derivport.analysis import price_fingerprint
from derivport import research
from derivport.research import ExperimentConfig, run_experiment, write_experiment
from derivport.research_models import MODEL_NAMES, walk_forward_models
from scripts import run_research as cli


@pytest.fixture(scope="module")
def prices():
    # Synthetic test fixture, not measured security performance.
    rng = np.random.default_rng(981)
    index = pd.bdate_range("2000-01-03", periods=975)
    shared = rng.normal(.0002, .006, (len(index), 1))
    returns = shared + rng.normal(0, .009, (len(index), 3))
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=["A", "B", "C"])


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig(validation_start="2001-01-02", test_start="2003-01-02", end="2003-08-29",
        seed=2026, bootstrap_samples=200, block_length=3)


def source_metadata(prices, currencies=("USD", "USD", "USD")):
    return {ticker: dict(currency=currency, source_url=None, retrieved="synthetic test fixture",
        snapshot_sha256=hashlib.sha256(prices[ticker].to_csv().encode()).hexdigest(), point_in_time_archive=False)
        for ticker, currency in zip(prices.columns, currencies)}


@pytest.fixture(scope="module")
def fitted(prices, config):
    return walk_forward_models(prices, validation_start=config.validation_start,
        test_start=config.test_start, end=config.end, seed=config.seed,
        rebalance_sessions=config.rebalance_sessions)


@pytest.fixture(scope="module")
def result(prices, config, fitted):
    # Cache the REAL fitted output across orchestration-only currency/export
    # checks rather than refitting identical forests for metadata variations.
    with patch.object(research, "walk_forward_models", return_value=fitted):
        return run_experiment(prices, config, source_metadata=source_metadata(prices),
            source_series={ticker: prices[ticker] for ticker in prices})


def test_manifest_records_exact_data_identity_settings_and_frozen_selection(prices, config, result):
    manifest = result["manifest"]
    selected = prices.loc[:config.end].sort_index(axis=1)
    assert manifest["config"]["seed"] == config.seed
    assert manifest["config"]["commission_bps"] == config.commission_bps
    assert manifest["data"]["aligned_sha256"] == price_fingerprint(selected)
    assert manifest["data"]["universe"] == list(selected.columns)
    assert pd.Timestamp(manifest["data"]["last_date"]) == selected.index[-1]
    assert manifest["models"]["seed"] == config.seed
    assert pd.Timestamp(manifest["models"]["parameter_selection_last_label_end"]) < pd.Timestamp(config.test_start)
    assert len(result["development_forecasts"].cutoff.unique()) >= 12
    assert (result["development_forecasts"].label_end < pd.Timestamp(config.test_start)).all()
    weight_fit = manifest["signals"]["weight_fit"]
    assert weight_fit["status"] == "available"
    assert pd.Timestamp(weight_fit["train_last_label_end"]) < pd.Timestamp(weight_fit["validation_first"])
    assert pd.Timestamp(weight_fit["final_fit_last_label_end"]) < pd.Timestamp(config.test_start)
    assert "scripts/run_research.py" in manifest["code"]["source_sha256"]
    assert all(len(value) == 64 for value in manifest["code"]["source_sha256"].values())
    assert len(manifest["run_id"]) == 16


def test_final_test_has_common_support_and_no_outputs_after_the_observed_end(config, result):
    forecasts = result["forecasts"]
    assert forecasts.cutoff.nunique() >= 2
    assert set(forecasts.model) == set(MODEL_NAMES)
    assert (forecasts.cutoff >= pd.Timestamp(config.test_start)).all()
    assert (forecasts.label_end <= pd.Timestamp(config.end)).all()
    assert (forecasts.training_last_label_end <= forecasts.cutoff).all()
    dates = set(forecasts.cutoff)
    for model, rows in forecasts.groupby("model"):
        assert set(rows.cutoff) == dates
        assert rows.groupby("cutoff").size().eq(3).all()
    for _, rows in result["scores"].groupby("strategy"):
        assert set(rows.cutoff) == dates
    for portfolio in result["portfolios"].values():
        assert portfolio["daily"].index[-1] <= pd.Timestamp(config.end)
        assert portfolio["daily"].index[-1] == forecasts.label_end.max()
    assert pd.Timestamp(result["manifest"]["portfolio_evaluation_end"]) == forecasts.label_end.max()
    parity = result["manifest"]["baseline_parity"]
    assert parity["matched_forecasts"] == parity["new_forecasts"] == len(forecasts.loc[forecasts.model.eq("Ridge")])
    assert parity["max_prediction_difference_pp"] == pytest.approx(0, abs=1e-12)
    assert parity["before_rmse_pp"] == pytest.approx(parity["after_rmse_pp"], abs=1e-12)


def test_zero_return_prediction_is_not_invested_through_arbitrary_tie_breaking(result):
    assert "Zero return" in set(result["forecasts"].model)
    assert result["forecasts"].loc[lambda frame: frame.model.eq("Zero return"), "predicted_return"].eq(0).all()
    assert "Zero return" not in result["portfolios"]
    cash = result["portfolios"]["Cash"]
    assert cash["daily"].net_equity.eq(1).all()
    assert cash["daily"].net_return.eq(0).all()
    assert cash["trades"].empty
    assert cash["metrics"]["sharpe"] is None
    assert cash["metrics"]["costs"] == cash["metrics"]["turnover"] == 0


@pytest.mark.parametrize("currencies", [("USD", "EUR", "USD"), ("Unknown", "Unknown", "Unknown"), (None, None, None)])
def test_mixed_or_unknown_currencies_disable_portfolios_but_preserve_predictions(prices, config, fitted, result, currencies):
    with patch.object(research, "walk_forward_models", return_value=fitted):
        unavailable = run_experiment(prices, config, source_metadata=source_metadata(prices, currencies),
            source_series={ticker: prices[ticker] for ticker in prices})
    assert unavailable["portfolios"] == {}
    assert unavailable["portfolio_summary"].empty
    assert unavailable["period_returns"].empty
    assert "unavailable" in unavailable["manifest"]["portfolio_status"].lower()
    pd.testing.assert_frame_equal(unavailable["model_summary"], result["model_summary"])
    pd.testing.assert_frame_equal(unavailable["ranking_summary"], result["ranking_summary"])


def test_exported_json_is_finite_and_every_csv_has_a_verified_checksum(result, tmp_path):
    target = write_experiment(result, tmp_path / "evidence")
    def reject_constant(value):
        pytest.fail("Nonstandard JSON numeric constant: " + value)
    manifest = json.loads((target / "manifest.json").read_text(), parse_constant=reject_constant)
    assert manifest["run_id"] == result["manifest"]["run_id"]
    assert {"forecasts.csv", "selection.csv", "folds.csv", "model_summary.csv", "ranking_summary.csv"}.issubset(manifest["output_sha256"])
    actual_files = {path.name for path in target.glob("*.csv")}
    assert actual_files == set(manifest["output_sha256"])
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    exported = pd.read_csv(target / "forecasts.csv")
    assert len(exported) == len(result["forecasts"])
    assert manifest["portfolio_file_order"] == sorted(result["portfolios"])


def test_cli_frozen_loader_checks_integrity_and_propagates_snapshot_provenance():
    prices, metadata, raw = cli.load_offline_dataset("frozen", ["AAPL", "MSFT"])
    digest = cli.verify_frozen_cache()
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert prices.index.max() == pd.Timestamp("2024-12-31")
    for ticker in prices:
        assert metadata[ticker]["currency"] == "USD"
        assert metadata[ticker]["snapshot_sha256"] == digest
        assert metadata[ticker]["retrieved"]
        assert "/363151b27e8bf71b377483b37fa49ee5ebb3f5e2/" in metadata[ticker]["source_url"]
        pd.testing.assert_series_equal(raw[ticker], prices[ticker])


def test_cli_missing_cache_cannot_trigger_a_provider_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "verify_frozen_cache", lambda: calls.append("verified") or "a" * 64)
    monkeypatch.setattr(cli, "read_cached_symbols", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(cli, "_read_cached", lambda symbol: None)
    def forbidden(*args, **kwargs):
        pytest.fail("Offline research attempted a market-data request")
    import derivport.market_data as market_data
    monkeypatch.setattr(market_data, "_provider", forbidden)
    with pytest.raises(ValueError, match="no network request"):
        cli.load_offline_dataset("cache", ["AAPL", "MSFT"])
    assert calls == ["verified"]
    with pytest.raises(ValueError, match="No offline snapshot"):
        cli.load_offline_dataset("cache", ["UNCACHED", "MSFT"])


def test_cli_records_the_exact_arguments_and_an_explicit_end_for_reproduction(prices, result, monkeypatch, tmp_path, capsys):
    captured = {}
    monkeypatch.setattr(cli, "load_offline_dataset", lambda dataset, symbols:
        (prices, source_metadata(prices), {ticker: prices[ticker] for ticker in prices}))
    def run(price_input, config, source_metadata=None, source_series=None):
        captured["config"] = config
        return deepcopy(result)
    def write(output, target):
        captured["manifest"] = output["manifest"]
        return target
    monkeypatch.setattr(cli, "run_experiment", run)
    monkeypatch.setattr(cli, "write_experiment", write)
    target = tmp_path / "cli-output"
    assert cli.main(["--dataset", "cache", "--symbols", "A,B,C", "--seed", "87", "--commission-bps", "25",
        "--execution-lag", "2", "--annual-rf", ".02", "--bootstrap-samples", "200", "--output", str(target)]) == 0
    assert captured["config"].seed == 87
    assert captured["config"].commission_bps == 25
    assert captured["config"].execution_lag == 2
    command = shlex.split(captured["manifest"]["reproduction_command"])
    assert command[:2] == [".venv/bin/python", "scripts/run_research.py"]
    assert command[command.index("--seed") + 1] == "87"
    assert command[command.index("--end") + 1] == "2003-08-29"
    assert command[command.index("--output") + 1] == str(target.resolve())
    assert captured["manifest"]["dataset"] == "cache"
    assert captured["manifest"]["reproduction_cwd"] == "Repository root"
    assert "scores are not return forecasts" in capsys.readouterr().out


def test_cli_rejects_duplicate_or_single_symbol_universes():
    for values in ("AAPL,aapl", "AAPL"):
        with pytest.raises(ValueError):
            cli._symbols(values)
