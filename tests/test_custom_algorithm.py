"""The downloadable standalone algorithm must match the app's actual math."""

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from derivport import apply_rule


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "algorithms/derivport_signal.py"
spec = importlib.util.spec_from_file_location("downloaded_signal", SCRIPT)
signal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal)

ROWS = [
    dict(ticker="AAPL", predicted_return=2.51, uncertainty=8.77, momentum=8.62, volatility=5.1),
    dict(ticker="SAP.DE", predicted_return=-1.5, uncertainty=3.0, momentum=-2.0, volatility=6.4),
    dict(ticker="7203.T", predicted_return=3.2, uncertainty=5.6, momentum=4.0, volatility=7.1),
]


@pytest.mark.parametrize("k", [0, .5, 1, 2, 10])
def test_downloaded_algorithm_matches_app_scores_and_ranks(k):
    app = apply_rule(pd.DataFrame(ROWS).set_index("ticker"), signal.FORMULA, k)
    standalone = signal.rank_stocks(ROWS, k)
    for row in standalone:
        assert row["score"] == pytest.approx(app.loc[row["ticker"], "score"])
        assert row["rank"] == app.loc[row["ticker"], "student_rank"]
    assert ROWS[0]["ticker"] == "AAPL"
    assert "score" not in ROWS[0]


def test_exact_ties_receive_average_rank_with_alphabetical_display():
    tied = dict(predicted_return=5, uncertainty=2, momentum=3, volatility=4)
    rows = signal.rank_stocks([dict(tied, ticker="Z"), dict(tied, ticker="A")])
    assert [row["ticker"] for row in rows] == ["A", "Z"]
    assert [row["rank"] for row in rows] == [1.5, 1.5]


@pytest.mark.parametrize("name,value", [("uncertainty", -1), ("volatility", -1),
    ("momentum", float("nan")), ("predicted_return", float("inf")), ("predicted_return", True)])
def test_invalid_inputs_are_rejected(name, value):
    with pytest.raises(ValueError):
        signal.score_row(dict(ROWS[0], **{name: value}))


@pytest.mark.parametrize("k", [-1, float("nan"), float("inf"), True])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError):
        signal.score_row(ROWS[0], k)


def test_zero_error_zero_volatility_and_negative_return_inputs_are_valid():
    assert signal.score_row(dict(predicted_return=-2, uncertainty=0, momentum=-3, volatility=0)) == pytest.approx(-2.4)


def test_duplicate_symbols_and_missing_inputs_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        signal.rank_stocks([ROWS[0], dict(ROWS[1], ticker="aapl")])
    with pytest.raises(ValueError, match="Missing input"):
        signal.score_row({"predicted_return": 2})
    with pytest.raises(ValueError, match="at least one"):
        signal.rank_stocks([])


def test_file_runs_outside_repository_with_only_standard_library(tmp_path):
    input_path, output_path = tmp_path / "features.csv", tmp_path / "ranked.csv"
    with input_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ticker", *signal.INPUT_FIELDS))
        writer.writeheader()
        writer.writerows(ROWS)
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--input", str(input_path),
        "--output", str(output_path), "--k", "1"], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    with output_path.open() as handle:
        result = list(csv.DictReader(handle))
    expected = signal.rank_stocks(ROWS)
    assert [row["ticker"] for row in result] == [row["ticker"] for row in expected]
    assert float(result[0]["score"]) == pytest.approx(expected[0]["score"])


def test_demo_is_explicitly_synthetic_and_csv_output_is_usable():
    completed = subprocess.run([sys.executable, "-I", str(SCRIPT), "--demo"], capture_output=True, text=True)
    assert completed.returncode == 0
    assert "SYNTHETIC INPUT DEMO" in completed.stderr
    assert completed.stdout.startswith("ticker,predicted_return,")
