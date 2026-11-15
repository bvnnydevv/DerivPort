"""Checks on the actual, bundled demonstration data (no network required)."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TICKERS = {"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"}
CUTOFFS = ["2023-08-04", "2023-11-03", "2024-05-03", "2024-08-02"]


@pytest.fixture(scope="module")
def prices():
    return pd.read_csv(ROOT / "data" / "prices.csv", parse_dates=["date"])


def test_five_stocks_have_aligned_positive_adjusted_prices(prices):
    assert set(prices["ticker"]) == TICKERS
    assert not prices.duplicated(["date", "ticker"]).any()
    assert np.isfinite(prices["adjusted_close"]).all()
    assert (prices["adjusted_close"] > 0).all()
    counts = prices.groupby("date")["ticker"].nunique()
    assert counts.eq(5).all(), "Every experiment needs the same five-stock calendar."


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_curated_cutoffs_have_features_and_complete_outcomes(prices, cutoff):
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    assert pd.Timestamp(cutoff) in dates
    position = dates.get_loc(pd.Timestamp(cutoff))
    assert position >= 252, "A meaningful training history must predate the experiment."
    assert position + 21 < len(dates), "Each replay needs 21 completed future sessions."


def test_curated_notes_are_dated_and_source_linked():
    notes = json.loads((ROOT / "data" / "notes.json").read_text())
    assert notes
    for note in notes:
        assert note["ticker"] in TICKERS
        pd.Timestamp(note["published"])
        assert note["url"].startswith("https://")
        assert all(note[field].strip() for field in (
            "source_title", "fact", "interpretation", "limitation"
        ))
    for cutoff in CUTOFFS:
        for ticker in TICKERS:
            eligible = [
                note for note in notes
                if note["ticker"] == ticker
                and pd.Timestamp(note["published"]) <= pd.Timestamp(cutoff)
            ]
            assert eligible, f"Missing dated evidence for {ticker} on {cutoff}."
            chosen = max(eligible, key=lambda note: note["published"])
            assert pd.Timestamp(chosen["published"]) <= pd.Timestamp(cutoff)


def test_credits_are_bundled():
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
    serialized = json.dumps(provenance).lower()
    assert "http" in serialized
    assert "adjust" in serialized


def test_original_frozen_bytes_match_the_recorded_immutable_snapshot():
    expected = '812922aec7ee2a760d2e8cb91a447e8ef36099c796477c15d1ddd272a20132a1'
    provenance = json.loads((ROOT / 'data/provenance.json').read_text())
    assert provenance['cache_sha256'] == expected
    assert hashlib.sha256((ROOT / 'data/prices.csv').read_bytes()).hexdigest() == expected
