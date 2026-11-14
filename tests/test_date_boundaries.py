"""Latest observed research remains valid without inventing pending outcomes."""

import numpy as np
import pandas as pd
import pytest

from derivport import analyze_cutoff, historical_forecasts, reveal_outcomes
from derivport.market_data import load_symbol_prices


@pytest.fixture(scope="module")
def extended_prices():
    series = [
        load_symbol_prices(symbol)[0].rename(symbol)
        for symbol in ("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")
    ]
    return pd.concat(series, axis=1, sort=True).dropna().sort_index()


def test_latest_cached_2026_close_can_be_researched_without_future_labels(extended_prices):
    latest = extended_prices.index[-1]
    assert latest.year == 2026
    history = historical_forecasts(extended_prices)
    analysis = analyze_cutoff(extended_prices, latest, history)
    assert len(analysis.stocks) == 5
    assert np.isfinite(analysis.stocks["predicted_return"]).all()
    assert np.isfinite(analysis.stocks["uncertainty"]).all()
    assert analysis.training["last_label_end"] <= latest
    assert "actual_return" not in analysis.stocks
    with pytest.raises(ValueError, match="21 subsequent trading sessions"):
        reveal_outcomes(extended_prices, latest)


def test_last_completed_replay_ends_at_latest_real_cached_close(extended_prices):
    last_complete_cutoff = extended_prices.index[-22]
    result = reveal_outcomes(extended_prices, last_complete_cutoff)
    assert result.attrs["end_date"] == extended_prices.index[-1]
    expected = (extended_prices.iloc[-1] / extended_prices.loc[last_complete_cutoff] - 1) * 100
    np.testing.assert_allclose(result.loc[expected.index, "actual_return"], expected)
