"""Histories cannot silently migrate between revised datasets or edited outcomes."""
from pathlib import Path

import pytest

from derivport.analysis import analyze_cutoff, historical_forecasts, load_prices
from derivport.backtest import backtest_rule

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def snapshot():
    prices = load_prices(ROOT / 'data/prices.csv')
    return prices, historical_forecasts(prices)


def test_revised_same_universe_prices_reject_old_history(snapshot):
    prices, history = snapshot
    revised = prices.copy()
    revised.iloc[500, 0] *= 1.001
    with pytest.raises(ValueError, match='price snapshot'):
        analyze_cutoff(revised, '2024-05-03', history)
    with pytest.raises(ValueError, match='price snapshot'):
        backtest_rule(revised, '2023-01-01', '2024-01-01', history=history)


@pytest.mark.parametrize('field', ['actual_return', 'predicted_return', 'uncertainty', 'training_n'])
def test_modified_forecast_evidence_rejects_history(snapshot, field):
    prices, history = snapshot
    modified = history.copy(deep=True)
    row = modified[field].first_valid_index()
    modified.loc[row, field] += 1
    with pytest.raises(ValueError, match='edited'):
        analyze_cutoff(prices, '2024-05-03', modified)


def test_history_without_snapshot_metadata_requires_recomputation(snapshot):
    prices, history = snapshot
    detached = history.copy()
    detached.attrs.clear()
    with pytest.raises(ValueError, match='Recompute'):
        backtest_rule(prices, '2023-01-01', '2024-01-01', history=detached)


def test_valid_history_still_produces_same_original_analysis(snapshot):
    prices, history = snapshot
    result = analyze_cutoff(prices, '2024-05-03', history)
    assert result.training['ridge_alpha'] == 25
    assert result.training['last_label_end'] <= result.cutoff
    assert result.stocks['uncertainty_n'].ge(6).all()
