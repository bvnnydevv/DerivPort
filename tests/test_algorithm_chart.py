"""Algorithm plots replay known inputs, never realized outcomes or fake prices."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from derivport.algorithm_chart import build_algorithm_history, history_score_chart


@pytest.fixture
def history():
    return pd.DataFrame([
        dict(cutoff=date, ticker=ticker, predicted_return=p, uncertainty=u, momentum=m, volatility=v,
            actual_return=999999, label_end='2099-12-31')
        for date, ticker, p, u, m, v in [
            ('2024-01-02','AAPL',4.,1.,3.,2.),
            ('2024-02-01','AAPL',6.,2.,5.,3.),
            ('2024-03-01','AAPL',8.,0.,7.,4.),
            ('2024-04-01','AAPL',100.,99.,98.,97.),
            ('2024-02-01','MSFT',1000.,999.,998.,997.),
        ]])


def test_score_curve_uses_exact_formula_inputs_and_no_outcome_columns(history):
    without_outcomes = history.drop(columns=['actual_return', 'label_end'])
    frame = build_algorithm_history(without_outcomes, 'AAPL', 'predicted_return - k * uncertainty', 2., '2024-03-01')
    assert frame.score.tolist() == [2., 2., 8.]
    assert list(frame.index) == list(pd.to_datetime(['2024-01-02','2024-02-01','2024-03-01']))
    assert 'actual_return' not in frame and 'label_end' not in frame
    assert frame.attrs['undefined_points'] == 0
    assert 'not a price forecast' in frame.attrs['status']


def test_future_and_other_stock_mutations_cannot_change_the_view(history):
    original = history.copy(deep=True)
    expected = build_algorithm_history(history, 'AAPL', 'predicted_return - k * uncertainty', 1., '2024-03-01')
    mutated = deepcopy(history)
    fields = ['predicted_return','uncertainty','momentum','volatility']
    mutated[fields] = mutated[fields].astype(object)
    mutated.loc[3:4, fields] = 'malformed future or other stock'
    mutated['actual_return'] = np.nan
    mutated['label_end'] = 'not a date'
    actual = build_algorithm_history(mutated, 'AAPL', 'predicted_return - k * uncertainty', 1., '2024-03-01')
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(history, original)


def test_current_snapshot_appends_once_and_invalid_snapshot_does_not_append(history):
    current = dict(predicted_return=9., uncertainty=2., momentum=6., volatility=3.)
    frame = build_algorithm_history(history, 'AAPL', 'predicted_return - k * uncertainty', 2., '2024-03-15', current_row=current)
    assert frame.score.tolist() == [2.,2.,8.,5.]
    assert frame.source.iloc[-1] == 'Selected cutoff'
    existing = build_algorithm_history(history, 'AAPL', 'predicted_return - k * uncertainty', 2., '2024-03-01', current_row=current)
    assert len(existing) == 3 and existing.score.iloc[-1] == 8.
    invalid = build_algorithm_history(history, 'AAPL', 'predicted_return / uncertainty', 2., '2024-03-15', current_row=dict(current, uncertainty=0.))
    assert len(invalid) == 3


def test_undefined_history_is_a_gap_not_a_line_through_an_invalid_score(history):
    fig, frame = history_score_chart(history, 'AAPL', 'predicted_return / uncertainty', 1., '2024-03-01', start='2024-02-01')
    assert len(frame) == 2 and frame.score.iloc[0] == 3.
    assert np.isnan(frame.score.iloc[1]) and frame.attrs['undefined_points'] == 1
    assert fig.data[0].y == (3., None)
    assert fig.data[0].connectgaps is False
    payload = json.loads(fig.to_json())
    assert isinstance(payload['data'][0]['x'], list) and isinstance(payload['data'][0]['y'], list)
    assert 'bdata' not in fig.to_json()
    assert fig.layout.yaxis.title.text == 'Ranking score'
    assert not fig.layout.yaxis.ticksuffix
    assert fig.layout.meta['kind'] == 'algorithm_score_history'


def test_unused_uncertainty_does_not_hide_a_valid_momentum_rule(history):
    history.loc[0, 'uncertainty'] = np.nan
    frame = build_algorithm_history(history, 'AAPL', 'momentum', 1., '2024-03-01')
    assert frame.score.tolist() == [3.,5.,7.]


def test_duplicate_known_origins_are_rejected_but_future_duplicates_are_irrelevant(history):
    future_duplicate = pd.concat([history, history.iloc[[3]]], ignore_index=True)
    assert len(build_algorithm_history(future_duplicate, 'AAPL', 'momentum', 1., '2024-03-01')) == 3
    known_duplicate = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='unique'):
        build_algorithm_history(known_duplicate, 'AAPL', 'momentum', 1., '2024-03-01')


@pytest.mark.parametrize('kwargs', [dict(k=np.inf), dict(k=True), dict(cutoff='bad'), dict(start='2025-01-01')])
def test_bad_display_parameters_are_rejected(history, kwargs):
    args = dict(history=history, ticker='AAPL', formula='momentum', k=1., cutoff='2024-03-01')
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_algorithm_history(**args)


def test_short_window_can_show_only_the_current_snapshot_without_invented_origins(history):
    fig, frame = history_score_chart(history, 'AAPL', 'momentum', 1., '2024-03-15',
        start='2024-03-10', current_row=dict(predicted_return=9., uncertainty=2., momentum=6., volatility=3.))
    assert len(frame) == 1
    assert frame.source.iloc[0] == 'Selected cutoff'
    assert list(fig.data[0].y) == [6.]
