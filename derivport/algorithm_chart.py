"""Retrospective ranking-score curves from then-known forecast inputs.

No realized returns or outcome labels are inputs. Reapplying a rule chosen now
is a mathematical replay, not an untouched backtest or a stock-price forecast.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .formula import FormulaError, evaluate_formula

INPUT_COLUMNS = ('predicted_return', 'uncertainty', 'momentum', 'volatility')


def _date(value, name):
    if value is None or isinstance(value, (bool, int, float, np.number)):
        raise ValueError(f'{name} must be a valid calendar date.')
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError('missing date')
        return stamp.tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a valid calendar date.') from exc


def _score(expression, values, k):
    try:
        score = float(evaluate_formula(expression, values, k))
        return score if np.isfinite(score) else np.nan
    except (FormulaError, TypeError, ValueError, OverflowError):
        return np.nan


def build_algorithm_history(history, ticker, formula, k, cutoff, start=None, current_row=None):
    """Replay one rule on selected historical forecast origins, without outcomes.

    Origin dates and ticker are filtered before reading/coercing feature values.
    Undefined scores stay as NaN gaps. Sparse forecast origins are not resampled
    into invented daily forecasts. A valid current input snapshot may add the
    selected cutoff, but never duplicates an existing origin on that date.
    """
    if not isinstance(history, pd.DataFrame) or not {'cutoff', 'ticker', *INPUT_COLUMNS}.issubset(history.columns):
        raise ValueError('Provide forecast history with origin dates, stocks and the four ranking inputs.')
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError('Choose a valid stock for the algorithm graph.')
    if not isinstance(formula, str) or not formula.strip() or len(formula) > 400:
        raise FormulaError('Provide a ranking formula of up to 400 characters.')
    if isinstance(k, (bool, np.bool_)):
        raise ValueError('The risk multiplier must be finite.')
    try:
        k = float(k)
        if not np.isfinite(k):
            raise ValueError('non-finite multiplier')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('The risk multiplier must be finite.') from exc
    boundary = _date(cutoff, 'Cutoff')
    beginning = _date(start, 'Chart start') if start is not None else None
    if beginning is not None and beginning > boundary:
        raise ValueError('The chart start must be on or before the cutoff.')

    # Never inspect outcome columns, or values belonging to another stock or a
    # later origin. The input history is a larger cache than this display view.
    selected = history.loc[history['ticker'].eq(ticker), ['cutoff']].copy()
    dates = pd.to_datetime(selected['cutoff'], errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    eligible = dates.notna() & dates.le(boundary)
    if beginning is not None:
        eligible &= dates.ge(beginning)
    positions = np.flatnonzero(history['ticker'].eq(ticker).to_numpy())[eligible.to_numpy()]
    known = history.iloc[positions].loc[:, list(INPUT_COLUMNS)].copy()
    known.index = pd.DatetimeIndex(dates.loc[eligible].to_numpy(), name='cutoff')
    if not known.index.is_unique:
        raise ValueError('Forecast origins must be unique for the selected stock.')
    known = known.sort_index()
    for column in INPUT_COLUMNS:
        known[column] = pd.to_numeric(known[column], errors='coerce')
    known['score'] = [_score(formula, values, k) for values in known.to_dict('records')]
    known['source'] = 'Forecast origin'

    if current_row is not None and boundary not in known.index:
        if not isinstance(current_row, (Mapping, pd.Series)):
            raise ValueError('Current inputs must be a stock input mapping.')
        values = {column: pd.to_numeric(current_row.get(column, np.nan), errors='coerce') for column in INPUT_COLUMNS}
        score = _score(formula, values, k)
        if np.isfinite(score):
            current = pd.DataFrame([{**values, 'score': score, 'source': 'Selected cutoff'}],
                index=pd.DatetimeIndex([boundary], name='cutoff'))
            known = current if known.empty else pd.concat([known, current]).sort_index()

    known.attrs.update(symbol=ticker, cutoff=boundary.date().isoformat(), formula=formula,
        k=k, undefined_points=int(known['score'].isna().sum()),
        status='Retrospective rule replay on then-known inputs; not a price forecast or performance backtest.',
        sampling='Historical forecast origins, plus a valid selected-cutoff snapshot; no daily interpolation.')
    return known


def history_score_chart(history, ticker, formula, k, cutoff, start=None, current_row=None):
    """Return a distinct orange score figure and its exact displayed input frame."""
    frame = build_algorithm_history(history, ticker, formula, k, cutoff, start, current_row)
    values = [None if pd.isna(value) else float(value) for value in frame['score']]
    fig = go.Figure(go.Scatter(x=frame.index.to_pydatetime().tolist(), y=values,
        mode='lines+markers', connectgaps=False, name=f'{ticker} algorithm score',
        line=dict(color='#c46d32', width=2.5), marker=dict(size=5),
        hovertemplate='%{x|%d %b %Y}<br>Ranking score %{y:.3f}<extra></extra>'))
    fig.update_layout(height=250, margin=dict(l=55, r=20, t=24, b=48),
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        showlegend=False, hovermode='x', meta={'kind': 'algorithm_score_history', **frame.attrs})
    fig.update_xaxes(title='Historical input date', type='date', automargin=True)
    fig.update_yaxes(title='Ranking score', automargin=True, zeroline=True)
    return fig, frame
