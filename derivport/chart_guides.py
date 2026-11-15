"""Bounded coach annotations computed from observed chart data, never model code."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


GUIDE_KINDS = ('trend', 'range', 'moving_average')
GUIDE_EXPLANATIONS = {
    'trend': 'The trend guide connects the first and last observed closes in the visible window. It describes that window; it does not extend into future prices.',
    'range': 'The range guide marks the lowest observed low and highest observed high in the visible window. A close-only chart uses close extremes instead. These window extremes are descriptive, not validated support or resistance levels.',
    'moving_average': 'The moving-average guide uses SMA 20: the mean of 20 observed closes at the selected bar interval. It describes past prices and is not a prediction.',
}


def guide_intent(question):
    """Map explicit user requests to three permitted actions, not coordinates."""
    text = str(question or '').casefold()
    if any(negative in text for negative in ("don't", 'don’t', 'do not', 'without drawing', 'no drawing')):
        return None
    if re.search(r'\b(clear|remove|erase)\b', text) and re.search(r'\b(guide|guides|annotation|annotations|drawing|drawings)\b', text):
        return 'clear'
    if not re.search(r'\b(draw|mark|show|add|plot)\b', text):
        return None
    if 'moving average' in text or re.search(r'\bsma\s*20\b', text):
        return 'moving_average'
    if re.search(r'\b(trend|trendline)\b', text):
        return 'trend'
    if re.search(r'\b(range|support|resistance)\b', text):
        return 'range'
    return None


def queue_chart_guide(state, kind, context):
    """Queue only a known local action, before the chart widgets are rendered."""
    if kind == 'clear':
        state.pop('coach_chart_guide', None)
        # Remove an indicator only when this coach action added it. Existing
        # user-selected indicators and chart-toolbar drawings are left alone.
        if state.pop('coach_added_sma20', False):
            state['chart_overlays'] = [item for item in state.get('chart_overlays', []) if item != 'SMA 20']
        return
    if kind not in GUIDE_KINDS:
        raise ValueError('Choose an observed trend, range or moving-average guide.')
    if kind == 'moving_average':
        overlays = list(state.get('chart_overlays', []))
        if 'SMA 20' not in overlays:
            state['chart_overlays'] = [*overlays, 'SMA 20']
            state['coach_added_sma20'] = True
    state['coach_chart_guide'] = {'kind': kind, 'context': str(context)}


def _clear_owned(fig):
    fig.layout.shapes = tuple(shape for shape in fig.layout.shapes if not str(shape.name or '').startswith('coach-'))
    fig.layout.annotations = tuple(annotation for annotation in fig.layout.annotations if not str(annotation.name or '').startswith('coach-'))
    fig.data = tuple(trace for trace in fig.data if not str(trace.name or '').startswith('Coach · '))


def _known_frame(frame, cutoff, kind):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Chart guides need dated, observed price bars.')
    try:
        cutoff = pd.Timestamp(cutoff)
        if pd.isna(cutoff):
            raise ValueError('Missing cutoff')
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        cutoff = cutoff.normalize()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Chart guides need a valid historical cutoff.') from exc
    known = frame.copy()
    if known.index.tz is not None:
        known.index = known.index.tz_localize(None)
    if known.index.hasnans:
        raise ValueError('Chart guides need valid observed session dates.')
    # Drop later bars before numeric validation, rolling calculations or extrema.
    known = known.loc[known.index <= cutoff].sort_index()
    known.columns = [str(column).lower() for column in known.columns]
    if 'close' not in known and 'adjusted_close' in known:
        known = known.rename(columns={'adjusted_close': 'close'})
    if known.empty:
        raise ValueError('No observed chart bars are available by this cutoff.')
    if known.index.duplicated().any() or known.columns.duplicated().any():
        raise ValueError('Chart guides need unique dates and price fields.')
    close_range = kind == 'range' and 'close' in known and not {'low', 'high'}.intersection(known.columns)
    fields = ['low', 'high'] if kind == 'range' and not close_range else ['close']
    known.attrs['coach_range_basis'] = 'close' if close_range else 'high_low'
    if not set(fields).issubset(known.columns):
        raise ValueError('A range guide needs genuine high/low bars.' if kind == 'range' else 'This chart needs observed closing prices for a guide.')
    try:
        known[fields] = known[fields].apply(pd.to_numeric, errors='raise')
        values = known[fields].to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Chart guide prices must be numeric observations.') from exc
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Chart guide prices must be finite and positive.')
    if kind == 'range' and not close_range and (known.low > known.high).any():
        raise ValueError('Observed chart high/low bars are inconsistent.')
    return known


def _label(fig, x, y, text, name, color):
    # Plotly's annotation coordinates use log10 on a log axis. Shape and trace
    # prices still use ordinary observed-price units.
    label_y = math.log10(float(y)) if fig.layout.yaxis.type == 'log' else float(y)
    fig.add_annotation(x=x, y=label_y, xref='x', yref='y', text=text,
        name=name, showarrow=False, xanchor='right', yanchor='bottom',
        bgcolor='#15242c', bordercolor=color, borderpad=3,
        font=dict(family='Trebuchet MS, sans-serif', size=11, color=color))


def apply_chart_guide(fig, frame, guide, context, cutoff):
    """Apply a local guide using only observed bars on/before cutoff.

    Coach-owned elements are replaced idempotently. Other figure shapes and
    traces are preserved; model output is never interpreted as drawing code.
    """
    if not isinstance(fig, go.Figure):
        raise ValueError('Provide a Plotly chart for the observed-price guide.')
    _clear_owned(fig)
    if not isinstance(guide, dict) or guide.get('context') != str(context):
        return fig
    kind = guide.get('kind')
    if kind not in GUIDE_KINDS:
        raise ValueError('Choose an observed trend, range or moving-average guide.')
    known = _known_frame(frame, cutoff, kind)
    first, last = known.index[0].to_pydatetime(), known.index[-1].to_pydatetime()
    cyan, amber = '#69d9ff', '#ffbd69'
    if kind == 'trend':
        start, end = float(known.close.iloc[0]), float(known.close.iloc[-1])
        fig.add_shape(type='line', x0=first, x1=last, y0=start, y1=end,
            xref='x', yref='y', name='coach-trend',
            line=dict(color=cyan, width=2, dash='dash'))
        _label(fig, last, end, 'Observed window trend', 'coach-trend-label', cyan)
    elif kind == 'range':
        close_only = known.attrs['coach_range_basis'] == 'close'
        low = float(known.close.min()) if close_only else float(known.low.min())
        high = float(known.close.max()) if close_only else float(known.high.max())
        prefix = 'Close window' if close_only else 'Window'
        for value, label in ((low, 'low'), (high, 'high')):
            fig.add_shape(type='line', x0=first, x1=last, y0=value, y1=value,
                xref='x', yref='y', name='coach-window-' + label,
                line=dict(color=amber, width=1.5, dash='dot'))
            _label(fig, last, value, f'{prefix} {label} {value:,.2f}', 'coach-window-' + label + '-label', amber)
    else:
        if 'sma 20' in known:
            # The chart preparer warms this causal indicator on all earlier
            # observed bars before cropping its visible window. Reuse it.
            try:
                mean = pd.to_numeric(known['sma 20'], errors='raise')
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError('The observed SMA 20 indicator must be numeric.') from exc
        else:
            mean = known.close.rolling(20, min_periods=20).mean()
        available = mean.dropna()
        if not np.isfinite(available.to_numpy(dtype=float)).all() or (available <= 0).any():
            raise ValueError('The observed SMA 20 indicator must be finite and positive.')
        if available.empty:
            _label(fig, last, float(known.close.iloc[-1]), 'Choose a wider window for a 20-bar guide',
                'coach-moving-average-label', cyan)
            return fig
        if not any(trace.name == 'SMA 20' for trace in fig.data):
            fig.add_trace(go.Scatter(x=available.index.to_pydatetime().tolist(),
                y=available.tolist(), mode='lines', name='Coach · SMA 20',
                line=dict(color=cyan, width=2), xaxis='x', yaxis='y',
                hovertemplate='Observed SMA 20 %{y:,.2f}<extra></extra>'))
        _label(fig, available.index[-1].to_pydatetime(), float(available.iloc[-1]),
            'Observed SMA 20', 'coach-moving-average-label', cyan)
    return fig
