"""Distinct views of hypothetical scenarios; observed prices stay at the anchor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


RISK_PRESETS = {
    'Low': {'volatility_multiplier': .75, 'annual_log_drift_percent': 6.0, 'correlation': .6},
    'Medium': {'volatility_multiplier': 1.0, 'annual_log_drift_percent': 6.0, 'correlation': .6},
    'High': {'volatility_multiplier': 1.5, 'annual_log_drift_percent': 6.0, 'correlation': .6},
}


def apply_risk_preset(state):
    """Change scenario dispersion only; retain the user's drift/correlation.

    The labels describe simulation assumptions, not an investment's suitability.
    Fixed log drift preserves analytical median growth. A lognormal model's
    mean simple return can still change with volatility through convexity.
    """
    selected = state.get('projection_risk', 'Medium')
    if selected == 'Custom':
        return
    if not isinstance(selected, str) or selected not in RISK_PRESETS:
        raise ValueError('Choose Low, Medium, High or Custom scenario risk.')
    state['projection_volatility'] = RISK_PRESETS[selected]['volatility_multiplier']


def _scenario_data(projected, ticker):
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError('Choose a valid stock for the simulated scenario chart.')
    try:
        if ticker not in projected.latest_prices.index or ticker not in projected.terminal_returns.columns:
            raise ValueError('The chosen stock is not in this simulated scenario.')
        dates = pd.DatetimeIndex(projected.dates)
        if dates.hasnans or len(dates) < 2 or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('The scenario needs ordered, unique anchor and target dates.')
        anchor = float(projected.latest_prices[ticker])
        if not np.isfinite(anchor) or anchor <= 0:
            raise ValueError('The observed scenario anchor must be finite and positive.')
        band = projected.quantiles.loc[projected.quantiles['ticker'].eq(ticker),
            ['date', 'p10', 'p50', 'p90']].copy()
        band['date'] = pd.to_datetime(band['date'], errors='raise')
        band = band.sort_values('date').set_index('date')
        if not pd.DatetimeIndex(band.index).equals(dates):
            raise ValueError('Scenario percentile dates must match its anchor and target checkpoints.')
        band = band.apply(pd.to_numeric, errors='raise')
        values = band.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Simulated scenario prices must be finite and positive.')
        if (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
            raise ValueError('Simulated scenario percentiles must remain ordered.')
        if not np.allclose(values[0], anchor, rtol=1e-10, atol=1e-10):
            raise ValueError('The scenario must start at its observed price anchor.')
        terminal = pd.to_numeric(projected.terminal_returns[ticker], errors='raise').to_numpy(dtype=float)
        if terminal.ndim != 1 or not len(terminal) or not np.isfinite(terminal).all():
            raise ValueError('Simulated terminal returns must be a nonempty finite sample.')
    except (AttributeError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Provide a valid simulated scenario and a stock included in it.') from exc
    return dates, anchor, band, terminal


def _layout(fig, dates, ticker, kind):
    fig.update_layout(font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif', size=12),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=66, r=18, t=44, b=62, autoexpand=True),
        hovermode='x', legend=dict(orientation='h', x=0, y=1.03, yanchor='bottom', font_size=11),
        meta={'kind': kind, 'status': 'Hypothetical simulation, not observed future prices or a validated forecast',
            'symbol': ticker, 'observed_anchor': str(dates[0].date()),
            'simulated_target': str(dates[-1].date())})
    return fig


def scenario_fan(projected, ticker):
    """Purple simulated percentile fan, with exactly one observed anchor point."""
    dates, anchor, band, _ = _scenario_data(projected, ticker)
    x = dates.to_pydatetime().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=band.p90.tolist(), mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=band.p10.tolist(), mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(188,155,255,.20)',
        name='Simulated 10th–90th range', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=band.p50.tolist(), mode='lines',
        line=dict(color='#bc9bff', width=2.5), name='Simulated median',
        hovertemplate='%{x|%d %b %Y}<br>Simulated median %{y:,.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[x[0]], y=[anchor], mode='markers',
        marker=dict(color='#dce7e4', size=8, line=dict(color='#bc9bff', width=1.5)),
        name='Last observed anchor',
        hovertemplate='%{x|%d %b %Y}<br>Observed anchor %{y:,.2f}<extra></extra>'))
    short_horizon = (dates[-1] - dates[0]).days < 730
    fig.update_xaxes(title=dict(text='Scenario date', standoff=12, font_size=11),
        type='date', automargin=True, nticks=5, tickangle=0,
        tickformat='%b %Y' if short_horizon else '%Y', tickfont_size=11)
    fig.update_yaxes(title=dict(text='Simulated price', standoff=12, font_size=11),
        automargin=True, tickfont_size=11)
    return _layout(fig, dates, ticker, 'simulated_percentile_fan')


def scenario_distribution(projected, ticker):
    """Amber histogram of the shared simulated terminal-return samples."""
    dates, _, _, terminal = _scenario_data(projected, ticker)
    p10, median, p90 = (float(value) for value in np.quantile(terminal, [.1, .5, .9]))
    fig = go.Figure(go.Histogram(x=terminal.tolist(),
        nbinsx=min(60, max(15, int(np.sqrt(len(terminal))))),
        marker=dict(color='#ffbd69', line=dict(color='#d69a50', width=.5)),
        opacity=.85, name='Simulated terminal returns', showlegend=False,
        hovertemplate='Hypothetical return %{x:+.1f}%<br>Simulated paths %{y}<extra></extra>'))
    fig.add_vline(x=median, line=dict(color='#f5dfbd', width=2),
        name=f'Median {median:+.3g}%', showlegend=True)
    for value, label in ((p10, 'P10'), (p90, 'P90')):
        fig.add_vline(x=value, line=dict(color='#ffbd69', width=1.5, dash='dash'),
            name=f'{label} {value:+.3g}%', showlegend=True)
    fig.update_xaxes(title=dict(text='Hypothetical return (%)', standoff=12, font_size=11),
        automargin=True, nticks=4, tickfont_size=11)
    fig.update_yaxes(title=dict(text='Simulated paths', standoff=10, font_size=11),
        rangemode='tozero', automargin=True, tickfont_size=11)
    _layout(fig, dates, ticker, 'simulated_terminal_distribution')
    # Percentile values live in a wrapping legend below the axis rather than
    # three colliding text labels at the top of a narrow histogram.
    fig.update_layout(showlegend=True, bargap=.04,
        margin=dict(l=60, r=14, t=24, b=94, autoexpand=True),
        legend=dict(orientation='h', x=0, xanchor='left', y=-.28,
            yanchor='top', font_size=10, itemwidth=30))
    return fig
