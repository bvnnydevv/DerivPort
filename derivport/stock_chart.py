"""Observed-price charts with cutoff-safe OHLC aggregation and indicators.

Candles require genuine open/high/low/close observations. The separate adjusted
close renderer intentionally cannot create candles or invent trading volume.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PERIODS = ('1M', '3M', '6M', '1Y', '5Y', 'All')
INTERVALS = ('Daily', 'Weekly', 'Monthly')
STYLES = ('Candles', 'OHLC', 'Line')
OVERLAYS = ('SMA 20', 'SMA 50', 'EMA 20')
UP_COLOR = '#34d399'
DOWN_COLOR = '#fb7185'
BACKGROUND = '#101820'
INDICATOR_COLORS = {'SMA 20': '#ffc966', 'SMA 50': '#8db6ff', 'EMA 20': '#d0a8ff'}


def _cutoff(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('Choose a valid historical cutoff for the chart.')
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if 'date' in result.columns:
        result = result.set_index('date')
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError('Chart observations need a dated index.')
    if result.index.hasnans:
        raise ValueError('Chart observations contain an invalid date.')
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize()
    if result.index.duplicated().any():
        raise ValueError('Chart observations contain duplicate trading dates.')
    result.index.name = 'date'
    return result.sort_index()


def _indicators(frame: pd.DataFrame, overlays: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for name in dict.fromkeys(overlays):
        if name not in OVERLAYS:
            raise ValueError(f'Unknown chart indicator: {name}.')
        periods = int(name.rsplit(' ', 1)[1])
        if name.startswith('SMA'):
            result[name] = result['close'].rolling(periods, min_periods=periods).mean()
        else:
            result[name] = result['close'].ewm(span=periods, adjust=False, min_periods=periods).mean()
    return result


def _window(frame: pd.DataFrame, cutoff: pd.Timestamp, period: str) -> pd.DataFrame:
    if period not in PERIODS:
        raise ValueError('Choose 1M, 3M, 6M, 1Y, 5Y, or All for the chart range.')
    if period == 'All':
        return frame.copy()
    amount = int(period[:-1])
    offset = pd.DateOffset(months=amount) if period.endswith('M') else pd.DateOffset(years=amount)
    return frame.loc[frame.index >= cutoff - offset].copy()


def prepare_chart_data(candles: pd.DataFrame, cutoff, period: str = '6M',
                       interval: str = 'Daily', overlays: Iterable[str] = ()) -> pd.DataFrame:
    """Aggregate actual OHLCV through cutoff, compute indicators, then crop.

    Weekly/monthly bars end on their last *observed* session, including the
    current partial period. No future Friday/month-end labels or future prices
    enter an incomplete bar. Indicator windows count selected-interval bars.
    """
    if interval not in INTERVALS:
        raise ValueError('Choose Daily, Weekly, or Monthly chart bars.')
    known = candles.copy()
    known.columns = [str(column).lower() for column in known.columns]
    if known.columns.duplicated().any():
        raise ValueError('Chart observations contain duplicate field names.')
    required = ('open', 'high', 'low', 'close')
    if not set(required).issubset(known.columns):
        raise ValueError('Candles require real open, high, low and close observations.')
    known = _dated(known)
    cutoff = _cutoff(cutoff)
    known = known.loc[known.index <= cutoff].copy()
    # Validate only observations available by the cutoff; future rows do not
    # influence this historical view or its validation outcome.
    fields = [*required, *(['volume'] if 'volume' in known else [])]
    known = known[fields].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(known.to_numpy()).all() or (known[list(required)] <= 0).any().any():
        raise ValueError('Chart prices must be finite, positive observed values.')
    if ((known['low'] > known[['open', 'close']].min(axis=1)) |
        (known['high'] < known[['open', 'close']].max(axis=1)) |
        (known['high'] < known['low'])).any():
        raise ValueError('Observed OHLC values have an invalid high/low range.')
    if 'volume' in known and (known['volume'] < 0).any():
        raise ValueError('Observed trading volume cannot be negative.')
    if interval != 'Daily' and not known.empty:
        known['_observed_end'] = known.index
        aggregation = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', '_observed_end': 'last'}
        if 'volume' in known:
            aggregation['volume'] = 'sum'
        known = known.resample('W-FRI' if interval == 'Weekly' else 'ME').agg(aggregation)
        known = known.dropna(subset=['close', '_observed_end']).set_index('_observed_end')
        known.index.name = 'date'
    return _window(_indicators(known, overlays), cutoff, period)


def _figure(ticker: str, cutoff, period: str, interval: str, style: str,
            show_volume: bool, log_scale: bool, show_grid: bool, height: int) -> go.Figure:
    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.035,
                            row_heights=[.79, .21])
    else:
        fig = make_subplots(rows=1, cols=1)
    fig.update_layout(
        height=height, margin=dict(l=12, r=72, t=22, b=12),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(family='Arial, sans-serif', size=11, color='#b9c7c4'),
        legend=dict(orientation='h', x=0, y=1.03, xanchor='left', yanchor='bottom',
                    bgcolor='rgba(0,0,0,0)', font_size=11, itemclick='toggle', itemdoubleclick='toggleothers'),
        hoverlabel=dict(bgcolor='#1d2b34', bordercolor='#47605e', font=dict(color='#edf4ee', size=12)),
        hovermode='x', spikedistance=-1, hoverdistance=80, dragmode='pan',
        uirevision=f'{ticker}|{_cutoff(cutoff).date()}|{period}|{interval}|{style}',
        modebar=dict(bgcolor=BACKGROUND, color='#90a6a0', activecolor=UP_COLOR,
                     add=['drawline', 'drawrect', 'eraseshape']),
        newshape=dict(line_color='#ffc966', line_width=1.5),
    )
    fig.update_xaxes(
        type='date', showgrid=show_grid, gridcolor='#20312f', gridwidth=1,
        zeroline=False, showline=False, rangeslider_visible=False,
        showspikes=True, spikecolor='#819b94', spikethickness=1, spikedash='dot',
        spikesnap='cursor', spikemode='across', tickformat='%d %b\n%Y', nticks=7,
        tickfont_color='#9bb0a8', fixedrange=False,
    )
    if interval == 'Daily':
        fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])
    fig.update_yaxes(showgrid=show_grid, gridcolor='#263a34', zeroline=False,
                     showline=False, side='right', tickformat=',.2f', nticks=6,
                     showspikes=True, spikecolor='#819b94', spikethickness=1,
                     spikedash='dot', spikesnap='cursor', spikemode='across',
                     fixedrange=False, row=1, col=1, type='log' if log_scale else 'linear')
    if show_volume:
        fig.update_yaxes(showgrid=False, zeroline=False, side='right', tickformat='.3s',
                         nticks=3, title=dict(text='Volume', font_size=10), row=2, col=1)
    return fig


def _add_indicators(fig: go.Figure, frame: pd.DataFrame, overlays: Iterable[str]) -> None:
    for name in dict.fromkeys(overlays):
        fig.add_trace(go.Scatter(x=frame.index.to_pydatetime().tolist(), y=frame[name].tolist(), name=name, mode='lines',
            line=dict(color=INDICATOR_COLORS[name], width=1.4), connectgaps=False,
            hovertemplate=f'{name} %{{y:,.2f}}<extra></extra>'), row=1, col=1)


def _last_price(fig: go.Figure, frame: pd.DataFrame, color: str) -> None:
    if frame.empty:
        fig.add_annotation(text='No observed prices in this date range', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False, font_color='#b9c7c4')
        return
    last = float(frame['close'].iloc[-1])
    fig.add_shape(type='line', x0=0, x1=1, xref='paper', y0=last, y1=last,
                  yref='y', line=dict(color=color, width=.7, dash='dot'), layer='below')
    # Plotly annotation positions use log10 coordinates on a logarithmic axis;
    # trace and shape prices retain their ordinary observed-price units.
    badge_y = float(np.log10(last)) if fig.layout.yaxis.type == 'log' else last
    fig.add_annotation(x=1.007, xref='paper', y=badge_y, yref='y', text=f'{last:,.2f}',
                       showarrow=False, xanchor='left', bgcolor=color,
                       borderpad=4, font=dict(size=11, color=BACKGROUND))


def make_stock_chart(candles: pd.DataFrame, ticker: str, cutoff, period: str = '6M',
                     interval: str = 'Daily', style: str = 'Candles',
                     overlays: Iterable[str] = (), show_volume: bool = True,
                     log_scale: bool = False, show_grid: bool = True,
                     up_color: str = UP_COLOR, down_color: str = DOWN_COLOR,
                     height: int = 460) -> go.Figure:
    """Render actual observed OHLCV, never close-derived synthetic candles."""
    if style not in STYLES:
        raise ValueError('Choose Candles, OHLC, or Line for the chart style.')
    overlays = tuple(overlays)
    frame = prepare_chart_data(candles, cutoff, period, interval, overlays)
    if show_volume and 'volume' not in frame:
        raise ValueError('Trading volume was not supplied; turn off the volume panel.')
    fig = _figure(ticker, cutoff, period, interval, style, show_volume, log_scale, show_grid, height)
    # Supply lists at construction: newer Plotly encodes NumPy arrays as bdata,
    # which older embedded Plotly.js renderers cannot draw. Reassigning an
    # equivalent list later may be optimized away, retaining the typed array.
    dates = frame.index.to_pydatetime().tolist()
    if style == 'Line':
        fig.add_trace(go.Scatter(x=dates, y=frame['close'].tolist(), name=f'{ticker} close', mode='lines',
            line=dict(color=up_color, width=2), hovertemplate='%{x|%d %b %Y}<br>Close %{y:,.2f}<extra></extra>'), row=1, col=1)
    else:
        cls = go.Candlestick if style == 'Candles' else go.Ohlc
        fig.add_trace(cls(x=dates, open=frame['open'].tolist(), high=frame['high'].tolist(), low=frame['low'].tolist(),
            close=frame['close'].tolist(), name=ticker,
            increasing=dict(line=dict(color=up_color, width=1), **({'fillcolor': up_color} if style == 'Candles' else {})),
            decreasing=dict(line=dict(color=down_color, width=1), **({'fillcolor': down_color} if style == 'Candles' else {})),
            hovertemplate='%{x|%d %b %Y}<br>Open %{open:,.2f} · High %{high:,.2f}<br>Low %{low:,.2f} · Close %{close:,.2f}<extra></extra>'), row=1, col=1)
    _add_indicators(fig, frame, overlays)
    if show_volume:
        volume_colors = np.where(frame['close'] >= frame['open'], up_color, down_color)
        fig.add_trace(go.Bar(x=dates, y=frame['volume'].tolist(), marker_color=volume_colors.tolist(),
            opacity=.55, name='Observed volume', showlegend=False,
            hovertemplate='Volume %{y:,.0f}<extra></extra>'), row=2, col=1)
    color = up_color if frame.empty or frame['close'].iloc[-1] >= frame['open'].iloc[-1] else down_color
    _last_price(fig, frame, color)
    fig.update_layout(meta=dict(basis='Observed OHLCV; chart-only vendor prices', interval=interval,
                                indicator_units='Selected-interval bars', cutoff=str(_cutoff(cutoff).date())))
    return fig


def make_adjusted_line_chart(series: pd.Series, ticker: str, cutoff, period: str = '6M',
                            overlays: Iterable[str] = (), log_scale: bool = False,
                            show_grid: bool = True, up_color: str = UP_COLOR,
                            height: int = 460) -> go.Figure:
    """Honest adjusted-close-only fallback with no OHLC or volume fabrication."""
    overlays = tuple(overlays)
    frame = _dated(series.rename('close').to_frame())
    stamp = _cutoff(cutoff)
    frame = frame.loc[frame.index <= stamp]
    frame['close'] = pd.to_numeric(frame['close'], errors='coerce')
    if not np.isfinite(frame.to_numpy()).all() or (frame['close'] <= 0).any():
        raise ValueError('Adjusted close prices must be finite and positive.')
    frame = _window(_indicators(frame, overlays), stamp, period)
    fig = _figure(ticker, cutoff, period, 'Daily', 'Adjusted close', False, log_scale, show_grid, height)
    fig.add_trace(go.Scatter(x=frame.index.to_pydatetime().tolist(), y=frame['close'].tolist(), name='Adjusted close', mode='lines',
        line=dict(color=up_color, width=2), hovertemplate='%{x|%d %b %Y}<br>Adjusted close %{y:,.2f}<extra></extra>'), row=1, col=1)
    _add_indicators(fig, frame, overlays)
    _last_price(fig, frame, up_color)
    fig.update_layout(meta=dict(basis='Observed adjusted close only; no OHLCV available', cutoff=str(stamp.date())))
    return fig
