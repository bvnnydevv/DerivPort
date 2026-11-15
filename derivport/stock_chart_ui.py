"""Interactive chart workspace; chart bars never replace research inputs."""
from __future__ import annotations

import html
import pandas as pd
import streamlit as st

from derivport.market_data import MarketDataError
from derivport.ohlc import load_candles
from derivport.stock_chart import (PERIODS, INTERVALS, STYLES, OVERLAYS,
    make_stock_chart, make_adjusted_line_chart, prepare_chart_data)


@st.cache_data(show_spinner=False)
def cached_candles(symbol):
    return load_candles(symbol)


def render_stock_chart(adjusted_prices, ticker, cutoff, metadata):
    """Render cutoff-safe observed OHLCV, with an honest close-only fallback."""
    candles, provenance, unavailable = None, {}, None
    try:
        candles, provenance = cached_candles(ticker)
        if candles.loc[:cutoff].empty:
            unavailable = 'No candle observations are cached on or before this cutoff.'
            candles = None
    except MarketDataError as exc:
        unavailable = str(exc)

    with st.container(key='market_hero'):
        known = adjusted_prices.loc[:cutoff]
        daily = candles.loc[:cutoff] if candles is not None else None
        value = daily.close.iloc[-1] if daily is not None else known.iloc[-1]
        previous = (daily.close.iloc[-2] if len(daily) > 1 else None) if daily is not None else (known.iloc[-2] if len(known) > 1 else None)
        last_date = daily.index[-1] if daily is not None else known.index[-1]
        basis = 'Daily OHLC · not dividend-adjusted' if daily is not None else 'Adjusted close'
        move = '' if previous is None else f'{100 * (value / previous - 1):+.2f}% vs previous session'
        st.html(f'''<div class="stock-header"><div><h1>{html.escape(metadata.get('name', ticker))}</h1>
            <p>{html.escape(metadata.get('exchange', ''))} · {html.escape(metadata.get('currency', 'USD'))} · Historical observations</p></div>
            <div class="price">{value:,.2f}<small>{basis} · {last_date:%d %b %Y}<br>{move}</small></div></div>''')
        with st.container(key='price_toolbar'):
            range_col, interval_col, style_col, settings_col = st.columns([1.1, 1.1, 1.1, 1], vertical_alignment='bottom')
            with range_col:
                period = st.selectbox('Chart range', PERIODS, index=2, key='chart_period')
            with interval_col:
                interval = st.selectbox('Candle interval', INTERVALS, key='chart_interval', disabled=candles is None)
            with style_col:
                style = st.selectbox('Chart type', STYLES, key='chart_style', disabled=candles is None)
            with settings_col:
                with st.popover('Customize', width='stretch'):
                    st.markdown('**Chart appearance**')
                    overlays = st.multiselect('Indicators', OVERLAYS, default=['SMA 20'], key='chart_overlays', disabled=candles is None)
                    show_volume = st.checkbox('Volume', value=True, key='chart_volume', disabled=candles is None)
                    show_grid = st.checkbox('Grid lines', value=True, key='chart_grid')
                    log_scale = st.checkbox('Log price scale', value=False, key='chart_log')
                    up_color = st.color_picker('Up candles / line', '#34d399', key='chart_up')
                    down_color = st.color_picker('Down candles', '#fb7185', key='chart_down')
                    st.caption('Indicators use the selected daily, weekly or monthly bars. The last weekly/monthly bar can be partial at the cutoff.')
                    if st.button('Download / refresh candles', key='refresh_candles', width='stretch'):
                        try:
                            with st.spinner(f'Loading daily OHLCV for {ticker}…'):
                                _, refreshed = load_candles(ticker, refresh=True, allow_download=True)
                            cached_candles.clear()
                            st.session_state.candle_notice = refreshed.get('warning') or f'{ticker} candle history refreshed.'
                            st.rerun()
                        except MarketDataError as exc:
                            st.error(str(exc))
        notice = st.session_state.pop('candle_notice', None)
        if notice:
            st.caption(notice)
        if daily is not None:
            last = daily.iloc[-1]
            st.html('<div class="ohlc-strip">' + ''.join(
                f'<span><b>{label}</b> {value:,.2f}</span>' for label, value in
                [('O', last.open), ('H', last.high), ('L', last.low), ('C', last.close)])
                + f'<span><b>VOL</b> {last.volume:,.0f}</span></div>')
            fig = make_stock_chart(candles, ticker, cutoff, period=period, interval=interval,
                style=style, overlays=overlays, show_volume=show_volume, log_scale=log_scale,
                show_grid=show_grid, up_color=up_color, down_color=down_color, height=510)
            exported = prepare_chart_data(candles, cutoff, period=period, interval=interval, overlays=overlays)
            summary = (f'{style} · {interval.lower()} bars · {len(exported):,} bars · {exported.index[0]:%d %b %Y}–{exported.index[-1]:%d %b %Y}'
                if not exported.empty else 'No cached candle observations in the selected range. Choose a longer range or refresh candles.')
            st.session_state.chart_data = exported.copy()
            st.session_state.chart_basis = 'observed_ohlcv'
        else:
            fig = make_adjusted_line_chart(adjusted_prices, ticker, cutoff, period=period,
                log_scale=log_scale, show_grid=show_grid, up_color=up_color, height=450)
            exported = pd.DataFrame({'adjusted_close': fig.data[0].y}, index=pd.DatetimeIndex(fig.data[0].x, name='date'))
            st.session_state.chart_data = exported.copy()
            st.session_state.chart_basis = 'adjusted_close_only'
            summary = 'Adjusted-close line · no OHLC candles available for this cutoff'
        fig.update_layout(uirevision=f'{ticker}|{cutoff}|{period}|{interval}|{style}|{log_scale}',
            newshape=dict(line_color='#f2cc60', line_width=1.5), dragmode='pan')
        config = {
            'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True,
            # Line/rectangle/erase are already added by the figure's modebar.
            # Registering a name twice aborts Plotly's entire render.
            'modeBarButtonsToAdd': ['drawopenpath'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'editable': False, 'edits': {'shapePosition': True},
            'toImageButtonOptions': {'filename': f'{ticker}_{cutoff}', 'scale': 2}}
        st.plotly_chart(fig, theme=None, width='stretch', key='known_price_hero', config=config)
        st.caption(summary + ' · Drag to pan; scroll to zoom. Use the toolbar to draw, erase, reset or save an image.')
        if unavailable:
            st.caption(unavailable + ' Open Customize to download genuine daily candles. Close-only data is never turned into invented bars.')
        if candles is not None:
            st.caption(f"Yahoo Finance OHLCV · cached through {candles.index[-1]:%d %b %Y}; only observations through {cutoff} are shown. Research models continue to use dividend-adjusted closes.")
        st.download_button('Export displayed chart data', exported.rename_axis('date').to_csv().encode(),
            file_name=f'{ticker}_{interval.lower()}_{cutoff}.csv', mime='text/csv', key='export_chart_data')
