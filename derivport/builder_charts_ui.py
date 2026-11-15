"""Mirror the Research price chart alongside an honest draft-rule score view."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .algorithm_chart import history_score_chart
from .formula import FormulaError
from .stock_chart_ui import fullscreen_chart
from .theme import style_plot


def render_builder_charts(history, current_row, ticker, cutoff, context, formula, k):
    snapshot = st.session_state.get('stock_chart_snapshot')
    if not snapshot or (snapshot['ticker'], snapshot['cutoff'], snapshot['context']) != (ticker, cutoff, context):
        st.info('The selected stock chart is unavailable for this research view.')
        return
    stock_area, score_area = st.columns([1.55, 1], gap='large')
    with stock_area:
        st.markdown(f'#### {ticker} · stock chart')
        st.caption(f'Observed prices through {cutoff}. Range, indicators and colors follow the Research chart.')
        price_figure = go.Figure(snapshot['figure'])
        st.plotly_chart(price_figure, theme=None, width='stretch',
            key='builder_stock_chart', config=snapshot['config'])
        if st.button('Fullscreen stock chart', key='builder_open_fullscreen_chart'):
            fullscreen_chart(price_figure, snapshot['config'], ticker, cutoff)

    with score_area:
        st.markdown('#### Your algorithm · score over time')
        frame = snapshot['frame']
        start = frame.index[0] if not frame.empty else None
        try:
            figure, scores = history_score_chart(history, ticker, formula, k, cutoff,
                start=start, current_row=current_row)
            style_plot(figure)
            figure.update_layout(height=510, uirevision=f'builder-score|{ticker}|{cutoff}|{start}')
            st.session_state.builder_algorithm_frame = scores.copy(deep=True)
            st.session_state.builder_algorithm_formula = formula
            st.plotly_chart(figure, theme=None, width='stretch', key='builder_algorithm_chart',
                config={'displayModeBar': True, 'displaylogo': False})
            st.caption('Draft recipe replayed on the model inputs available at each historical forecast date. '
                'Higher scores rank higher; this curve is a score, not a stock price or investment return. '
                'Apply the recipe, then Run backtest below to compare performance.')
            if scores.attrs.get('undefined_points', 0):
                st.caption('Gaps mark dates where this recipe could not produce a finite score.')
        except (ValueError, FormulaError) as exc:
            st.info(f'The score curve is unavailable: {exc}')
