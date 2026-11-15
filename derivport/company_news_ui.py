"""Explicit current-news reading and cutoff-bound observed event windows."""
from __future__ import annotations

from hashlib import sha256
from html import escape
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .company_news import CompanyNewsError, fetch_company_news, safe_article_url
from .news_events import analyze_news_event
from .coach import strip_emojis


MAX_VISIBLE_HEADLINES = 10
MAX_DISPLAY_HEADLINES = 50


def _stock_prices(prices, symbol):
    if isinstance(prices, pd.DataFrame):
        if symbol not in prices:
            raise ValueError('Adjusted price history is unavailable for this stock.')
        return prices[symbol].dropna()
    if isinstance(prices, pd.Series):
        return prices.dropna()
    raise ValueError('Adjusted price history is unavailable for this stock.')


def _date_label(article):
    if article.published_at is None:
        return 'Publication date not supplied'
    if article.timestamp_precision == 'date':
        return article.published_at.strftime('%d %b %Y') + ' · date only'
    return article.published_at.strftime('%d %b %Y, %H:%M UTC')


def _render_headline(article):
    url = safe_article_url(article.url)
    if url is None:
        return
    st.html('<article class="news-item"><a href="' + escape(url, quote=True)
        + '" target="_blank" rel="noopener noreferrer">' + escape(strip_emojis(article.title))
        + '</a><p>' + escape(strip_emojis(article.publisher)) + ' · ' + escape(_date_label(article))
        + '</p></article>')


def _event_date(article):
    # Exact publication instants use the US session calendar; date-only sources
    # retain their reported date because their timezone is unknown.
    if article.timestamp_precision == 'second':
        return article.published_at.astimezone(ZoneInfo('America/New_York')).date()
    return article.published_at.date()


def _event_chart(result):
    frame = result.window
    if result.event_session is None or result.baseline_date is None or frame.empty or frame['return_from_baseline_pct'].isna().all():
        return None
    fig = go.Figure(go.Scatter(x=frame.index.to_pydatetime().tolist(),
        y=frame['return_from_baseline_pct'].tolist(), mode='lines',
        line=dict(color='#b9a4e8', width=2.5), name='Observed adjusted-price change',
        hovertemplate='%{x|%d %b %Y}<br>%{y:+.2f}% from baseline<extra></extra>'))
    fig.add_hline(y=0, line=dict(color='#8d9690', width=1, dash='dot'))
    if result.event_session is not None:
        fig.add_shape(type='line', x0=result.event_session, x1=result.event_session,
            y0=0, y1=1, yref='paper', line=dict(color='#ff633d', width=1.5, dash='dash'))
    fig.update_layout(height=280, margin=dict(l=50, r=18, t=18, b=45),
        font=dict(family='Helvetica Neue, Helvetica, Arial, sans-serif'),
        showlegend=False, hovermode='x',
        meta={'kind': 'observed_news_event', **result.metadata})
    fig.update_xaxes(title='Observed session date', automargin=True)
    fig.update_yaxes(title='Change from baseline (%)', ticksuffix='%', automargin=True)
    return fig


def _render_event_analysis(symbol, prices, cutoff, articles, plotstyle):
    with st.expander('Price move around an event'):
        st.caption(f'Observed prices through {pd.Timestamp(cutoff):%d %b %Y}. '
            'Current headlines can be newer than this research cutoff.')
        dated = [article for article in articles if article.published_at is not None]
        choices = ['manual'] + list(range(len(dated)))
        choice_key = f'news_event_choice_{symbol}'
        if st.session_state.get(choice_key, 'manual') not in choices:
            st.session_state[choice_key] = 'manual'
        selected = st.selectbox('Choose an event', choices,
            format_func=lambda choice: 'Enter a date' if choice == 'manual' else strip_emojis(dated[choice].title),
            key=choice_key)
        if selected == 'manual':
            event_date = st.date_input('Event publication date', value=pd.Timestamp(cutoff).date(),
                key=f'news_event_date_{symbol}')
            label = st.text_input('Event label', placeholder='Earnings release, product launch…',
                max_chars=160, key=f'news_event_label_{symbol}') or 'Selected event'
        else:
            article = dated[selected]
            event_date, label = _event_date(article), article.title
            st.caption(f'{_date_label(article)} · {strip_emojis(article.publisher)}')
            calendar = 'New York calendar date' if article.timestamp_precision == 'second' else 'Reported publication date'
            st.caption(f'{calendar}: {event_date:%d %b %Y}.')
        st.caption('Timing: use the first observed session after the publication date. '
            'The baseline is its preceding adjusted close; publication-day moves are excluded.')
        try:
            series = _stock_prices(prices, symbol)
            # Identity includes actual input bytes: results cannot survive a changed cache.
            fingerprint = sha256(pd.util.hash_pandas_object(series, index=True).values.tobytes()).hexdigest()
            identity = (symbol, str(pd.Timestamp(cutoff)), str(event_date), label, fingerprint)
        except (ValueError, TypeError, KeyError) as exc:
            st.info(str(exc))
            return
        if st.button('Measure price move', key=f'news_event_run_{symbol}'):
            try:
                result = analyze_news_event(series, event_date, cutoff=cutoff)
                st.session_state.news_event_result = (identity, result)
                st.session_state.pop('news_event_error', None)
            except ValueError as exc:
                st.session_state.news_event_error = str(exc)
                st.session_state.pop('news_event_result', None)
        if st.session_state.get('news_event_error'):
            st.info(st.session_state.news_event_error)
        saved = st.session_state.get('news_event_result')
        if not saved or saved[0] != identity:
            return
        result = saved[1]
        st.markdown('**Observed change after the event**')
        columns = st.columns(3)
        for column, horizon in zip(columns, (1, 5, 21)):
            value = result.returns[horizon]
            with column:
                st.metric(f'+{horizon} session' + ('s' if horizon != 1 else ''),
                    'Pending / unavailable' if value is None else f'{value:+.2f}%')
        if result.baseline_date is not None:
            st.caption(f'Baseline close: {result.baseline_date:%d %b %Y}. '
                f'Price cache available through {result.metadata["known_through"]}.')
        figure = _event_chart(result)
        if figure is not None:
            if plotstyle is not None:
                figure = plotstyle(figure)
            st.plotly_chart(figure, width='stretch', theme=None,
                config={'displaylogo': False}, key=f'news_event_chart_{symbol}')
        if result.event_session is None:
            st.info('No post-publication session is available through this cutoff. The event outcome is pending.')
        elif result.baseline_date is None:
            st.info('The cache has no preceding close for this event. Relative returns are unavailable.')
        st.caption('Observed association, not proof that the news caused the move. '
            'Other company events and market changes may contribute. No benchmark-relative return is inferred.')


def render_company_news(company, symbol, prices, cutoff, plotstyle=None):
    """No news requests run until Update news is submitted by the user."""
    company = str(company or symbol)
    symbol = str(symbol)
    identity = (company, symbol)
    if st.session_state.get('company_news_identity') != identity:
        st.session_state.company_news_identity = identity
        for key in ('company_news_digest', 'company_news_error', 'news_event_result', 'news_event_error'):
            st.session_state.pop(key, None)
        st.session_state.company_news_topic = ''
    with st.container(border=True, key='company_news_panel'):
        st.markdown('### Company news · Past 7 days')
        st.caption(f'{company} · latest provider headlines, separate from historical research.')
        with st.form('company_news_search', border=False):
            query_column, update_column = st.columns([4, 1.2], vertical_alignment='bottom')
            with query_column:
                query = st.text_input('Search company news', placeholder='Optional topic: earnings, products, leadership…',
                    key='company_news_topic', max_chars=120, label_visibility='collapsed')
            with update_column:
                update = st.form_submit_button('Update news', width='stretch')
        if update:
            try:
                with st.spinner('Finding company headlines…'):
                    digest = fetch_company_news(company, symbol, query)
                st.session_state.company_news_digest = digest
                st.session_state.pop('company_news_error', None)
            except CompanyNewsError as exc:
                st.session_state.company_news_error = str(exc)
        error = st.session_state.get('company_news_error')
        if error:
            st.info(error)
            st.link_button('Open provider news', f'https://finance.yahoo.com/quote/{quote(symbol, safe="")}/news/')
        digest = st.session_state.get('company_news_digest')
        articles = () if digest is None else digest.articles
        if digest is not None:
            publishers = {article.publisher for article in articles}
            dated = [article.published_at for article in articles if article.published_at is not None]
            oldest = f' · oldest supplied {min(dated):%d %b %Y}' if dated else ''
            st.caption(f'{len(articles)} headlines · {len(publishers)} publishers{oldest}')
            st.caption(f'Updated {digest.fetched_at:%d %b %Y, %H:%M UTC} · {digest.source}')
            window_start = getattr(digest, 'window_start', None)
            window_end = getattr(digest, 'window_end', None)
            if window_start is not None and window_end is not None:
                st.caption(f'7-day snapshot: {window_start:%d %b %Y, %H:%M}–{window_end:%d %b %Y, %H:%M UTC}.')
            if len(articles) > MAX_DISPLAY_HEADLINES:
                st.caption(f'Showing the {MAX_DISPLAY_HEADLINES} most recent of {len(articles)} supplied headlines.')
            if digest.query:
                st.caption(f'Search topic: {digest.query}')
            if not articles:
                st.info('No matching headlines were supplied for the past 7 days. Try another topic or update later.')
            for article in articles[:MAX_VISIBLE_HEADLINES]:
                _render_headline(article)
            more = articles[MAX_VISIBLE_HEADLINES:MAX_DISPLAY_HEADLINES]
            if more:
                with st.expander(f'More headlines ({len(more)})'):
                    for article in more:
                        _render_headline(article)
            st.caption('A recent provider feed, not a complete 7-day news archive. Available coverage varies by company.')
            st.caption('Only headlines mentioning the company name or ticker are shown; aliases may be missed. Headlines, not AI summaries. '
                'Latest news never enters historical forecasts, rankings or the coach context.')
        else:
            st.caption('Load a 7-day headline snapshot with Update news. Coverage depends on the provider.')
        _render_event_analysis(symbol, prices, cutoff, articles, plotstyle)
