"""Current news is explicit and event-price views respect the research cutoff."""
from datetime import datetime, timezone
import json

from streamlit.testing.v1 import AppTest

from derivport.company_news import CompanyNewsDigest, CompanyNewsError, NewsArticle
import derivport.company_news_ui as news_ui


APP = '''
import pandas as pd
import streamlit as st
from derivport.company_news_ui import render_company_news
symbol=st.selectbox('Test stock', ['AAPL','MSFT'], key='test_symbol')
cutoff=st.selectbox('Test cutoff',['2024-02-15','2024-01-31'], key='test_cutoff')
index=pd.bdate_range('2024-01-02','2024-04-30')
prices=pd.Series([100+i for i in range(len(index))], index=index)
# Later outcomes exist in the cache; the event view must discard them.
prices.loc[prices.index > cutoff] = 99999999.
render_company_news('Apple' if symbol=='AAPL' else 'Microsoft',symbol,prices,cutoff)
'''


def digest(symbol='AAPL', query=''):
    articles=tuple(NewsArticle(f'Headline {i} <script>alert(1)</script>', 'Example News',
        f'https://example.com/news/{i}', datetime(2024,3,1,tzinfo=timezone.utc), 'second', (symbol,))
        for i in range(8))
    return CompanyNewsDigest('Apple',symbol,query,datetime(2024,3,2,12,tzinfo=timezone.utc),articles)


def update_button(app):
    return next(button for button in app.button if button.label == 'Update news')


def test_news_never_fetches_on_load_or_typing_only_explicit_update(monkeypatch):
    calls=[]
    def fetch(company,symbol,query=''):
        calls.append((company,symbol,query))
        return digest(symbol,query)
    monkeypatch.setattr(news_ui,'fetch_company_news',fetch)
    app=AppTest.from_string(APP,default_timeout=20).run()
    assert not app.exception
    assert not calls
    app.text_input(key='company_news_topic').set_value('earnings').run()
    assert not calls
    # Form fields are sent together with the explicit submit action.
    app.text_input(key='company_news_topic').set_value('earnings')
    update_button(app).click().run()
    assert not app.exception
    assert calls==[('Apple','AAPL','earnings')]
    assert app.session_state['company_news_digest'].query=='earnings'
    assert len(app.get('html'))==6
    html=' '.join(element.proto.body for element in app.get('html'))
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert '01 Mar 2024, 00:00 UTC' in html
    assert any('Updated 02 Mar 2024, 12:00 UTC' in item.value for item in app.caption)
    app.run()
    assert len(calls)==1
    app.selectbox(key='test_symbol').set_value('MSFT').run()
    assert not app.exception
    assert 'company_news_digest' not in app.session_state
    assert not app.get('html')
    assert len(calls)==1


def test_provider_failure_retains_timestamped_previous_results(monkeypatch):
    monkeypatch.setattr(news_ui,'fetch_company_news',lambda *args: digest())
    app=AppTest.from_string(APP,default_timeout=20).run()
    update_button(app).click().run()
    def unavailable(*args):
        raise CompanyNewsError('News provider unavailable. Try later.')
    monkeypatch.setattr(news_ui,'fetch_company_news',unavailable)
    update_button(app).click().run()
    assert not app.exception
    assert any('News provider unavailable' in item.value for item in app.info)
    assert app.session_state['company_news_digest'].fetched_at.year==2024
    assert len(app.get('html'))==6


def test_event_analysis_filters_later_prices_and_resets_when_cutoff_changes(monkeypatch):
    monkeypatch.setattr(news_ui,'fetch_company_news',lambda *args: digest())
    app=AppTest.from_string(APP,default_timeout=20).run()
    app.date_input(key='news_event_date_AAPL').set_value(datetime(2024,1,15).date()).run()
    assert 'news_event_result' not in app.session_state
    app.button(key='news_event_run_AAPL').click().run()
    assert not app.exception
    _,result=app.session_state['news_event_result']
    assert result.metadata['known_through']=='2024-02-15'
    assert result.window.index.max().date()<=datetime(2024,2,15).date()
    assert max(result.window.adjusted_close)<99999999
    assert all(item.value!='Pending / unavailable' for item in app.metric)
    chart=json.loads(app.get('plotly_chart')[0].proto.spec)
    assert chart['layout']['meta']['kind']=='observed_news_event'
    assert max(chart['data'][0]['y'])<100
    app.selectbox(key='test_cutoff').set_value('2024-01-31').run()
    assert not app.get('plotly_chart'), 'Old results must not persist under another cutoff.'


def test_current_article_after_research_cutoff_has_pending_outcomes(monkeypatch):
    monkeypatch.setattr(news_ui,'fetch_company_news',lambda *args: digest())
    app=AppTest.from_string(APP,default_timeout=20).run()
    update_button(app).click().run()
    app.selectbox(key='news_event_choice_AAPL').set_value(0).run()
    app.button(key='news_event_run_AAPL').click().run()
    assert not app.exception
    _,result=app.session_state['news_event_result']
    assert result.event_session is None
    assert all(value is None for value in result.returns.values())
    assert not app.get('plotly_chart'), 'A pending future event must not show an outcome chart.'
    assert [metric.value for metric in app.metric]==['Pending / unavailable']*3
    assert any('outcome is pending' in item.value for item in app.info)
    assert any('Latest news never enters historical forecasts' in item.value for item in app.caption)


def test_precise_article_instant_uses_new_york_event_calendar(monkeypatch):
    article=NewsArticle('After-hours announcement', 'Example News', 'https://example.com/after-hours',
        datetime(2024,2,15,1,tzinfo=timezone.utc), 'second', ('AAPL',))
    supplied=CompanyNewsDigest('Apple','AAPL','',datetime(2024,2,16,tzinfo=timezone.utc),(article,))
    monkeypatch.setattr(news_ui,'fetch_company_news',lambda *args: supplied)
    app=AppTest.from_string(APP,default_timeout=20).run()
    update_button(app).click().run()
    app.selectbox(key='news_event_choice_AAPL').set_value(0).run()
    app.button(key='news_event_run_AAPL').click().run()
    assert not app.exception
    _,result=app.session_state['news_event_result']
    assert result.publication_date.date()==datetime(2024,2,14).date()
    assert result.event_session.date()==datetime(2024,2,15).date()
    assert result.returns[1] is not None and result.returns[5] is None
    # A date-only report does not carry an instant/timezone to convert.
    date_only=NewsArticle('Date-only report','Example News','https://example.com/date-only',
        datetime(2024,2,15,tzinfo=timezone.utc),'date',('AAPL',))
    supplied=CompanyNewsDigest('Apple','AAPL','',datetime(2024,2,16,tzinfo=timezone.utc),(date_only,))
    monkeypatch.setattr(news_ui,'fetch_company_news',lambda *args: supplied)
    update_button(app).click().run()
    app.button(key='news_event_run_AAPL').click().run()
    assert not app.exception
    _,result=app.session_state['news_event_result']
    assert result.publication_date.date()==datetime(2024,2,15).date()
    assert result.event_session is None
