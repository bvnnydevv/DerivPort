"""News sources remain bounded, explicit, and outside historical evidence."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from derivport import company_news as news


def story(title='Apple announces update', url='https://example.com/story', timestamp=1735833600, tickers=None):
    return dict(title=title, publisher='Example News', link=url,
        providerPublishTime=timestamp, relatedTickers=['AAPL'] if tickers is None else tickers)


def test_normalization_keeps_sources_and_utc_without_sentiment_or_body():
    records = [story(), story(url='https://example.com/older', timestamp=1735747200),
        story(url='https://example.com/story'), story(url='https://example.com/unrelated', tickers=['MSFT'])]
    original = [dict(record) for record in records]
    articles = news.normalize_articles(records, 'AAPL')
    assert len(articles) == 2
    assert articles[0].title == 'Apple announces update'
    assert articles[0].publisher == 'Example News'
    assert articles[0].published_at == datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    assert articles[0].timestamp_precision == 'second'
    assert not hasattr(articles[0], 'sentiment')
    assert not hasattr(articles[0], 'body')
    assert records == original


def test_current_schema_and_date_only_news_do_not_fabricate_exact_times():
    articles = news.normalize_articles([{'content': dict(title='<b>Product release</b>',
        provider={'displayName': 'Issuer'}, canonicalUrl={'url':'https://example.com/release'}, pubDate='2025-01-03')},
        story(url='https://example.com/unknown', timestamp='not a date'),
        story(url='https://example.com/naive', timestamp='2025-01-03T12:00:00')], 'AAPL')
    assert articles[0].title == 'Product release'
    assert articles[0].timestamp_precision == 'date'
    assert articles[0].published_at.date().isoformat() == '2025-01-03'
    assert all(article.published_at is None and article.timestamp_precision == 'unknown' for article in articles[1:])


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'http://example.com/article',
    'https://localhost/x', 'https://127.0.0.1/x', 'https://user:pass@example.com/x',
    'https://example.local/x', 'https://example.com:8443/x', 'https://example.com/\nattack'])
def test_unsafe_article_links_are_dropped(url):
    assert news.normalize_articles([story(url=url)], 'AAPL') == ()


def test_response_bounds_and_missing_fields():
    articles = news.normalize_articles([story(url=f'https://example.com/{i}', timestamp=1735833600+i)
        for i in range(150)] + [None, {}, {'content':'wrong'}], 'AAPL')
    assert len(articles) == news.MAX_ARTICLES
    assert articles[0].url == 'https://example.com/99'
    assert news.normalize_articles([{}, None, {'content':'wrong'}, story(title='')], 'AAPL') == ()
    with pytest.raises(news.CompanyNewsError, match='unexpected'):
        news.normalize_articles({'not':'a list'}, 'AAPL')


def test_fetch_is_explicit_and_search_contract_has_timeout_and_no_extra_results(monkeypatch):
    import yfinance
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(news=[story()])
    monkeypatch.setattr(yfinance, 'Search', search)
    assert calls == []
    digest = news.fetch_company_news('Apple Inc.', 'aapl', 'earnings outlook')
    assert len(calls) == 1
    assert calls[0][0] == 'AAPL earnings outlook'
    assert calls[0][1]['timeout'] == 8
    assert calls[0][1]['news_count'] == 20
    assert calls[0][1]['max_results'] == calls[0][1]['lists_count'] == 0
    assert digest.symbol == 'AAPL'
    assert digest.fetched_at.tzinfo == timezone.utc
    assert 'not a point-in-time' in digest.notice
    assert 'historical forecasts' in digest.notice
    assert digest.search_url == 'https://finance.yahoo.com/quote/AAPL/news/'


def test_failures_return_no_fabricated_news_and_invalid_inputs_make_no_request(monkeypatch):
    calls = []
    def unavailable(query):
        calls.append(query)
        raise RuntimeError('private internal details')
    monkeypatch.setattr(news, '_search', unavailable)
    for args in [('Apple','AAPL','a'*121), ('Apple','AAPL','a\nb'), ('Apple','../../x',''), ('','AAPL','')]:
        with pytest.raises(news.CompanyNewsError):
            news.fetch_company_news(*args)
    assert calls == []
    with pytest.raises(news.CompanyNewsError, match='unavailable') as error:
        news.fetch_company_news('Apple', 'AAPL')
    assert 'private' not in str(error.value)
    assert calls == ['AAPL']


def test_empty_feed_is_an_empty_digest_not_a_generated_summary(monkeypatch):
    monkeypatch.setattr(news, '_search', lambda query: [])
    digest = news.fetch_company_news('Apple', 'AAPL')
    assert digest.articles == ()
    assert digest.source == 'Yahoo Finance search via yfinance'


def test_fetch_removes_broad_provider_tags_without_a_literal_company_title_mention(monkeypatch):
    titles = ['Qualcomm announces new chip', 'Warren Buffett changes Berkshire holdings',
        'Rich mom seeks a trustee', 'Apple announces a product update',
        'AAPL shares rise after earnings', 'Pineapple prices recover',
        'Company group stock class A update']
    monkeypatch.setattr(news, '_search', lambda query: [story(title=title,
        url=f'https://example.com/{i}') for i, title in enumerate(titles)])
    digest = news.fetch_company_news('Apple Inc. - Common Stock', 'AAPL')
    assert {article.title for article in digest.articles} == {
        'Apple announces a product update', 'AAPL shares rise after earnings'}
    assert 'aliases and indirect stories may be missed' in digest.notice


@pytest.mark.parametrize('title,company,symbol,expected', [
    ('NVIDIA launches a chip', 'NVIDIA Corporation - Common Stock', 'NVDA', True),
    ('QCOM reports earnings', 'QUALCOMM Incorporated', 'QCOM', True),
    ('An earnings update', 'QUALCOMM Incorporated', 'QCOM', False),
    ('General Motors launches model', 'General Motors Company', 'GM', True),
    ('AT&T faces new coverage', 'AT&T Inc.', 'T', True),
    ('Shares go up on news', 'ON Semiconductor Corporation', 'ON', False),
    ('$ON releases a statement', 'ON Semiconductor Corporation', 'ON', True),
    ('Company class A shares rise', 'Apple Inc. - Class A Common Stock', 'AAPL', False),
    ('Pineapple output rises', 'Apple Inc.', 'AAPL', False),
    ('META reports results', 'Meta Platforms, Inc.', 'META', True),
])
def test_conservative_company_name_and_ticker_mentions(title, company, symbol, expected):
    assert news.title_mentions_company(title, company, symbol) is expected
