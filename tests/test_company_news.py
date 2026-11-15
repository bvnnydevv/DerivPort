"""News sources remain bounded, explicit, and outside historical evidence."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from derivport import company_news as news


@pytest.fixture(autouse=True)
def fixed_news_clock(monkeypatch):
    monkeypatch.setattr(news, '_utc_now', lambda: datetime(2025, 1, 3, 16, tzinfo=timezone.utc))


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
    assert calls[0][1]['news_count'] == 100
    assert calls[0][1]['max_results'] == calls[0][1]['lists_count'] == 0
    assert digest.symbol == 'AAPL'
    assert digest.fetched_at.tzinfo == timezone.utc
    assert 'not a point-in-time' in digest.notice
    assert 'historical forecasts' in digest.notice
    assert digest.search_url == 'https://finance.yahoo.com/quote/AAPL/news/'
    assert digest.window_end == digest.fetched_at
    assert digest.window_end - digest.window_start == timedelta(days=7)
    assert digest.fetched_count == 1


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


def test_seven_day_window_includes_exact_boundaries_and_excludes_future_old_unknown_dates():
    end = datetime(2025, 1, 10, 16, tzinfo=timezone.utc)
    start = end - timedelta(days=7)
    def article(name, published, precision='second'):
        return news.NewsArticle(name, 'Issuer', 'https://example.com/' + name, published, precision)
    items = [article('start', start), article('end', end),
        article('old', start - timedelta(seconds=1)), article('future', end + timedelta(seconds=1)),
        article('unknown', None, 'unknown'),
        article('local', datetime(2025, 1, 3, 11, tzinfo=timezone(timedelta(hours=-5)))),
        article('date-only', datetime(2025, 1, 5, tzinfo=timezone.utc), 'date')]
    selected = news.articles_in_window(items, end)
    assert [article.title for article in selected] == ['start', 'end', 'local', 'date-only']
    assert selected[-1].timestamp_precision == 'date'
    assert selected[-1].published_at.hour == 0


def test_fetch_applies_date_window_to_real_provider_records_and_discloses_coverage(monkeypatch):
    now = news._utc_now()
    records = [story(title='Apple current update', timestamp=now.timestamp()),
        story(title='Apple older report', url='https://example.com/old', timestamp=(now - timedelta(days=8)).timestamp()),
        story(title='Apple future report', url='https://example.com/future', timestamp=(now + timedelta(hours=1)).timestamp()),
        story(title='Apple undated report', url='https://example.com/unknown', timestamp='unknown')]
    monkeypatch.setattr(news, '_search', lambda query: records)
    digest = news.fetch_company_news('Apple Inc.', 'AAPL')
    assert [article.title for article in digest.articles] == ['Apple current update']
    assert digest.fetched_count == 4
    assert 'coverage can be incomplete' in digest.notice
