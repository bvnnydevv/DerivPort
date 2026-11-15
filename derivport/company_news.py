"""Explicit, read-only company news searches, separate from historical research.

Uses the project's existing yfinance provider and its documented Search API:
https://ranaroussi.github.io/yfinance/reference/api/yfinance.Search.html
Only headlines and source links are retained; articles are never scraped.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import ipaddress
import re
from urllib.parse import quote, urlsplit, urlunsplit

from .market_data import MarketDataError, normalize_symbol

MAX_ARTICLES = 20
REQUEST_TIMEOUT = 8
NEWS_NOTICE = ('Latest provider search results, not a point-in-time news archive. '
    'Only titles mentioning the selected company name or ticker are shown; aliases and indirect stories may be missed. '
    'Publication timestamps are supplied by the provider. '
    'These headlines do not enter historical forecasts, rankings or the coach context.')


class CompanyNewsError(ValueError):
    """A safe, actionable provider or input error."""


@dataclass(frozen=True)
class NewsArticle:
    title: str
    publisher: str
    url: str
    published_at: datetime | None
    timestamp_precision: str
    related_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompanyNewsDigest:
    company: str
    symbol: str
    query: str
    fetched_at: datetime
    articles: tuple[NewsArticle, ...]
    source: str = 'Yahoo Finance search via yfinance'
    notice: str = NEWS_NOTICE

    @property
    def search_url(self):
        return f'https://finance.yahoo.com/quote/{quote(self.symbol, safe="")}/news/'


class _TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _text(value, maximum=300):
    if not isinstance(value, str):
        return ''
    parser = _TextOnly()
    parser.feed(value[:4000])
    return ' '.join(''.join(parser.parts).split())[:maximum]


def safe_article_url(value):
    """Allow public HTTPS article links only; never fetch their destinations."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname or ''
        if parts.scheme != 'https' or not host or parts.username or parts.password or parts.port not in (None, 443):
            return None
        if '.' not in host or host.endswith(('.local', '.localhost')) or host == 'localhost':
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        if any(ord(c) < 32 for c in value):
            return None
        return urlunsplit(parts)
    except (TypeError, ValueError):
        return None


def _published(value):
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, timezone.utc), 'second'
        if isinstance(value, str):
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                return datetime.fromisoformat(value).replace(tzinfo=timezone.utc), 'date'
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            # A missing timezone is not enough evidence for an exact event time.
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc), 'second'
    except (ValueError, TypeError, OverflowError, OSError):
        pass
    return None, 'unknown'


def normalize_articles(records, symbol):
    """Normalize old Search and current Ticker news schemas without inference."""
    if not isinstance(records, list):
        raise CompanyNewsError('The news provider returned an unexpected response. Try again later.')
    found, seen = [], set()
    for record in records[:100]:
        if not isinstance(record, dict):
            continue
        content = record.get('content') or record
        if not isinstance(content, dict):
            continue
        title = _text(content.get('title'))
        provider = content.get('provider') or {}
        publisher = _text(content.get('publisher') or (provider.get('displayName') if isinstance(provider, dict) else None), 120)
        canonical = content.get('canonicalUrl') or {}
        url = safe_article_url(content.get('link') or (canonical.get('url') if isinstance(canonical, dict) else None))
        if not title or not url or url in seen:
            continue
        related = content.get('relatedTickers') or record.get('relatedTickers') or []
        related = tuple(str(s).upper() for s in related[:40]) if isinstance(related, list) else ()
        # Explicit provider ticker tags can rule out an unrelated search hit.
        if related and symbol not in related:
            continue
        published, precision = _published(content.get('providerPublishTime') or content.get('pubDate') or content.get('displayTime'))
        found.append(NewsArticle(title, publisher or 'Publisher not supplied', url, published, precision, related))
        seen.add(url)
    floor = datetime.min.replace(tzinfo=timezone.utc)
    found.sort(key=lambda article: article.published_at or floor, reverse=True)
    return tuple(found[:MAX_ARTICLES])


_LEGAL_WORDS = frozenset(('the inc incorporated corp corporation company co ltd limited plc '
    'sa ag nv lp llc common ordinary preferred stock stocks share shares class adr ads '
    'american depositary depository representing a b c').split())
_GENERIC_NAME_WORDS = frozenset(('and of for international global holding holdings group '
    'financial finance services service investment investments capital technology technologies '
    'systems solutions industries industry resources energy partners trust bank bancorp '
    'united states general motor motors advanced micro devices platforms trade').split())
_AMBIGUOUS_SYMBOLS = frozenset(('A I IT ON FOR ARE ALL AM AN AS AT BE BY GO HE IN IS '
    'ME MY NO OF OR SO TO UP US WE').split())


def title_mentions_company(title, company, symbol):
    """Conservative lexical relevance, not provider tags or inferred article content.

    Require an explicit title mention of a distinctive company-name word, its
    core name phrase, or its ticker. Short ordinary-word tickers require a cash
    tag such as $ON. This can miss aliases; it cannot establish article relevance
    or sentiment beyond the literal mention.
    """
    title = _text(title)
    company = _text(company, 240).split(' - ')[0]
    title_words = re.findall(r'[\w]+', title.casefold())
    company_words = [word for word in re.findall(r'[\w]+', company.casefold())
        if word not in _LEGAL_WORDS]
    title_set = set(title_words)
    distinctive = [word for word in company_words if len(word) >= 4
        and word not in _GENERIC_NAME_WORDS and not word.isdigit()]
    if any(word in title_set for word in distinctive):
        return True
    if len(company_words) > 1 and f' {" ".join(company_words)} ' in f' {" ".join(title_words)} ':
        return True
    token = re.escape(symbol.upper())
    if re.search(rf'(?<![A-Za-z0-9])\${token}(?![A-Za-z0-9])', title, re.IGNORECASE):
        return True
    if symbol.upper() in _AMBIGUOUS_SYMBOLS:
        return False
    flags = re.IGNORECASE if len(symbol) >= 4 else 0
    return re.search(rf'(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])', title, flags) is not None


def _search(requested_query):
    import yfinance as yf
    return yf.Search(requested_query, max_results=0, news_count=MAX_ARTICLES,
        lists_count=0, recommended=0, include_cb=False, timeout=REQUEST_TIMEOUT,
        raise_errors=True).news


def fetch_company_news(company, symbol, query=''):
    """Fetch on explicit caller action only. No imports or cache reads fetch news."""
    try:
        symbol = normalize_symbol(symbol)
    except MarketDataError as exc:
        raise CompanyNewsError(str(exc)) from exc
    if not isinstance(company, str) or not company.strip() or len(company) > 240:
        raise CompanyNewsError('Choose a company from the stock search first.')
    if not isinstance(query, str) or len(query) > 120 or any(ord(c) < 32 for c in query):
        raise CompanyNewsError('Use a news search of up to 120 characters.')
    query = ' '.join(query.split())
    requested_query = f'{symbol} {query}'.strip()
    try:
        records = _search(requested_query)
    except Exception as exc:
        raise CompanyNewsError('Company news is unavailable from the provider right now. Try again later or open the company news source.') from exc
    articles = tuple(article for article in normalize_articles(records, symbol)
        if title_mentions_company(article.title, company, symbol))
    return CompanyNewsDigest(company=_text(company, 240), symbol=symbol, query=query,
        fetched_at=datetime.now(timezone.utc), articles=articles)
