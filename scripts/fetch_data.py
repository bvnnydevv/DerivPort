#!/usr/bin/env python3
"""Rebuild the bundled historical cache from a pinned public Yahoo Finance mirror.

Not used by the app. Network access is needed only when running this script.
Run: python scripts/fetch_data.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '363151b27e8bf71b377483b37fa49ee5ebb3f5e2'
BASE = f'https://raw.githubusercontent.com/Akma86/us-tech-giants-market-analytics/{COMMIT}/data/raw'
TICKERS = ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
START, END = '2018-01-01', '2024-12-31'
EXPECTED = {
    'AAPL': '99044a83dc9cb1799c4fb7019e6bb07a7b100c6b5ab8ffaad4b007571a893059',
    'MSFT': 'c9e2e892c76bfd2a5c8dcf794b861ad074e711505aebaf1b388f5c69a87cc115',
    'NVDA': 'a168c7f15b74ade3b5b7ec3518d0f97c486a7a2914a7c8be837694998854e507',
    'AMZN': '6b03b9539aa4c2923c9ae53f5b6a49e4e8c829f14aafb124e171a1aeccc34a3f',
    'GOOGL': '75b48cddaca5a6c4fcd72894a7c4777d1d1569165da968aa37ee4c21b3ec4b1d',
}


def download(url: str) -> bytes:
    # certifi is bundled with Streamlit's HTTP dependencies; system roots are a fallback.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={'User-Agent': 'DerivPort educational data cache/1.0'})
    with urllib.request.urlopen(request, timeout=45, context=context) as response:
        return response.read()


def main() -> None:
    rows = []
    calendars = []
    for ticker in TICKERS:
        payload = download(f'{BASE}/{ticker}.csv')
        digest = hashlib.sha256(payload).hexdigest()
        if digest != EXPECTED[ticker]:
            raise RuntimeError(f'{ticker}: source checksum changed; cache not overwritten.')
        calendar = []
        for record in csv.DictReader(io.StringIO(payload.decode('utf-8'))):
            day = record['Date']
            if START <= day <= END:
                value = float(record['Adj Close'])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{ticker} has invalid adjusted close on {day}')
                calendar.append(day)
                rows.append((day, ticker, format(value, '.10f')))
        if len(calendar) != 1761 or len(set(calendar)) != len(calendar):
            raise RuntimeError(f'{ticker}: unexpected date coverage; cache not overwritten.')
        calendars.append(calendar)
        print(f'{ticker}: {len(calendar):,} validated adjusted daily prices')
    if any(calendar != calendars[0] for calendar in calendars):
        raise RuntimeError('Stock calendars differ; cache not overwritten.')
    rows.sort()
    target = ROOT / 'data' / 'prices.csv'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['date', 'ticker', 'adjusted_close'])
        writer.writerows(rows)
    temporary.replace(target)
    print(f'Saved {len(rows):,} rows to {target}')


if __name__ == '__main__':
    main()
