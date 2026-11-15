"""Explicitly download authentic daily OHLCV chart caches."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from derivport.ohlc import load_candles
from derivport.market_data import MarketDataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('symbols', nargs='+', help='US listing symbols, for example AAPL NVDA JPM')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    failures = 0
    for symbol in args.symbols:
        try:
            bars, meta = load_candles(symbol, refresh=args.refresh, allow_download=True)
            print(f'{symbol}: {len(bars):,} real daily candles, {meta["start"]} to {meta["end"]}, {meta["cache_status"]}')
            if meta.get('warning'):
                failures += 1
                print(meta['warning'], file=sys.stderr)
        except MarketDataError as exc:
            failures += 1
            print(str(exc), file=sys.stderr)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
