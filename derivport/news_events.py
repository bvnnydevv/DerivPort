"""Observed price changes around a dated article or company announcement.

This is a descriptive event window, not a causal estimate or an abnormal-return
study. Unknown publication times use the first observed session strictly after
the publication date; the event's publication-day move is deliberately excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


HORIZONS = (1, 5, 21)
TIMING_RULE = 'First session after publication date'


@dataclass
class NewsEventAnalysis:
    publication_date: pd.Timestamp
    event_session: pd.Timestamp | None
    baseline_date: pd.Timestamp | None
    baseline_price: float | None
    returns: dict[int, float | None]
    outcome_dates: dict[int, pd.Timestamp | None]
    available_sessions: int
    window: pd.DataFrame
    metadata: dict


def _daily_date(value, label):
    if isinstance(value, (bool, int, float, np.number)) or value is None:
        raise ValueError(f'{label} must be a valid calendar date.')
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError('missing date')
        # Price indices and publication dates are calendar dates. No intraday
        # timing is inferred even if a caller passes a timestamp.
        return result.tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a valid calendar date.') from exc


def analyze_news_event(prices, event_date, cutoff=None, before=20, after=21):
    """Measure close-to-close changes over 1, 5 and 21 observed sessions.

    ``prices`` is a single stock's observed adjusted daily close Series. With a
    cutoff, later values are discarded before price validation or arithmetic.
    The baseline is the close immediately before the event session. Horizon 1
    is that event session's close; horizon 5 is its fifth observed session.
    Missing horizons stay ``None``. No dates or prices are interpolated.

    ``window`` has adjusted_close, return_from_baseline_pct and session_offset
    columns. Baseline offset is 0 and the first post-publication session is 1.
    No-baseline or pending-event windows retain observed prices but use missing
    relative returns/offsets where the anchor cannot be established.
    """
    if not isinstance(prices, pd.Series) or prices.empty:
        raise ValueError('Provide a stock’s observed adjusted daily close Series.')
    if pd.api.types.is_numeric_dtype(prices.index.dtype):
        raise ValueError('Observed prices need valid daily session dates, not numeric positions.')
    for size, label in ((before, 'Before'), (after, 'After')):
        if type(size) is not int or not 1 <= size <= 252:
            raise ValueError(f'{label} window must be between 1 and 252 sessions.')
    publication = _daily_date(event_date, 'Publication date')
    boundary = _daily_date(cutoff, 'Cutoff') if cutoff is not None else None
    try:
        dates = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Observed prices need valid daily session dates.') from exc
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Observed session dates must be valid, unique and increasing.')
    known = prices.copy()
    known.index = dates
    if boundary is not None:
        known = known.loc[known.index <= boundary]
    if known.empty:
        raise ValueError('No observed prices are available through the cutoff.')
    try:
        if pd.api.types.is_bool_dtype(known.dtype):
            raise ValueError('boolean prices')
        known = pd.to_numeric(known, errors='raise').astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Observed prices must be finite positive numbers.') from exc
    if not np.isfinite(known.to_numpy()).all() or (known <= 0).any():
        raise ValueError('Observed prices must be finite positive numbers; missing prices are not filled.')

    event_position = int(known.index.searchsorted(publication, side='right'))
    event_session = known.index[event_position] if event_position < len(known) else None
    baseline_position = event_position - 1
    baseline_date = known.index[baseline_position] if baseline_position >= 0 else None
    baseline_price = float(known.iloc[baseline_position]) if baseline_date is not None else None
    available = max(0, len(known) - event_position)
    if baseline_price is not None:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            relative_returns = (known / baseline_price - 1) * 100
        if not np.isfinite(relative_returns.to_numpy()).all():
            raise ValueError('These observed price ratios exceed the supported numerical range.')
    returns, outcome_dates = {}, {}
    for horizon in HORIZONS:
        position = event_position + horizon - 1
        complete = baseline_price is not None and position < len(known)
        returns[horizon] = float(relative_returns.iloc[position]) if complete else None
        outcome_dates[horizon] = known.index[position] if complete else None

    start = max(0, event_position - before)
    stop = min(len(known), event_position + after)
    window = known.iloc[start:stop].to_frame('adjusted_close')
    if baseline_price is not None:
        window['return_from_baseline_pct'] = relative_returns.iloc[start:stop]
    else:
        window['return_from_baseline_pct'] = np.nan
    window['session_offset'] = (np.arange(start, stop) - baseline_position) if event_session is not None and baseline_date is not None else pd.array([None] * len(window), dtype='Int64')
    window.index.name = 'date'
    status = 'pending_event_session' if event_session is None else ('insufficient_baseline' if baseline_price is None else ('complete' if available >= 21 else 'partial'))
    metadata = {
        'status': status,
        'timing_rule': TIMING_RULE,
        'publication_time_used': False,
        'known_through': known.index[-1].date().isoformat(),
        'requested_cutoff': boundary.date().isoformat() if boundary is not None else None,
        'price_basis': 'Observed adjusted daily close',
        'horizon_definition': 'Session 1 is the first observed session strictly after publication; all returns start at its preceding close.',
        'interpretation': 'Observed association only. The selected news does not establish the cause of a price move; market and other company events may contribute.',
        'timing_limit': 'Publication time is unknown or not used. The publication-day price move is excluded, and adjusted snapshots can reflect later corporate actions or revisions.',
        'missing_data': 'Missing horizons are pending or unavailable in this cache. No prices, dates or benchmark-relative returns are invented.',
    }
    return NewsEventAnalysis(publication, event_session, baseline_date, baseline_price,
        returns, outcome_dates, available, window, metadata)
