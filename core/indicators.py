"""
Indicator computations using the 'ta' library.
"""
from __future__ import annotations

import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator


def compute_indicators(close: pd.Series) -> dict:
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    macd_diff = MACD(close).macd_diff()
    rsi14 = RSIIndicator(close, window=14).rsi()
    return {
        'ema21': ema21,
        'macd_diff': macd_diff,
        'rsi14': rsi14,
    }
