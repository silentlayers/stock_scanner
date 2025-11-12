"""
Market signal evaluation logic.
"""
from __future__ import annotations

import pandas as pd


def evaluate_signal(spy_close: pd.Series, ema21: pd.Series, macd_diff: pd.Series, rsi14: pd.Series, vix_close: pd.Series) -> bool:
    """Return simple boolean decision for deployment readiness.

    Optimal backtest configuration (empirically validated):
    - RSI > 55 (strong momentum confirmation)
    - VIX < 20 (low volatility regime)

    Backtest results show EMA21 and MACD filters provide NO additional edge
    and only reduce trade frequency without improving win rate or returns.

    Strategy: Bull call debit spread
    - Buy 0.40 delta call, sell 0.20 delta call
    - Target DTE: 30-45 days
    """
    latest_rsi = float(rsi14.iloc[-1])
    latest_vix_close = float(vix_close.iloc[-1])

    return (
        latest_rsi > 55 and
        latest_vix_close < 20
    )


def signal_diagnostics(spy_close: pd.Series, ema21: pd.Series, macd_diff: pd.Series, rsi14: pd.Series, vix_close: pd.Series) -> dict:
    """Return latest metrics and per-condition booleans for transparency.

    Note: Only RSI > 55 and VIX < 20 are used for entry signal.
    EMA21 and MACD are displayed for information but don't affect entry.
    """
    latest = {
        'close': float(spy_close.iloc[-1]),
        'ema21': float(ema21.iloc[-1]),
        'macd': float(macd_diff.iloc[-1]),
        'rsi': float(rsi14.iloc[-1]),
        'vix_close': float(vix_close.iloc[-1]),
    }
    checks = {
        'price_above_ema21': latest['close'] > latest['ema21'],  # Info only
        'macd_positive': latest['macd'] > 0,  # Info only
        'rsi_above_55': latest['rsi'] > 55,  # REQUIRED
        'vix_below_20': latest['vix_close'] < 20,  # REQUIRED
    }
    # Only check the required filters for signal
    all_ok = checks['rsi_above_55'] and checks['vix_below_20']
    return {'metrics': latest, 'checks': checks, 'all_ok': all_ok}
