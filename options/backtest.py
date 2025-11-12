"""
Local backtesting for the SPY signal using a simplified options model.

Approach:
- Use yfinance daily SPY and VIX series already in the app
- Generate the daily boolean signal (Close>EMA21, MACD>0, RSI>50, VIX<20)
- On a signal day with no open position, sell a put credit spread with target deltas
  (sell ~ -0.50, buy ~ -0.25) with DTE ~ 45 calendar days.
- Approximate strikes by inverting Black-Scholes delta, using VIX/100 as annualized IV.
- Approximate credit using Black-Scholes put prices.
- Settle PnL at expiration using underlying close S_T (no early exits/slippage).

Notes:
- This is a rough, model-based backtest. It does not use historical options chains.
- Using VIX as IV proxy is a simplification; results are indicative, not precise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import pandas as pd

from core.indicators import compute_indicators


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse CDF for standard normal. Acklam's approximation.
    Accurate enough for delta inversion. p must be in (0,1).
    """
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
        (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1))


def _black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Return (price, delta) for a European put under Black-Scholes.
    S: spot, K: strike, T: years, r: rate, sigma: vol
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return (max(K - S, 0.0) * math.exp(-r*T), -1.0 if S < K else 0.0)
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    Nd1 = _norm_cdf(-d1)  # for put
    Nd2 = _norm_cdf(-d2)
    price = K * math.exp(-r * T) * Nd2 - S * Nd1
    delta = Nd1 - 1.0  # put delta, negative
    return price, delta


def _black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Return (price, delta) for a European call under Black-Scholes.
    S: spot, K: strike, T: years, r: rate, sigma: vol
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        # Intrinsic value for call; delta ~ 1 if ITM else 0 (approx)
        return (max(S - K, 0.0), 1.0 if S > K else 0.0)
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vsqrt
    d2 = d1 - vsqrt
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    price = S * Nd1 - K * math.exp(-r * T) * Nd2
    delta = Nd1
    return price, delta


def _strike_for_put_delta(S: float, T: float, r: float, sigma: float, target_delta: float) -> float:
    """Solve for strike K that yields a put delta ~= target_delta using BS.
    For a put: delta = N(d1) - 1 -> N(d1) = delta + 1 -> d1 = Phi^{-1}(delta+1)
    Then ln(S/K) = d1*sigma*sqrt(T) - (r + 0.5*sigma^2)T
    -> K = S * exp(-(d1*sigma*sqrt(T) - (r + 0.5*sigma^2)T))
    """
    p = target_delta + 1.0
    p = min(max(p, 1e-6), 1 - 1e-6)  # clamp
    d1 = _norm_ppf(p)
    vsqrt = sigma * math.sqrt(T)
    ln_S_over_K = d1 * vsqrt - (r + 0.5 * sigma * sigma) * T
    K = S * math.exp(-ln_S_over_K)
    return max(0.01, K)


def _strike_for_call_delta(S: float, T: float, r: float, sigma: float, target_delta: float) -> float:
    """Solve for strike K that yields a call delta ~= target_delta using BS.
    For a call: delta = N(d1) -> d1 = Phi^{-1}(delta)
    Then ln(S/K) = d1*sigma*sqrt(T) - (r + 0.5*sigma^2)T
    -> K = S * exp(-(d1*sigma*sqrt(T) - (r + 0.5*sigma^2)T))
    """
    p = min(max(target_delta, 1e-6), 1.0 - 1e-6)
    d1 = _norm_ppf(p)
    vsqrt = sigma * math.sqrt(T)
    ln_S_over_K = d1 * vsqrt - (r + 0.5 * sigma * sigma) * T
    K = S * math.exp(-ln_S_over_K)
    return max(0.01, K)


def generate_signal_series(spy_close: pd.Series, vix_close: pd.Series) -> pd.Series:
    """Vectorized daily signal series: True when all conditions are met."""
    inds = compute_indicators(spy_close)
    ema21 = inds['ema21']
    macd_diff = inds['macd_diff']
    rsi14 = inds['rsi14']
    # Align indices
    common_idx = spy_close.index.intersection(vix_close.index)
    spy_c = spy_close.reindex(common_idx)
    ema_c = ema21.reindex(common_idx)
    macd_c = macd_diff.reindex(common_idx)
    rsi_c = rsi14.reindex(common_idx)
    vix_c = vix_close.reindex(common_idx)
    sig = (spy_c > ema_c) & (macd_c > 0) & (rsi_c > 50) & (vix_c < 20)
    # Drop initial NaNs where EMA/MACD need warmup
    sig = sig & ema_c.notna() & macd_c.notna() & rsi_c.notna() & vix_c.notna()
    return sig.astype(bool)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    S_entry: float
    VIX_entry: float
    K_sell: float
    K_buy: float
    credit: float
    max_loss: float
    S_expiry: float
    pnl: float


def run_backtest(
    spy_close: pd.Series,
    vix_close: pd.Series,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    dte_days: int = 45,
    dte_min_days: Optional[int] = None,
    dte_max_days: Optional[int] = None,
    sell_delta: float = -0.50,
    buy_delta: float = -0.25,
    r_annual: float = 0.02,
    iv_scale: float = 1.0,
    # VIX regime filters
    vix_threshold: float = 18.0,
    vix_ma_window: int = 5,
    require_vix_ma_down: bool = True,
    avoid_vix_spike: bool = True,
    vix_spike_pct: float = 0.05,
    # Entry timing filters
    rsi_threshold: float = 55.0,
    require_ema21_slope_up: bool = False,
    require_reclaim_ema21: bool = False,
    reclaim_lookback_days: int = 2,
    require_price_above_ema21: bool = False,  # New: Price > EMA21 check
    require_macd_positive: bool = False,  # New: MACD > 0 check
    allowed_weekdays: Optional[List[int]] = None,  # e.g., [1,2,3] for Tue-Thu
    avoid_big_up_day: bool = False,
    big_up_day_pct: float = 0.015,
    # Optional exit strategy (disabled by default)
    # e.g., 0.5 for +50% of credit
    take_profit_pct_of_credit: Optional[float] = None,
    # e.g., 1.0 for -100% of credit
    stop_loss_pct_of_credit: Optional[float] = None,
    # e.g., 10 to close after 10 trading days
    exit_after_days_in_trade: Optional[int] = None,
    # e.g., 21 to close at 21 DTE
    exit_at_days_to_expiration: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """Simulate non-overlapping put credit spreads on signal days.

    Returns: (trades_df, equity_curve, summary)
    """
    spy_close = spy_close.sort_index()
    vix_close = vix_close.sort_index()
    if start is not None:
        spy_close = spy_close[spy_close.index >= start]
        vix_close = vix_close[vix_close.index >= start]
    if end is not None:
        spy_close = spy_close[spy_close.index <= end]
        vix_close = vix_close[vix_close.index <= end]

    signal = generate_signal_series(spy_close, vix_close)
    if signal.empty:
        return pd.DataFrame(), pd.Series(dtype=float), {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'total_pnl': 0.0
        }

    dates = signal.index
    # Precompute VIX moving average and prior day for slope check
    vix_ma = vix_close.sort_index().rolling(vix_ma_window, min_periods=vix_ma_window).mean(
    ) if vix_ma_window and vix_ma_window > 1 else None
    vix_ma_prev = vix_ma.shift(1) if vix_ma is not None else None
    # Precompute indicators for timing filters
    inds_all = compute_indicators(spy_close)
    ema21 = inds_all['ema21']
    ema21_prev = ema21.shift(1)
    rsi14 = inds_all['rsi14']
    macd_diff = inds_all['macd_diff']  # Add MACD for optional filter
    # Daily returns for big up-day filter
    ret = spy_close.pct_change()

    # Debug: Print filter configuration
    print(
        f"🔍 BACKTEST FILTERS: require_price_above_ema21={require_price_above_ema21}, require_macd_positive={require_macd_positive}, require_ema21_slope_up={require_ema21_slope_up}")

    # Default allowed weekdays to Tue-Thu if not provided
    if allowed_weekdays is None:
        allowed_weekdays = [1, 2, 3]
    trades: List[Trade] = []

    for dt in dates:
        if not signal.loc[dt]:
            continue
        # Removed non-overlapping restriction to allow multiple concurrent trades
        # New: VIX regime filters
        try:
            vix_today = float(vix_close.reindex([dt]).iloc[0])
        except Exception:
            continue
        if not (vix_today < float(vix_threshold)):
            continue
        if require_vix_ma_down and vix_ma is not None and vix_ma_prev is not None:
            v_ma = vix_ma.reindex([dt]).iloc[0]
            v_ma_prev = vix_ma_prev.reindex([dt]).iloc[0]
            # if MA not available yet, skip
            if pd.isna(v_ma) or pd.isna(v_ma_prev) or not (v_ma < v_ma_prev):
                continue
        if avoid_vix_spike and vix_ma is not None:
            v_ma = vix_ma.reindex([dt]).iloc[0]
            if pd.isna(v_ma) or vix_today > (1.0 + float(vix_spike_pct)) * float(v_ma):
                continue
        # Entry timing filters
        # Weekday filter
        if allowed_weekdays and (dt.weekday() not in set(allowed_weekdays)):
            continue
        # RSI threshold (stricter than base signal)
        rsi_val = rsi14.reindex([dt]).iloc[0]
        if pd.isna(rsi_val) or not (float(rsi_val) > float(rsi_threshold)):
            continue

        # New: Price > EMA21 check
        if require_price_above_ema21:
            e_today = ema21.reindex([dt]).iloc[0]
            if pd.isna(e_today) or not (float(spy_close.loc[dt]) > float(e_today)):
                continue

        # New: MACD > 0 check
        if require_macd_positive:
            macd_val = macd_diff.reindex([dt]).iloc[0]
            if pd.isna(macd_val) or not (float(macd_val) > 0):
                continue

        # EMA21 slope up
        if require_ema21_slope_up:
            e_today = ema21.reindex([dt]).iloc[0]
            e_prev = ema21_prev.reindex([dt]).iloc[0]
            if pd.isna(e_today) or pd.isna(e_prev) or not (e_today > e_prev):
                continue
        # Reclaim EMA21 after recent touch (approximate with close <= EMA21 in lookback, then today's close > EMA21)
        if require_reclaim_ema21:
            pos = ema21.index.get_loc(dt)
            start_pos = max(0, pos - max(1, reclaim_lookback_days))
            look_idx = ema21.index[start_pos:pos]
            if len(look_idx) == 0:
                continue
            touched = (spy_close.reindex(look_idx) <=
                       ema21.reindex(look_idx)).any()
            if not bool(touched):
                continue
            e_today = ema21.reindex([dt]).iloc[0]
            if pd.isna(e_today) or not (float(spy_close.loc[dt]) > float(e_today)):
                continue
        # Avoid entering on large up day
        if avoid_big_up_day:
            r = ret.reindex([dt]).iloc[0]
            if not pd.isna(r) and float(r) >= float(big_up_day_pct):
                continue
        # Entry
        S = float(spy_close.loc[dt])
        vix = vix_today

        # Determine expiry: if dte_min_days/dte_max_days provided, pick first date in range; else use dte_days
        if dte_min_days is not None and dte_max_days is not None:
            target_min = dt + pd.Timedelta(days=dte_min_days)
            target_max = dt + pd.Timedelta(days=dte_max_days)
            future_window = spy_close.index[(spy_close.index >= target_min) & (
                spy_close.index <= target_max)]
            if len(future_window) == 0:
                # Fallback: no date in window, skip trade
                continue
            expiry_dt = future_window[0]
        else:
            # Original logic: next date >= dt + dte_days
            expiry_target = dt + pd.Timedelta(days=dte_days)
            future_idx = spy_close.index[spy_close.index >= expiry_target]
            if len(future_idx) == 0:
                break
            expiry_dt = future_idx[0]

        # Compute actual T in years from entry to expiry
        T_years = (expiry_dt - dt).days / 365.0
        sigma = max(1e-4, (vix / 100.0) * iv_scale)
        r = r_annual

        K_sell = _strike_for_put_delta(S, T_years, r, sigma, sell_delta)
        K_buy = _strike_for_put_delta(S, T_years, r, sigma, buy_delta)
        # Ensure proper ordering: K_sell > K_buy for put credit spread
        if K_buy > K_sell:
            K_sell, K_buy = K_buy, K_sell

        p_sell, _ = _black_scholes_put(S, K_sell, T_years, r, sigma)
        p_buy, _ = _black_scholes_put(S, K_buy, T_years, r, sigma)
        credit = max(0.0, p_sell - p_buy)
        width = K_sell - K_buy
        max_loss = max(0.0, width - credit)

        S_T = float(spy_close.loc[expiry_dt])

        # Default: hold to expiry
        pnl_at_expiry = None
        payoff = credit - max(0.0, K_sell - S_T) + max(0.0, K_buy - S_T)
        pnl_at_expiry = max(-max_loss, min(credit, payoff))

        # Early-exit simulation (optional) via daily mark-to-market using BS
        exit_dt = None
        pnl = pnl_at_expiry
        # Determine trade path days (trading days only)
        path_idx = spy_close.index[(spy_close.index > dt) & (
            spy_close.index <= expiry_dt)]
        # Helper thresholds based on credit
        tp_threshold = None
        sl_threshold = None
        if isinstance(take_profit_pct_of_credit, (int, float)) and take_profit_pct_of_credit is not None and take_profit_pct_of_credit > 0:
            # Exit when PnL >= tp% * credit  <=>  debit_to_close <= credit * (1 - tp%)
            tp_threshold = float(credit) * \
                (1.0 - float(take_profit_pct_of_credit))
        if isinstance(stop_loss_pct_of_credit, (int, float)) and stop_loss_pct_of_credit is not None and stop_loss_pct_of_credit > 0:
            # Exit when PnL <= -sl% * credit  <=>  debit_to_close >= credit * (1 + sl%)
            sl_threshold = float(credit) * \
                (1.0 + float(stop_loss_pct_of_credit))

        # Iterate through each day to check exit conditions, first satisfied wins
        for i, day in enumerate(path_idx, start=1):
            # Exit at N trading days in trade
            if isinstance(exit_after_days_in_trade, int) and exit_after_days_in_trade is not None and exit_after_days_in_trade > 0:
                if i >= int(exit_after_days_in_trade):
                    # Close at prior day's mark if available; else continue to compute today's mark
                    exit_dt = day
                    # Compute today's mark and PnL
                    S_d = float(spy_close.loc[day])
                    T_years_d = max(1e-6, (expiry_dt - day).days / 365.0)
                    sigma_d = max(
                        1e-4, (float(vix_close.reindex([day]).iloc[0]) / 100.0) * iv_scale)
                    p_sell_d, _ = _black_scholes_put(
                        S_d, K_sell, T_years_d, r, sigma_d)
                    p_buy_d, _ = _black_scholes_put(
                        S_d, K_buy, T_years_d, r, sigma_d)
                    debit_to_close = max(0.0, p_sell_d - p_buy_d)
                    pnl = credit - debit_to_close
                    break

            # Exit at a specific remaining DTE
            if isinstance(exit_at_days_to_expiration, int) and exit_at_days_to_expiration is not None and exit_at_days_to_expiration > 0:
                rem_dte = (expiry_dt - day).days
                if rem_dte <= int(exit_at_days_to_expiration):
                    exit_dt = day
                    S_d = float(spy_close.loc[day])
                    T_years_d = max(1e-6, rem_dte / 365.0)
                    sigma_d = max(
                        1e-4, (float(vix_close.reindex([day]).iloc[0]) / 100.0) * iv_scale)
                    p_sell_d, _ = _black_scholes_put(
                        S_d, K_sell, T_years_d, r, sigma_d)
                    p_buy_d, _ = _black_scholes_put(
                        S_d, K_buy, T_years_d, r, sigma_d)
                    debit_to_close = max(0.0, p_sell_d - p_buy_d)
                    pnl = credit - debit_to_close
                    break

            # Take-profit / Stop-loss checks
            if tp_threshold is not None or sl_threshold is not None:
                S_d = float(spy_close.loc[day])
                rem_dte = (expiry_dt - day).days
                T_years_d = max(1e-6, rem_dte / 365.0)
                # Use same-day VIX as IV proxy
                try:
                    vix_d = float(vix_close.reindex([day]).iloc[0])
                except Exception:
                    vix_d = float(vix_close.dropna().reindex(spy_close.index, method='ffill').reindex(
                        [day]).iloc[0]) if not vix_close.dropna().empty else (vix_close.mean() if not pd.isna(vix_close.mean()) else 20.0)
                sigma_d = max(1e-4, (vix_d / 100.0) * iv_scale)
                p_sell_d, _ = _black_scholes_put(
                    S_d, K_sell, T_years_d, r, sigma_d)
                p_buy_d, _ = _black_scholes_put(
                    S_d, K_buy, T_years_d, r, sigma_d)
                debit_to_close = max(0.0, p_sell_d - p_buy_d)

                # Check TP first (lock profits), then SL
                if tp_threshold is not None and debit_to_close <= tp_threshold:
                    exit_dt = day
                    pnl = credit - debit_to_close
                    break
                if sl_threshold is not None and debit_to_close >= sl_threshold:
                    exit_dt = day
                    pnl = credit - debit_to_close
                    break

        trades.append(Trade(
            entry_date=dt,
            expiry_date=exit_dt if exit_dt is not None else expiry_dt,
            S_entry=S,
            VIX_entry=vix,
            K_sell=K_sell,
            K_buy=K_buy,
            credit=credit,
            max_loss=max_loss,
            S_expiry=float(spy_close.loc[exit_dt]
                           ) if exit_dt is not None else S_T,
            pnl=pnl,
        ))
        # Removed: open_until tracking (was restricting to non-overlapping trades)

    if not trades:
        return pd.DataFrame(), pd.Series(dtype=float), {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'total_pnl': 0.0
        }

    df = pd.DataFrame([t.__dict__ for t in trades])
    df['return_on_risk'] = df['pnl'] / df['max_loss'].replace(0, math.nan)
    # Equity curve in units of $ per spread
    eq = df.set_index('expiry_date')['pnl'].cumsum()

    wins = int((df['pnl'] > 0).sum())
    total = int(len(df))

    # Calculate maximum concurrent positions
    # For each date, count how many trades are active (entry <= date < expiry)
    if total > 0:
        all_dates = pd.date_range(
            start=df['entry_date'].min(),
            end=df['expiry_date'].max(),
            freq='D'
        )
        max_concurrent = 0
        for date in all_dates:
            concurrent = ((df['entry_date'] <= date) &
                          (df['expiry_date'] > date)).sum()
            max_concurrent = max(max_concurrent, int(concurrent))
    else:
        max_concurrent = 0

    # Calculate compounding metrics
    avg_capital = float(df['max_loss'].mean()) if total else 0.0
    avg_return_pct = (float(df['pnl'].mean()) / avg_capital *
                      100.0) if total and avg_capital > 0 else 0.0

    # Simulate compounding: start with avg capital, reinvest gains
    compound_balance = avg_capital
    for pnl in df['pnl']:
        compound_balance += pnl
    compound_growth = ((compound_balance / avg_capital) -
                       1.0) * 100.0 if avg_capital > 0 else 0.0

    summary = {
        'trades': total,
        'wins': wins,
        'win_rate': wins / total if total else 0.0,
        'avg_pnl': float(df['pnl'].mean()) if total else 0.0,
        'total_pnl': float(df['pnl'].sum()) if total else 0.0,
        'avg_ror': float(df['return_on_risk'].mean()) if total else 0.0,
        'avg_capital_per_trade': avg_capital,
        'total_capital_required': float(df['max_loss'].sum()) if total else 0.0,
        'return_on_capital': (float(df['pnl'].sum()) / float(df['max_loss'].sum())) if total and df['max_loss'].sum() > 0 else 0.0,
        'avg_return_pct': avg_return_pct,
        'compound_growth_pct': compound_growth,
        'max_concurrent_positions': max_concurrent,
    }
    return df, eq, summary


def run_backtest_bull_call(
    spy_close: pd.Series,
    vix_close: pd.Series,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    dte_days: int = 45,
    dte_min_days: Optional[int] = None,
    dte_max_days: Optional[int] = None,
    buy_delta: float = 0.40,
    sell_delta: float = 0.20,
    r_annual: float = 0.02,
    iv_scale: float = 1.0,
    # VIX regime filters
    vix_threshold: float = 18.0,
    vix_ma_window: int = 5,
    require_vix_ma_down: bool = False,
    avoid_vix_spike: bool = False,
    vix_spike_pct: float = 0.05,
    # Entry timing filters
    rsi_threshold: float = 55.0,
    require_ema21_slope_up: bool = False,
    require_price_above_ema21: bool = False,  # New: Price > EMA21 check
    require_macd_positive: bool = False,  # New: MACD > 0 check
    allowed_weekdays: Optional[List[int]] = None,
    # Optional exit strategy
    take_profit_pct_of_debit: Optional[float] = None,
    stop_loss_pct_of_debit: Optional[float] = None,
    exit_after_days_in_trade: Optional[int] = None,
    exit_at_days_to_expiration: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """Bull call debit spread backtest (long lower strike call, short higher strike call).

    Returns: (trades_df, equity_curve, summary)
    """
    spy_close = spy_close.sort_index()
    vix_close = vix_close.sort_index()
    if start is not None:
        spy_close = spy_close[spy_close.index >= start]
        vix_close = vix_close[vix_close.index >= start]
    if end is not None:
        spy_close = spy_close[spy_close.index <= end]
        vix_close = vix_close[vix_close.index <= end]

    signal = generate_signal_series(spy_close, vix_close)
    if signal.empty:
        return pd.DataFrame(), pd.Series(dtype=float), {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'total_pnl': 0.0
        }

    dates = signal.index
    # VIX MA for filters
    vix_ma = vix_close.sort_index().rolling(vix_ma_window, min_periods=vix_ma_window).mean(
    ) if vix_ma_window and vix_ma_window > 1 else None
    vix_ma_prev = vix_ma.shift(1) if vix_ma is not None else None
    inds_all = compute_indicators(spy_close)
    ema21 = inds_all['ema21']
    ema21_prev = ema21.shift(1)
    rsi14 = inds_all['rsi14']
    macd_diff = inds_all['macd_diff']  # Add MACD for optional filter

    # Debug: Print filter configuration
    print(
        f"🔍 BULL CALL BACKTEST FILTERS: require_price_above_ema21={require_price_above_ema21}, require_macd_positive={require_macd_positive}, require_ema21_slope_up={require_ema21_slope_up}")

    if allowed_weekdays is None:
        allowed_weekdays = [1, 2, 3]
    trades: List[Trade] = []

    for dt in dates:
        if not signal.loc[dt]:
            continue
        # Removed non-overlapping restriction to allow multiple concurrent trades
        # VIX filters
        try:
            vix_today = float(vix_close.reindex([dt]).iloc[0])
        except Exception:
            continue
        if not (vix_today < float(vix_threshold)):
            continue
        if require_vix_ma_down and vix_ma is not None and vix_ma_prev is not None:
            v_ma = vix_ma.reindex([dt]).iloc[0]
            v_ma_prev = vix_ma_prev.reindex([dt]).iloc[0]
            if pd.isna(v_ma) or pd.isna(v_ma_prev) or not (v_ma < v_ma_prev):
                continue
        if avoid_vix_spike and vix_ma is not None:
            v_ma = vix_ma.reindex([dt]).iloc[0]
            if pd.isna(v_ma) or vix_today > (1.0 + float(vix_spike_pct)) * float(v_ma):
                continue
        # Timing filters
        if allowed_weekdays and (dt.weekday() not in set(allowed_weekdays)):
            continue
        rsi_val = rsi14.reindex([dt]).iloc[0]
        if pd.isna(rsi_val) or not (float(rsi_val) > float(rsi_threshold)):
            continue

        # New: Price > EMA21 check
        if require_price_above_ema21:
            e_today = ema21.reindex([dt]).iloc[0]
            if pd.isna(e_today) or not (float(spy_close.loc[dt]) > float(e_today)):
                continue

        # New: MACD > 0 check
        if require_macd_positive:
            macd_val = macd_diff.reindex([dt]).iloc[0]
            if pd.isna(macd_val) or not (float(macd_val) > 0):
                continue

        if require_ema21_slope_up:
            e_today = ema21.reindex([dt]).iloc[0]
            e_prev = ema21_prev.reindex([dt]).iloc[0]
            if pd.isna(e_today) or pd.isna(e_prev) or not (e_today > e_prev):
                continue

        # Entry
        S = float(spy_close.loc[dt])

        # Determine expiry: if dte_min_days/dte_max_days provided, pick first date in range; else use dte_days
        if dte_min_days is not None and dte_max_days is not None:
            target_min = dt + pd.Timedelta(days=dte_min_days)
            target_max = dt + pd.Timedelta(days=dte_max_days)
            future_window = spy_close.index[(spy_close.index >= target_min) & (
                spy_close.index <= target_max)]
            if len(future_window) == 0:
                continue
            expiry_dt = future_window[0]
        else:
            expiry_target = dt + pd.Timedelta(days=dte_days)
            future_idx = spy_close.index[spy_close.index >= expiry_target]
            if len(future_idx) == 0:
                break
            expiry_dt = future_idx[0]

        T_years = (expiry_dt - dt).days / 365.0
        sigma = max(1e-4, (vix_today / 100.0) * iv_scale)
        r = r_annual

        K_long = _strike_for_call_delta(S, T_years, r, sigma, buy_delta)
        K_short = _strike_for_call_delta(S, T_years, r, sigma, sell_delta)
        # Ensure proper ordering: long lower strike, short higher strike
        if K_short <= K_long:
            # if inversion gives reversed order, adjust by swapping deltas or recompute
            K_long, K_short = min(K_long, K_short), max(K_long, K_short)

        p_long, _ = _black_scholes_call(S, K_long, T_years, r, sigma)
        p_short, _ = _black_scholes_call(S, K_short, T_years, r, sigma)
        debit = max(0.0, p_long - p_short)
        width = max(0.0, K_short - K_long)
        max_gain = max(0.0, width - debit)
        max_loss = debit

        S_T = float(spy_close.loc[expiry_dt])

        # Hold-to-expiry PnL
        payoff_spread = max(0.0, S_T - K_long) - max(0.0, S_T - K_short)
        pnl_at_expiry = payoff_spread - debit

        exit_dt = None
        pnl = pnl_at_expiry

        path_idx = spy_close.index[(spy_close.index > dt) & (
            spy_close.index <= expiry_dt)]
        # Thresholds
        tp_val = None
        sl_val = None
        if isinstance(take_profit_pct_of_debit, (int, float)) and take_profit_pct_of_debit is not None and take_profit_pct_of_debit > 0:
            # TP when value_today >= debit * (1 + tp%)  <=> PnL >= debit*tp%
            tp_val = float(debit) * (1.0 + float(take_profit_pct_of_debit))
        if isinstance(stop_loss_pct_of_debit, (int, float)) and stop_loss_pct_of_debit is not None and stop_loss_pct_of_debit > 0:
            # SL when value_today <= debit * (1 - sl%)  <=> PnL <= -debit*sl%
            sl_val = float(debit) * (1.0 - float(stop_loss_pct_of_debit))

        for i, day in enumerate(path_idx, start=1):
            # Exit after N trading days
            if isinstance(exit_after_days_in_trade, int) and exit_after_days_in_trade is not None and exit_after_days_in_trade > 0:
                if i >= int(exit_after_days_in_trade):
                    exit_dt = day
                    S_d = float(spy_close.loc[day])
                    rem_dte = (expiry_dt - day).days
                    T_years_d = max(1e-6, rem_dte / 365.0)
                    vix_d = float(vix_close.reindex(
                        [day]).iloc[0]) if not vix_close.empty else vix_today
                    sigma_d = max(1e-4, (vix_d / 100.0) * iv_scale)
                    pL, _ = _black_scholes_call(
                        S_d, K_long, T_years_d, r, sigma_d)
                    pS, _ = _black_scholes_call(
                        S_d, K_short, T_years_d, r, sigma_d)
                    value_today = max(0.0, pL - pS)
                    pnl = value_today - debit
                    break

            # Exit at specific remaining DTE
            if isinstance(exit_at_days_to_expiration, int) and exit_at_days_to_expiration is not None and exit_at_days_to_expiration > 0:
                rem_dte = (expiry_dt - day).days
                if rem_dte <= int(exit_at_days_to_expiration):
                    exit_dt = day
                    S_d = float(spy_close.loc[day])
                    T_years_d = max(1e-6, rem_dte / 365.0)
                    vix_d = float(vix_close.reindex(
                        [day]).iloc[0]) if not vix_close.empty else vix_today
                    sigma_d = max(1e-4, (vix_d / 100.0) * iv_scale)
                    pL, _ = _black_scholes_call(
                        S_d, K_long, T_years_d, r, sigma_d)
                    pS, _ = _black_scholes_call(
                        S_d, K_short, T_years_d, r, sigma_d)
                    value_today = max(0.0, pL - pS)
                    pnl = value_today - debit
                    break

            # TP/SL checks via current spread value
            if tp_val is not None or sl_val is not None:
                S_d = float(spy_close.loc[day])
                rem_dte = (expiry_dt - day).days
                T_years_d = max(1e-6, rem_dte / 365.0)
                vix_d = float(vix_close.reindex(
                    [day]).iloc[0]) if not vix_close.empty else vix_today
                sigma_d = max(1e-4, (vix_d / 100.0) * iv_scale)
                pL, _ = _black_scholes_call(S_d, K_long, T_years_d, r, sigma_d)
                pS, _ = _black_scholes_call(
                    S_d, K_short, T_years_d, r, sigma_d)
                value_today = max(0.0, pL - pS)
                # TP first, then SL
                if tp_val is not None and value_today >= tp_val:
                    exit_dt = day
                    pnl = value_today - debit
                    break
                if sl_val is not None and value_today <= sl_val:
                    exit_dt = day
                    pnl = value_today - debit
                    break

        trades.append(Trade(
            entry_date=dt,
            expiry_date=exit_dt if exit_dt is not None else expiry_dt,
            S_entry=S,
            VIX_entry=vix_today,
            K_sell=K_short,  # short/higher strike stored in K_sell for consistency
            K_buy=K_long,    # long/lower strike stored in K_buy
            credit=-debit,   # negative credit to indicate debit paid
            max_loss=max_loss,
            S_expiry=float(spy_close.loc[exit_dt]
                           ) if exit_dt is not None else S_T,
            pnl=pnl,
        ))
        # Removed: open_until tracking (was restricting to non-overlapping trades)

    if not trades:
        return pd.DataFrame(), pd.Series(dtype=float), {
            'trades': 0, 'wins': 0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'total_pnl': 0.0
        }

    df = pd.DataFrame([t.__dict__ for t in trades])
    df['return_on_risk'] = df['pnl'] / df['max_loss'].replace(0, math.nan)
    eq = df.set_index('expiry_date')['pnl'].cumsum()
    wins = int((df['pnl'] > 0).sum())
    total = int(len(df))

    # Calculate maximum concurrent positions
    # For each date, count how many trades are active (entry <= date < expiry)
    if total > 0:
        all_dates = pd.date_range(
            start=df['entry_date'].min(),
            end=df['expiry_date'].max(),
            freq='D'
        )
        max_concurrent = 0
        for date in all_dates:
            concurrent = ((df['entry_date'] <= date) &
                          (df['expiry_date'] > date)).sum()
            max_concurrent = max(max_concurrent, int(concurrent))
    else:
        max_concurrent = 0

    # Calculate compounding metrics
    avg_capital = float(df['max_loss'].mean()) if total else 0.0
    avg_return_pct = (float(df['pnl'].mean()) / avg_capital *
                      100.0) if total and avg_capital > 0 else 0.0

    # Simulate compounding: start with avg capital, reinvest gains
    compound_balance = avg_capital
    for pnl in df['pnl']:
        compound_balance += pnl
    compound_growth = ((compound_balance / avg_capital) -
                       1.0) * 100.0 if avg_capital > 0 else 0.0

    summary = {
        'trades': total,
        'wins': wins,
        'win_rate': wins / total if total else 0.0,
        'avg_pnl': float(df['pnl'].mean()) if total else 0.0,
        'total_pnl': float(df['pnl'].sum()) if total else 0.0,
        'avg_ror': float(df['return_on_risk'].mean()) if total else 0.0,
        'avg_capital_per_trade': avg_capital,
        'total_capital_required': float(df['max_loss'].sum()) if total else 0.0,
        'return_on_capital': (float(df['pnl'].sum()) / float(df['max_loss'].sum())) if total and df['max_loss'].sum() > 0 else 0.0,
        'avg_return_pct': avg_return_pct,
        'compound_growth_pct': compound_growth,
        'max_concurrent_positions': max_concurrent,
    }
    return df, eq, summary
