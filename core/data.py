"""
Data utilities: yfinance download and robust Close series extraction.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
import yfinance as yf


def download_symbol(symbol: str, period: str = '60d', interval: str = '1d', auto_adjust: bool = False) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=auto_adjust)
    if df is None or (hasattr(df, 'empty') and df.empty):
        raise RuntimeError(f"Failed to download data for {symbol}")
    return df


def get_close_series(df: pd.DataFrame, symbol_hint: Optional[str] = None) -> pd.Series:
    if isinstance(df, pd.Series):
        return df.astype(float)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            if symbol_hint and (symbol_hint in df.columns.get_level_values(0)):
                obj = df[(symbol_hint, 'Close')]
                if isinstance(obj, pd.DataFrame):
                    series = pd.Series(obj.to_numpy()[:, 0], index=obj.index)
                else:
                    series = obj
                return series.astype(float)
            close_df = df.xs('Close', axis=1, level=1)
            if isinstance(close_df, pd.DataFrame):
                series = pd.Series(close_df.to_numpy()[
                                   :, 0], index=close_df.index)
            else:
                series = close_df
            return series.astype(float)
        except Exception:
            pass

    if 'Close' in df.columns:
        close_col = df['Close']
        if isinstance(close_col, pd.DataFrame):
            series = pd.Series(close_col.to_numpy()[
                               :, 0], index=close_col.index)
        else:
            series = close_col
        return series.astype(float)

    if 'Adj Close' in df.columns:
        adj_col = df['Adj Close']
        if isinstance(adj_col, pd.DataFrame):
            series = pd.Series(adj_col.to_numpy()[:, 0], index=adj_col.index)
        else:
            series = adj_col
        return series.astype(float)

    raise ValueError('No Close or Adj Close column found in data frame.')
