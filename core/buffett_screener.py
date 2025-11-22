"""
Warren Buffett Stock Screener
Implements Buffett's investment criteria and intrinsic value calculation
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
import numpy as np


class BuffettScreener:
    """Screen stocks using Warren Buffett's investment criteria"""

    # Buffett's core screening criteria
    CRITERIA = {
        'roe_min': 15,  # Return on Equity >= 15%
        'debt_to_equity_max': 0.5,  # Conservative debt levels
        'current_ratio_min': 1.5,  # Strong liquidity
        'profit_margin_min': 10,  # Profit margin >= 10%
        'pe_ratio_max': 25,  # Reasonable P/E ratio
        'earnings_growth_min': 7,  # Consistent earnings growth >= 7%
    }

    def __init__(self):
        self.results = []

    def screen_stock(self, ticker: str) -> Optional[Dict]:
        """
        Screen a single stock against Buffett's criteria

        Returns:
            Dictionary with stock data and pass/fail status, or None if data unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get key metrics
            roe = info.get('returnOnEquity', 0) * \
                100 if info.get('returnOnEquity') else 0
            debt_to_equity = info.get(
                'debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            current_ratio = info.get('currentRatio', 0)
            profit_margin = info.get(
                'profitMargins', 0) * 100 if info.get('profitMargins') else 0
            pe_ratio = info.get('trailingPE', 0)

            # Get earnings growth (5-year average if available)
            earnings_growth = info.get(
                'earningsQuarterlyGrowth', 0) * 100 if info.get('earningsQuarterlyGrowth') else 0

            # Check each criterion
            passes = {
                'roe': roe >= self.CRITERIA['roe_min'],
                'debt': debt_to_equity <= self.CRITERIA['debt_to_equity_max'],
                'liquidity': current_ratio >= self.CRITERIA['current_ratio_min'],
                'margin': profit_margin >= self.CRITERIA['profit_margin_min'],
                'pe': 0 < pe_ratio <= self.CRITERIA['pe_ratio_max'],
                'growth': earnings_growth >= self.CRITERIA['earnings_growth_min'],
            }

            result = {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'profit_margin': profit_margin,
                'pe_ratio': pe_ratio,
                'earnings_growth': earnings_growth,
                'passes': passes,
                'total_pass': sum(passes.values()),
                'passed_all': all(passes.values()),
            }

            return result

        except Exception as e:
            print(f"Error screening {ticker}: {e}")
            return None

    def screen_multiple(self, tickers: List[str]) -> pd.DataFrame:
        """Screen multiple stocks and return results as DataFrame"""
        results = []
        for ticker in tickers:
            result = self.screen_stock(ticker)
            if result:
                results.append(result)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.sort_values('total_pass', ascending=False)

    def calculate_intrinsic_value(self, ticker: str, discount_rate: float = 0.09,
                                  growth_rate: float = 0.07, terminal_growth: float = 0.03) -> Dict:
        """
        Calculate intrinsic value using Discounted Cash Flow (DCF) method
        Based on Buffett's preference for owner earnings

        Args:
            ticker: Stock ticker symbol
            discount_rate: Required rate of return (default 9%)
            growth_rate: Expected growth rate for next 10 years (default 7%)
            terminal_growth: Perpetual growth rate after 10 years (default 3%)

        Returns:
            Dictionary with intrinsic value calculation details
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get free cash flow (proxy for owner earnings)
            cash_flow = stock.cashflow
            if cash_flow.empty:
                return {'error': 'Cash flow data not available'}

            # Get most recent free cash flow
            fcf = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0

            # Ensure fcf is a scalar value
            if isinstance(fcf, pd.Series):
                fcf = fcf.item() if len(fcf) > 0 else 0

            if fcf <= 0:
                return {'error': 'Negative or zero free cash flow'}

            # Project cash flows for 10 years
            projected_cf = []
            for year in range(1, 11):
                cf = fcf * ((1 + growth_rate) ** year)
                discounted_cf = cf / ((1 + discount_rate) ** year)
                projected_cf.append(discounted_cf)

            # Calculate terminal value
            terminal_cf = fcf * ((1 + growth_rate) ** 10) * \
                (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            discounted_terminal = terminal_value / ((1 + discount_rate) ** 10)

            # Total intrinsic value
            total_value = sum(projected_cf) + discounted_terminal

            # Get shares outstanding
            shares = info.get('sharesOutstanding', 0)
            if shares <= 0:
                return {'error': 'Shares outstanding not available'}

            intrinsic_value_per_share = total_value / shares
            current_price = info.get('currentPrice', 0)

            # Calculate margin of safety (Buffett likes 25-30% discount)
            margin_of_safety = ((intrinsic_value_per_share - current_price) /
                                intrinsic_value_per_share * 100) if intrinsic_value_per_share > 0 else 0

            return {
                'ticker': ticker,
                'current_price': current_price,
                'intrinsic_value': intrinsic_value_per_share,
                'margin_of_safety': margin_of_safety,
                'fcf': fcf,
                'projected_10yr_value': sum(projected_cf),
                'terminal_value': discounted_terminal,
                'total_value': total_value,
                'shares_outstanding': shares,
                'discount_rate': discount_rate,
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'recommendation': self._get_recommendation(margin_of_safety),
            }

        except Exception as e:
            return {'error': f'Calculation failed: {str(e)}'}

    def _get_recommendation(self, margin_of_safety: float) -> str:
        """Get investment recommendation based on margin of safety"""
        if margin_of_safety >= 30:
            return "🟢 Strong Buy - Excellent margin of safety"
        elif margin_of_safety >= 20:
            return "🟢 Buy - Good margin of safety"
        elif margin_of_safety >= 10:
            return "🟡 Hold - Moderate margin of safety"
        elif margin_of_safety >= 0:
            return "🟡 Hold - Fair value"
        else:
            return "🔴 Avoid - Overvalued"


def get_sp500_tickers() -> List[str]:
    """Get list of S&P 500 tickers"""
    try:
        # Using a simple list of well-known stocks for demo
        # In production, you could fetch from Wikipedia or other sources
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'JNJ', 'V', 'PG', 'MA', 'NVDA',
            'HD', 'DIS', 'PYPL', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'KO', 'PEP',
            'INTC', 'CSCO', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'CVX', 'NKE', 'WMT',
            'MCD', 'UNH', 'DHR', 'NEE', 'LIN', 'UPS', 'BMY', 'QCOM', 'TXN', 'AMT',
            'HON', 'SBUX', 'MDT', 'LLY', 'IBM', 'CAT', 'GE', 'BA', 'MMM', 'GS'
        ]
    except Exception:
        return []


def get_buffett_portfolio() -> List[str]:
    """Get Warren Buffett's (Berkshire Hathaway) top holdings"""
    # As of recent filings - these are BRK's major positions
    return [
        'AAPL', 'BAC', 'AXP', 'KO', 'CVX', 'OXY', 'KHC', 'MCO', 'VZ', 'C',
        'BK', 'USB', 'GM', 'KR', 'AON', 'MA', 'V', 'DVA', 'ALLY', 'CE'
    ]
