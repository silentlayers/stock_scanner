"""
Automated Signal Detection and Trade Execution

This module monitors market conditions and automatically executes trades
when predefined criteria are met.
"""
import logging
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Tuple
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketConditions:
    """Check market conditions for trading signals"""

    @staticmethod
    def get_rsi(symbol: str = 'SPY', period: int = 14) -> Optional[float]:
        """
        Calculate RSI for a symbol

        Args:
            symbol: Ticker symbol (default: SPY)
            period: RSI period (default: 14)

        Returns:
            Current RSI value or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='60d', interval='1d')

            if len(hist) < period + 1:
                logger.warning(
                    f"Insufficient data for RSI calculation: {len(hist)} days")
                return None

            # Calculate RSI - using numpy operations to satisfy type checker
            import numpy as np

            delta = hist['Close'].diff()
            # Separate gains and losses
            gains = delta.clip(lower=0).rolling(window=period).mean()
            losses = (-delta).clip(lower=0).rolling(window=period).mean()

            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]
            logger.info(f"{symbol} RSI({period}): {current_rsi:.2f}")

            return float(current_rsi)

        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return None

    @staticmethod
    def get_vix() -> Optional[float]:
        """
        Get current VIX value

        Returns:
            Current VIX close price or None if error
        """
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='1d', interval='1d')

            if len(hist) == 0:
                logger.warning("No VIX data available")
                return None

            current_vix = hist['Close'].iloc[-1]
            logger.info(f"VIX: {current_vix:.2f}")

            return float(current_vix)

        except Exception as e:
            logger.error(f"Failed to get VIX: {e}")
            return None

    @staticmethod
    def check_signal(rsi_threshold: float = 55.0, vix_threshold: float = 20.0) -> Tuple[bool, Dict]:
        """
        Check if market conditions meet trading signal criteria

        Args:
            rsi_threshold: Minimum RSI for bullish signal (default: 55)
            vix_threshold: Maximum VIX for low volatility (default: 20)

        Returns:
            (signal_triggered: bool, conditions: dict)
        """
        conditions = {
            'timestamp': datetime.now().isoformat(),
            'rsi': None,
            'vix': None,
            'rsi_threshold': rsi_threshold,
            'vix_threshold': vix_threshold,
            'signal': False,
            'reason': ''
        }

        # Get RSI
        rsi = MarketConditions.get_rsi()
        if rsi is None:
            conditions['reason'] = 'Failed to fetch RSI'
            return False, conditions

        conditions['rsi'] = rsi

        # Get VIX
        vix = MarketConditions.get_vix()
        if vix is None:
            conditions['reason'] = 'Failed to fetch VIX'
            return False, conditions

        conditions['vix'] = vix

        # Check signal
        rsi_ok = rsi >= rsi_threshold
        vix_ok = vix <= vix_threshold

        if rsi_ok and vix_ok:
            conditions['signal'] = True
            conditions['reason'] = f'✅ Signal triggered: RSI {rsi:.2f} >= {rsi_threshold}, VIX {vix:.2f} <= {vix_threshold}'
            logger.info(conditions['reason'])
            return True, conditions
        else:
            reasons = []
            if not rsi_ok:
                reasons.append(f'RSI {rsi:.2f} < {rsi_threshold}')
            if not vix_ok:
                reasons.append(f'VIX {vix:.2f} > {vix_threshold}')

            conditions['reason'] = f'❌ No signal: {", ".join(reasons)}'
            logger.info(conditions['reason'])
            return False, conditions


class TradingSchedule:
    """Manage trading schedule and execution windows"""

    @staticmethod
    def is_market_hours(tz: str = 'America/New_York') -> bool:
        """
        Check if currently within market hours (9:30 AM - 4:00 PM ET)

        Args:
            tz: Timezone (default: America/New_York)

        Returns:
            True if within market hours
        """
        try:
            now = datetime.now(ZoneInfo(tz))

            # Check if weekday (0 = Monday, 6 = Sunday)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False

            current_time = now.time()
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)

            return market_open <= current_time <= market_close

        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            return False

    @staticmethod
    def is_execution_window(
        start_time: str = "10:30",
        end_time: str = "10:45",
        tz: str = 'America/New_York'
    ) -> bool:
        """
        Check if currently within execution window

        Args:
            start_time: Start time in HH:MM format (default: 10:30)
            end_time: End time in HH:MM format (default: 10:45)
            tz: Timezone (default: America/New_York)

        Returns:
            True if within execution window
        """
        try:
            now = datetime.now(ZoneInfo(tz))

            # Check if weekday
            if now.weekday() >= 5:
                return False

            current_time = now.time()

            # Parse start and end times
            start_hour, start_min = map(int, start_time.split(':'))
            end_hour, end_min = map(int, end_time.split(':'))

            window_start = dt_time(start_hour, start_min)
            window_end = dt_time(end_hour, end_min)

            in_window = window_start <= current_time <= window_end

            if in_window:
                logger.info(
                    f"✅ Within execution window: {start_time}-{end_time} ET")
            else:
                logger.info(
                    f"⏳ Outside execution window: {start_time}-{end_time} ET (current: {current_time.strftime('%H:%M')})")

            return in_window

        except Exception as e:
            logger.error(f"Failed to check execution window: {e}")
            return False

    @staticmethod
    def should_trade_now(
        execution_window: Tuple[str, str] = ("10:30", "10:45"),
        require_market_hours: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if trading should occur right now

        Args:
            execution_window: (start_time, end_time) tuple
            require_market_hours: Enforce market hours check

        Returns:
            (should_trade: bool, reason: str)
        """
        # Check market hours
        if require_market_hours and not TradingSchedule.is_market_hours():
            return False, "Market is closed"

        # Check execution window
        start_time, end_time = execution_window
        if not TradingSchedule.is_execution_window(start_time, end_time):
            return False, f"Outside execution window ({start_time}-{end_time} ET)"

        return True, f"✅ Ready to trade (within {start_time}-{end_time} ET window)"


class AutomationEngine:
    """Main automation engine that coordinates signal detection and execution"""

    def __init__(
        self,
        rsi_threshold: float = 55.0,
        vix_threshold: float = 20.0,
        execution_window: Tuple[str, str] = ("10:30", "10:45"),
        check_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize automation engine

        Args:
            rsi_threshold: Minimum RSI for signal (default: 55)
            vix_threshold: Maximum VIX for signal (default: 20)
            execution_window: Trading window tuple (start, end)
            check_interval_seconds: How often to check conditions (default: 300s = 5min)
        """
        self.rsi_threshold = rsi_threshold
        self.vix_threshold = vix_threshold
        self.execution_window = execution_window
        self.check_interval = check_interval_seconds

        self.last_check_time = None
        self.last_signal_conditions = None
        self.trades_today = []

    def check_and_log_conditions(self) -> Tuple[bool, Dict]:
        """
        Check market conditions and log results

        Returns:
            (signal_triggered: bool, conditions: dict)
        """
        logger.info("Checking market conditions...")

        # Check trading schedule
        can_trade, schedule_reason = TradingSchedule.should_trade_now(
            self.execution_window)

        # Check signal
        signal_triggered, conditions = MarketConditions.check_signal(
            self.rsi_threshold,
            self.vix_threshold
        )

        # Add schedule info to conditions
        conditions['can_trade_now'] = can_trade
        conditions['schedule_reason'] = schedule_reason

        # Update tracking
        self.last_check_time = datetime.now()
        self.last_signal_conditions = conditions

        # Overall decision
        ready_to_execute = signal_triggered and can_trade

        if ready_to_execute:
            logger.info("🚀 SIGNAL CONFIRMED - Ready to execute trade")
        else:
            if signal_triggered and not can_trade:
                logger.info(
                    f"⏳ Signal detected but cannot trade: {schedule_reason}")
            else:
                logger.info(f"⏳ Waiting for signal: {conditions['reason']}")

        conditions['ready_to_execute'] = ready_to_execute

        return ready_to_execute, conditions

    def should_check_now(self) -> bool:
        """
        Determine if it's time to check conditions again

        Returns:
            True if enough time has passed since last check
        """
        if self.last_check_time is None:
            return True

        elapsed = (datetime.now() - self.last_check_time).total_seconds()
        return elapsed >= self.check_interval

    def get_status(self) -> Dict:
        """
        Get current automation status

        Returns:
            Status dictionary with conditions and timing
        """
        return {
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'next_check_in_seconds': max(0, self.check_interval - (
                (datetime.now() - self.last_check_time).total_seconds()
                if self.last_check_time else 0
            )),
            'last_conditions': self.last_signal_conditions,
            'trades_today': len(self.trades_today),
            'config': {
                'rsi_threshold': self.rsi_threshold,
                'vix_threshold': self.vix_threshold,
                'execution_window': self.execution_window,
                'check_interval_seconds': self.check_interval
            }
        }


# Convenience functions for Streamlit integration

def check_signal_now(rsi_threshold: float = 55.0, vix_threshold: float = 20.0) -> Tuple[bool, Dict]:
    """
    Quick check if signal conditions are met right now

    Args:
        rsi_threshold: Minimum RSI (default: 55)
        vix_threshold: Maximum VIX (default: 20)

    Returns:
        (signal_triggered: bool, conditions: dict)
    """
    return MarketConditions.check_signal(rsi_threshold, vix_threshold)


def can_trade_now(execution_window: Tuple[str, str] = ("10:30", "10:45")) -> Tuple[bool, str]:
    """
    Quick check if trading is allowed right now

    Args:
        execution_window: (start_time, end_time) tuple

    Returns:
        (can_trade: bool, reason: str)
    """
    return TradingSchedule.should_trade_now(execution_window)


def get_current_market_snapshot() -> Dict:
    """
    Get current market conditions snapshot

    Returns:
        Dictionary with RSI, VIX, market hours, execution window status
    """
    rsi = MarketConditions.get_rsi()
    vix = MarketConditions.get_vix()
    market_open = TradingSchedule.is_market_hours()
    in_window = TradingSchedule.is_execution_window()

    return {
        'timestamp': datetime.now().isoformat(),
        'rsi': rsi,
        'vix': vix,
        'market_open': market_open,
        'in_execution_window': in_window
    }
