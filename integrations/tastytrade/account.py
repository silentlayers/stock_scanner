"""
Tastytrade account and position management.

This module provides functions to:
- Get account balance and net liquidation value
- Fetch open positions
- Calculate position P&L (realized and unrealized)
- Monitor portfolio utilization

API Documentation: https://developer.tastytrade.com/
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime, date
import requests

from integrations.tastytrade.config import get_api_base_url


def get_account_numbers(session: requests.Session) -> List[str]:
    """
    Get list of account numbers for the authenticated user.

    Returns:
        List of account numbers (e.g., ['5WT00000'])
    """
    base_url = get_api_base_url()
    url = f"{base_url}/customers/me/accounts"
    resp = session.get(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to get accounts: {resp.status_code} {resp.text}")

    data = resp.json()
    items = data.get('data', {}).get('items', [])

    return [item['account']['account-number'] for item in items if 'account' in item]


def get_account_balance(session: requests.Session, account_number: str) -> Dict[str, Any]:
    """
    Get account balance information.

    Args:
        session: Authenticated requests session
        account_number: Account number (e.g., '5WT00000')

    Returns:
        Dict with balance info (see API docs for full structure)
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/balances"
    resp = session.get(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to get balance: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


def get_account_positions(session: requests.Session, account_number: str) -> List[Dict[str, Any]]:
    """
    Get all open positions for an account.

    A position with quantity 0 is considered closed.

    Args:
        session: Authenticated requests session
        account_number: Account number

    Returns:
        List of position dicts (see API docs for structure)
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/positions"
    resp = session.get(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to get positions: {resp.status_code} {resp.text}")

    data = resp.json()
    items = data.get('data', {}).get('items', [])

    # Filter out closed positions (quantity = 0)
    open_positions = [pos for pos in items if float(
        pos.get('quantity', 0)) != 0]

    return open_positions


def get_options_positions(session: requests.Session, account_number: str) -> List[Dict[str, Any]]:
    """
    Get only options positions (filters out equity, futures, etc.).

    Returns:
        List of options positions
    """
    all_positions = get_account_positions(session, account_number)
    return [pos for pos in all_positions if pos.get('instrument-type') == 'Equity Option']


def calculate_position_current_value(position: Dict[str, Any], mark: float) -> float:
    """
    Calculate the current value (net liquidation value) of a position.

    This is the value you'd receive/pay if you closed the position right now.

    Formula:
        mark * quantity * multiplier * direction
        direction = 1 for long, -1 for short

    Args:
        position: Position dict from get_account_positions()
        mark: Current market price (use bid for long, ask for short, or mid)

    Returns:
        Current value in dollars (positive = credit, negative = debit)
    """
    quantity = float(position.get('quantity', 0))
    multiplier = int(position.get('multiplier', 1))
    quantity_direction = position.get('quantity-direction', 'Long')

    # Direction: 1 for long, -1 for short
    direction = 1 if quantity_direction == 'Long' else -1

    return mark * quantity * multiplier * direction


def calculate_unrealized_gain(position: Dict[str, Any], mark: float) -> float:
    """
    Calculate unrealized P&L since position was opened.

    For long: (mark - average_open_price) * quantity * multiplier
    For short: (average_open_price - mark) * quantity * multiplier

    Args:
        position: Position dict
        mark: Current market price

    Returns:
        Unrealized P&L in dollars (positive = profit, negative = loss)
    """
    quantity = float(position.get('quantity', 0))
    multiplier = int(position.get('multiplier', 1))
    avg_open_price = float(position.get('average-open-price', 0))
    quantity_direction = position.get('quantity-direction', 'Long')

    if quantity_direction == 'Long':
        return (mark - avg_open_price) * quantity * multiplier
    else:  # Short
        return (avg_open_price - mark) * quantity * multiplier


def calculate_account_net_liq(
    session: requests.Session,
    account_number: str,
    position_marks: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate account net liquidation value (total account value).

    Formula:
        sum(position values) + cash_balance + pending_cash

    Args:
        session: Authenticated session
        account_number: Account number
        position_marks: Optional dict of {symbol: mark_price} for positions.
                       If None, will use close_price from positions (less accurate).

    Returns:
        Net liquidation value in dollars
    """
    # Get balance
    balance = get_account_balance(session, account_number)
    cash_balance = float(balance.get('cash-balance', 0))
    pending_cash = float(balance.get('pending-cash', 0))
    pending_effect = balance.get('pending-cash-effect', 'None')

    # Adjust pending cash sign
    if pending_effect == 'Debit':
        pending_cash = -pending_cash

    # Get positions
    positions = get_account_positions(session, account_number)

    # Calculate total position value
    total_position_value = 0.0
    for pos in positions:
        symbol = pos.get('symbol')

        # Use provided mark or fall back to close price
        if position_marks and symbol in position_marks:
            mark = position_marks[symbol]
        else:
            mark = float(pos.get('close-price', 0))

        position_value = calculate_position_current_value(pos, mark)
        total_position_value += position_value

    return total_position_value + cash_balance + pending_cash


def get_spread_positions(session: requests.Session, account_number: str, underlying: str = 'SPY') -> List[Dict[str, Any]]:
    """
    Get all options positions for a specific underlying (e.g., SPY spreads).

    This helps identify existing spreads to avoid over-deployment.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: Underlying symbol (default: 'SPY')

    Returns:
        List of options positions for the underlying
    """
    options_positions = get_options_positions(session, account_number)
    return [pos for pos in options_positions if pos.get('underlying-symbol') == underlying]


def count_open_spreads(session: requests.Session, account_number: str, underlying: str = 'SPY') -> int:
    """
    Count number of open spreads for a given underlying.

    Assumes each spread has 2 legs (sell + buy).

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: Underlying symbol (default: 'SPY')

    Returns:
        Number of open spreads (total positions / 2)
    """
    spread_positions = get_spread_positions(
        session, account_number, underlying)
    # Each spread typically has 2 legs, so divide by 2
    return len(spread_positions) // 2


def calculate_portfolio_utilization(
    session: requests.Session,
    account_number: str,
    capital_at_risk: float
) -> float:
    """
    Calculate what % of portfolio would be used if opening a new position.

    Args:
        session: Authenticated session
        account_number: Account number
        capital_at_risk: Max loss of proposed new position

    Returns:
        Portfolio utilization as decimal (0.10 = 10%)
    """
    net_liq = calculate_account_net_liq(session, account_number)

    if net_liq <= 0:
        return 1.0  # 100% utilized if no liquidity

    return capital_at_risk / net_liq


def check_can_open_position(
    session: requests.Session,
    account_number: str,
    capital_at_risk: float,
    max_portfolio_pct: float = 0.10
) -> tuple[bool, str]:
    """
    Check if a new position can be opened within portfolio limits.

    Args:
        session: Authenticated session
        account_number: Account number
        capital_at_risk: Max loss of proposed position
        max_portfolio_pct: Maximum portfolio deployment (default: 0.10 = 10%)

    Returns:
        (can_open: bool, reason: str)
    """
    try:
        # Get current positions and calculate total risk
        positions = get_account_positions(session, account_number)
        net_liq = calculate_account_net_liq(session, account_number)

        if net_liq <= 0:
            return False, "Account has no available liquidity"

        # Calculate total capital currently deployed
        # This is a simplified calculation - you may want to track this more precisely
        total_deployed = sum(
            abs(calculate_position_current_value(
                pos, float(pos.get('close-price', 0))))
            for pos in positions
        )

        # Check if new position would exceed limit
        new_total = total_deployed + capital_at_risk
        new_utilization = new_total / net_liq

        if new_utilization > max_portfolio_pct:
            return False, f"Would exceed {max_portfolio_pct*100:.0f}% portfolio limit ({new_utilization*100:.1f}% total)"

        return True, f"OK - {new_utilization*100:.1f}% portfolio utilization"

    except Exception as e:
        return False, f"Error checking position limits: {e}"


# Helper function to get spread P&L for monitoring exits
def get_spread_pnl(
    position_legs: List[Dict[str, Any]],
    current_marks: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate P&L for a spread (2-leg position).

    Args:
        position_legs: List of 2 position dicts (short leg + long leg)
        current_marks: Dict of {symbol: current_mark_price}

    Returns:
        {
            'entry_credit': float,  # Original credit received
            'current_debit': float,  # Cost to close now
            'unrealized_pnl': float,  # Profit/loss
            'pnl_pct_of_credit': float  # P&L as % of original credit
        }
    """
    if len(position_legs) != 2:
        raise ValueError("Spread must have exactly 2 legs")

    # Calculate entry credit (what we received when opening)
    entry_credit = 0.0
    current_debit = 0.0

    for pos in position_legs:
        symbol = pos['symbol']
        quantity = float(pos['quantity'])
        multiplier = int(pos['multiplier'])
        avg_open_price = float(pos['average-open-price'])
        quantity_direction = pos['quantity-direction']

        # Entry value
        if quantity_direction == 'Short':
            # We sold this leg, received credit
            entry_credit += avg_open_price * abs(quantity) * multiplier
        else:
            # We bought this leg, paid debit
            entry_credit -= avg_open_price * abs(quantity) * multiplier

        # Current value to close
        current_mark = current_marks.get(symbol, avg_open_price)
        if quantity_direction == 'Short':
            # To close short, we buy back (debit)
            current_debit += current_mark * abs(quantity) * multiplier
        else:
            # To close long, we sell (credit)
            current_debit -= current_mark * abs(quantity) * multiplier

    unrealized_pnl = entry_credit - current_debit
    pnl_pct = (unrealized_pnl / entry_credit) if entry_credit > 0 else 0.0

    return {
        'entry_credit': entry_credit,
        'current_debit': current_debit,
        'unrealized_pnl': unrealized_pnl,
        'pnl_pct_of_credit': pnl_pct
    }


def get_account_summary(session: requests.Session, account_number: str) -> Dict[str, Any]:
    """
    Get key account metrics in one call.

    Returns:
        {
            'account_number': str,
            'net_liquidating_value': float,
            'cash_balance': float,
            'buying_power': float,
            'open_positions_count': int,
            'day_trade_count': int,
            'equity_buying_power': float
        }
    """
    balance = get_account_balance(session, account_number)
    positions = get_account_positions(session, account_number)

    return {
        'account_number': account_number,
        'net_liquidating_value': float(balance.get('net-liquidating-value', 0)),
        'cash_balance': float(balance.get('cash-balance', 0)),
        'buying_power': float(balance.get('derivative-buying-power', 0)),
        'open_positions_count': len(positions),
        'day_trade_count': int(balance.get('day-trade-count', 0)),
        'equity_buying_power': float(balance.get('equity-buying-power', 0))
    }
