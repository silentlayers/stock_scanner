"""
Tastytrade order management for automated trading.

This module provides functions to:
- Place spread orders (open positions)
- Close spread orders (exit positions)  
- Place bracket orders (OTOCO - one triggers OCO)
- Dry run orders (validate before submitting)
- Cancel orders
- Monitor order status

API Documentation: https://developer.tastytrade.com/
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal
import requests

from integrations.tastytrade.config import get_api_base_url


OrderType = Literal["Limit", "Market", "Stop", "Stop Limit"]
TimeInForce = Literal["Day", "GTC", "GTD"]
Action = Literal["Buy to Open", "Sell to Open",
                 "Buy to Close", "Sell to Close"]
PriceEffect = Literal["Credit", "Debit"]


def format_option_symbol(
    underlying: str,
    expiration_date: str,
    option_type: Literal["C", "P"],
    strike: float
) -> str:
    """
    Format an option symbol for Tastytrade API.

    Format: SYMBOL YYMMDDCPPPPPPPPP
    Example: SPY   251219P00500000

    Args:
        underlying: Underlying symbol (e.g., 'SPY')
        expiration_date: Expiration date in YYYY-MM-DD format
        option_type: 'C' for call, 'P' for put
        strike: Strike price (e.g., 500.0)

    Returns:
        Formatted option symbol
    """
    # Parse date: 2025-12-19 -> 251219
    year, month, day = expiration_date.split('-')
    date_part = f"{year[2:]}{month}{day}"

    # Format strike: 500.0 -> 00500000 (8 digits, strike * 1000)
    strike_part = f"{int(strike * 1000):08d}"

    # Pad underlying to 6 characters
    underlying_part = underlying.ljust(6)

    return f"{underlying_part}{date_part}{option_type}{strike_part}"


def create_spread_order(
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    credit: float,
    time_in_force: TimeInForce = "Day"
) -> Dict[str, Any]:
    """
    Create a put credit spread order for submission.

    Args:
        underlying: Underlying symbol (e.g., 'SPY')
        sell_leg: Dict with keys: 'symbol', 'strike-price', 'expiration-date'
        buy_leg: Dict with keys: 'symbol', 'strike-price', 'expiration-date'
        num_contracts: Number of contracts to trade
        credit: Limit price (credit received)
        time_in_force: Order duration ('Day', 'GTC', 'GTD')

    Returns:
        Order dict ready for submission
    """
    return {
        "time-in-force": time_in_force,
        "order-type": "Limit",
        "price": f"{credit:.2f}",
        "price-effect": "Credit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": sell_leg['symbol'],
                "quantity": num_contracts,
                "action": "Sell to Open"
            },
            {
                "instrument-type": "Equity Option",
                "symbol": buy_leg['symbol'],
                "quantity": num_contracts,
                "action": "Buy to Open"
            }
        ]
    }


def create_close_spread_order(
    sell_leg_symbol: str,
    buy_leg_symbol: str,
    num_contracts: int,
    debit: float,
    time_in_force: TimeInForce = "Day"
) -> Dict[str, Any]:
    """
    Create an order to close an existing put credit spread.

    Args:
        sell_leg_symbol: Symbol of the short put leg
        buy_leg_symbol: Symbol of the long put leg
        num_contracts: Number of contracts to close
        debit: Limit price (debit paid to close)
        time_in_force: Order duration

    Returns:
        Order dict ready for submission
    """
    return {
        "time-in-force": time_in_force,
        "order-type": "Limit",
        "price": f"{debit:.2f}",
        "price-effect": "Debit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": sell_leg_symbol,
                "quantity": num_contracts,
                "action": "Buy to Close"  # Close the short leg
            },
            {
                "instrument-type": "Equity Option",
                "symbol": buy_leg_symbol,
                "quantity": num_contracts,
                "action": "Sell to Close"  # Close the long leg
            }
        ]
    }


def dry_run_order(
    session: requests.Session,
    account_number: str,
    order: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate an order without submitting it (dry run).

    This checks:
    - Order validity (symbols, strikes, expiration)
    - Account buying power
    - Existing positions and conflicts

    Args:
        session: Authenticated session
        account_number: Account number
        order: Order dict from create_spread_order() or create_close_spread_order()

    Returns:
        Response dict with:
        {
            'order': {...},
            'warnings': [...],
            'buying-power-effect': {...},
            'fee-calculation': {...}
        }

    Raises:
        RuntimeError if dry run fails
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/orders/dry-run"
    resp = session.post(url, json=order)

    if resp.status_code != 200 and resp.status_code != 201:
        raise RuntimeError(f"Dry run failed: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


def submit_order(
    session: requests.Session,
    account_number: str,
    order: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Submit an order to open or close a spread.

    WARNING: This actually places a live order!
    Always run dry_run_order() first to validate.

    Args:
        session: Authenticated session
        account_number: Account number
        order: Order dict (same format as dry run)

    Returns:
        Response dict with order details including 'id' for tracking

    Raises:
        RuntimeError if order submission fails
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/orders"
    resp = session.post(url, json=order)

    if resp.status_code != 200 and resp.status_code != 201:
        raise RuntimeError(
            f"Order submission failed: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


def create_otoco_bracket_order(
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Create an OTOCO bracket order (One Triggers OCO).

    This places:
    1. Trigger order: Opens the spread at entry_credit
    2. Take profit: Closes at 50% of credit (default)
    3. Stop loss: Closes at 100% loss (default)

    Args:
        underlying: Underlying symbol
        sell_leg: Short put leg details
        buy_leg: Long put leg details
        num_contracts: Number of contracts
        entry_credit: Credit to receive when opening
        take_profit_pct: % of credit to take profit (0.5 = 50%)
        stop_loss_pct: % of credit for stop loss (1.0 = 100% loss)

    Returns:
        OTOCO order dict ready for submission
    """
    # Calculate profit and loss prices
    take_profit_debit = entry_credit * (1 - take_profit_pct)
    stop_loss_debit = entry_credit * (1 + stop_loss_pct)

    return {
        "type": "OTOCO",
        "trigger-order": {
            "time-in-force": "Day",
            "order-type": "Limit",
            "underlying-symbol": underlying,
            "price": f"{entry_credit:.2f}",
            "price-effect": "Credit",
            "legs": [
                {
                    "instrument-type": "Equity Option",
                    "symbol": sell_leg['symbol'],
                    "quantity": num_contracts,
                    "action": "Sell to Open"
                },
                {
                    "instrument-type": "Equity Option",
                    "symbol": buy_leg['symbol'],
                    "quantity": num_contracts,
                    "action": "Buy to Open"
                }
            ],
            "advanced-instructions": {
                "strict-position-effect-validation": False
            }
        },
        "orders": [
            # Take profit order (Limit - closes at profit target)
            {
                "time-in-force": "GTC",
                "order-type": "Limit",
                "underlying-symbol": underlying,
                "price": f"{take_profit_debit:.2f}",
                "price-effect": "Debit",
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": sell_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Buy to Close"
                    },
                    {
                        "instrument-type": "Equity Option",
                        "symbol": buy_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Sell to Close"
                    }
                ],
                "advanced-instructions": {
                    "strict-position-effect-validation": False
                }
            },
            # Stop loss order (Stop Limit - closes at loss limit)
            {
                "time-in-force": "GTC",
                "order-type": "Stop Limit",
                "underlying-symbol": underlying,
                "stop-trigger": f"{stop_loss_debit:.2f}",
                "price": f"{stop_loss_debit:.2f}",
                "price-effect": "Debit",
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": sell_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Buy to Close"
                    },
                    {
                        "instrument-type": "Equity Option",
                        "symbol": buy_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Sell to Close"
                    }
                ],
                "advanced-instructions": {
                    "strict-position-effect-validation": False
                }
            }
        ]
    }


def submit_complex_order(
    session: requests.Session,
    account_number: str,
    complex_order: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Submit a bracket order (OTOCO, OTO, or OCO).

    Args:
        session: Authenticated session
        account_number: Account number
        complex_order: Complex order dict from create_otoco_bracket_order()

    Returns:
        Response with complex order id and nested order ids

    Raises:
        RuntimeError if submission fails
    """
    import time

    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/complex-orders"

    # Retry logic for transient server errors
    max_retries = 3
    retry_delay = 2  # seconds

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = session.post(url, json=complex_order)

            # Success
            if resp.status_code in (200, 201):
                data = resp.json()
                return data.get('data', {})

            # Transient server errors - retry
            if resp.status_code in (502, 503, 504):
                last_error = f"{resp.status_code} {resp.text}"
                if attempt < max_retries - 1:
                    print(
                        f"⚠️ Server error {resp.status_code}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise RuntimeError(
                        f"Complex order submission failed after {max_retries} retries: {last_error}")

            # All other errors - raise immediately
            raise RuntimeError(
                f"Complex order submission failed: {resp.status_code} {resp.text}")

        except Exception as e:
            if "Complex order submission failed" in str(e):
                raise  # Re-raise our own errors
            # Network errors or other issues
            last_error = str(e)
            if attempt < max_retries - 1:
                print(
                    f"⚠️ Request error: {e}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise RuntimeError(
                    f"Complex order submission failed after {max_retries} retries: {last_error}")

    # Should not reach here, but just in case
    raise RuntimeError(f"Complex order submission failed: {last_error}")


def get_order_status(
    session: requests.Session,
    account_number: str,
    order_id: int
) -> Dict[str, Any]:
    """
    Get the status of an order.

    Args:
        session: Authenticated session
        account_number: Account number
        order_id: Order ID from submit_order response

    Returns:
        Order dict with current status
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/orders/{order_id}"
    resp = session.get(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to get order status: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


def get_live_orders(
    session: requests.Session,
    account_number: str
) -> List[Dict[str, Any]]:
    """
    Get all orders created or updated today.

    Returns orders with various statuses: Live, Filled, Cancelled, etc.

    Args:
        session: Authenticated session
        account_number: Account number

    Returns:
        List of order dicts
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/orders/live"
    resp = session.get(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to get live orders: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {}).get('items', [])


def cancel_order(
    session: requests.Session,
    account_number: str,
    order_id: int
) -> Dict[str, Any]:
    """
    Cancel an order.

    Args:
        session: Authenticated session
        account_number: Account number
        order_id: Order ID to cancel

    Returns:
        Order dict with status "Cancel Requested"

    Raises:
        RuntimeError if order cannot be cancelled (e.g., already filled)
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/orders/{order_id}"
    resp = session.delete(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to cancel order: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


def cancel_complex_order(
    session: requests.Session,
    account_number: str,
    complex_order_id: int
) -> Dict[str, Any]:
    """
    Cancel a bracket order (OTOCO/OTO/OCO).

    Args:
        session: Authenticated session
        account_number: Account number
        complex_order_id: Complex order ID (not the trigger or nested order IDs)

    Returns:
        Complex order dict with cancelled status
    """
    base_url = get_api_base_url()
    url = f"{base_url}/accounts/{account_number}/complex-orders/{complex_order_id}"
    resp = session.delete(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to cancel complex order: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get('data', {})


# Helper function for automation
def open_spread_with_bracket(
    session: requests.Session,
    account_number: str,
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 1.0,
    dry_run_first: bool = True
) -> Dict[str, Any]:
    """
    Open a spread with automatic profit/loss exit orders (OTOCO bracket).

    This is the main function for automated trading.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: 'SPY'
        sell_leg: Short put leg from scanner
        buy_leg: Long put leg from scanner
        num_contracts: Number of contracts (from position sizing)
        entry_credit: Credit to collect
        take_profit_pct: Close at 50% profit (default)
        stop_loss_pct: Close at 100% loss (default)
        dry_run_first: Validate before submitting (recommended)

    Returns:
        Complex order response with all IDs for tracking

    Raises:
        RuntimeError if order fails validation or submission
    """
    # Create bracket order
    bracket = create_otoco_bracket_order(
        underlying=underlying,
        sell_leg=sell_leg,
        buy_leg=buy_leg,
        num_contracts=num_contracts,
        entry_credit=entry_credit,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct
    )

    # Optional: Dry run the trigger order first
    if dry_run_first:
        trigger_order = bracket['trigger-order']
        dry_result = dry_run_order(session, account_number, trigger_order)

        # Log warnings but don't fail (sandbox may have expected warnings)
        if dry_result.get('warnings'):
            warnings = dry_result['warnings']
            print("⚠️ Dry run warnings:")
            for w in warnings:
                print(f"  - {w}")

    # Submit the bracket order
    return submit_complex_order(session, account_number, bracket)


def open_spread_simple(
    session: requests.Session,
    account_number: str,
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    dry_run_first: bool = True
) -> Dict[str, Any]:
    """
    Open a spread with a simple limit order (no bracket/OCO).

    Use this for sandbox testing when OTOCO orders don't work.
    After the order fills, you can manually place OCO exit orders.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: 'SPY'
        sell_leg: Short put leg from scanner
        buy_leg: Long put leg from scanner
        num_contracts: Number of contracts
        entry_credit: Credit to collect
        dry_run_first: Validate before submitting

    Returns:
        Order response with order ID

    Raises:
        RuntimeError if order fails
    """
    # Create simple opening order
    opening_order = {
        "time-in-force": "Day",
        "order-type": "Limit",
        "underlying-symbol": underlying,
        "price": f"{entry_credit:.2f}",
        "price-effect": "Credit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": sell_leg['symbol'],
                "quantity": num_contracts,
                "action": "Sell to Open"
            },
            {
                "instrument-type": "Equity Option",
                "symbol": buy_leg['symbol'],
                "quantity": num_contracts,
                "action": "Buy to Open"
            }
        ]
    }

    # Optional: Dry run first
    if dry_run_first:
        dry_result = dry_run_order(session, account_number, opening_order)

        # Log warnings but don't fail
        if dry_result.get('warnings'):
            warnings = dry_result['warnings']
            print("⚠️ Dry run warnings:")
            for w in warnings:
                print(f"  - {w}")

    # Submit the simple order
    return submit_order(session, account_number, opening_order)


def create_oco_exit_orders(
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Create OCO (One Cancels Other) exit orders for an existing spread position.

    Use this to place profit target and stop loss after opening a position,
    since OTOCO orders don't work in sandbox.

    Args:
        underlying: Underlying symbol
        sell_leg: Short put leg details
        buy_leg: Long put leg details  
        num_contracts: Number of contracts in the position
        entry_credit: Credit received when opening
        take_profit_pct: % of credit to take profit (0.5 = 50%)
        stop_loss_pct: % of credit for stop loss (1.0 = 100% loss)

    Returns:
        OCO order dict ready for submission
    """
    # Calculate exit prices
    take_profit_debit = entry_credit * (1 - take_profit_pct)
    stop_loss_debit = entry_credit * (1 + stop_loss_pct)

    return {
        "type": "OCO",
        "orders": [
            # Take profit order (Limit - closes at profit target)
            {
                "time-in-force": "GTC",
                "order-type": "Limit",
                "underlying-symbol": underlying,
                "price": f"{take_profit_debit:.2f}",
                "price-effect": "Debit",
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": sell_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Buy to Close"
                    },
                    {
                        "instrument-type": "Equity Option",
                        "symbol": buy_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Sell to Close"
                    }
                ]
            },
            # Stop loss order (Stop Limit - closes at loss limit)
            {
                "time-in-force": "GTC",
                "order-type": "Stop Limit",
                "underlying-symbol": underlying,
                "stop-trigger": f"{stop_loss_debit:.2f}",
                "price": f"{stop_loss_debit:.2f}",
                "price-effect": "Debit",
                "legs": [
                    {
                        "instrument-type": "Equity Option",
                        "symbol": sell_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Buy to Close"
                    },
                    {
                        "instrument-type": "Equity Option",
                        "symbol": buy_leg['symbol'],
                        "quantity": num_contracts,
                        "action": "Sell to Close"
                    }
                ]
            }
        ]
    }


def place_oco_exit_for_position(
    session: requests.Session,
    account_number: str,
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Place OCO exit orders for an existing spread position.

    This is the workaround for sandbox not supporting OTOCO orders.
    First open the position with open_spread_simple(), then call this
    to place the profit/loss exit orders.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: 'SPY'
        sell_leg: Short put leg from scanner
        buy_leg: Long put leg from scanner
        num_contracts: Number of contracts in the open position
        entry_credit: Credit received when opening
        take_profit_pct: Close at X% profit (0.5 = 50%)
        stop_loss_pct: Close at X% loss (1.0 = 100%)

    Returns:
        Complex order response with OCO order IDs

    Raises:
        RuntimeError if OCO submission fails
    """
    # Create OCO exit orders
    oco = create_oco_exit_orders(
        underlying=underlying,
        sell_leg=sell_leg,
        buy_leg=buy_leg,
        num_contracts=num_contracts,
        entry_credit=entry_credit,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct
    )

    # Submit the OCO order
    return submit_complex_order(session, account_number, oco)


def place_separate_exit_orders(
    session: requests.Session,
    account_number: str,
    underlying: str,
    sell_leg: Dict[str, Any],
    buy_leg: Dict[str, Any],
    num_contracts: int,
    entry_credit: float,
    take_profit_pct: float = 0.5,
    stop_loss_pct: float = 1.0
) -> Dict[str, Any]:
    """
    Place separate (non-OCO) exit orders for sandbox testing.

    Since sandbox doesn't support OCO orders, this places two independent orders.
    You'll need to manually cancel the other one when one fills.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: 'SPY'
        sell_leg: Short put leg from scanner
        buy_leg: Long put leg from scanner
        num_contracts: Number of contracts in the open position
        entry_credit: Credit received when opening
        take_profit_pct: Close at X% profit (0.5 = 50%)
        stop_loss_pct: Close at X% loss (1.0 = 100%)

    Returns:
        Dict with both order responses

    Raises:
        RuntimeError if order submission fails
    """
    # Calculate exit prices
    take_profit_debit = entry_credit * (1 - take_profit_pct)
    stop_loss_debit = entry_credit * (1 + stop_loss_pct)

    # Take profit order (Limit)
    tp_order = {
        "time-in-force": "GTC",
        "order-type": "Limit",
        "underlying-symbol": underlying,
        "price": f"{take_profit_debit:.2f}",
        "price-effect": "Debit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": sell_leg['symbol'],
                "quantity": num_contracts,
                "action": "Buy to Close"
            },
            {
                "instrument-type": "Equity Option",
                "symbol": buy_leg['symbol'],
                "quantity": num_contracts,
                "action": "Sell to Close"
            }
        ]
    }

    # Stop loss order (Stop Limit)
    sl_order = {
        "time-in-force": "GTC",
        "order-type": "Stop Limit",
        "underlying-symbol": underlying,
        "stop-trigger": f"{stop_loss_debit:.2f}",
        "price": f"{stop_loss_debit:.2f}",
        "price-effect": "Debit",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": sell_leg['symbol'],
                "quantity": num_contracts,
                "action": "Buy to Close"
            },
            {
                "instrument-type": "Equity Option",
                "symbol": buy_leg['symbol'],
                "quantity": num_contracts,
                "action": "Sell to Close"
            }
        ]
    }

    # Submit both orders separately
    tp_result = submit_order(session, account_number, tp_order)
    sl_result = submit_order(session, account_number, sl_order)

    return {
        "take_profit_order": tp_result,
        "stop_loss_order": sl_result,
        "note": "These are independent orders - you must manually cancel one when the other fills"
    }
