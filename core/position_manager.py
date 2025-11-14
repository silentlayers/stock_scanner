"""
Position Manager - Monitor and manage options positions

Handles:
- Monitoring positions for DTE thresholds
- Automatic closing at 21 DTE
- P&L tracking
- Position health checks
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import logging
import requests

from integrations.tastytrade.account import get_account_positions, get_options_positions
from integrations.tastytrade.orders import (
    create_close_spread_order,
    submit_order,
    get_live_orders,
    cancel_order,
    cancel_complex_order
)

logger = logging.getLogger(__name__)


def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse an option symbol to extract expiration date and other details.

    Format: SYMBOL YYMMDDCPPPPPPPPP
    Example: SPY   251219P00500000

    Args:
        symbol: Option symbol string

    Returns:
        Dict with:
        - underlying: str (e.g., 'SPY')
        - expiration_date: date object
        - option_type: str ('C' or 'P')
        - strike: float
    """
    # Strip whitespace
    symbol = symbol.strip()

    # Extract components
    # First 6 chars: underlying (may have spaces)
    underlying = symbol[:6].strip()

    # Next 6 chars: YYMMDD
    date_str = symbol[6:12]
    year = 2000 + int(date_str[0:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    expiration_date = date(year, month, day)

    # Next char: C or P
    option_type = symbol[12]

    # Last 8 chars: strike price (in cents, so divide by 1000)
    strike_str = symbol[13:21]
    strike = int(strike_str) / 1000.0

    return {
        'underlying': underlying,
        'expiration_date': expiration_date,
        'option_type': option_type,
        'strike': strike,
        'symbol': symbol
    }


def calculate_dte(expiration_date: date) -> int:
    """
    Calculate days to expiration from today.

    Args:
        expiration_date: Expiration date

    Returns:
        Number of calendar days to expiration
    """
    today = date.today()
    delta = expiration_date - today
    return delta.days


def get_positions_by_dte(
    session: requests.Session,
    account_number: str,
    dte_threshold: int = 21,
    underlying: str = 'SPY'
) -> List[Dict[str, Any]]:
    """
    Get all positions that are at or below the DTE threshold.

    Args:
        session: Authenticated session
        account_number: Account number
        dte_threshold: DTE threshold (default: 21)
        underlying: Filter by underlying (default: 'SPY')

    Returns:
        List of positions that should be closed
    """
    # Get all options positions
    positions = get_options_positions(session, account_number)

    # Filter by underlying
    positions = [p for p in positions if p.get(
        'underlying-symbol') == underlying]

    positions_to_close = []

    for pos in positions:
        try:
            symbol = pos.get('symbol', '')
            parsed = parse_option_symbol(symbol)
            dte = calculate_dte(parsed['expiration_date'])

            if dte <= dte_threshold:
                pos['parsed_symbol'] = parsed
                pos['dte'] = dte
                positions_to_close.append(pos)
                logger.info(
                    f"Position {symbol} has {dte} DTE (≤ {dte_threshold})")

        except Exception as e:
            logger.error(f"Failed to parse symbol {pos.get('symbol')}: {e}")
            continue

    return positions_to_close


def group_positions_into_spreads(
    positions: List[Dict[str, Any]]
) -> List[Tuple[Dict, Dict]]:
    """
    Group positions into spreads (pairs of short + long with same expiration).

    Args:
        positions: List of position dicts

    Returns:
        List of (short_leg, long_leg) tuples
    """
    spreads = []

    # Group by expiration date
    by_expiration = {}
    for pos in positions:
        exp_date = pos['parsed_symbol']['expiration_date']
        if exp_date not in by_expiration:
            by_expiration[exp_date] = []
        by_expiration[exp_date].append(pos)

    # Within each expiration, pair short + long
    for exp_date, positions_list in by_expiration.items():
        # Sort by strike (descending for puts)
        positions_list.sort(
            key=lambda p: p['parsed_symbol']['strike'], reverse=True)

        # Find pairs: short should have higher strike than long (for put credit spread)
        for i in range(len(positions_list)):
            for j in range(i + 1, len(positions_list)):
                pos1 = positions_list[i]
                pos2 = positions_list[j]

                # Check if one is short and one is long
                dir1 = pos1.get('quantity-direction')
                dir2 = pos2.get('quantity-direction')

                if dir1 != dir2:
                    # One is short, one is long - this is a spread
                    if dir1 == 'Short':
                        short_leg = pos1
                        long_leg = pos2
                    else:
                        short_leg = pos2
                        long_leg = pos1

                    spreads.append((short_leg, long_leg))
                    break

    return spreads


def close_position_at_market(
    session: requests.Session,
    account_number: str,
    short_leg: Dict[str, Any],
    long_leg: Dict[str, Any],
    reason: str = "21 DTE threshold"
) -> Dict[str, Any]:
    """
    Close a spread position at market price.

    This will:
    1. Find and cancel any related OTOCO/OCO orders (take profit/stop loss)
    2. Submit a market order to close the spread

    Args:
        session: Authenticated session
        account_number: Account number
        short_leg: Short leg position dict
        long_leg: Long leg position dict
        reason: Reason for closing

    Returns:
        Order response dict with cancellation info
    """
    short_symbol = short_leg['symbol']
    long_symbol = long_leg['symbol']
    num_contracts = int(abs(float(short_leg['quantity'])))

    logger.info(f"Closing spread: {short_symbol} / {long_symbol}")
    logger.info(f"Reason: {reason}")

    # Step 1: Cancel any existing OTOCO/OCO orders for this spread
    cancelled_orders = cancel_related_exit_orders(
        session, account_number, short_symbol, long_symbol
    )

    if cancelled_orders:
        logger.info(
            f"✅ Cancelled {len(cancelled_orders)} related exit order(s)")

    # Step 2: Calculate close price
    short_close_price = float(short_leg.get('close-price', 0))
    long_close_price = float(long_leg.get('close-price', 0))

    # Spread debit = (short close - long close) for put credit spread closing
    debit = short_close_price - long_close_price

    # Add some buffer to ensure fill (5% worse than mid)
    debit_with_buffer = debit * 1.05

    logger.info(f"Contracts: {num_contracts}, Debit: ${debit_with_buffer:.2f}")

    # Step 3: Create and submit close order
    close_order = create_close_spread_order(
        sell_leg_symbol=short_symbol,
        buy_leg_symbol=long_symbol,
        num_contracts=num_contracts,
        debit=debit_with_buffer,
        time_in_force="Day"
    )

    try:
        result = submit_order(session, account_number, close_order)
        logger.info(f"✅ Close order submitted: Order ID {result.get('id')}")

        # Add cancellation info to result
        result['cancelled_exit_orders'] = cancelled_orders

        return result
    except Exception as e:
        logger.error(f"❌ Failed to submit close order: {e}")
        raise


def cancel_related_exit_orders(
    session: requests.Session,
    account_number: str,
    short_symbol: str,
    long_symbol: str
) -> List[Dict[str, Any]]:
    """
    Find and cancel any OTOCO/OCO exit orders related to a spread.

    This looks for:
    - Live orders with matching symbols (take profit, stop loss)
    - Orders with "Buy to Close" and "Sell to Close" actions

    Args:
        session: Authenticated session
        account_number: Account number
        short_symbol: Short leg symbol
        long_symbol: Long leg symbol

    Returns:
        List of cancelled order responses
    """
    try:
        # Get all live orders (working, not filled)
        live_orders = get_live_orders(session, account_number)

        cancelled = []

        for order in live_orders:
            # Check if this order involves our spread symbols
            legs = order.get('legs', [])
            if not legs:
                continue

            # Get symbols from legs
            order_symbols = {leg.get('symbol') for leg in legs}

            # Check if this order contains both our spread legs
            if short_symbol in order_symbols and long_symbol in order_symbols:
                # Check if it's a closing order (Buy to Close, Sell to Close)
                actions = {leg.get('action') for leg in legs}
                is_close_order = 'Buy to Close' in actions or 'Sell to Close' in actions

                if is_close_order:
                    order_id = order.get('id')
                    order_status = order.get('status')

                    # Only cancel if order is still live and has valid ID
                    if order_id and order_status in ['Live', 'Received', 'Routed', 'In Flight']:
                        logger.info(
                            f"Cancelling exit order: ID {order_id} (Status: {order_status})")

                        try:
                            # Check if it's a complex order (OTOCO/OCO)
                            if order.get('complex-order-id'):
                                # This is part of a complex order - cancel the complex order
                                complex_id = order['complex-order-id']
                                logger.info(
                                    f"Cancelling complex order: ID {complex_id}")
                                cancel_result = cancel_complex_order(
                                    session, account_number, complex_id)
                            else:
                                # Simple order - cancel directly
                                cancel_result = cancel_order(
                                    session, account_number, order_id)

                            cancelled.append({
                                'order_id': order_id,
                                'complex_order_id': order.get('complex-order-id'),
                                'status': 'cancelled',
                                'result': cancel_result
                            })

                        except Exception as e:
                            logger.warning(
                                f"Failed to cancel order {order_id}: {e}")
                            cancelled.append({
                                'order_id': order_id,
                                'status': 'failed',
                                'error': str(e)
                            })

        return cancelled

    except Exception as e:
        logger.error(f"Failed to retrieve/cancel exit orders: {e}")
        return []


def monitor_and_close_positions(
    session: requests.Session,
    account_number: str,
    dte_threshold: int = 21,
    underlying: str = 'SPY',
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    Monitor all positions and close any at/below DTE threshold.

    Args:
        session: Authenticated session
        account_number: Account number
        dte_threshold: Close positions at this DTE or less (default: 21)
        underlying: Filter by underlying (default: 'SPY')
        dry_run: If True, don't actually submit orders (default: False)

    Returns:
        List of close order results (or dry run info)
    """
    logger.info(
        f"Monitoring positions for {underlying} with DTE ≤ {dte_threshold}")

    # Get positions at/below threshold
    positions_to_close = get_positions_by_dte(
        session, account_number, dte_threshold, underlying
    )

    if not positions_to_close:
        logger.info(f"No positions found with DTE ≤ {dte_threshold}")
        return []

    logger.info(f"Found {len(positions_to_close)} positions to close")

    # Group into spreads
    spreads = group_positions_into_spreads(positions_to_close)

    if not spreads:
        logger.warning("Could not group positions into spreads")
        return []

    logger.info(f"Identified {len(spreads)} spreads to close")

    results = []

    for short_leg, long_leg in spreads:
        short_symbol = short_leg['symbol']
        long_symbol = long_leg['symbol']
        dte = short_leg['dte']

        logger.info(f"\nSpread: {short_symbol} / {long_symbol}")
        logger.info(f"DTE: {dte} days")

        if dry_run:
            logger.info("DRY RUN - Would close this spread")
            results.append({
                'dry_run': True,
                'short_leg': short_symbol,
                'long_leg': long_symbol,
                'dte': dte,
                'action': 'would_close'
            })
        else:
            try:
                order_result = close_position_at_market(
                    session, account_number, short_leg, long_leg,
                    reason=f"{dte} DTE (threshold: {dte_threshold})"
                )
                results.append(order_result)
            except Exception as e:
                logger.error(f"Failed to close spread: {e}")
                results.append({
                    'error': str(e),
                    'short_leg': short_symbol,
                    'long_leg': long_symbol
                })

    return results


def get_positions_summary(
    session: requests.Session,
    account_number: str,
    underlying: str = 'SPY'
) -> List[Dict[str, Any]]:
    """
    Get a summary of all open positions with DTE information.

    Args:
        session: Authenticated session
        account_number: Account number
        underlying: Filter by underlying

    Returns:
        List of position summaries with DTE
    """
    positions = get_options_positions(session, account_number)
    positions = [p for p in positions if p.get(
        'underlying-symbol') == underlying]

    summary = []

    for pos in positions:
        try:
            symbol = pos.get('symbol', '')
            parsed = parse_option_symbol(symbol)
            dte = calculate_dte(parsed['expiration_date'])

            summary.append({
                'symbol': symbol,
                'underlying': parsed['underlying'],
                'expiration': parsed['expiration_date'].isoformat(),
                'dte': dte,
                'option_type': parsed['option_type'],
                'strike': parsed['strike'],
                'quantity': float(pos.get('quantity', 0)),
                'direction': pos.get('quantity-direction'),
                'close_price': float(pos.get('close-price', 0)),
                'unrealized_pnl': float(pos.get('unrealized-day-gain-effect-absolute', 0))
            })

        except Exception as e:
            logger.error(f"Failed to parse position {pos.get('symbol')}: {e}")
            continue

    # Sort by DTE
    summary.sort(key=lambda x: x['dte'])

    return summary
