"""Tastytrade authentication and token management."""

from integrations.tastytrade.auth import authenticate
from integrations.tastytrade.token_manager import (
    save_token,
    load_token,
    clear_saved_token,
    get_persistent_session
)
from integrations.tastytrade.config import (
    get_oauth_settings,
    get_oauth_scopes,
    get_api_base_url,
    get_streamer_url,
    is_sandbox
)
from integrations.tastytrade.account import (
    get_account_numbers,
    get_account_balance,
    get_account_positions,
    get_options_positions,
    calculate_account_net_liq,
    count_open_spreads,
    check_can_open_position,
    get_spread_pnl
)
from integrations.tastytrade.orders import (
    create_spread_order,
    create_close_spread_order,
    create_otoco_bracket_order,
    dry_run_order,
    submit_order,
    submit_complex_order,
    get_order_status,
    get_live_orders,
    cancel_order,
    open_spread_with_bracket
)

__all__ = [
    'authenticate',
    'save_token',
    'load_token',
    'clear_saved_token',
    'get_persistent_session',
    'get_oauth_settings',
    'get_oauth_scopes',
    'get_api_base_url',
    'get_streamer_url',
    'is_sandbox',
    'get_account_numbers',
    'get_account_balance',
    'get_account_positions',
    'get_options_positions',
    'calculate_account_net_liq',
    'count_open_spreads',
    'check_can_open_position',
    'get_spread_pnl',
    'create_spread_order',
    'create_close_spread_order',
    'create_otoco_bracket_order',
    'dry_run_order',
    'submit_order',
    'submit_complex_order',
    'get_order_status',
    'get_live_orders',
    'cancel_order',
    'open_spread_with_bracket'
]
