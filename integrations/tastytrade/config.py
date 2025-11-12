"""
Configuration and environment loading for the stock_scanner app.
Loads .env, sets OAuth-related constants, and exposes helper getters.
"""
from __future__ import annotations

import os
from typing import Tuple, List


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except Exception:
        # It's fine if python-dotenv isn't installed; we'll rely on process env
        print("⚠️  python-dotenv not installed. Set env vars in your shell or install python-dotenv.")


def _configure_oauthlib_for_local_dev() -> None:
    # Allow http://localhost redirect URIs during local development
    os.environ.setdefault('OAUTHLIB_INSECURE_TRANSPORT', '1')


def get_api_base_url() -> str:
    """Return the API base URL based on environment.

    Honors TASTYTRADE_ENV = 'sandbox' | 'production' (default production).
    Sandbox: api.cert.tastyworks.com (resets every 24 hours)
    Production: api.tastyworks.com (live trading)
    """
    env = os.getenv('TASTYTRADE_ENV', 'production').lower()
    if env == 'sandbox':
        return 'https://api.cert.tastyworks.com'
    else:
        return 'https://api.tastyworks.com'


def get_streamer_url() -> str:
    """Return the WebSocket streamer URL based on environment.

    Honors TASTYTRADE_ENV = 'sandbox' | 'production' (default production).
    """
    env = os.getenv('TASTYTRADE_ENV', 'production').lower()
    if env == 'sandbox':
        return 'wss://streamer.cert.tastyworks.com'
    else:
        return 'wss://streamer.tastyworks.com'


def is_sandbox() -> bool:
    """Check if running in sandbox environment."""
    return os.getenv('TASTYTRADE_ENV', 'production').lower() == 'sandbox'


def get_oauth_settings() -> Tuple[str, str, str, str, str]:
    """Return OAuth settings (authorization URL, token URL, client_id, client_secret, redirect_uri).

    Honors TASTYTRADE_ENV = 'sandbox' | 'production' (default production).

    For sandbox, tries TASTYTRADE_SANDBOX_CLIENT_ID first, falls back to TASTYTRADE_CLIENT_ID.
    Note: Sandbox requires separate OAuth credentials from Tastytrade support.
    Contact api.support@tastytrade.com to request sandbox client credentials.
    """
    env = os.getenv('TASTYTRADE_ENV', 'production').lower()
    if env == 'sandbox':
        authorization_base_url = 'https://cert-my.staging-tasty.works/auth.html'
        token_url = 'https://api.cert.tastyworks.com/oauth/token'
        # Sandbox uses separate credentials
        client_id = os.getenv('SANDBOX_TASTYTRADE_CLIENT_ID')
        client_secret = os.getenv('SANDBOX_TASTYTRADE_CLIENT_SECRET')

        if not client_id or not client_secret:
            print("❌ Sandbox mode requires separate OAuth credentials:")
            print(
                "   Set SANDBOX_TASTYTRADE_CLIENT_ID and SANDBOX_TASTYTRADE_CLIENT_SECRET")
            print("\n💡 Contact api.support@tastytrade.com to request sandbox credentials")
            print(
                "   Or test with paper trading (unset TASTYTRADE_ENV or set to 'production')")
            raise SystemExit(1)
    else:
        authorization_base_url = 'https://my.tastytrade.com/auth.html'
        token_url = 'https://api.tastyworks.com/oauth/token'
        # Production uses standard credentials
        client_id = os.getenv('TASTYTRADE_CLIENT_ID')
        client_secret = os.getenv('TASTYTRADE_CLIENT_SECRET')

    redirect_uri = os.getenv('TASTYTRADE_REDIRECT_URI')

    if not client_id or not client_secret or not redirect_uri:
        print("❌ Missing required environment variables")
        if env == 'sandbox':
            print("   Need: SANDBOX_TASTYTRADE_CLIENT_ID, SANDBOX_TASTYTRADE_CLIENT_SECRET, TASTYTRADE_REDIRECT_URI")
        else:
            print(
                "   Need: TASTYTRADE_CLIENT_ID, TASTYTRADE_CLIENT_SECRET, TASTYTRADE_REDIRECT_URI")
        raise SystemExit(1)

    return authorization_base_url, token_url, str(client_id), str(client_secret), str(redirect_uri)


def get_oauth_scopes() -> List[str]:
    """Return list of OAuth scopes to request.

    Configure via env var TASTYTRADE_SCOPES using space- or comma-separated values.
    Defaults to 'openid offline_access'. If you plan to stream market data, include
    'market-data' (or the scope name provided by tastytrade) in this list.
    """
    # Default to the scopes you specified: read, trade, openid
    # (order doesn't matter)
    raw = os.getenv('TASTYTRADE_SCOPES', 'openid read trade')
    # Support both comma and whitespace separators
    raw = raw.replace(',', ' ')
    scopes = [s for s in (raw.split()) if s]
    return scopes


# Side effects on import: load .env and configure oauthlib
_try_load_dotenv()
_configure_oauthlib_for_local_dev()
