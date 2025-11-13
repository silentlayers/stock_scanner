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

    Honors TASTYTRADE_USE_PRODUCTION = 'true' | 'false' (default true).
    Production (true): api.tastyworks.com (live trading)
    Sandbox (false): api.cert.tastyworks.com (resets every 24 hours)
    """
    use_production = os.getenv(
        'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')
    if use_production:
        return 'https://api.tastyworks.com'
    else:
        return 'https://api.cert.tastyworks.com'


def get_streamer_url() -> str:
    """Return the WebSocket streamer URL based on environment.

    Honors TASTYTRADE_USE_PRODUCTION = 'true' | 'false' (default true).
    """
    use_production = os.getenv(
        'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')
    if use_production:
        return 'wss://streamer.tastyworks.com'
    else:
        return 'wss://streamer.cert.tastyworks.com'


def is_sandbox() -> bool:
    """Check if running in sandbox environment."""
    use_production = os.getenv(
        'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')
    return not use_production


def get_oauth_settings() -> Tuple[str, str, str, str, str]:
    """Return OAuth settings (authorization URL, token URL, client_id, client_secret, redirect_uri).

    Honors TASTYTRADE_USE_PRODUCTION = 'true' | 'false' (default true).

    For sandbox, tries SANDBOX_TASTYTRADE_CLIENT_ID first.
    Note: Sandbox requires separate OAuth credentials from Tastytrade support.
    Contact api.support@tastytrade.com to request sandbox client credentials.
    """
    use_production = os.getenv(
        'TASTYTRADE_USE_PRODUCTION', 'true').lower() in ('true', '1', 'yes')

    if use_production:
        authorization_base_url = 'https://my.tastytrade.com/auth.html'
        token_url = 'https://api.tastyworks.com/oauth/token'
        # Production uses standard credentials
        client_id = os.getenv('TASTYTRADE_CLIENT_ID')
        client_secret = os.getenv('TASTYTRADE_CLIENT_SECRET')
    else:
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
                "   Or test with production (set TASTYTRADE_USE_PRODUCTION=true)")
            raise SystemExit(1)

    # Auto-detect redirect URI based on environment
    # Check if running on Streamlit Cloud first
    hostname = os.getenv('HOSTNAME', '')
    if hostname.startswith('streamlit'):
        # On Streamlit Cloud - use STREAMLIT_APP_URL
        redirect_uri = os.getenv('STREAMLIT_APP_URL')
        if not redirect_uri:
            print("⚠️  Running on Streamlit Cloud but STREAMLIT_APP_URL not set")
            print("   Please set STREAMLIT_APP_URL in Streamlit Cloud secrets")
            print("   Example: STREAMLIT_APP_URL = 'https://your-app.streamlit.app'")
            # Fall back to manual setting
            redirect_uri = os.getenv('TASTYTRADE_REDIRECT_URI')
    else:
        # Local development - check manual setting first, then default
        redirect_uri = os.getenv('TASTYTRADE_REDIRECT_URI')
        if not redirect_uri:
            redirect_uri = 'http://localhost:3000/callback'
            print(f"ℹ️  Using default local redirect URI: {redirect_uri}")

    if not client_id or not client_secret or not redirect_uri:
        print("❌ Missing required environment variables")
        if not use_production:
            print(
                "   Need: SANDBOX_TASTYTRADE_CLIENT_ID, SANDBOX_TASTYTRADE_CLIENT_SECRET")
        else:
            print(
                "   Need: TASTYTRADE_CLIENT_ID, TASTYTRADE_CLIENT_SECRET")
        print("   And either: TASTYTRADE_REDIRECT_URI or STREAMLIT_APP_URL (on cloud)")
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
