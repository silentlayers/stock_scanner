"""
Token persistence and management for OAuth authentication.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import requests
from requests_oauthlib import OAuth2Session

from integrations.tastytrade.config import get_oauth_settings


def get_token_file_path() -> Path:
    """Get the path to the token storage file within the project directory."""
    # Get the directory where this script is located (stock_scanner folder)
    project_dir = Path(__file__).parent
    return project_dir / '.tastytrade_tokens.json'


def save_token(token: Dict[str, Any]) -> None:
    """Save token to local file with restricted permissions."""
    token_file = get_token_file_path()

    # Add timestamp for expiry tracking
    token['saved_at'] = time.time()

    try:
        with open(token_file, 'w') as f:
            json.dump(token, f, indent=2)

        # Set restrictive permissions (user read/write only)
        os.chmod(token_file, 0o600)
        print(f"✅ Token saved to {token_file}")
    except Exception as e:
        print(f"⚠️ Failed to save token: {e}")


def load_token() -> Optional[Dict[str, Any]]:
    """Load token from local file if it exists and is valid."""
    token_file = get_token_file_path()

    if not token_file.exists():
        return None

    try:
        with open(token_file, 'r') as f:
            token = json.load(f)

        # Check if token has expired (with 5-minute buffer)
        if 'expires_in' in token and 'saved_at' in token:
            expires_at = token['saved_at'] + \
                token['expires_in'] - 300  # 5 min buffer
            if time.time() > expires_at:
                print("⚠️ Saved token has expired")
                return None

        print("✅ Loaded saved token")
        return token
    except Exception as e:
        print(f"⚠️ Failed to load token: {e}")
        return None


def create_session_from_token(token: Dict[str, Any]) -> requests.Session:
    """Create a requests session with the bearer token."""
    session = requests.Session()
    session.headers.update({
        'Authorization': f"Bearer {token['access_token']}",
        'Content-Type': 'application/json',
    })
    return session


def refresh_token_if_needed(token: Dict[str, Any]) -> Optional[Tuple[requests.Session, Dict[str, Any]]]:
    """Attempt to refresh the token if it has a refresh_token."""
    if 'refresh_token' not in token:
        print("⚠️ No refresh token available")
        return None

    try:
        _, token_url, client_id, client_secret, redirect_uri = get_oauth_settings()

        oauth = OAuth2Session(client_id)
        new_token = oauth.refresh_token(
            token_url,
            refresh_token=token['refresh_token'],
            client_id=client_id,
            client_secret=client_secret
        )

        save_token(new_token)
        session = create_session_from_token(new_token)
        print("✅ Token refreshed successfully")
        return session, new_token
    except Exception as e:
        print(f"⚠️ Token refresh failed: {e}")
        return None


def get_persistent_session() -> Optional[Tuple[requests.Session, Dict[str, Any]]]:
    """Get an authenticated session using saved/refreshed token if possible."""
    # Try to load saved token
    token = load_token()
    if not token:
        return None

    # Create session from saved token
    session = create_session_from_token(token)

    # Test if the token still works by making a simple API call
    try:
        test_response = session.get('https://api.tastyworks.com/accounts')
        if test_response.status_code == 401:
            # Token is invalid, try to refresh
            print("⚠️ Token invalid, attempting refresh...")
            return refresh_token_if_needed(token)
        elif test_response.status_code == 200:
            print("✅ Existing token is valid")
            return session, token
        else:
            print(f"⚠️ Unexpected API response: {test_response.status_code}")
            return refresh_token_if_needed(token)
    except Exception as e:
        print(f"⚠️ Token validation failed: {e}")
        return refresh_token_if_needed(token)


def clear_saved_token() -> None:
    """Clear the saved token file (for logout)."""
    token_file = get_token_file_path()
    if token_file.exists():
        try:
            token_file.unlink()
            print("✅ Saved token cleared")
        except Exception as e:
            print(f"⚠️ Failed to clear token: {e}")
