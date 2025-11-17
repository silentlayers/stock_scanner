"""
Cloud-friendly OAuth authentication for Streamlit Cloud deployments.
Uses URL query parameters instead of a local callback server.
"""
from __future__ import annotations

import streamlit as st
from typing import Tuple, Optional
import requests
from requests_oauthlib import OAuth2Session

from integrations.tastytrade.config import get_oauth_settings, get_oauth_scopes
from integrations.tastytrade.token_manager import save_token


def authenticate_cloud() -> Optional[Tuple[requests.Session, dict]]:
    """
    Perform OAuth flow suitable for Streamlit Cloud.
    Uses query parameters from the URL instead of a local server.

    Returns:
        Tuple of (session, token) if authentication successful, None otherwise.
    """
    authorization_base_url, token_url, client_id, client_secret, redirect_uri = get_oauth_settings()

    # Check if we have an authorization code in the URL
    query_params = st.query_params
    auth_code = query_params.get('code')

    if auth_code:
        # We have a code! Exchange it for a token
        st.info("🔄 Exchanging authorization code for access token...")

        try:
            oauth = OAuth2Session(
                client_id,
                redirect_uri=redirect_uri,
                scope=get_oauth_scopes()
            )

            # Exchange authorization code for token
            token = oauth.fetch_token(
                token_url,
                code=auth_code,
                client_secret=client_secret,
                include_client_id=True
            )

            # Save token for future use
            save_token(token)

            # Create authenticated session
            session = requests.Session()
            session.headers.update({
                'Authorization': f"Bearer {token['access_token']}"
            })

            # Clear the code from URL
            st.query_params.clear()

            st.success("✅ Authentication successful!")
            return session, token

        except Exception as e:
            st.error(f"❌ Token exchange failed: {e}")
            # Clear query params on error
            st.query_params.clear()
            raise

    else:
        # No code yet, show authorization link
        oauth = OAuth2Session(
            client_id,
            redirect_uri=redirect_uri,
            scope=get_oauth_scopes()
        )

        authorization_url, state = oauth.authorization_url(
            authorization_base_url)

        st.warning("🔐 Authorization Required")

        # Show authorization button
        st.link_button(
            "🔓 Authorize with TastyTrade",
            authorization_url,
            help="Opens TastyTrade authorization page"
        )

        return None
