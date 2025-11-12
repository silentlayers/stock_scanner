"""
OAuth2 authentication for Tastytrade with automatic local callback and token persistence.
"""
from __future__ import annotations

import threading
import time
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Tuple, Optional

import requests
from requests_oauthlib import OAuth2Session

from integrations.tastytrade.config import get_oauth_settings, get_oauth_scopes
from integrations.tastytrade.token_manager import save_token, get_persistent_session


_authorization_code: Optional[str] = None
_auth_error: Optional[str] = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _authorization_code, _auth_error
        parsed_url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_url.query)
        if 'code' in query:
            _authorization_code = query['code'][0]
            self._send_html(
                200,
                """
                <html>
                  <head><title>Authorization Successful</title></head>
                  <body>
                    <h1>Authorization Successful</h1>
                    <p>You can close this window and return to your application.</p>
                    <script>setTimeout(function(){window.close();}, 1500);</script>
                  </body>
                </html>
                """,
            )
        elif 'error' in query:
            _auth_error = query['error'][0]
            self._send_html(
                400, f"<h1>Authorization Failed</h1><p>Error: {_auth_error}</p>")
        else:
            self._send_html(400, "<h1>Invalid callback</h1>")

    def log_message(self, format, *args):  # suppress logs
        pass

    def _send_html(self, status: int, html: str):
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))


def authenticate() -> Tuple[requests.Session, dict]:
    """Perform OAuth flow and return (requests session, token dict).

    First tries to use saved/refreshed token, falls back to full OAuth flow.
    """
    # Try to use existing token first
    existing = get_persistent_session()
    if existing:
        return existing

    print("🔄 Starting new OAuth flow...")
    return _perform_oauth_flow()


def _perform_oauth_flow() -> Tuple[requests.Session, dict]:
    """Perform the full OAuth authorization flow."""
    global _authorization_code, _auth_error
    _authorization_code = None
    _auth_error = None

    authorization_base_url, token_url, client_id, client_secret, redirect_uri = get_oauth_settings()

    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    port = parsed_redirect.port or 3000

    oauth = OAuth2Session(
        client_id, redirect_uri=redirect_uri, scope=get_oauth_scopes())

    # Start local HTTP server to catch callback
    server = HTTPServer(('localhost', port), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    auth_url, state = oauth.authorization_url(authorization_base_url)
    print("🌐 Opening authorization URL in your default browser...")
    print(f"   If the browser doesn't open automatically, go to: {auth_url}")
    webbrowser.open(auth_url)

    print("⏳ Waiting for authorization... (this may take a few moments)")
    timeout_s = 180
    start = time.time()
    while _authorization_code is None and _auth_error is None:
        if time.time() - start > timeout_s:
            server.shutdown()
            raise SystemExit("Authorization timed out")
        time.sleep(0.25)

    server.shutdown()

    if _auth_error:
        raise SystemExit(f"Authorization failed: {_auth_error}")

    assert _authorization_code is not None
    try:
        auth_response = f"{redirect_uri}?code={_authorization_code}&state={state}"
        token = oauth.fetch_token(
            token_url,
            authorization_response=auth_response,
            client_secret=client_secret,
            include_client_id=True,
        )
    except Exception as e:
        raise SystemExit(f"Token exchange failed: {e}")

    print("✅ Access token obtained successfully!")

    # Save token for future use
    save_token(token)

    # Return a plain requests.Session with the bearer token header set
    session = requests.Session()
    session.headers.update({
        'Authorization': f"Bearer {token['access_token']}",
        'Content-Type': 'application/json',
    })
    return session, token
