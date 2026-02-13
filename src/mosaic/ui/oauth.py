"""Manual OAuth flow for HF Spaces with Docker SDK.

On HF Spaces with sdk:docker, Gradio's built-in OAuth (gr.LoginButton /
gr.OAuthProfile) doesn't work because the HF reverse proxy injects the
Space owner's identity into every session.  This module implements the
Authorization Code flow directly against HF's OAuth provider.

The HF proxy strips all custom cookies — only its own forged ``session``
cookie survives.  To persist the visitor's session ID across page loads,
the callback stores it in the browser's ``localStorage`` via an
intermediate HTML page.  The Gradio ``demo.load()`` handler reads it back
via a ``js=`` parameter and passes it to the Python handler, which looks
up the server-side session.

Environment variables (set automatically by HF Spaces):
    OAUTH_CLIENT_ID:     OAuth application client ID
    OAUTH_CLIENT_SECRET: OAuth application client secret
    OAUTH_SCOPES:        Space-separated scopes (default: "openid profile")
    SPACE_HOST:           Public hostname of the Space (e.g. "user-space.hf.space")

Routes mounted on the Gradio ASGI app:
    GET /api/auth/login    -> Redirect to HF authorize endpoint
    GET /api/auth/callback -> Exchange code for token, store session via localStorage
    GET /api/auth/logout   -> Clear session, clear localStorage, redirect to /
"""

import contextlib
import html as html_module
import json
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

from loguru import logger
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse, HTMLResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES", "openid profile")
SPACE_HOST = os.environ.get("SPACE_HOST", "")

HF_AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
HF_TOKEN_URL = "https://huggingface.co/oauth/token"

# ---------------------------------------------------------------------------
# Server-side session store
# ---------------------------------------------------------------------------

# {session_id: {"userinfo": {...}, "created_at": float}}
_sessions: dict[str, dict] = {}
_SESSION_TTL_SEC = 24 * 60 * 60  # 24 hours
_SESSION_COOKIE = "mosaic_auth"  # kept for backward-compat with tests

# localStorage key used by the browser to persist the session ID
_LOCALSTORAGE_KEY = "mosaic_session_id"

# In-memory CSRF state tokens: {state_value: created_at_float}
_pending_states: dict[str, float] = {}
_STATE_TTL_SEC = 10 * 60  # 10 minutes


def _prune_expired() -> None:
    """Remove expired sessions and CSRF state tokens."""
    now = time.time()
    expired_sessions = [
        k for k, v in _sessions.items() if now - v["created_at"] > _SESSION_TTL_SEC
    ]
    for k in expired_sessions:
        del _sessions[k]
    expired_states = [k for k, v in _pending_states.items() if now - v > _STATE_TTL_SEC]
    for k in expired_states:
        del _pending_states[k]


def get_user_from_server_session(request) -> Optional[dict]:
    """Look up user info from the server-side session store.

    Checks the ``mosaic_auth`` cookie in the request and returns the
    stored userinfo dict, or None if the cookie is missing/expired.

    Args:
        request: Starlette/Gradio request object (needs .cookies or .headers)

    Returns:
        userinfo dict with at least ``preferred_username`` key, or None
    """
    _prune_expired()

    cookie_val = None
    # Gradio's gr.Request wraps cookies/headers through dict_to_obj() which
    # returns a gradio.route_utils.Obj, NOT a plain dict.  Use hasattr/get
    # instead of isinstance(dict) to support both dict and Obj.
    if hasattr(request, "cookies"):
        cookies = request.cookies
        if hasattr(cookies, "get"):
            cookie_val = cookies.get(_SESSION_COOKIE)
        logger.debug(
            f"Session lookup: cookies type={type(cookies).__name__}, "
            f"cookie_val={cookie_val!r}"
        )
    # Fallback: parse Cookie header
    if cookie_val is None and hasattr(request, "headers"):
        headers = request.headers
        cookie_header = ""
        if hasattr(headers, "get"):
            cookie_header = headers.get("cookie", "") or ""
        if cookie_header:
            for part in cookie_header.split(";"):
                part = part.strip()
                if part.startswith(f"{_SESSION_COOKIE}="):
                    cookie_val = part[len(f"{_SESSION_COOKIE}=") :]
                    break
        logger.debug(
            f"Session lookup header fallback: has_cookie_header={bool(cookie_header)}, "
            f"cookie_val={cookie_val!r}"
        )

    if not cookie_val:
        logger.debug(
            f"No {_SESSION_COOKIE} cookie found. "
            f"Active sessions: {len(_sessions)}"
        )
        return None

    entry = _sessions.get(cookie_val)
    if entry is None:
        logger.debug(
            f"{_SESSION_COOKIE} cookie present ({cookie_val[:8]}...) "
            f"but no matching server session. "
            f"Active sessions: {list(_sessions.keys())[:3]}"
        )
        return None

    if time.time() - entry["created_at"] > _SESSION_TTL_SEC:
        del _sessions[cookie_val]
        return None

    return entry.get("userinfo")


def get_user_by_session_id(session_id: Optional[str]) -> Optional[dict]:
    """Look up user info by a raw session ID (from localStorage).

    This is the primary lookup method on HF Spaces where the HF proxy
    strips custom cookies.  The session ID is stored in the browser's
    localStorage by the callback page and passed to Gradio handlers via
    the ``js=`` parameter on ``demo.load()``.

    Args:
        session_id: The session ID string (from localStorage), or empty/None.

    Returns:
        userinfo dict with at least ``preferred_username`` key, or None.
    """
    _prune_expired()

    if not session_id:
        return None

    entry = _sessions.get(session_id)
    if entry is None:
        logger.debug(
            f"Session ID ({session_id[:8]}...) not found. "
            f"Active sessions: {len(_sessions)}"
        )
        return None

    if time.time() - entry["created_at"] > _SESSION_TTL_SEC:
        del _sessions[session_id]
        return None

    return entry.get("userinfo")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def _login(request: Request):
    """Redirect to HF OAuth authorize endpoint."""
    if not OAUTH_CLIENT_ID or not SPACE_HOST:
        return JSONResponse(
            {"error": "OAuth not configured (missing OAUTH_CLIENT_ID or SPACE_HOST)"},
            status_code=500,
        )

    _prune_expired()
    state = secrets.token_urlsafe(32)
    _pending_states[state] = time.time()

    # Build redirect URI pointing back to our callback
    redirect_uri = f"https://{SPACE_HOST}/api/auth/callback"

    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": OAUTH_SCOPES,
        "state": state,
    }
    authorize_url = f"{HF_AUTHORIZE_URL}?{urlencode(params)}"
    logger.info(f"OAuth login: redirecting to HF authorize endpoint")
    return RedirectResponse(authorize_url, status_code=302)


async def _callback(request: Request):
    """Handle OAuth callback: exchange code for token, set session."""
    import httpx

    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state:
        return JSONResponse(
            {"error": "Missing code or state parameter"}, status_code=400
        )

    # Validate CSRF state
    _prune_expired()
    created_at = _pending_states.pop(state, None)
    if created_at is None:
        return JSONResponse(
            {"error": "Invalid or expired state parameter"}, status_code=400
        )

    # Exchange authorization code for access token
    redirect_uri = f"https://{SPACE_HOST}/api/auth/callback"
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                HF_TOKEN_URL,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            token_response = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"OAuth token exchange failed: {e.response.status_code} {e.response.text}"
        )
        return JSONResponse({"error": "Token exchange failed"}, status_code=502)
    except Exception as e:
        logger.error(f"OAuth token exchange error: {e}")
        return JSONResponse({"error": "Token exchange failed"}, status_code=502)

    # Extract userinfo from the id_token (JWT) or use the userinfo endpoint
    access_token = token_response.get("access_token")
    userinfo = None

    # Try to decode the id_token (JWT) for userinfo
    id_token = token_response.get("id_token")
    if id_token:
        try:
            # JWT is base64url-encoded: header.payload.signature
            # We only need the payload (claims) — no signature verification
            # since we just received this directly from HF's token endpoint
            import base64

            payload_b64 = id_token.split(".")[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            userinfo = json.loads(payload_bytes)
        except Exception as e:
            logger.warning(f"Failed to decode id_token: {e}")

    # Fallback: call userinfo endpoint
    if userinfo is None and access_token:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://huggingface.co/oauth/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                userinfo = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch userinfo: {e}")
            return JSONResponse({"error": "Failed to get user info"}, status_code=502)

    if not userinfo:
        return JSONResponse({"error": "No user info received"}, status_code=502)

    # Extract username
    username = userinfo.get("preferred_username") or userinfo.get("sub", "unknown")
    logger.info(f"OAuth callback: authenticated user '{username}'")

    # Store in server-side session dict (keyed by random session ID)
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "userinfo": userinfo,
        "created_at": time.time(),
    }

    # The HF proxy strips custom cookies, so we can't use Set-Cookie to
    # persist the session ID.  Instead, return a small HTML page that
    # stores the session ID in localStorage and then redirects to /.
    safe_sid = html_module.escape(session_id)
    page = f"""<!DOCTYPE html>
<html><head><title>Signing in...</title></head>
<body><p>Signing in&hellip;</p>
<script>
try {{ localStorage.setItem('{_LOCALSTORAGE_KEY}', '{safe_sid}'); }}
catch(e) {{ console.warn('Could not store session:', e); }}
window.location.replace('/');
</script>
<noscript><a href="/">Click here to continue</a></noscript>
</body></html>"""
    return HTMLResponse(page)


async def _logout(request: Request):
    """Clear server-side session and browser localStorage, then redirect."""
    # Try to clear server-side session from query param (set by JS)
    sid = request.query_params.get("sid", "")
    if sid and sid in _sessions:
        del _sessions[sid]
        logger.info(f"Cleared server-side session for logout ({sid[:8]}...)")

    # Return HTML that clears localStorage and redirects
    page = f"""<!DOCTYPE html>
<html><head><title>Signing out...</title></head>
<body><p>Signing out&hellip;</p>
<script>
try {{ localStorage.removeItem('{_LOCALSTORAGE_KEY}'); }}
catch(e) {{ console.warn('Could not clear session:', e); }}
window.location.replace('/');
</script>
<noscript><a href="/">Click here to continue</a></noscript>
</body></html>"""
    return HTMLResponse(page)


# ---------------------------------------------------------------------------
# Mount helper
# ---------------------------------------------------------------------------

_oauth_routes = [
    Route("/api/auth/login", _login, methods=["GET"]),
    Route("/api/auth/callback", _callback, methods=["GET"]),
    Route("/api/auth/logout", _logout, methods=["GET"]),
]

def mount_oauth_routes(app) -> None:
    """Mount custom OAuth routes on the Gradio ASGI app.

    .. deprecated::
        Gradio's ``launch()`` recreates ``demo.app`` from scratch, discarding
        any routes prepended beforehand.  Use :func:`create_oauth_lifespan`
        instead, which mounts routes on the **final** app instance via the
        lifespan protocol.

    Args:
        app: The Starlette/FastAPI app (``demo.app``)
    """
    if not OAUTH_CLIENT_ID:
        logger.warning(
            "OAuth routes not mounted: OAUTH_CLIENT_ID not set. "
            "Custom login will not work."
        )
        return

    # Insert our routes at the beginning so they take priority
    app.routes[:0] = _oauth_routes
    logger.info(
        f"Mounted custom OAuth routes: /api/auth/login, /api/auth/callback, /api/auth/logout"
    )


@contextlib.asynccontextmanager
async def create_oauth_lifespan(app):
    """ASGI lifespan handler that mounts custom OAuth routes.

    Gradio's ``launch()`` recreates ``demo.app`` via ``App.create_app()``,
    discarding any routes prepended before launch.  However,
    ``create_lifespan_handler()`` (in Gradio's ``route_utils.py``) wraps
    user-supplied lifespan and calls it with the **final** App instance.
    Routes added here are on the real app with the full middleware stack
    (including ``SessionMiddleware``).

    Pass this to ``demo.launch(app_kwargs={"lifespan": create_oauth_lifespan})``.
    """
    if OAUTH_CLIENT_ID:
        app.routes[:0] = _oauth_routes
        logger.info(
            "Custom OAuth routes mounted via lifespan: "
            "/api/auth/login, /api/auth/callback, /api/auth/logout"
        )
    yield
