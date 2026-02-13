"""Manual OAuth flow for HF Spaces with Docker SDK.

On HF Spaces with sdk:docker, Gradio's built-in OAuth (gr.LoginButton /
gr.OAuthProfile) doesn't work because the HF reverse proxy injects the
Space owner's identity into every session.  This module implements the
Authorization Code flow directly against HF's OAuth provider and writes
the visitor's real identity into the Starlette session so that Gradio's
existing OAuthProfile injection works transparently.

Environment variables (set automatically by HF Spaces):
    OAUTH_CLIENT_ID:     OAuth application client ID
    OAUTH_CLIENT_SECRET: OAuth application client secret
    OAUTH_SCOPES:        Space-separated scopes (default: "openid profile")
    SPACE_HOST:           Public hostname of the Space (e.g. "user-space.hf.space")

Routes mounted on the Gradio ASGI app:
    GET /api/auth/login    -> Redirect to HF authorize endpoint
    GET /api/auth/callback -> Exchange code for token, set session
    GET /api/auth/logout   -> Clear session, redirect to /
"""

import contextlib
import json
import os
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

from loguru import logger
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse
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

# {cookie_value: {"userinfo": {...}, "created_at": float}}
_sessions: dict[str, dict] = {}
_SESSION_TTL_SEC = 24 * 60 * 60  # 24 hours
_SESSION_COOKIE = "mosaic_auth"

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
    # Gradio's gr.Request wraps a Starlette Request; try .cookies first
    if hasattr(request, "cookies"):
        cookies = request.cookies
        if isinstance(cookies, dict):
            cookie_val = cookies.get(_SESSION_COOKIE)
    # Fallback: parse Cookie header
    if cookie_val is None and hasattr(request, "headers"):
        headers = request.headers
        cookie_header = None
        if isinstance(headers, dict):
            cookie_header = headers.get("cookie", "")
        elif hasattr(headers, "get"):
            cookie_header = headers.get("cookie", "")
        if cookie_header:
            for part in cookie_header.split(";"):
                part = part.strip()
                if part.startswith(f"{_SESSION_COOKIE}="):
                    cookie_val = part[len(f"{_SESSION_COOKIE}=") :]
                    break

    if not cookie_val:
        return None

    entry = _sessions.get(cookie_val)
    if entry is None:
        return None

    if time.time() - entry["created_at"] > _SESSION_TTL_SEC:
        del _sessions[cookie_val]
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

    # Write into Gradio's Starlette session so OAuthProfile picks it up
    # Gradio expects session["oauth_info"]["userinfo"] with at least
    # "preferred_username" and optionally "name", "picture", etc.
    oauth_info = {
        "userinfo": userinfo,
        "access_token": access_token,
    }
    request.session["oauth_info"] = oauth_info

    # Also store in our server-side session as fallback
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "userinfo": userinfo,
        "created_at": time.time(),
    }

    # Build redirect response with cookie
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        _SESSION_COOKIE,
        session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=_SESSION_TTL_SEC,
    )
    return response


async def _logout(request: Request):
    """Clear session and redirect to /."""
    # Clear Gradio's session
    if "oauth_info" in request.session:
        del request.session["oauth_info"]

    # Clear server-side session
    cookie_val = request.cookies.get(_SESSION_COOKIE)
    if cookie_val and cookie_val in _sessions:
        del _sessions[cookie_val]

    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(_SESSION_COOKIE)
    return response


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
