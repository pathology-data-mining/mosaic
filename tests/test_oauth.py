"""Tests for the custom OAuth module (mosaic.ui.oauth)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mosaic.ui.oauth import (
    _prune_expired,
    _sessions,
    _pending_states,
    _SESSION_COOKIE,
    _SESSION_TTL_SEC,
    _STATE_TTL_SEC,
    create_oauth_lifespan,
    get_user_from_server_session,
    mount_oauth_routes,
)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clear session and state stores before each test."""
    _sessions.clear()
    _pending_states.clear()
    yield
    _sessions.clear()
    _pending_states.clear()


# ---------------------------------------------------------------------------
# Session store tests
# ---------------------------------------------------------------------------


class TestSessionStore:
    """Tests for the server-side session store."""

    def test_store_and_retrieve_session(self):
        """Test basic session CRUD."""
        _sessions["abc123"] = {
            "userinfo": {"preferred_username": "alice"},
            "created_at": time.time(),
        }

        request = MagicMock()
        request.cookies = {_SESSION_COOKIE: "abc123"}
        request.headers = {}

        userinfo = get_user_from_server_session(request)
        assert userinfo is not None
        assert userinfo["preferred_username"] == "alice"

    def test_missing_cookie_returns_none(self):
        """Test that missing cookie returns None."""
        request = MagicMock()
        request.cookies = {}
        request.headers = {}

        assert get_user_from_server_session(request) is None

    def test_unknown_session_id_returns_none(self):
        """Test that unknown session ID returns None."""
        request = MagicMock()
        request.cookies = {_SESSION_COOKIE: "unknown"}
        request.headers = {}

        assert get_user_from_server_session(request) is None

    def test_expired_session_returns_none(self):
        """Test that expired sessions are pruned."""
        _sessions["expired"] = {
            "userinfo": {"preferred_username": "bob"},
            "created_at": time.time() - _SESSION_TTL_SEC - 1,
        }

        request = MagicMock()
        request.cookies = {_SESSION_COOKIE: "expired"}
        request.headers = {}

        assert get_user_from_server_session(request) is None
        assert "expired" not in _sessions

    def test_cookie_from_header_fallback(self):
        """Test parsing cookie from raw Cookie header."""
        _sessions["header_val"] = {
            "userinfo": {"preferred_username": "carol"},
            "created_at": time.time(),
        }

        request = MagicMock()
        request.cookies = {}  # Empty dict, fallback to header
        request.headers = {"cookie": f"other=x; {_SESSION_COOKIE}=header_val; foo=bar"}

        userinfo = get_user_from_server_session(request)
        assert userinfo is not None
        assert userinfo["preferred_username"] == "carol"

    def test_none_request_returns_none(self):
        """Test that None request returns None gracefully."""
        # get_user_from_server_session checks hasattr, so pass an object
        # without cookies or headers
        assert get_user_from_server_session(None) is None


# ---------------------------------------------------------------------------
# Prune tests
# ---------------------------------------------------------------------------


class TestPruneExpired:
    """Tests for _prune_expired."""

    def test_prune_expired_sessions(self):
        """Test that expired sessions are removed."""
        _sessions["old"] = {
            "userinfo": {"preferred_username": "old_user"},
            "created_at": time.time() - _SESSION_TTL_SEC - 100,
        }
        _sessions["fresh"] = {
            "userinfo": {"preferred_username": "new_user"},
            "created_at": time.time(),
        }

        _prune_expired()

        assert "old" not in _sessions
        assert "fresh" in _sessions

    def test_prune_expired_states(self):
        """Test that expired CSRF states are removed."""
        _pending_states["old_state"] = time.time() - _STATE_TTL_SEC - 100
        _pending_states["fresh_state"] = time.time()

        _prune_expired()

        assert "old_state" not in _pending_states
        assert "fresh_state" in _pending_states


# ---------------------------------------------------------------------------
# mount_oauth_routes tests
# ---------------------------------------------------------------------------


class TestMountOAuthRoutes:
    """Tests for mount_oauth_routes."""

    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "")
    def test_no_routes_when_client_id_missing(self):
        """Test that routes are not mounted when OAUTH_CLIENT_ID is empty."""
        app = MagicMock()
        app.routes = []

        mount_oauth_routes(app)

        assert len(app.routes) == 0

    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "test-client-id")
    def test_routes_mounted_when_configured(self):
        """Test that routes are mounted when OAuth is configured."""
        app = MagicMock()
        app.routes = []

        mount_oauth_routes(app)

        assert len(app.routes) == 3
        paths = [r.path for r in app.routes]
        assert "/api/auth/login" in paths
        assert "/api/auth/callback" in paths
        assert "/api/auth/logout" in paths


# ---------------------------------------------------------------------------
# Login route tests
# ---------------------------------------------------------------------------


class TestLoginRoute:
    """Tests for the /api/auth/login route."""

    @pytest.mark.asyncio
    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "test-client-id")
    @patch("mosaic.ui.oauth.SPACE_HOST", "user-space.hf.space")
    async def test_login_redirects_to_hf(self):
        """Test that login redirects to HF authorize URL."""
        from mosaic.ui.oauth import _login

        request = MagicMock()
        response = await _login(request)

        assert response.status_code == 302
        location = response.headers["location"]
        assert "huggingface.co/oauth/authorize" in location
        assert "client_id=test-client-id" in location
        assert "redirect_uri=" in location
        assert "state=" in location
        # Should have stored a CSRF state
        assert len(_pending_states) == 1

    @pytest.mark.asyncio
    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "")
    async def test_login_returns_error_when_not_configured(self):
        """Test that login returns error when OAuth is not configured."""
        from mosaic.ui.oauth import _login

        request = MagicMock()
        response = await _login(request)

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Callback route tests
# ---------------------------------------------------------------------------


class TestCallbackRoute:
    """Tests for the /api/auth/callback route."""

    @pytest.mark.asyncio
    async def test_callback_missing_params(self):
        """Test callback with missing code/state parameters."""
        from mosaic.ui.oauth import _callback

        request = MagicMock()
        request.query_params = {}

        response = await _callback(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self):
        """Test callback with invalid CSRF state."""
        from mosaic.ui.oauth import _callback

        request = MagicMock()
        request.query_params = {"code": "test-code", "state": "invalid-state"}

        response = await _callback(request)
        assert response.status_code == 400

    @pytest.mark.asyncio
    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "test-client-id")
    @patch("mosaic.ui.oauth.OAUTH_CLIENT_SECRET", "test-secret")
    @patch("mosaic.ui.oauth.SPACE_HOST", "user-space.hf.space")
    async def test_callback_exchanges_code_for_token(self):
        """Test successful callback with mocked token exchange."""
        import json
        import base64

        from mosaic.ui.oauth import _callback

        # Set up a valid CSRF state
        valid_state = "valid-state-token"
        _pending_states[valid_state] = time.time()

        # Create a mock id_token JWT
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload_data = json.dumps(
            {"preferred_username": "visitor123", "sub": "visitor123"}
        )
        payload = base64.urlsafe_b64encode(payload_data.encode()).rstrip(b"=").decode()
        mock_id_token = f"{header}.{payload}.fake_signature"

        # Mock httpx — json() is a regular method (not async) on httpx.Response
        mock_response = AsyncMock()
        mock_response.json = MagicMock(
            return_value={
                "access_token": "test-access-token",
                "id_token": mock_id_token,
            }
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        request = MagicMock()
        request.query_params = {"code": "auth-code", "state": valid_state}
        request.session = {}

        with patch("httpx.AsyncClient", return_value=mock_client):
            response = await _callback(request)

        # Should redirect to /
        assert response.status_code == 302

        # Session should have oauth_info
        assert "oauth_info" in request.session
        assert (
            request.session["oauth_info"]["userinfo"]["preferred_username"]
            == "visitor123"
        )

        # Server-side session should be stored
        assert len(_sessions) == 1

        # CSRF state should be consumed
        assert valid_state not in _pending_states


# ---------------------------------------------------------------------------
# Logout route tests
# ---------------------------------------------------------------------------


class TestLogoutRoute:
    """Tests for the /api/auth/logout route."""

    @pytest.mark.asyncio
    async def test_logout_clears_session(self):
        """Test that logout clears session data."""
        from mosaic.ui.oauth import _logout

        # Set up server-side session
        _sessions["session_id"] = {
            "userinfo": {"preferred_username": "user"},
            "created_at": time.time(),
        }

        request = MagicMock()
        request.session = {"oauth_info": {"userinfo": {"preferred_username": "user"}}}
        request.cookies = {_SESSION_COOKIE: "session_id"}

        response = await _logout(request)

        assert response.status_code == 302
        assert "oauth_info" not in request.session
        assert "session_id" not in _sessions


# ---------------------------------------------------------------------------
# get_user_from_server_session integration
# ---------------------------------------------------------------------------


class TestGetUserFromServerSession:
    """Integration tests for the server-session fallback."""

    def test_works_with_gradio_request_like_object(self):
        """Test with an object mimicking gr.Request."""
        _sessions["gr_session"] = {
            "userinfo": {"preferred_username": "gradio_user", "name": "Test User"},
            "created_at": time.time(),
        }

        class MockGradioRequest:
            cookies = {_SESSION_COOKIE: "gr_session"}
            headers = {}

        result = get_user_from_server_session(MockGradioRequest())
        assert result is not None
        assert result["preferred_username"] == "gradio_user"
        assert result["name"] == "Test User"

    def test_returns_none_for_object_without_cookies(self):
        """Test graceful handling of objects without cookies."""

        class Bare:
            pass

        assert get_user_from_server_session(Bare()) is None


# ---------------------------------------------------------------------------
# create_oauth_lifespan tests
# ---------------------------------------------------------------------------


class TestCreateOAuthLifespan:
    """Tests for the create_oauth_lifespan async context manager."""

    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "test-client-id")
    def test_routes_mounted_when_configured(self):
        """Test that OAuth routes are mounted on app when OAUTH_CLIENT_ID is set."""
        import asyncio

        app = MagicMock()
        app.routes = []

        async def _run():
            async with create_oauth_lifespan(app):
                assert len(app.routes) == 3
                paths = [r.path for r in app.routes]
                assert "/api/auth/login" in paths
                assert "/api/auth/callback" in paths
                assert "/api/auth/logout" in paths

        asyncio.run(_run())

    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "")
    def test_routes_skipped_when_not_configured(self):
        """Test that no routes are mounted when OAUTH_CLIENT_ID is empty."""
        import asyncio

        app = MagicMock()
        app.routes = []

        async def _run():
            async with create_oauth_lifespan(app):
                assert len(app.routes) == 0

        asyncio.run(_run())

    @patch("mosaic.ui.oauth.OAUTH_CLIENT_ID", "test-client-id")
    def test_routes_persist_after_yield(self):
        """Test that routes remain on the app after the lifespan exits."""
        import asyncio

        app = MagicMock()
        app.routes = []

        async def _run():
            async with create_oauth_lifespan(app):
                pass
            # Routes should still be present after context manager exits
            assert len(app.routes) == 3

        asyncio.run(_run())
