"""
GitHub OAuth authentication for the scalping bot dashboard.

Flow:
  1. GET  /auth/login    → redirect to GitHub OAuth
  2. GET  /auth/callback → exchange code for token, verify user, set session cookie
  3. GET  /auth/me       → return current user info (used by UI to check auth state)
  4. POST /auth/logout   → clear session cookie

Session is a signed cookie (itsdangerous) — no server-side session store needed.
Cookie is HttpOnly + SameSite=Lax. Set SECURE=true in production (HTTPS).

Required env vars:
  GITHUB_CLIENT_ID      — from your GitHub App
  GITHUB_CLIENT_SECRET  — from your GitHub App
  AUTHORIZED_USERS      — comma-separated GitHub usernames allowed access
  SESSION_SECRET        — random string used to sign cookies (generate once, keep secret)
  APP_URL               — public URL of the dashboard e.g. https://scalper.yourdomain.com
"""

import os
import hmac
import hashlib
import logging
import secrets
from datetime import datetime, timezone, timedelta

import httpx
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import Request, Response
from fastapi.responses import RedirectResponse, JSONResponse

log = logging.getLogger("auth")

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_CLIENT_ID     = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
AUTHORIZED_USERS     = {
    u.strip().lower()
    for u in os.environ.get("AUTHORIZED_USERS", "").split(",")
    if u.strip()
}
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))
APP_URL        = os.environ.get("APP_URL", "").rstrip("/")

COOKIE_NAME    = "scalper_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
SESSION_MAX_AGE = COOKIE_MAX_AGE

# Pending OAuth states — maps state token → expiry. Prevents CSRF.
_pending_states: dict[str, datetime] = {}

_serializer = URLSafeTimedSerializer(SESSION_SECRET)


def _validate_config():
    missing = []
    if not GITHUB_CLIENT_ID:     missing.append("GITHUB_CLIENT_ID")
    if not GITHUB_CLIENT_SECRET: missing.append("GITHUB_CLIENT_SECRET")
    if not AUTHORIZED_USERS:     missing.append("AUTHORIZED_USERS")
    if not APP_URL:              missing.append("APP_URL")
    if missing:
        log.warning(f"Auth config incomplete — missing: {missing}. Dashboard will be inaccessible.")
    else:
        log.info(f"Auth ready. Authorized users: {AUTHORIZED_USERS}")


_validate_config()


# ── Session helpers ───────────────────────────────────────────────────────────

def _make_session(username: str, avatar_url: str) -> str:
    return _serializer.dumps({"u": username, "a": avatar_url})


def _read_session(token: str) -> dict | None:
    try:
        return _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> dict | None:
    """Return {"username": ..., "avatar_url": ...} if session is valid, else None."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    data = _read_session(token)
    if not data:
        return None
    username = data.get("u", "")
    if username.lower() not in AUTHORIZED_USERS:
        return None
    return {"username": username, "avatar_url": data.get("a", "")}


def require_auth(request: Request) -> dict:
    """Dependency — raises 401 JSON if not authenticated."""
    user = get_current_user(request)
    if not user:
        raise JSONResponse(status_code=401, content={"detail": "Not authenticated"})
    return user


# ── OAuth routes ──────────────────────────────────────────────────────────────

def login_route(request: Request):
    """Redirect user to GitHub OAuth."""
    if not GITHUB_CLIENT_ID:
        return JSONResponse(
            status_code=503,
            content={"detail": "GitHub OAuth not configured. Set GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, APP_URL."}
        )

    state = secrets.token_urlsafe(32)
    _pending_states[state] = datetime.now(timezone.utc) + timedelta(minutes=10)

    # Clean up expired states
    now = datetime.now(timezone.utc)
    expired = [s for s, exp in _pending_states.items() if exp < now]
    for s in expired:
        del _pending_states[s]

    callback_url = f"{APP_URL}/auth/callback"
    github_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={callback_url}"
        f"&scope=read:user"
        f"&state={state}"
    )
    return RedirectResponse(url=github_url)


async def callback_route(request: Request):
    """Handle GitHub OAuth callback — exchange code for token, verify user."""
    code  = request.query_params.get("code")
    state = request.query_params.get("state")

    # Validate state to prevent CSRF
    if not state or state not in _pending_states:
        return RedirectResponse(url="/?error=invalid_state")
    if _pending_states[state] < datetime.now(timezone.utc):
        del _pending_states[state]
        return RedirectResponse(url="/?error=state_expired")
    del _pending_states[state]

    if not code:
        return RedirectResponse(url="/?error=no_code")

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        try:
            token_resp = await client.post(
                "https://github.com/login/oauth/access_token",
                json={
                    "client_id":     GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code":          code,
                    "redirect_uri":  f"{APP_URL}/auth/callback",
                },
                headers={"Accept": "application/json"},
                timeout=10,
            )
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                log.warning(f"No access token in GitHub response: {token_data}")
                return RedirectResponse(url="/?error=no_token")

            # Fetch GitHub user info
            user_resp = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
            user_data = user_resp.json()
        except Exception as e:
            log.error(f"GitHub OAuth request failed: {e}")
            return RedirectResponse(url="/?error=github_error")

    username   = (user_data.get("login") or "").lower()
    avatar_url = user_data.get("avatar_url") or ""

    if not username:
        return RedirectResponse(url="/?error=no_username")

    if username not in AUTHORIZED_USERS:
        log.warning(f"Unauthorized login attempt: {username}")
        return RedirectResponse(url="/?error=unauthorized")

    log.info(f"Authenticated: {username}")
    session_token = _make_session(username, avatar_url)

    response = RedirectResponse(url="/")
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=APP_URL.startswith("https"),
    )
    return response


def me_route(request: Request):
    """Return current user info or 401."""
    user = get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"authenticated": False})
    return JSONResponse(content={"authenticated": True, **user})


def logout_route(request: Request):
    """Clear session cookie and redirect to login."""
    user = get_current_user(request)
    if user:
        log.info(f"Logout: {user['username']}")
    response = RedirectResponse(url="/")
    response.delete_cookie(COOKIE_NAME)
    return response
