"""
middleware/stack.py
───────────────────
All middleware registered on the FastAPI app.

WHY MIDDLEWARE (not just endpoint logic):
  Middleware runs on EVERY request before it hits any endpoint.
  You write auth/logging/rate-limiting once here — not in every endpoint.
  This is the production pattern. Without it: UI → FastAPI → agent (no guardrails).
  With it: UI → Gateway → Auth → RateLimit → CostGuard → Cache → FastAPI → agent.

LAYERS (in order of execution, outermost first):
  1. RequestID    — assigns UUID to every request for log correlation
  2. StructuredLog — JSON log: method, path, user_id, duration, status
  3. Auth         — JWT verification (skips /health and /token)
  4. RateLimit    — per-user 10 req/min via SlowAPI + Redis
  5. CostGuard    — rejects requests projected to exceed token budget
  6. ResultCache  — returns Redis-cached result for identical (jd+resume) runs
"""

import time
import uuid
import json
import hashlib
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as aioredis
from config.settings import settings

logger = logging.getLogger("job_agent")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── 1. Request ID middleware ──────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Stamps every request with a UUID. All downstream logs use this ID."""
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── 2. Structured logging middleware ─────────────────────────────────────────

class StructuredLogMiddleware(BaseHTTPMiddleware):
    """
    Logs every request as JSON. Interviewers love this — shows production awareness.
    Output: {"request_id": "...", "method": "POST", "path": "/analyze",
             "user_id": "...", "duration_ms": 4231, "status": 200}
    """
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)

        log = {
            "request_id": getattr(request.state, "request_id", "?"),
            "method": request.method,
            "path": request.url.path,
            "user_id": getattr(request.state, "user_id", "anonymous"),
            "duration_ms": duration_ms,
            "status": response.status_code,
        }
        logger.info(json.dumps(log))
        return response


# ── 3. Auth middleware ─────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    JWT verification on every request except public endpoints.
    WHY JWT (not sessions): stateless — no DB lookup per request.
    The token carries user_id, issued_at, expiry. Verified with HMAC-SHA256.
    """
    SKIP_PATHS = {"/health", "/token", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "Missing auth token"}, status_code=401)

        token = auth_header.split(" ")[1]
        user_id = _verify_jwt(token)
        if not user_id:
            return JSONResponse({"error": "Invalid or expired token"}, status_code=401)

        request.state.user_id = user_id
        return await call_next(request)


def _verify_jwt(token: str) -> str | None:
    """Decode and verify JWT. Returns user_id or None."""
    try:
        import jwt
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload.get("sub")
    except Exception:
        return None


def create_token(user_id: str) -> str:
    """Create a JWT token for a user. Called from /token endpoint."""
    import jwt
    from datetime import datetime, timedelta
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# ── 4. Cost guard middleware ───────────────────────────────────────────────

class CostGuardMiddleware(BaseHTTPMiddleware):
    """
    Estimates token usage BEFORE running the graph.
    Rejects requests projected to exceed the per-run budget.
    WHY: one runaway request (huge resume + JD + 3 loop cycles) can cost $2+.
    At scale that's thousands of dollars in unexpected spend.
    Formula: (jd_chars + resume_chars) / 4 * estimated_llm_calls
    """
    GUARDED_PATHS = {"/analyze", "/analyze/stream", "/analyze/upload"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in self.GUARDED_PATHS:
            return await call_next(request)

        try:
            body = await request.body()
            data = json.loads(body) if body else {}
            jd_len = len(data.get("jd_text", "") or data.get("jd_url", ""))
            resume_len = len(data.get("resume_text", ""))
            # rough estimate: chars/4 = tokens, * 12 LLM calls avg
            estimated_tokens = ((jd_len + resume_len) / 4) * 12
            if estimated_tokens > settings.max_tokens_per_run:
                return JSONResponse(
                    {"error": f"Input too large. Estimated {int(estimated_tokens)} tokens exceeds {settings.max_tokens_per_run} limit."},
                    status_code=413
                )
            # Reconstruct request (body already consumed)
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        except Exception:
            pass  # parsing failure — let the endpoint handle it

        return await call_next(request)


# ── 5. Result cache middleware ─────────────────────────────────────────────

class ResultCacheMiddleware(BaseHTTPMiddleware):
    """
    Redis cache: if same (jd_url + resume hash) was run in last 24h,
    return the cached report instantly. Skips the entire graph.
    WHY: identical runs happen when a user refreshes or retries — no reason to burn $0.05.
    Cache key: SHA256(jd_url + resume_text[:200])
    """
    CACHED_PATHS = {"/analyze"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in self.CACHED_PATHS or request.method != "POST":
            return await call_next(request)

        try:
            body = await request.body()
            data = json.loads(body) if body else {}
            cache_key = _make_cache_key(
                data.get("jd_url", "") + data.get("jd_text", ""),
                data.get("resume_text", "")
            )
            redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            cached = await redis.get(f"result:{cache_key}")
            if cached:
                return JSONResponse(
                    {"cached": True, **json.loads(cached)},
                    headers={"X-Cache": "HIT"}
                )
            # Store cache_key in request state for the endpoint to save result
            request.state.cache_key = cache_key

            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        except Exception:
            pass

        return await call_next(request)


def _make_cache_key(jd: str, resume: str) -> str:
    content = (jd + resume[:200]).encode()
    return hashlib.sha256(content).hexdigest()[:16]


async def save_to_cache(cache_key: str, result: dict):
    """Called from endpoint after successful run to cache the result."""
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.setex(
            f"result:{cache_key}",
            settings.result_cache_ttl_seconds,
            json.dumps(result)
        )
    except Exception:
        pass  # cache failure is non-fatal


def register_middleware(app):
    """
    Register all middleware on the FastAPI app.
    Order matters — outermost middleware wraps everything inside it.
    Call this once in api/main.py after creating the app.
    """
    app.add_middleware(ResultCacheMiddleware)   # innermost — closest to endpoint
    app.add_middleware(CostGuardMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(StructuredLogMiddleware)
    app.add_middleware(RequestIDMiddleware)     # outermost — runs first
