"""
gateway/router.py
──────────────────
API Gateway layer — sits in front of FastAPI.

WHY AN API GATEWAY:
  Without a gateway: every client talks directly to the FastAPI server.
  One endpoint change breaks all clients. No versioning. No circuit breaker.
  With a gateway: clients talk to /v1/... routes.
  The gateway handles: versioning, request routing, circuit breaking, health checks.

IN PRODUCTION: This would be Kong, AWS API Gateway, or Nginx.
IN THIS PROJECT: We implement a lightweight gateway as FastAPI routers
  so the concept is demonstrated without infrastructure overhead.

WHAT THE GATEWAY ADDS:
  - /v1/analyze, /v1/analyze/stream  — versioned routes (clients stay stable)
  - /health and /ready               — health + readiness probes for Kubernetes
  - /metrics                         — basic token cost + latency counters
  - Circuit breaker                  — if Anthropic API is down, fail fast
    instead of queuing requests that will all timeout
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import time
import asyncio
from collections import deque
from config.settings import settings

# ── Versioned router ──────────────────────────────────────────────────────────
# All API endpoints are mounted under /v1/ via this router.
# When v2 is needed, add a v2 router without breaking v1 clients.
v1_router = APIRouter(prefix="/v1", tags=["v1"])


# ── Circuit breaker ───────────────────────────────────────────────────────────
# WHY: If the Anthropic API is down, every request will hang for 30s then fail.
# A circuit breaker detects consecutive failures and fast-fails subsequent requests
# for a cooldown period, protecting the system from cascade failures.

class CircuitBreaker:
    """
    States: CLOSED (normal) → OPEN (failing fast) → HALF_OPEN (testing recovery)
    Opens after 5 consecutive failures. Resets after 60 second cooldown.
    """
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"   # CLOSED = normal, OPEN = fast-fail, HALF_OPEN = testing

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"

    def can_attempt(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True   # allow one test request
            return False      # fast-fail
        return True   # HALF_OPEN: allow the test request


# Global circuit breaker for Anthropic API calls
anthropic_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


# ── Metrics collector ─────────────────────────────────────────────────────────
# Simple in-memory metrics. In production: Prometheus + Grafana.

class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.latencies_ms = deque(maxlen=100)   # last 100 requests
        self.errors = 0

    def record(self, tokens: int, latency_ms: int, cost: float, error: bool = False):
        self.total_requests += 1
        self.total_tokens_used += tokens
        self.total_cost_usd += cost
        self.latencies_ms.append(latency_ms)
        if error:
            self.errors += 1

    def summary(self) -> dict:
        lats = list(self.latencies_ms)
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "error_rate": round(self.errors / max(self.total_requests, 1), 3),
            "avg_latency_ms": int(sum(lats) / len(lats)) if lats else 0,
            "p95_latency_ms": int(sorted(lats)[int(len(lats) * 0.95)]) if lats else 0,
        }


metrics = MetricsCollector()


# ── Gateway routes ────────────────────────────────────────────────────────────

@v1_router.get("/health")
async def health():
    """Basic health check — is the server alive?"""
    return {"status": "ok", "model": settings.model_name}


@v1_router.get("/ready")
async def readiness():
    """
    Readiness probe — is the server ready to handle traffic?
    Checks: Redis reachable, Qdrant reachable, circuit breaker state.
    Kubernetes uses this to know when to send traffic.
    """
    checks = {}

    # Check Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unreachable"

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception:
        checks["qdrant"] = "unreachable"

    checks["circuit_breaker"] = anthropic_circuit.state
    checks["mcp_enabled"] = settings.mcp_enabled

    all_ok = checks["redis"] == "ok" and checks["qdrant"] == "ok"
    return JSONResponse(
        {"ready": all_ok, "checks": checks},
        status_code=200 if all_ok else 503
    )


@v1_router.get("/metrics")
async def get_metrics():
    """Basic token cost and latency metrics. Replace with Prometheus in production."""
    return metrics.summary()


@v1_router.get("/circuit")
async def circuit_status():
    """Check circuit breaker state. Useful for debugging when Anthropic API is flaky."""
    return {
        "state": anthropic_circuit.state,
        "failures": anthropic_circuit.failures,
        "threshold": anthropic_circuit.failure_threshold,
    }
