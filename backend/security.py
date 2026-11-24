"""
Small, dependency-free protections for a public deployment:

* rate limiting for the expensive /api/evaluate endpoint
  (every call makes ~5 Groq requests on YOUR api key)
* an admin token guard for /api/ingest
"""

import hmac
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import Header, HTTPException, Request


def _int_env(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def client_ip(request: Request):
    # Behind Render's proxy the real client is the first X-Forwarded-For hop.
    forwarded = request.headers.get("x-forwarded-for", "")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


class EvaluationLimiter:
    """Per-IP sliding window + a global per-day cap (resets on restart)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits = {}          # ip -> [timestamps]
        self._day = None
        self._day_count = 0

    def check(self, ip):
        per_minute = _int_env("RATE_LIMIT_PER_MINUTE", 5)
        daily_cap = _int_env("DAILY_EVAL_CAP", 300)

        now = time.time()
        today = datetime.now(timezone.utc).date()

        with self._lock:
            if self._day != today:
                self._day = today
                self._day_count = 0

            if daily_cap > 0 and self._day_count >= daily_cap:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "Daily evaluation limit reached for this demo. "
                        "Please try again tomorrow."
                    ),
                    headers={"Retry-After": "3600"}
                )

            if per_minute > 0:
                window = [
                    t for t in self._hits.get(ip, [])
                    if now - t < 60
                ]

                if len(window) >= per_minute:
                    retry = max(1, int(60 - (now - window[0])))

                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "Too many requests. Please wait a moment "
                            "and try again."
                        ),
                        headers={"Retry-After": str(retry)}
                    )

                window.append(now)
                self._hits[ip] = window

                # keep the dict from growing forever
                if len(self._hits) > 5000:
                    self._hits = {
                        k: v for k, v in self._hits.items()
                        if v and now - v[-1] < 60
                    }

            self._day_count += 1


_limiter = EvaluationLimiter()


def rate_limit_evaluate(request: Request):
    """FastAPI dependency for /api/evaluate."""
    _limiter.check(client_ip(request))


def require_admin(x_admin_token: str | None = Header(default=None)):
    """
    FastAPI dependency for /api/ingest.
    Disabled entirely unless ADMIN_TOKEN is set in the environment.
    """

    expected = os.getenv("ADMIN_TOKEN", "")

    if not expected:
        raise HTTPException(
            status_code=403,
            detail=(
                "Ingestion is disabled on this server. "
                "Set ADMIN_TOKEN to enable it."
            )
        )

    if not x_admin_token or not hmac.compare_digest(
        x_admin_token.encode("utf-8"),
        expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token."
        )
