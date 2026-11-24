import pytest
from fastapi import HTTPException

from backend import security


def test_rate_limit_blocks_after_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("DAILY_EVAL_CAP", "0")

    limiter = security.EvaluationLimiter()
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")

    with pytest.raises(HTTPException) as error:
        limiter.check("1.2.3.4")

    assert error.value.status_code == 429

    # a different visitor is unaffected
    limiter.check("5.6.7.8")


def test_daily_cap(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("DAILY_EVAL_CAP", "2")

    limiter = security.EvaluationLimiter()
    limiter.check("a")
    limiter.check("b")

    with pytest.raises(HTTPException) as error:
        limiter.check("c")

    assert error.value.status_code == 429


def test_admin_disabled_without_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    with pytest.raises(HTTPException) as error:
        security.require_admin("anything")

    assert error.value.status_code == 403


def test_admin_token_check(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")

    security.require_admin("s3cret")   # accepted

    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as error:
            security.require_admin(bad)

        assert error.value.status_code == 401
