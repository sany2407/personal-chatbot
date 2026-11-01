from app.core.ratelimit import SlidingWindowLimiter


def test_allows_requests_up_to_the_limit():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    assert [limiter.check("1.2.3.4")[0] for _ in range(3)] == [True, True, True]


def test_blocks_after_the_limit_with_retry_after():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")

    allowed, retry_after = limiter.check("1.2.3.4")
    assert allowed is False
    assert 1 <= retry_after <= 61


def test_clients_are_tracked_independently():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60)
    assert limiter.check("a")[0] is True
    assert limiter.check("b")[0] is True
    assert limiter.check("a")[0] is False


def test_zero_limit_disables_limiting():
    limiter = SlidingWindowLimiter(limit=0, window_seconds=60)
    assert all(limiter.check("a")[0] for _ in range(100))


def test_window_expiry_allows_new_requests(monkeypatch):
    import app.core.ratelimit as module

    clock = {"now": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False

    clock["now"] += 11
    assert limiter.check("a")[0] is True


def test_idle_clients_are_swept(monkeypatch):
    import app.core.ratelimit as module

    clock = {"now": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    limiter = SlidingWindowLimiter(limit=5, window_seconds=10)
    limiter.check("stale")
    clock["now"] += 11
    limiter.check("fresh")

    assert "stale" not in limiter._events
    assert "fresh" in limiter._events
