"""In-process, per-client rate limiting.

The chat endpoint costs money on every call (provider tokens), so a public
deployment needs at least a speed bump against scripted abuse. This limiter is
dependency-free and deliberately simple: it only protects the chat routes, and
it is per process. Run a shared store (Redis) if you ever scale to multiple
instances.
"""

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Allow at most ``limit`` events per ``window_seconds`` for each key."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def check(self, key: str) -> tuple[bool, int]:
        """Register a hit for ``key``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is 0 when the
        request is allowed.
        """
        if self.limit <= 0:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._events.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(bucket[0] + self.window_seconds - now) + 1)
                return False, retry_after

            bucket.append(now)
            self._sweep(now, cutoff)
            return True, 0

    def _sweep(self, now: float, cutoff: float) -> None:
        """Drop idle clients so memory cannot grow with unique IPs."""
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        for key in [k for k, v in self._events.items() if not v or v[-1] < cutoff]:
            self._events.pop(key, None)
