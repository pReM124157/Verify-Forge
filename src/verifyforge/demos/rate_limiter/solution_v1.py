import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window rate limiter: 60 requests per 60 s plus a burst of 10 (70 total)."""

    WINDOW = 60.0
    LIMIT = 70

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        hits = self._hits[key]
        cutoff = self._clock() - self.WINDOW
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.LIMIT:
            return False
        hits.append(self._clock())
        return True
