# Thread-Safe Sliding Window Rate Limiter

Module: rate_limiter

## Requirements
- Provide a `RateLimiter` class with `allow(key: str) -> bool` for an in-memory sliding-window limiter.
- Allow 60 standard requests per rolling 60-second window per client key.
- Allow an additional burst capacity of 10, so no more than 70 accepted requests may exist inside one active window.
- State is isolated between client keys.
- Expired timestamps are automatically removed.
- Rejected requests do not consume capacity.
- Concurrent calls from multiple threads never allow capacity to be exceeded.
- Support deterministic time injection for tests via a `clock` callable passed to the constructor (default `time.monotonic`).
