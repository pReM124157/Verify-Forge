import pytest

from rate_limiter import RateLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make():
    c = Clock()
    return c, RateLimiter(clock=c)


def test_R1_R2_accepts_up_to_70_then_rejects():
    c, rl = make()
    assert all(rl.allow("a") for _ in range(70))
    assert not rl.allow("a")


def test_R3_keys_are_isolated():
    c, rl = make()
    assert all(rl.allow("a") for _ in range(70))
    assert not rl.allow("a")
    assert rl.allow("b")


def test_R4_expired_requests_free_capacity():
    c, rl = make()
    assert all(rl.allow("a") for _ in range(70))
    c.t += 61
    assert all(rl.allow("a") for _ in range(70))
    assert not rl.allow("a")


def test_R5_rejected_requests_do_not_consume_capacity():
    c, rl = make()
    assert all(rl.allow("a") for _ in range(70))
    c.t += 30
    assert not any(rl.allow("a") for _ in range(100))
    c.t += 30.5  # the original 70 are now expired; the 100 rejections must not still be counted
    assert rl.allow("a")


def test_R6_window_slides():
    c, rl = make()
    assert all(rl.allow("a") for _ in range(35))
    c.t += 30
    assert all(rl.allow("a") for _ in range(35))
    c.t += 31  # only the first 35 have expired
    assert sum(rl.allow("a") for _ in range(100)) == 35
