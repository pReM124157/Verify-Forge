import threading
import time

from rate_limiter import RateLimiter


def slow_clock():
    time.sleep(0.001)  # a clock that yields, as real clocks and schedulers do
    return 1000.0


def hammer(rl, keys, threads_per_key=100):
    accepted = {k: 0 for k in keys}
    lock = threading.Lock()
    barrier = threading.Barrier(len(keys) * threads_per_key)

    def worker(key):
        barrier.wait()
        if rl.allow(key):
            with lock:
                accepted[key] += 1

    ts = [threading.Thread(target=worker, args=(k,)) for k in keys for _ in range(threads_per_key)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return accepted


def test_ADV_concurrent_capacity():
    accepted = hammer(RateLimiter(clock=slow_clock), ["a"])["a"]
    assert accepted <= 70, f"expected <= 70 accepted, observed {accepted}"


def test_ADV_concurrent_capacity_per_key():
    accepted = hammer(RateLimiter(clock=slow_clock), ["a", "b"])
    for key, n in accepted.items():
        assert n <= 70, f"key {key}: expected <= 70 accepted, observed {n}"
