"""Metric definitions used by the benchmark and tests."""
from __future__ import annotations


def tokens_per_second(tokens: int, seconds: float) -> float:
    if tokens < 0 or seconds <= 0:
        raise ValueError("tokens must be nonnegative and seconds positive")
    return tokens / seconds


def per_request_latency_ms(batch_latency_ms: float, batch_size: int) -> float:
    if batch_latency_ms < 0 or batch_size < 1:
        raise ValueError("latency must be nonnegative and batch_size positive")
    return batch_latency_ms / batch_size


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 100:
        raise ValueError("values must be nonempty and q must be between 0 and 100")
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)
