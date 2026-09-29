import pytest

from inference_lab.metrics import percentile, per_request_latency_ms, tokens_per_second


def test_token_rate_and_batch_latency():
    assert tokens_per_second(20, 4) == 5
    assert per_request_latency_ms(120, 4) == 30


def test_invalid_metrics_fail_fast():
    with pytest.raises(ValueError):
        tokens_per_second(2, 0)
    with pytest.raises(ValueError):
        per_request_latency_ms(2, 0)


def test_percentile_interpolates():
    assert percentile([1, 3, 5], 50) == 3
    assert percentile([1, 3, 5], 95) == pytest.approx(4.8)
