import numpy as np
import pytest

from inference_lab.model import ModelConfig, TinyDecoder, theoretical_kv_bytes


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_heads=3)


def test_prefill_decode_cache_shapes_and_generation():
    model = TinyDecoder(ModelConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=2))
    prompts = np.array([[1, 2, 3], [4, 5, 6]])
    logits, cache = model.prefill(prompts)
    assert logits.shape == (2, 32)
    assert cache[0][0].shape == (2, 4, 3, 4)
    next_logits, next_cache = model.decode(np.array([7, 8]), cache)
    assert next_logits.shape == (2, 32)
    assert next_cache[0][0].shape[2] == 4
    assert model.generate(prompts[:1], 3).shape == (1, 3)


def test_cache_and_recompute_greedy_outputs_match():
    model = TinyDecoder(ModelConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=1))
    prompt = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(model.generate(prompt, 4, True), model.generate(prompt, 4, False))


def test_parameter_and_kv_payload_accounting():
    cfg = ModelConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=2, dtype="float16")
    model = TinyDecoder(cfg)
    assert model.parameter_bytes() > 0
    # KV arrays are held in FP32 even when the weights are FP16.
    assert theoretical_kv_bytes(cfg, batch=2, sequence=10) == 2 * 2 * 2 * 10 * 16 * 4
