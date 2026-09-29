"""A tiny randomly initialized decoder-only Transformer for systems measurement.

This is a microbenchmark kernel, not a language model: it has no tokenizer,
pretraining, instruction tuning, or meaningful language quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    seed: int = 17
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("model dimensions must be positive and vocab_size >= 2")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("supported dtypes are float32 and float16")

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)


class TinyDecoder:
    """Pre-norm Transformer decoder with explicit per-layer KV state."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.dtype = config.numpy_dtype
        d = config.d_model
        scale = 1.0 / np.sqrt(d)
        self.embedding = (self.rng.standard_normal((config.vocab_size, d)) * scale).astype(self.dtype)
        self.layers = []
        for _ in range(config.n_layers):
            self.layers.append({
                "qkv": (self.rng.standard_normal((d, 3 * d)) * scale).astype(self.dtype),
                "out": (self.rng.standard_normal((d, d)) * scale).astype(self.dtype),
                "ff1": (self.rng.standard_normal((d, 4 * d)) * scale).astype(self.dtype),
                "ff2": (self.rng.standard_normal((4 * d, d)) * scale).astype(self.dtype),
            })
        self.lm_head = (self.rng.standard_normal((d, config.vocab_size)) * scale).astype(self.dtype)
        self.head_dim = d // config.n_heads

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        x32 = x.astype(np.float32, copy=False)
        mean = x32.mean(axis=-1, keepdims=True)
        var = ((x32 - mean) ** 2).mean(axis=-1, keepdims=True)
        return ((x32 - mean) / np.sqrt(var + 1e-5)).astype(x.dtype, copy=False)

    def _block(self, x: np.ndarray, layer: dict[str, np.ndarray], cache: tuple[np.ndarray, np.ndarray] | None
               ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        b, t, d = x.shape
        h, hd = self.config.n_heads, self.head_dim
        qkv = self._norm(x) @ layer["qkv"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, h, hd).transpose(0, 2, 1, 3).astype(np.float32)
        k = k.reshape(b, t, h, hd).transpose(0, 2, 1, 3).astype(np.float32)
        v = v.reshape(b, t, h, hd).transpose(0, 2, 1, 3).astype(np.float32)
        past = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            k = np.concatenate((cache[0], k), axis=2)
            v = np.concatenate((cache[1], v), axis=2)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
        # A query at offset past may only see keys up through past + its own index.
        qi = np.arange(t)[:, None] + past
        ki = np.arange(k.shape[2])[None, :]
        scores = np.where(ki <= qi, scores, -1e9)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(b, t, d).astype(x.dtype)
        x = x + attended @ layer["out"]
        ff = self._norm(x) @ layer["ff1"]
        x = x + np.maximum(ff, 0) @ layer["ff2"]
        return x, (k, v)

    def prefill(self, token_ids: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("token_ids must have shape [batch, nonzero_sequence]")
        x = self.embedding[ids]
        cache = []
        for layer in self.layers:
            x, kv = self._block(x, layer, None)
            cache.append(kv)
        return (x[:, -1, :] @ self.lm_head).astype(np.float32), cache

    def decode(self, token_ids: np.ndarray, cache: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1, 1)
        if len(cache) != self.config.n_layers or ids.shape[0] != cache[0][0].shape[0]:
            raise ValueError("token batch/cache shape mismatch")
        x = self.embedding[ids]
        new_cache = []
        for layer, kv in zip(self.layers, cache, strict=True):
            x, updated = self._block(x, layer, kv)
            new_cache.append(updated)
        return (x[:, -1, :] @ self.lm_head).astype(np.float32), new_cache

    def generate(self, prompts: np.ndarray, new_tokens: int, use_cache: bool = True) -> np.ndarray:
        if new_tokens < 1:
            raise ValueError("new_tokens must be positive")
        logits, cache = self.prefill(prompts)
        token = logits.argmax(axis=-1)
        out = [token]
        for _ in range(1, new_tokens):
            if use_cache:
                logits, cache = self.decode(token, cache)
            else:
                full = np.concatenate((prompts, np.stack(out, axis=1)), axis=1)
                logits, _ = self.prefill(full)
            token = logits.argmax(axis=-1)
            out.append(token)
        return np.stack(out, axis=1)

    def parameter_bytes(self) -> int:
        arrays = [self.embedding, self.lm_head]
        arrays.extend(arr for layer in self.layers for arr in layer.values())
        return sum(int(a.nbytes) for a in arrays)


def theoretical_kv_bytes(config: ModelConfig, batch: int, sequence: int) -> int:
    """Bytes for two [layers,batch,heads,sequence,head_dim] K/V tensors."""
    return 2 * config.n_layers * batch * sequence * config.d_model * np.dtype("float32").itemsize
