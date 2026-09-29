"""Controlled, repeatable CPU microbenchmarks for prefill and autoregressive decode."""
from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import percentile, per_request_latency_ms, tokens_per_second
from .model import ModelConfig, TinyDecoder, theoretical_kv_bytes
from .system_info import collect_system_info


def _timed(callable_: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - start


def _measure_case(model: TinyDecoder, prompt_len: int, batch: int, gen_tokens: int,
                  repeats: int, workload: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260929 + prompt_len * 17 + batch)
    prompts = rng.integers(0, model.config.vocab_size, (batch, prompt_len), dtype=np.int64)
    model.prefill(prompts)  # one untimed warm-up for import, BLAS and allocator effects
    prefill_times: list[float] = []
    decode_times: list[float] = []
    e2e_times: list[float] = []
    for _ in range(repeats):
        (logits, cache), prefill_s = _timed(lambda: model.prefill(prompts))
        token = logits.argmax(axis=-1)
        decode_s = 0.0
        start = time.perf_counter()
        for _step in range(gen_tokens - 1):
            logits, cache = model.decode(token, cache)
            token = logits.argmax(axis=-1)
        decode_s = time.perf_counter() - start
        prefill_times.append(prefill_s)
        decode_times.append(decode_s)
        e2e_times.append(prefill_s + decode_s)
    p50 = statistics.median(e2e_times)
    p95 = percentile(e2e_times, 95)
    output_count = batch * gen_tokens
    return [{
        "workload": workload,
        "model_id": "synthetic-tiny-decoder-v1",
        "trained_model": False,
        "device": "cpu",
        "precision": model.config.dtype,
        "prompt_tokens": prompt_len,
        "generated_tokens_per_request": gen_tokens,
        "batch_size": batch,
        "repeats": repeats,
        "prefill_ms_p50": statistics.median(prefill_times) * 1000,
        "decode_ms_p50": statistics.median(decode_times) * 1000,
        "end_to_end_ms_p50": p50 * 1000,
        "end_to_end_ms_p95": p95 * 1000,
        "generation_tokens_per_second": tokens_per_second(batch * max(gen_tokens - 1, 1), max(statistics.median(decode_times), 1e-12)),
        "aggregate_tokens_per_second": tokens_per_second(output_count, p50),
        "per_request_latency_ms": per_request_latency_ms(p50 * 1000, batch),
        "model_parameter_bytes": model.parameter_bytes(),
        "kv_cache_bytes_theoretical": theoretical_kv_bytes(model.config, batch, prompt_len + gen_tokens),
        "memory_note": "tensor payload estimate only; excludes Python/NumPy/BLAS/runtime overhead",
    } for _ in [0]]


def run_benchmark(mode: str = "quick", out_dir: str | Path = "results", seed: int = 17) -> dict[str, Any]:
    if mode not in {"quick", "standard"}:
        raise ValueError("mode must be 'quick' or 'standard'")
    cfg = ModelConfig(seed=seed)
    contexts = [16, 32, 64] if mode == "quick" else [16, 32, 64, 128, 256]
    batches = [1, 2, 4] if mode == "quick" else [1, 2, 4, 8]
    gen_tokens = 4 if mode == "quick" else 8
    repeats = 3 if mode == "quick" else 5
    model = TinyDecoder(cfg)
    rows = []
    for context in contexts:
        rows.extend(_measure_case(model, context, 1, gen_tokens, repeats, "context_scaling"))
    for batch in batches:
        rows.extend(_measure_case(model, 64, batch, gen_tokens, repeats, "batch_scaling"))
    for width, heads, label in ((32, 4, "small"), (64, 4, "base")):
        scaled_cfg = ModelConfig(d_model=width, n_heads=heads, n_layers=2, seed=seed)
        scaled_model = TinyDecoder(scaled_cfg)
        size_rows = _measure_case(scaled_model, 64, 1, gen_tokens, repeats, "model_scaling")
        for row in size_rows:
            row["model_variant"] = label
            row["model_width"] = width
        rows.extend(size_rows)
    precision_models = [TinyDecoder(ModelConfig(seed=seed, dtype=dtype)) for dtype in ("float32", "float16")]
    precision_prompt = np.random.default_rng(91).integers(0, cfg.vocab_size, (1, 64), dtype=np.int64)
    reference_logits, _ = precision_models[0].prefill(precision_prompt)
    reference_token = reference_logits.argmax(axis=-1)
    for precision_model in precision_models:
        logits, _ = precision_model.prefill(precision_prompt)
        row = _measure_case(precision_model, 64, 1, gen_tokens, repeats, "precision_comparison")[0]
        row["max_abs_logit_difference_vs_fp32"] = float(np.max(np.abs(logits - reference_logits)))
        row["greedy_next_token_agreement_vs_fp32"] = bool(np.array_equal(logits.argmax(axis=-1), reference_token))
        rows.append(row)
    for cached in (True, False):
        prompt = np.random.default_rng(91).integers(0, cfg.vocab_size, (1, 64), dtype=np.int64)
        def timed_cache_path() -> tuple[float, float]:
            (logits, cache), prefill_s = _timed(lambda: model.prefill(prompt))
            token = logits.argmax(axis=-1)
            generated = [token]
            start = time.perf_counter()
            for _step in range(1, gen_tokens):
                if cached:
                    logits, cache = model.decode(token, cache)
                else:
                    full = np.concatenate((prompt, np.stack(generated, axis=1)), axis=1)
                    logits, _ = model.prefill(full)
                token = logits.argmax(axis=-1)
                generated.append(token)
            return prefill_s, time.perf_counter() - start
        timed_cache_path()
        samples: list[tuple[float, float]] = []
        for _ in range(repeats):
            samples.append(timed_cache_path())
        totals = [a + b for a, b in samples]
        elapsed = statistics.median(totals)
        decode_elapsed = statistics.median([b for _, b in samples])
        rows.append({
            "workload": "cache_comparison", "model_id": "synthetic-tiny-decoder-v1", "trained_model": False,
            "device": "cpu", "precision": cfg.dtype, "prompt_tokens": 64,
            "generated_tokens_per_request": gen_tokens, "batch_size": 1, "repeats": repeats,
            "prefill_ms_p50": statistics.median([a for a, _ in samples]) * 1000,
            "decode_ms_p50": decode_elapsed * 1000, "end_to_end_ms_p50": elapsed * 1000,
            "end_to_end_ms_p95": percentile([s * 1000 for s in totals], 95),
            "generation_tokens_per_second": tokens_per_second(gen_tokens - 1, decode_elapsed),
            "aggregate_tokens_per_second": tokens_per_second(gen_tokens, elapsed),
            "per_request_latency_ms": elapsed * 1000, "kv_cache_enabled": cached,
            "model_parameter_bytes": model.parameter_bytes(),
            "kv_cache_bytes_theoretical": theoretical_kv_bytes(cfg, 1, 64 + gen_tokens) if cached else None,
            "memory_note": "tensor payload estimate only; excludes Python/NumPy/BLAS/runtime overhead",
        })
    results = {"schema_version": 1, "mode": mode, "model_config": asdict(cfg),
               "system_info": collect_system_info(), "rows": rows,
               "caveat": "Randomly initialized Transformer; CPU microbenchmark, not a pretrained language-model evaluation."}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out / "system_info.json").write_text(json.dumps(results["system_info"], indent=2) + "\n", encoding="utf-8")
    if rows:
        with (out / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    return results
