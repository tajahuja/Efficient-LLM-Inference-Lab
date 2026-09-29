# Research report: Efficient LLM inference microbenchmarks

## Abstract

This study measures context length, fixed batch size, model width, weight precision, and explicit KV reuse in a small NumPy decoder-only Transformer on one CPU host. The model has seeded random weights and synthetic token IDs; it is not a pretrained language model. In the standard run, prefill median increased from 0.316 ms at a 16-token prompt to 7.220 ms at 256 tokens. Aggregate throughput rose from 3,050.9 tokens/s at batch 1 to 7,178.2 at batch 4, then measured 6,569.8 at batch 8. FP16 halved weight-array payload but was slower than FP32 in this CPU NumPy implementation. Incremental KV reuse was faster than recomputing the prefix in this implementation. These measurements characterize only the executed code path and environment; they do not establish pretrained-model quality or GPU/serving behavior.

## Introduction

Inference cost depends on prompt processing, autoregressive decode, cache state, workload shape, and hardware. This repository makes a small part of that cost inspectable by implementing explicit prefill and cached decode, then varying prompt length, fixed batch size, model width, and parameter dtype on CPU.

## Research Questions

1. How do prompt length and fixed batch size relate to latency and throughput in this implementation?
2. What difference does the explicit per-layer KV cache make relative to recomputing the complete prefix at every output step?
3. How do model width and weight precision change parameter payload and measured latency here?
4. Which comparisons remain unmeasured because the inspected environment lacks a GPU/framework and the model is untrained?

## Hypotheses

The hypotheses were recorded in `research_notes.md` before the run: longer prompts raise prefill cost and cache payload; larger batches may trade per-request latency for aggregate throughput; cache reuse may reduce repeated prefix work; lower-precision weights reduce weight payload but can change runtime and output behavior. The experiment tests them as bounded observations.

## Related Work

Autoregressive Transformers retain previous keys and values to avoid recomputation during token-by-token decoding. Runtime libraries expose multiple cache policies, each with memory and compilation trade-offs. Yandex Research's 2026 article explores cache state as an interaction/runtime mechanism; it provides conceptual motivation only. This project neither implements nor reproduces the article's approach. References appear in the README.

## Experimental Methodology

A seeded, 2-layer, width-64, 4-head synthetic decoder with 256 integer token IDs was implemented in NumPy. A single machine ran context-scaling, fixed-batch, model-width, precision, and cache-reuse measurements. Standard mode uses five timed repeats after an untimed prefill warm-up. See `methodology.md` for controls, definitions, and exact code settings.

## Hardware and Software Environment

See `results/system_info.json` for captured metadata. The reported run is CPU-only; no CUDA-capable GPU, PyTorch, or Transformers was available in the inspected environment. The host CPU model was exposed as part of the system metadata after inspecting `/proc/cpuinfo`.

## Models

The executed widths were 32 and 64, both with two layers and four heads; the base model width is 64. Weights are seeded random arrays. There is no tokenizer, trained checkpoint, or external model license to assess. These are microbenchmark configurations rather than language models selected for generation quality.

## Workloads

The workloads are synthetic integer token sequences with greedy generation. Context lengths are 16–256. Batch sizes are 1–8. The cache test compares explicit incremental KV state with repeated full-prefix prefill. Precision compares FP32 and FP16 parameter arrays with an FP32 KV cache. There is no natural-language request mix, request queue, or service process.

## Metrics

The result JSON and CSV report prefill, decode, and end-to-end timing; generation and aggregate tokens/s; amortized per-request latency; exact parameter array bytes; theoretical FP32 KV payload; and for the precision comparison the maximum absolute logit difference and greedy next-token agreement versus FP32. No peak RSS or GPU memory metric is reported.

## Experiments and Results

The standard run produced 15 rows across five workload types. Detailed tables are in `results/summary.md`.

- Prefill median rose from 0.316 ms (16 prompt tokens) to 7.220 ms (256 tokens), batch size 1.
- Aggregate throughput rose from 3,050.9 tokens/s at batch 1 to 7,178.2 at batch 4, and was 6,569.8 at batch 8. Amortized per-request latency was not monotonic: 2.622 ms, 1.114 ms, 1.218 ms for batch sizes 1, 4, and 8, respectively.
- Width 32 used 163,840 bytes of parameter arrays and measured 1.961 ms end-to-end; width 64 used 524,288 bytes and measured 2.193 ms under the same prompt and batch.
- FP16 used 262,144 parameter bytes versus 524,288 in FP32, while end-to-end latency was 39.254 ms versus 2.202 ms. On the fixed synthetic prompt, next-token argmax agreed and max absolute logit difference was 0.007082.
- Incremental KV reuse measured 2.197 ms end-to-end versus 6.807 ms for full-prefix recomputation.

## Analysis

Longer contexts increase prefill work and cached tensor payload in this dense implementation. Larger fixed batches improved aggregate throughput through batch 4, but throughput dipped at batch 8 and amortized latency rose from the batch-4 value. FP16 parameter arrays reduced weight payload but were considerably slower on this CPU/NumPy path; the KV tensors remained FP32. Incremental KV reuse avoided repeated full-prefix work in this implementation. Plausible alternative explanations for individual timing differences include BLAS threading, temporary allocation, CPU frequency, thermal state, scheduling, and system load. The small number of runs and one synthetic prompt for precision agreement preclude broad statistical or quality conclusions.

## Limitations

This study uses one CPU host, tiny untrained weights, synthetic token IDs, contexts capped at 256, fixed batches, and a handful of repeats. There is no GPU, pretrained model, INT8/4-bit quantization, memory profiler, online request trace, continuous batching, model-quality benchmark, or significance test. Tensor payload estimates are not measured peak memory. See `limitations.md` for the claims boundary.

## Reproducibility

Install `.[test,plots]`, run `pytest`, execute `python experiments/run_benchmarks.py --mode quick` (or `standard`), then `python experiments/generate_figures.py`. Timing depends on hardware and BLAS settings; system metadata is saved with results.

## Future Research

Add a licensed pretrained small model, verify output quality, repeat on an actual GPU, measure peak device memory, compare INT8/4-bit weights and cache with quality checks, and test request arrivals/queueing/TTFT with a serving engine.
