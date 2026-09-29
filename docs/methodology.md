# Methodology and execution environment

## Environment inspected

The benchmark run for this repository used Linux x86_64, Python 3.12.14, 9 visible logical CPUs, and an 8 GiB cgroup memory limit. NumPy and Matplotlib were available. PyTorch, Transformers, `nvidia-smi`, and a CUDA GPU were not available. The host kernel reports roughly 10 GiB total memory, but this process is constrained by the 8 GiB cgroup limit. The exact environment inventory is saved in `results/system_info.json`.

## Why this model

A small decoder-only Transformer is implemented in NumPy with seed 17, vocabulary size 256, width 64, four attention heads, and two layers. It is initialized from seeded random weights and does not download checkpoints. This permits a genuine, inspectable prefill/decode/KV-cache experiment on CPU without implying pretrained language behavior or external weight licensing. Its code is covered by this repository's MIT license. It is a systems microbenchmark, not a substitute for evaluating an open language model.

## Controlled runs

`quick` uses prompt lengths 16, 32, and 64; batch sizes 1, 2, and 4; four generated tokens; three timed repeats. `standard` uses lengths 16, 32, 64, 128, and 256; batch sizes 1, 2, 4, and 8; eight generated tokens; five repeats. Both modes also execute a two-width (32 and 64) model-size comparison and an FP32/FP16 weight precision comparison at batch 1 and prompt length 64. The KV cache remains FP32 in both precision cases. All models use the same seed and request generation rule. One untimed prefill warms each measurement case. The benchmark reports medians and a nearest-rank-like linear p95 from the small repeat set; those p95 values are descriptive and not confidence intervals.

The workloads measure:

- Context scaling: prefill and cached decode are timed separately; end-to-end is their sum.
- Batch scaling: same 64-token prompt length and generation length, varying batch size. Aggregate output throughput is total generated tokens divided by batch wall time. Per-request latency is batch wall time divided by batch size; this is an amortized quantity, not each request's observed latency in a concurrent server.
- Model scaling: widths 32 and 64 with two layers, same prompt length, batch, and output count. Model parameter bytes are exact array payloads.
- Precision comparison: FP32 versus FP16 weights in the same NumPy implementation and model shape, with an FP32 KV cache. Greedy next-token agreement and maximum absolute logit difference are recorded for one fixed prompt; this is not a quality evaluation.
- Cache comparison: greedy generation with incremental per-layer KV state versus recomputing the entire growing prefix for each next token. These are two paths in this small implementation, not a benchmark of an optimized serving engine.

Prompts are arrays of deterministic integer token IDs; generated tokens use greedy argmax. No natural-language tokenizer, sampling, chat template, or quality metric is used. All measurements are single-process CPU measurements on one machine.

## Metric and memory definitions

- Prefill latency: wall-clock time around the full prompt forward pass.
- Decode latency: cumulative wall-clock time for the requested output tokens after prefill.
- End-to-end latency: prefill plus decode; medians are computed per repeat.
- Generation throughput: batch output tokens divided by decode-only time.
- Aggregate throughput: batch output tokens divided by end-to-end time.
- Per-request latency: batch end-to-end wall time divided by batch size; only an amortized comparison.
- KV payload bytes: exact logical K and V tensor payload size from layer count, batch, context plus output length, model width, and FP32 cache dtype. This excludes allocator padding and all framework/runtime overhead.
- Model parameter bytes: exact NumPy parameter-array payload size, not process memory or a checkpoint file size.
- FP16 agreement: max absolute logit difference and greedy next-token equality for a single fixed prompt against the FP32 baseline; not a quality score.

No peak RSS or device memory value is presented as measured. This avoids attributing Python process memory or host-level usage to the model. GPU utilization, GPU memory, streaming TTFT, INT8/4-bit quantization, and language-model quality were not measured. The current Python function times a full prefill and then a decode step; a true streaming TTFT callback is not instrumented.

## Reproducibility

The model and request generation are seeded. Runtime still depends on CPU, NumPy build, BLAS threading, system load, and interpreter. Re-run from a fresh process and retain the saved system metadata when comparing machines. Do not compare these results directly with GPU or pretrained model numbers.
