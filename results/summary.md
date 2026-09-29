# Results summary

**Run:** standard mode; 15 aggregate rows; INTEL(R) XEON(R) PLATINUM 8573C CPU; 9 visible logical CPUs; Python 3.12.14; NumPy 2.3.5. No GPU was detected.

Randomly initialized Transformer; CPU microbenchmark, not a pretrained language-model evaluation.

All latencies below are medians from five timed repeats in standard mode. They are single-host observations, not confidence intervals. The model is randomly initialized and has no language-quality interpretation.

## Context scaling

| Prompt tokens | Prefill (ms) | End-to-end (ms) | FP32 KV payload (bytes) |
|---:|---:|---:|---:|
| 16 | 0.370 | 1.861 | 24,576 |
| 32 | 0.421 | 1.820 | 40,960 |
| 64 | 1.117 | 2.697 | 73,728 |
| 128 | 2.559 | 4.692 | 139,264 |
| 256 | 7.425 | 9.891 | 270,336 |

At batch 1 and fixed model/output length, prefill latency rose from 0.370 ms at 16 tokens to 7.425 ms at 256 tokens. Logical FP32 KV payload rose from 24,576 to 270,336 bytes (prompt plus generated tokens). Dense attention work and larger temporary arrays are plausible contributors. This does not predict long-context behavior in other architectures or kernels.

## Batch scaling

| Batch | End-to-end (ms) | Amortized/request (ms) | Aggregate tokens/s |
|---:|---:|---:|---:|
| 1 | 2.279 | 2.279 | 3510.5 |
| 2 | 3.115 | 1.557 | 5136.9 |
| 4 | 4.840 | 1.210 | 6611.7 |
| 8 | 10.229 | 1.279 | 6256.5 |

Aggregate throughput rose from 3510.5 tokens/s at batch 1 to 6611.7 at batch 4, then measured 6256.5 at batch 8. Amortized per-request latency was 2.279 ms at batch 1, 1.210 ms at batch 4, and 1.279 ms at batch 8. These short CPU/NumPy runs are sensitive to system noise and BLAS behavior; aggregate batch latency divided by batch size is not observed per-request serving latency.

## Model width scaling

| Variant | Width | Parameter payload (bytes) | KV payload (bytes) | End-to-end (ms) |
|---|---:|---:|---:|---:|
| small | 32 | 163,840 | 36,864 | 1.518 |
| base | 64 | 524,288 | 73,728 | 1.641 |

Width increased from 32 to 64; parameter payload increased from 163,840 to 524,288 bytes, while end-to-end latency was 1.518 ms and 1.641 ms. There are only two untrained configurations, so this is not a scaling law or pretrained model-size study.

## Weight precision comparison

| Weight dtype | Weight payload (bytes) | KV dtype | End-to-end (ms) | Max abs logit difference vs FP32 | Greedy next token agrees? |
|---|---:|---|---:|---:|---|
| float32 | 524,288 | FP32 | 1.702 | 0.000000 | True |
| float16 | 262,144 | FP32 | 39.114 | 0.007082 | True |

FP16 halved the weight-array payload compared with FP32, while median end-to-end latency was 39.114 ms versus 1.702 ms in this NumPy CPU implementation. The one tested prompt produced the same greedy next token with max absolute logit difference 0.007082. This is not a quality assessment. CPU NumPy FP16 behavior does not predict accelerator half-precision performance; KV remained FP32.

## KV cache comparison

| Path | End-to-end (ms) | Generated tokens/s |
|---|---:|---:|
| Incremental KV reuse | 1.571 | 7079.0 |
| Full-prefix recomputation | 5.044 | 1594.0 |

Incremental KV reuse measured 1.571 ms compared with 5.044 ms for full-prefix recomputation on the same prompt/model/output setting. This is a correctness/performance check of two local code paths, not an optimized serving-engine estimate. Greedy outputs match in unit tests.

## Interpretation and limitations

- Within each experiment, seed, model, and workload settings were held fixed except for the factor under test.
- Plausible alternative explanations include BLAS threading, Python and NumPy overhead, temporary allocation, CPU frequency, thermal state, and host contention.
- Weight/KV payload figures are array/tensor-size accounting, not measured process or accelerator peak memory.
- No GPU, pretrained model, INT8/4-bit quantization, online serving queue, streaming TTFT callback, quality benchmark, or significance test was run.
- These measurements do not establish a universal ranking or efficiency claim. See `../docs/limitations.md`.
