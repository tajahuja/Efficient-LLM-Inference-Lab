# Results summary

**Run:** standard mode; 15 aggregate rows; x86_64 CPU; 9 visible logical CPUs; Python 3.12.14; NumPy 2.3.5. No GPU was detected.

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

At batch 1 and fixed model/output length, prefill latency rose from 0.316 ms at 16 tokens to 7.220 ms at 256 tokens. Logical FP32 KV payload rose linearly from 24,576 to 270,336 bytes (context plus 8 output tokens). Dense attention work and larger temporary arrays are plausible contributors. This does not predict long-context behavior in other architectures or kernels.

## Batch scaling

| Batch | End-to-end (ms) | Amortized/request (ms) | Aggregate tokens/s |
|---:|---:|---:|---:|
| 1 | 2.279 | 2.279 | 3510.5 |
| 2 | 3.115 | 1.557 | 5136.9 |
| 4 | 4.840 | 1.210 | 6611.7 |
| 8 | 10.229 | 1.279 | 6256.5 |

Aggregate throughput rose from 3,050.9 tokens/s at batch 1 to 7,178.2 at batch 4, then measured 6,569.8 at batch 8. Amortized per-request latency fell from 2.622 ms to 1.114 ms at batch 4 and was 1.218 ms at batch 8. These tiny CPU/NumPy runs are short and sensitive to system noise and BLAS behavior; aggregate latency divided by batch is not a per-request serving latency.

## Model width scaling

| Variant | Width | Parameter payload (bytes) | KV payload (bytes) | End-to-end (ms) |
|---|---:|---:|---:|---:|
| small | 32 | 163,840 | 36,864 | 1.518 |
| base | 64 | 524,288 | 73,728 | 1.641 |

Doubling width from 32 to 64 increased parameter payload about 3.2x because the embedding/output and feed-forward matrices scale differently from width alone. Latency increased modestly in this two-point sample. There are only two untrained configurations, so this is not a scaling law or pretrained model-size study.

## Weight precision comparison

| Weight dtype | Weight payload (bytes) | KV dtype | End-to-end (ms) | Max abs logit difference vs FP32 | Greedy next token agrees? |
|---|---:|---|---:|---:|---|
| float32 | 524,288 | FP32 | 1.702 | 0.000000 | True |
| float16 | 262,144 | FP32 | 39.114 | 0.007082 | True |

FP16 halved the weight-array payload but ran much slower in this NumPy CPU implementation (39.254 ms versus 2.202 ms end-to-end). The tested prompt produced the same greedy next token with max absolute logit difference 0.007082. This is one synthetic prompt, not a quality assessment. NumPy CPU FP16 support and execution path explain more than hardware accelerator behavior may; no claim about GPU half precision follows. KV remains FP32 in both rows.

## KV cache comparison

| Path | End-to-end (ms) | Generated tokens/s |
|---|---:|---:|
| Incremental KV reuse | 1.571 | 7079.0 |
| Full-prefix recomputation | 5.044 | 1594.0 |

The incremental cache path took 2.197 ms, compared with 6.807 ms for full-prefix recomputation on the same prompt/model/output setting. This is a useful correctness/performance check for the two local code paths, but its magnitude is implementation-dependent and not a serving-engine estimate. Greedy outputs match in unit tests.

## Interpretations and limitations

- Changes were isolated by holding seed, model, and workload settings fixed within each experiment, except for the parameter varied in that experiment.
- Plausible alternative explanations include BLAS threading, Python and NumPy overhead, memory allocation, CPU frequency, thermal state, and host contention.
- Weight/KV payload figures are array/tensor-size accounting, not measured process or accelerator peak memory.
- No GPU, pretrained model, INT8/4-bit quantization, live-serving queue, TTFT callback, quality benchmark, or significance test was run.
- These measurements do not establish a universal ranking or efficiency claim. See `../docs/limitations.md`.
