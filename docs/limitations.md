# Limitations and claims boundary

1. The decoder has random untrained weights and synthetic token IDs. No language understanding, generation quality, or user-facing behavior is evaluated.
2. All executed experiments are CPU-only and use one host. There are no GPU, CUDA, VRAM, utilization, or cross-device results.
3. FP32 and FP16 weights were timed in this NumPy implementation, while the KV cache remained FP32. The FP16 result is implementation-specific and is not evidence about accelerator half-precision performance. INT8 and 4-bit quantization were not executed.
4. Batch scaling is fixed-batch offline compute, not a queueing/serving test. The per-request metric divides batch elapsed time by batch size and hides scheduling and tail latency.
5. Context range is capped at 256 tokens in standard mode. It does not probe long-context capacity or architecture-specific context limits.
6. The model is deliberately tiny and NumPy/BLAS performance is not representative of optimized inference engines such as vLLM, llama.cpp, or Transformers serving.
7. Latencies are sensitive to CPU frequency, BLAS implementation, thread count, co-tenancy, and thermal state. The run uses a small number of repeats; no statistical inference is intended.
8. Memory numbers are exact tensor payload estimates, not measured peak resident memory. They omit activations, allocator overhead, temporary arrays, Python objects, BLAS workspace, and OS memory.
9. Only two synthetic model widths were compared; this is not a real pretrained-model scaling study. No trained model, tokenizer, network download, response cache, prefix-cache server, speculative decoder, or continuous batching system was tested.
10. Cache-on and full-prefix recomputation paths are implemented in this microbenchmark. Their comparison should not be treated as a general product-level estimate of KV-cache benefits.

## Do not claim

- “GPU inference was benchmarked” or any GPU/VRAM performance figure.
- “Quantization reduced memory or improved speed” from this evidence; the FP16 comparison concerns the toy model’s weight dtype only and was slower on this CPU.
- “This model produces useful language” or any quality result.
- General claims that larger batches, longer contexts, or a cache improve performance across LLMs.
- Reproduction or implementation of Yandex Research's work.
- A serving-level p95/SLO from the small repeat set.
