# Inference concepts used in this project

- **Prefill** processes the prompt and computes per-layer key/value tensors. Its work depends on prompt length and implementation.
- **Autoregressive decode** produces tokens one at a time; each next-token step consumes earlier context.
- **Transformer KV cache** stores attention keys and values from prior tokens so decode need not recompute those token representations. In this implementation the cache is explicit per layer and is FP32.
- **Application response cache** maps an exact request to a prior completed response. It avoids a model call on a cache hit; it does not reuse intermediate Transformer state.
- **Prompt/prefix cache** reuses computation for shared prompt prefixes across requests, subject to model/runtime-specific constraints. It is not the same as returning a cached final answer.
- **Batching** groups multiple requests into a model call. It can alter total throughput and individual response latency; this project's per-request figure is a batch-time average, not a serving queue measurement.
- **Quantization** represents weights or cache values with fewer bits. It can lower payload size and can change runtime and output behavior. No quantization experiment is claimed here.
- **Model compression** broadly includes quantization, pruning, distillation, and low-rank methods; these are not interchangeable.
- **Speculative decoding** uses a draft process to propose tokens and a target model to verify them. It is not implemented here.
- **Continuous batching** admits and retires requests dynamically during generation; this fixed-batch offline microbenchmark does not implement serving or scheduling.

The Yandex Research article *The KV cache as an agent runtime* motivates a broader systems question: cached state may be relevant to runtime interaction protocols, not only repeated arithmetic avoidance. This repo is an independent conceptual exploration of the narrower mechanics of prefill, decode, batching, and KV-state reuse. It neither implements nor reproduces that work.
