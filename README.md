# Efficient-LLM-Inference-Lab

**An independent, reproducible CPU microbenchmark study of autoregressive Transformer inference mechanics.**

> This repository is an independent experimental study of LLM inference efficiency. Results are hardware-, model-, workload-, and implementation-dependent.

This third portfolio project studies inference systems rather than agent orchestration or tabular prediction. It implements a small decoder-only Transformer in NumPy so prefill, autoregressive decoding, batching, and per-layer KV state can be measured in a CPU-only environment without downloading model weights.

**Scope first:** this is a randomly initialized, untrained model. The experiment can teach us about this implementation's compute paths; it cannot establish language-model quality or represent modern optimized LLM serving. No GPU or pretrained checkpoint results are claimed.

## Research Question

How do context length, batching, cache reuse, model size, and inference configuration affect latency, throughput, and memory behavior? This run tests context length, fixed batch size, model width, FP16 weight precision, and KV-cache reuse on one CPU host. INT8/4-bit quantization, trained-model evaluation, and GPU comparisons are explicit future work.

## Motivation

Interactive systems have at least two compute phases: prompt prefill and incremental decode. Decode can reuse key/value states from earlier tokens. Batching can change total compute efficiency while affecting latency per request. Measuring those behaviors under controlled conditions is more informative than reporting one aggregate “fast” number.

## Key Concepts

- **Prefill:** process the prompt and create attention keys and values.
- **Decode:** generate autoregressively one token at a time.
- **KV caching:** keep prior attention keys/values so cached decoding does not recompute all prior token representations.
- **Response caching:** return a stored completed answer for an identical request. It avoids inference on a hit and is not KV caching.
- **Prefix caching:** reuse model computation for a shared prompt prefix, subject to an inference engine's constraints.
- **Batching:** process multiple requests together; aggregate throughput and interactive latency are different outcomes.
- **Quantization:** lower-bit representation for weights or cache. It may affect memory, latency, and outputs; not measured here.

More distinctions are in [inference concepts](docs/inference_concepts.md).

## Experimental Setup

| Item | Executed setup |
|---|---|
| Model | 2-layer decoder, width 64, 4 heads, vocabulary 256 |
| Weights | Seeded random initialization; no pretrained checkpoint |
| Engine | Explicit NumPy implementation; FP32 baseline weights, FP32 KV tensors (FP16 weights tested separately) |
| Device | CPU only |
| Contexts | 16, 32, 64, 128, 256 tokens in standard run |
| Batches | 1, 2, 4, 8 in standard run |
| Generation | Greedy token selection; 8 output tokens in standard run |
| Repeats | 5 timed repeats per case, after an untimed prefill warm-up |

No model files are downloaded or committed. The implementation's toy token IDs are synthetic and have no tokenizer or language quality.

## Hardware

The checked-in run metadata is in [`results/system_info.json`](results/system_info.json). The inspected run environment had Linux x86_64, Python 3.12.14, NumPy 2.3.5, 9 logical CPUs, and no `nvidia-smi` executable or detectable CUDA GPU. The environment's cgroup memory limit was 8 GiB. CPU: Intel Xeon Platinum 8573C. This is not a GPU benchmark.

## Models

Only the synthetic `synthetic-tiny-decoder-v1` family was executed at widths 32 and 64. It is random and untrained, selected because it makes a true prefill/decode/cache path runnable without large downloads. No external weights or license terms apply to its seeded random arrays; project code is MIT-licensed. A small licensed pretrained checkpoint should be added as a separate follow-up only on an environment with the required framework and download access.

## Workloads and Experiments

- **Context scaling:** change input length while keeping model, batch, and output length fixed; measure prefill and decode latency and theoretical KV payload.
- **Batch scaling:** change fixed batch size with 64-token prompts and the same model/output length; measure batch wall time, amortized per-request latency, and aggregate throughput.
- **Cache comparison:** compare incremental per-layer KV reuse with full-prefix recomputation at each decode step, on the same model/prompt.
- **Model width:** compare width 32 and 64 with two layers and four heads.
- **Weight precision:** compare FP32 and FP16 weight arrays; the KV cache stays FP32. One fixed prompt records logit difference and greedy next-token agreement.

The standard command produced 15 aggregate measurement rows. All raw measurement rows and settings are in [`results/benchmark_results.json`](results/benchmark_results.json) and [`results/benchmark_results.csv`](results/benchmark_results.csv).

### Measured results (this host only)

Standard run, medians from five timed repeats; lower latency is better. Full scope and caveats are in [`results/summary.md`](results/summary.md).

Context scaling, batch size 1:

| Prompt tokens | Prefill median (ms) | End-to-end median (ms) | Estimated KV payload (bytes) |
|---:|---:|---:|---:|
| 16 | 0.370 | 1.861 | 24,576 |
| 32 | 0.421 | 1.820 | 40,960 |
| 64 | 1.117 | 2.697 | 73,728 |
| 128 | 2.559 | 4.692 | 139,264 |
| 256 | 7.425 | 9.891 | 270,336 |

Fixed 64-token prompt, batch scaling:

| Batch size | End-to-end median (ms) | Amortized per-request latency (ms) | Aggregate throughput (tokens/s) |
|---:|---:|---:|---:|
| 1 | 2.279 | 2.279 | 3510.5 |
| 2 | 3.115 | 1.557 | 5136.9 |
| 4 | 4.840 | 1.210 | 6611.7 |
| 8 | 10.229 | 1.279 | 6256.5 |

Model width, precision, and cache checks:

| Comparison | Tensor payload (bytes) | End-to-end median (ms) |
|---|---:|---:|
| Width 32 | 200,704 | 1.518 |
| Width 64 | 598,016 | 1.641 |
| float32 weights (KV stays FP32) | 598,016 | 1.702 |
| float16 weights (KV stays FP32) | 335,872 | 39.114 |
| KV cache | 598,016 | 1.571 |
| Full-prefix recomputation | 524,288 | 5.044 |

The width-32/width-64 cases used 163,840/524,288 bytes of weights, with end-to-end medians of 1.518/1.641 ms. FP16 weights halved the weight payload, but median end-to-end latency was 39.114 ms versus 1.702 ms in this CPU NumPy run. On the single fixed prompt, the greedy next token agreed with FP32 and maximum absolute logit difference was 0.007082; this is not a language-quality assessment.

### Analysis

- Prefill latency increased from 0.370 ms at 16 tokens to 7.425 ms at 256 tokens; logical FP32 KV payload also increased. The exact timing curve depends on this implementation and host.
- Aggregate throughput rose from 3510.5 tokens/s at batch 1 to 6611.7 at batch 4, then was 6256.5 at batch 8. Amortized per-request latency was 2.279, 1.210, and 1.279 ms at batches 1, 4, and 8. This fixed-batch result is not a serving optimum.
- Incremental KV reuse measured 1.571 ms versus 5.044 ms for full-prefix recomputation in this implementation; the magnitude is not generalizable to optimized model servers.
- Parameter and KV payload are tensor-size estimates, not measured peak memory.

The exact hypotheses, controls, alternate explanations, and metric definitions are in [`docs/methodology.md`](docs/methodology.md), [`docs/research_notes.md`](docs/research_notes.md), and [`results/summary.md`](results/summary.md). Nine figures are linked below.

## Metrics

- **Prefill latency:** wall time for full prompt forward pass; the first next-token logits are available at its end. No streaming callback is measured, so this is not an instrumented serving TTFT.
- **Decode latency:** wall time for the remaining decode steps after the first token.
- **End-to-end latency:** prefill plus decode.
- **Generation throughput:** output tokens after prefill divided by decode time.
- **Aggregate throughput:** total batch output tokens divided by end-to-end time.
- **Per-request latency:** batch elapsed time divided by batch size; this is amortized latency, not individually observed request latency.
- **Memory:** exact NumPy parameter payload and theoretical FP32 KV tensor payload. Neither is peak process RSS or GPU memory.

## Figures

- [Context length vs. prefill latency](results/figures/context_prefill_latency.png)
- [Context length vs. estimated KV payload](results/figures/context_kv_memory.png)
- [Batch size vs. aggregate throughput](results/figures/batch_throughput.png)
- [Batch size vs. amortized latency](results/figures/batch_latency.png)
- [KV cache vs. full-prefix recomputation](results/figures/cache_comparison.png)
- [Model size vs. latency](results/figures/model_size_latency.png)
- [Model size vs. estimated memory](results/figures/model_size_memory.png)
- [Precision vs. estimated memory](results/figures/precision_memory.png)
- [Precision vs. latency](results/figures/precision_latency.png)

## Limitations

The result set has one CPU host, two tiny untrained model widths, synthetic token IDs, contexts no longer than 256, fixed batches, and five repeats. There is no CUDA/GPU run, pretrained model, INT8/4-bit quantization, memory profiler, online request load, continuous batching, or output-quality evaluation. See [limitations](docs/limitations.md) for the full claim boundary.

## Reproducibility

Python 3.11+ is required. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[test,plots]"
pytest
python experiments/run_benchmarks.py --mode quick
python experiments/generate_figures.py
```

`--mode standard` adds more context and batch points plus the model-width and precision checks. It remains intentionally small and does not download models. Timing can vary with CPU, BLAS library/thread count, thermal state, and system load.

## Future Work

1. Add a verified, licensed small pretrained model backend and compare output quality as well as speed.
2. Repeat on a GPU-equipped machine, record CUDA/driver/runtime metadata, and measure device memory with the framework allocator.
3. Add INT8/4-bit weight and KV quantization with matched output-quality checks.
4. Add variable-arrival serving workloads, TTFT callbacks, queueing, tail latency, and continuous batching.
5. Compare model sizes and inference engines using isolated fresh-process measurements.

## Research context and Yandex connection

Yandex Research's *The KV cache as an agent runtime* (published September 5, 2026) discusses cache sharing and scheduling as runtime mechanisms for interactive inference. It motivates asking how inference state shapes systems behavior. This repo is an **independent conceptual exploration** of basic prefill/decode and cache reuse. It does not implement Yandex's approach and does not reproduce that work.

## References

1. Vaswani, A. et al. “Attention Is All You Need.” NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Hugging Face Transformers, “KV cache strategies.” https://huggingface.co/docs/transformers/kv_cache
3. Yakushev, G. “The KV cache as an agent runtime.” Yandex Research, September 5, 2026. https://research.yandex.com/blog/the-kv-cache-as-an-agent-runtime

## Project Structure

```text
configs/                 Benchmark and model declarations
src/inference_lab/       Model, metrics, runner, system inventory
experiments/             Benchmark and figure entry points
results/                  Machine-readable results, metadata, figures
  figures/
docs/                     Methodology, concepts, report, limitations
tests/                    Metric, model/cache, serialization tests
.github/workflows/        Python CI
```
