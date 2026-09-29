"""Generate labeled CPU microbenchmark figures from result JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results/benchmark_results.json").read_text())["rows"]
figdir = ROOT / "results/figures"
figdir.mkdir(parents=True, exist_ok=True)

def plot(workload: str, x: str, y: str, xlabel: str, ylabel: str, filename: str, title: str) -> None:
    data = [r for r in rows if r["workload"] == workload and r.get(x) is not None and r.get(y) is not None]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot([r[x] for r in data], [r[y] for r in data], marker="o", color="#2463a6")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=.25)
    fig.text(.5, .01, "CPU • seeded untrained NumPy Transformer • this environment only", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(figdir / filename, dpi=160)
    plt.close(fig)

plot("context_scaling", "prompt_tokens", "prefill_ms_p50", "Prompt tokens", "Prefill latency (ms, median)", "context_prefill_latency.png", "Prompt length and prefill latency")
plot("context_scaling", "prompt_tokens", "kv_cache_bytes_theoretical", "Prompt tokens", "Theoretical KV payload (bytes)", "context_kv_memory.png", "Prompt length and estimated KV payload")
plot("batch_scaling", "batch_size", "aggregate_tokens_per_second", "Batch size", "Aggregate generation throughput (tokens/s)", "batch_throughput.png", "Batch size and aggregate throughput")
plot("batch_scaling", "batch_size", "per_request_latency_ms", "Batch size", "Per-request end-to-end latency (ms)", "batch_latency.png", "Batch size and per-request latency")
cache = [r for r in rows if r["workload"] == "cache_comparison"]
if cache:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["KV cache", "Recompute prefix"], [next(r["end_to_end_ms_p50"] for r in cache if r.get("kv_cache_enabled") is True), next(r["end_to_end_ms_p50"] for r in cache if r.get("kv_cache_enabled") is False)], color=["#268a6a", "#cf7534"])
    ax.set(ylabel="End-to-end latency (ms, median)", title="Decode with cache vs repeated full-prefix recomputation")
    ax.text(.5, -.2, "Same synthetic CPU model; random weights; implementation-specific", transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "cache_comparison.png", dpi=160)
    plt.close(fig)

plot("model_scaling", "model_parameter_bytes", "end_to_end_ms_p50", "Model parameter payload (bytes)", "End-to-end latency (ms, median)", "model_size_latency.png", "Model size and CPU latency")
model_rows = [r for r in rows if r["workload"] == "model_scaling"]
if model_rows:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["model_variant"] for r in model_rows], [r["model_parameter_bytes"] + r["kv_cache_bytes_theoretical"] for r in model_rows], marker="o", color="#7854a5")
    ax.set(xlabel="Model variant", ylabel="Estimated weight + KV payload (bytes)", title="Model size and tensor payload estimate")
    ax.text(.5, -.2, "KV payload held constant at 64-token prompt, batch 1", transform=ax.transAxes, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "model_size_memory.png", dpi=160)
    plt.close(fig)
precision_rows = [r for r in rows if r["workload"] == "precision_comparison"]
if precision_rows:
    labels = [r["precision"] for r in precision_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["model_parameter_bytes"] + r["kv_cache_bytes_theoretical"] for r in precision_rows], color=["#2463a6", "#d48432"])
    ax.set(xlabel="Weight precision (KV remains FP32)", ylabel="Estimated weight + KV payload (bytes)", title="Precision and tensor payload estimate")
    fig.tight_layout()
    fig.savefig(figdir / "precision_memory.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, [r["end_to_end_ms_p50"] for r in precision_rows], color=["#2463a6", "#d48432"])
    ax.set(xlabel="Weight precision (KV remains FP32)", ylabel="End-to-end latency (ms, median)", title="Precision and CPU latency")
    fig.tight_layout()
    fig.savefig(figdir / "precision_latency.png", dpi=160)
    plt.close(fig)

print(f"Generated figures in {figdir}")
