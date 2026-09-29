from __future__ import annotations

import argparse

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU Transformer inference microbenchmarks")
    parser.add_argument("--mode", choices=["quick", "standard"], default="quick")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    result = run_benchmark(args.mode, args.out)
    print(f"Wrote {len(result['rows'])} measurement rows to {args.out}; device=CPU; trained_model=false")
