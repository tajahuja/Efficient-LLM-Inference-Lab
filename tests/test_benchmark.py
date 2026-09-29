import json

from inference_lab.benchmark import run_benchmark


def test_quick_run_serializes_machine_readable_results(tmp_path):
    result = run_benchmark("quick", tmp_path)
    assert len(result["rows"]) == 12
    assert {"context_scaling", "batch_scaling", "cache_comparison", "model_scaling", "precision_comparison"} == {r["workload"] for r in result["rows"]}
    reread = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert reread["caveat"].startswith("Randomly initialized")
    assert (tmp_path / "benchmark_results.csv").is_file()
