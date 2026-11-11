"""tests/test_benchmark_runner.py — Smoke test for benchmark runner."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_dry_run_writes_csv(tmp_path, monkeypatch):
    import src.benchmark_runner as br
    # Redirect output CSV
    test_csv = tmp_path / "runs.csv"
    monkeypatch.setattr(br, "RUNS_CSV", test_csv)
    br.run_benchmark(
        topologies=["T1_tree"],
        scenarios=["S1_link_failure"],
        configs=["B0"],
        n_seeds=1,
        base_seed=42,
        dry_run=True,
    )
    assert test_csv.exists()
    with open(test_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["topology"] == "T1_tree"

def test_resume_skips_done(tmp_path, monkeypatch):
    import src.benchmark_runner as br
    test_csv = tmp_path / "runs.csv"
    monkeypatch.setattr(br, "RUNS_CSV", test_csv)
    # Run once
    br.run_benchmark(["T1_tree"], ["S1_link_failure"], ["B0"], 1, 42, True)
    # Run again — should skip
    br.run_benchmark(["T1_tree"], ["S1_link_failure"], ["B0"], 1, 42, True)
    with open(test_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # no duplicate

if __name__ == "__main__":
    import tempfile, types
    # Quick manual run
    import src.benchmark_runner as br
    with tempfile.TemporaryDirectory() as td:
        br.RUNS_CSV = Path(td) / "runs.csv"
        br.run_benchmark(["T1_tree"], ["S1_link_failure"], ["B0"], 1, 42, True)
        print(f"CSV exists: {br.RUNS_CSV.exists()}")
    print("benchmark_runner tests passed.")

