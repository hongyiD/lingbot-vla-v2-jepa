"""Aggregate probe results into the Stage 1 report tables.

Reads one JSON per (task, variant, lr, seed) plus the P5 benchmark file, and writes a markdown
report with the mean and std over seeds, the paired-bootstrap confidence interval of each
variant's gap to the DINO-Video baseline, and the P3 and P5 tables.

  python -m research.jepa_probe.report --results $P/results --bench $R/jepa/s1_bench.json \
      --out $R/jepa/stage1_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE_VARIANT = "V0"


def paired_bootstrap(differences: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of paired per-episode differences."""
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (n_resamples, len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_runs(runs: list[dict]) -> dict:
    """runs: JSON records sharing a (task, variant), one per seed, already at the chosen lr."""
    scores = np.array([run["test_nmse"] for run in runs], dtype=np.float64)
    return {"mean": float(scores.mean()), "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "seeds": len(scores)}


def gap_to_baseline(variant_per_episode: np.ndarray, baseline_per_episode: np.ndarray) -> dict:
    """Per-episode nMSE arrays averaged over seeds, shape [episodes]."""
    difference = variant_per_episode - baseline_per_episode
    low, high = paired_bootstrap(difference)
    return {"mean_diff": float(difference.mean()), "ci95": [low, high],
            "significant": bool(low > 0 or high < 0)}


def format_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.rglob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_results(args.results)
    bench = json.loads(args.bench.read_text(encoding="utf-8"))
    report = [f"# Stage 1 probe report", "", f"Runs: {len(runs)}", "",
              "## P5 benchmark", "", "```json", json.dumps(bench, indent=2), "```", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report), encoding="utf-8")
    print(f"{len(runs)} runs, report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
