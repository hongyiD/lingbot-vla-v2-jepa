"""Aggregate probe results into the Stage 1 report (plan step 6).

For every (task, input) the learning rate with the lowest mean val nMSE is chosen, then the test
nMSE is reported as mean and std over seeds. The gap to the DINO-Video baseline (V0) is a paired
bootstrap over test episodes of the per-episode nMSE averaged over seeds; the relative gap divides
by V0's mean. Also written: the gripper-event table, P3, P4, P5, the Spearman correlation of the
linear and attention probe rankings, and PCA images of the visualization samples.

  python -m research.jepa_probe.report --results $P/results --bench $R/jepa/s1_bench.json \
      --features $P/features --index $P/index --out $R/jepa/stage1_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE_VARIANT = "V0"
TASK_ORDER = ("P1s", "P1", "P1d", "P2")
VIZ_VARIANTS = ("V0", "V1", "V3")


def paired_bootstrap(differences: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of paired per-episode differences."""
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (n_resamples, len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_runs(runs: list[dict], metric: str = "nmse") -> dict:
    """runs: JSON records sharing a (task, variant), one per seed, already at the chosen lr."""
    scores = np.array([run["test"][metric] for run in runs], dtype=np.float64)
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
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.rglob("*.json"))
            if "_jobs" not in path.parts and not path.name.startswith(".")]


# ---------------------------------------------------------------- selection


def choose_runs(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """(task, input) -> the attention-probe runs at the lr with the best mean val nMSE."""
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for record in records:
        if record.get("probe") != "attention":
            continue
        grouped.setdefault((record["task"], record["input"]), {}).setdefault(str(record["lr"]), []).append(record)
    chosen = {}
    for key, by_lr in grouped.items():
        best = min(by_lr.values(), key=lambda runs: np.mean([run["val"]["nmse"] for run in runs]))
        chosen[key] = sorted(best, key=lambda run: run["seed"])
    return chosen


def per_episode(runs: list[dict], keys: list[str]) -> np.ndarray:
    return np.mean([[run["test"]["per_episode"][key] for key in keys] for run in runs], axis=0)


def _mean_std(runs: list[dict], metric: str) -> str:
    runs = [run for run in runs if run["test"].get(metric) is not None]
    if not runs:
        return "-"
    summary = summarize_runs(runs, metric)
    if summary["seeds"] == 1:
        return f"{summary['mean']:.4f}"
    return f"{summary['mean']:.4f} ± {summary['std']:.4f}"


def task_rows(task: str, chosen: dict, metric: str = "nmse", with_gap: bool = True) -> list[dict]:
    names = sorted({name for (t, name) in chosen if t == task}, key=_input_order)
    baseline = chosen.get((task, BASELINE_VARIANT))
    keys = sorted(baseline[0]["test"]["per_episode"]) if baseline else []
    base_mean = float(np.mean([run["test"][metric] for run in baseline])) if baseline else None
    rows = []
    for name in names:
        runs = chosen[(task, name)]
        row = {"input": name, "lr": runs[0]["lr"] if runs[0]["lr"] is not None else "-", "seeds": len(runs),
               "test nMSE": _mean_std(runs, metric)}
        if metric == "nmse":
            row["arm"] = _mean_std(runs, "nmse_arm")
            row["gripper"] = _mean_std(runs, "nmse_gripper")
        if with_gap and baseline and name != BASELINE_VARIANT and keys:
            gap = gap_to_baseline(per_episode(runs, keys), per_episode(baseline, keys))
            low, high = gap["ci95"]
            row["Δ vs V0"] = f"{gap['mean_diff']:+.4f} ({gap['mean_diff'] / base_mean:+.1%})"
            row["95% CI"] = f"[{low:+.4f}, {high:+.4f}] ([{low / base_mean:+.1%}, {high / base_mean:+.1%}])"
            row["significant"] = "yes" if gap["significant"] else "no"
        rows.append(row)
    return rows


def _input_order(name: str) -> tuple:
    if name.startswith("B-"):
        return (0, name)
    return (1 if "+" not in name else 2, name)


def spearman(a: list[float], b: list[float]) -> float:
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def rank_agreement(chosen: dict, ridge: list[dict]) -> list[dict]:
    rows = []
    for task in TASK_ORDER:
        linear = {record["input"]: record["test"]["nmse"] for record in ridge if record["task"] == task}
        attention = {name: np.mean([run["test"]["nmse"] for run in runs])
                     for (t, name), runs in chosen.items() if t == task and name in linear}
        common = sorted(attention)
        if len(common) >= 3:
            rho = spearman([linear[name] for name in common], [attention[name] for name in common])
            rows.append({"task": task, "inputs": len(common), "Spearman ρ": f"{rho:.2f}"})
    return rows


# ---------------------------------------------------------------- visualization


def viz_tokens(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    ids, tokens = [], []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as data:
            if "viz_ids" in data:
                ids.append(data["viz_ids"])
                tokens.append(data["viz_cur"])
    if not ids:
        return None
    return np.concatenate(ids), np.concatenate(tokens).astype(np.float32)


def pca_rgb(tokens: np.ndarray) -> np.ndarray:
    """[N, 256, C] -> [N, 16, 16, 3] uint8: the first three principal components as RGB."""
    flat = tokens.reshape(-1, tokens.shape[-1])
    flat = flat - flat.mean(axis=0)
    sample = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 20000), replace=False)]
    _, _, components = np.linalg.svd(sample, full_matrices=False)
    projected = flat @ components[:3].T
    low, high = np.percentile(projected, [1, 99], axis=0)
    scaled = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    side = int(round(tokens.shape[1] ** 0.5))
    return (scaled.reshape(len(tokens), side, side, 3) * 255).astype(np.uint8)


def write_viz(features: Path, index: dict, out_dir: Path, episodes: int = 3, steps: int = 8) -> list[Path]:
    """One image per episode: a row per variant, a column per time step."""
    from PIL import Image, ImageDraw

    images = {}
    for name in VIZ_VARIANTS:
        loaded = viz_tokens(features / name)
        if loaded is not None:
            ids, tokens = loaded
            images[name] = dict(zip(ids.tolist(), pca_rgb(tokens)))
    if not images:
        return []
    shared = sorted(set.intersection(*(set(by_id) for by_id in images.values())))
    keys = index["episode_key"][shared]
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    cell, label = 96, 40
    for key in list(dict.fromkeys(keys.tolist()))[:episodes]:
        ids = [sample for sample, sample_key in zip(shared, keys) if sample_key == key]
        ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(steps, len(ids))).astype(int)]
        canvas = Image.new("RGB", (label + cell * len(ids), cell * len(images)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, (name, by_id) in enumerate(images.items()):
            draw.text((4, row * cell + cell // 2), name, fill="black")
            for column, sample in enumerate(ids):
                tile = Image.fromarray(by_id[sample]).resize((cell, cell), Image.NEAREST)
                canvas.paste(tile, (label + column * cell, row * cell))
        path = out_dir / f"{key.replace('/', '_ep')}.png"
        canvas.save(path)
        written.append(path)
    return written


# ---------------------------------------------------------------- report


def build_report(records: list[dict], bench: dict | None, viz_paths: list[Path], out: Path) -> str:
    chosen = choose_runs(records)
    ridge = [record for record in records if record.get("probe") == "ridge"]
    p3 = [record for record in records if record.get("task") == "P3"]
    lines = ["# Stage 1 probe report", "",
             f"Attention runs: {sum(len(runs) for runs in chosen.values())} at the chosen lr "
             f"(of {sum(1 for r in records if r.get('probe') == 'attention')}); ridge runs: {len(ridge)}.", ""]
    columns = ["input", "lr", "seeds", "test nMSE", "arm", "gripper", "Δ vs V0", "95% CI", "significant"]
    lines += ["## 1. Main table (test nMSE, attention probe)", ""]
    for task in TASK_ORDER:
        rows = task_rows(task, chosen)
        if rows:
            lines += [f"### {task}", "", format_table([r for r in rows if "+" not in r["input"]], columns), ""]
    lines += ["## 2. Gripper-event subset (test nMSE on samples whose chunk opens or closes a gripper)", ""]
    for task in ("P1s", "P1"):
        rows = task_rows(task, chosen, metric="nmse_event", with_gap=False)
        if rows:
            lines += [f"### {task}", "", format_table([r for r in rows if "+" not in r["input"]],
                                                       ["input", "lr", "seeds", "test nMSE"]), ""]
    if p3:
        lines += ["## 3. P3: predictability of the future target", "",
                  format_table([{"input": r["input"], "R² cur": f"{r['cur']:.3f}", "R² cur+act": f"{r['cur_act']:.3f}",
                                 "ΔR²": f"{r['delta']:+.3f}"} for r in sorted(p3, key=lambda r: _input_order(r["input"]))],
                               ["input", "R² cur", "R² cur+act", "ΔR²"]), ""]
    concat = [row for row in task_rows("P1s", chosen) if "+" in row["input"]]
    if concat:
        alone = {row["input"]: row["test nMSE"] for row in task_rows("P1s", chosen, with_gap=False)}
        for row in concat:
            row["alone"] = alone.get(row["input"].split("+", 1)[1], "-")
        lines += ["## 4. P4: V0 concatenated with JEPA (P1s)", "",
                  format_table(concat, ["input", "test nMSE", "alone", "Δ vs V0", "95% CI", "significant"]), ""]
    if bench:
        rows = []
        for name, entry in bench.get("variants", {}).items():
            compiled = entry.get("compiled", {})
            rows.append({
                "variant": name, "ms/batch": f"{entry['eager']['median_s'] * 1e3:.1f}",
                "p90 ms": f"{entry['eager']['p90_s'] * 1e3:.1f}",
                "compiled ms": f"{compiled['median_s'] * 1e3:.1f}" if "median_s" in compiled else compiled.get("error", "-"),
                "peak GB": f"{entry['peak_mem_gb']:.1f}",
                "StepTime ×, both": f"{entry['both']['step_time_ratio']:.3f}",
                "StepTime ×, replace": f"{entry['replace']['step_time_ratio']:.3f}" if "replace" in entry else "-",
                "extra frames": entry.get("decode", {}).get("extra_frames", "-"),
                "decode worker share": f"{entry['decode']['worker_share']:.0%}" if "decode" in entry else "-",
            })
        lines += [f"## 5. P5: teacher cost (batch {bench.get('batch')}, {bench.get('dtype')}, {bench.get('gpu', '')})", "",
                  format_table(rows, list(rows[0]) if rows else ["variant"]), ""]
    agreement = rank_agreement(chosen, ridge)
    if agreement:
        lines += ["## 6. Linear vs attention probe ranking", "", format_table(agreement, ["task", "inputs", "Spearman ρ"]), ""]
    if viz_paths:
        lines += ["## 7. PCA of current-frame tokens", ""]
        lines += [f"![{path.stem}]({path.relative_to(out.parent)})" for path in viz_paths]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--features", type=Path, help="feature root; enables the PCA images")
    parser.add_argument("--index", type=Path, help="index dir; needed with --features")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    records = load_results(args.results)
    bench = json.loads(args.bench.read_text(encoding="utf-8")) if args.bench and args.bench.is_file() else None
    viz_paths = []
    if args.features and args.index:
        from research.jepa_probe.common import load_index
        viz_paths = write_viz(args.features, load_index(args.index), args.out.parent / f"{args.out.stem}_viz")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(records, bench, viz_paths, args.out), encoding="utf-8")
    print(f"{len(records)} result files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
