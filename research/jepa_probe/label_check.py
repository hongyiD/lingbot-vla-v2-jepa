"""Which label should the action-conditioned head learn? Offline check on the Stage 1 features.

A candidate label is one teacher stream ("V2.fut") or a difference of two ("V0.fut-V0.cur"). Every
token is LayerNorm-ed over channels without affine parameters first, as the training head will see
its target, then pooled to 4x4 and PCA-projected to 256 dims on the train split.

  forward   R^2 of the label from the current frame and state, and with the action chunk added.
            dR2 = R2(cur+act) - R2(cur) is the share of the label that the action explains beyond
            the scene: the part the head is meant to learn. Fitted with ridge and with a small MLP
            (the MLP can use action x scene interactions).
  inverse   nMSE of the action chunk from the label plus state (ridge), next to state alone.

The current-frame input is a fixed teacher stream (`--current`, V0 first: the student VLM is
trained to predict it), so every label is judged from the same view of the scene. dR2 of each
label is compared with `--reference` by a paired bootstrap over test episodes.

Decision rule (fixed before looking at results): take the label with the largest dR2 under the
first `--current` input and the MLP fit; keep the reference unless the winner's dR2 minus the
reference's has a 95% interval above 0.

  python -m research.jepa_probe.label_check --features $P/features --index $P/index \
      --out $R/jepa/label_check --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from research.jepa_probe.common import load_index, prediction_metrics, split_ids
from research.jepa_probe.ridge_probe import pca_project, pool_grid, ridge_fit_predict, standardize

DEFAULT_LABELS = (
    "dino_delta=V0.fut-V0.cur",
    "dino_future=V0.fut",
    "v21_delta=V3h.fut-V3h.cur",
    "v2_2f_sep_delta=V1i.fut-V1i.cur",
    "v2_7f_joint_delta=V2.fut-V2.cur",
    "v2_7f_sep_delta=V2.fut-V1i.cur",
    "v2_7f_future=V2.fut",
)
LABEL_DIMS = 256
INPUT_DIMS = 512
SPLITS = ("train", "val", "test")
_TERM_SPLIT = re.compile(r"(?<=\.cur)-|(?<=\.fut)-")  # variant names may contain '-' (B-pixel)


def parse_label(spec: str) -> tuple[str, list[tuple[str, str, float]]]:
    """'name=A.fut-B.cur' -> (name, [(A, 'fut', 1.0), (B, 'cur', -1.0)])."""
    name, _, expr = spec.partition("=")
    parts = _TERM_SPLIT.split(expr) if expr else []
    if not name or not 1 <= len(parts) <= 2:
        raise ValueError(f"bad label {spec!r}; expected name=VARIANT.stream or name=VARIANT.stream-VARIANT.stream")
    terms = []
    for sign, part in zip((1.0, -1.0), parts):
        variant, _, stream = part.rpartition(".")
        if not variant or stream not in ("cur", "fut"):
            raise ValueError(f"bad term {part!r} in {spec!r}; streams are 'cur' and 'fut'")
        terms.append((variant, stream, sign))
    return name, terms


def layer_norm(tokens: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = tokens.mean(axis=-1, keepdims=True)
    return (tokens - mean) / np.sqrt(tokens.var(axis=-1, keepdims=True) + eps)


def load_stream(root: Path, variant: str, stream: str, n_samples: int) -> np.ndarray:
    """[N, 16 * C] fp16: each 8x8 token LayerNorm-ed over channels, then pooled to 4x4."""
    directory = root / variant
    out = None
    filled = 0
    for path in sorted(directory.glob("shard_*.npz")):
        start = int(path.stem.split("_")[1])
        with np.load(path) as data:
            pooled = pool_grid(layer_norm(data[stream].astype(np.float32)))
        if out is None:
            out = np.zeros((n_samples, pooled.shape[1]), dtype=np.float16)
        out[start:start + len(pooled)] = pooled
        filled += len(pooled)
    if filled != n_samples:
        raise ValueError(f"{directory}: shards hold {filled} samples, index has {n_samples}")
    return out


class StreamCache:
    """Loads each (variant, stream) once and drops it after its last planned use."""

    def __init__(self, root: Path, n_samples: int, uses: list[tuple[str, str]]):
        self.root, self.n_samples = root, n_samples
        self.remaining = Counter(uses)
        self.arrays: dict[tuple[str, str], np.ndarray] = {}

    def get(self, key: tuple[str, str]) -> np.ndarray:
        if key not in self.arrays:
            started = time.time()
            self.arrays[key] = load_stream(self.root, *key, self.n_samples)
            print(f"loaded {key[0]}.{key[1]} in {time.time() - started:.0f}s", flush=True)
        return self.arrays[key]

    def release(self, key: tuple[str, str]) -> None:
        self.remaining[key] -= 1
        if self.remaining[key] <= 0:
            self.arrays.pop(key, None)


def by_split(array: np.ndarray, ids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {split: array[ids[split]] for split in SPLITS}


def project(array: np.ndarray, ids: dict[str, np.ndarray], dims: int) -> dict[str, np.ndarray]:
    return dict(zip(SPLITS, pca_project(array[ids["train"]], [array[ids["val"]], array[ids["test"]]], dims)))


def fit_mlp(x: dict, y: dict, *, device: str, seed: int, epochs: int, patience: int,
            width: int = 1024, batch: int = 512, lr: float = 1e-3) -> np.ndarray:
    """2-hidden-layer MLP, early-stopped on val MSE; returns test predictions in y's units."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    scaled = dict(zip(SPLITS, standardize(x["train"], [x["val"], x["test"]])))
    y_mean, y_std = y["train"].mean(axis=0), float(y["train"].std()) + 1e-6
    as_tensor = lambda array: torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)  # noqa: E731
    inputs = {split: as_tensor(scaled[split]) for split in SPLITS}
    targets = {split: as_tensor((y[split] - y_mean) / y_std) for split in ("train", "val")}
    model = nn.Sequential(
        nn.Linear(inputs["train"].shape[1], width), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(width, width), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(width, targets["train"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_train = len(inputs["train"])
    best_val, best_state, idle = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        for start in range(0, n_train, batch):
            rows = order[start:start + batch]
            loss = nn.functional.mse_loss(model(inputs["train"][rows]), targets["train"][rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(inputs["val"]), targets["val"]).item()
        if val < best_val:
            best_val, best_state, idle = val, copy.deepcopy(model.state_dict()), 0
        else:
            idle += 1
            if idle >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(inputs["test"]).cpu().numpy()
    return prediction * y_std + y_mean


def episode_sums(prediction: np.ndarray, target: np.ndarray, codes: np.ndarray, n_episodes: int):
    """Per test episode: summed squared residuals and summed squared deviations from the test mean."""
    residual = np.square(prediction - target).sum(axis=1)
    deviation = np.square(target - target.mean(axis=0)).sum(axis=1)
    return (np.bincount(codes, residual, minlength=n_episodes),
            np.bincount(codes, deviation, minlength=n_episodes))


def r2_from_sums(sums, weights: np.ndarray | None = None):
    residual, deviation = sums
    if weights is None:
        return float(1.0 - residual.sum() / max(deviation.sum(), 1e-12))
    return 1.0 - (weights @ residual) / np.maximum(weights @ deviation, 1e-12)


def forward_fits(current: dict, state: dict, action: dict, label: dict, *, codes, n_episodes,
                 models: tuple[str, ...], device: str, seed: int, epochs: int, patience: int) -> dict:
    """{model: {'cur': sums, 'cur_act': sums}} for the test split."""
    out = {}
    for model in models:
        out[model] = {}
        for name, parts in (("cur", (current, state)), ("cur_act", (current, state, action))):
            x = {split: np.concatenate([part[split] for part in parts], axis=1) for split in SPLITS}
            if model == "ridge":
                prediction = ridge_fit_predict(x, label)[1]["test"]
            else:
                prediction = fit_mlp(x, label, device=device, seed=seed, epochs=epochs, patience=patience)
            out[model][name] = episode_sums(prediction, label["test"], codes, n_episodes)
    return out


def inverse_metrics(inputs: list[dict], action: dict, *, episode_keys, events) -> dict:
    x = {split: np.concatenate([part[split] for part in inputs], axis=1) for split in SPLITS}
    prediction = ridge_fit_predict(x, action)[1]["test"]
    metrics = prediction_metrics(prediction, action["test"], episode_keys=episode_keys, events=events)
    return {key: metrics[key] for key in ("nmse", "nmse_arm", "nmse_gripper", "nmse_event")}


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize(sums: dict, reference: str, n_boot: int, n_episodes: int, seed: int) -> dict:
    """sums[label][current][model] -> point values and paired-bootstrap intervals."""
    weights = np.random.default_rng(seed).multinomial(n_episodes, np.full(n_episodes, 1 / n_episodes),
                                                       size=n_boot).astype(np.float64)
    out: dict = {}
    for label, by_current in sums.items():
        out[label] = {}
        for current, by_model in by_current.items():
            out[label][current] = {}
            for model, fits in by_model.items():
                r2_cur, r2_act = r2_from_sums(fits["cur"]), r2_from_sums(fits["cur_act"])
                boot = r2_from_sums(fits["cur_act"], weights) - r2_from_sums(fits["cur"], weights)
                entry = {"r2_cur": r2_cur, "r2_cur_act": r2_act, "dr2": r2_act - r2_cur,
                         "dr2_ci": interval(boot),
                         "dr2_of_rest": (r2_act - r2_cur) / max(1.0 - r2_cur, 1e-12)}
                ref_fits = sums.get(reference, {}).get(current, {}).get(model)
                if ref_fits is not None and label != reference:
                    ref_boot = r2_from_sums(ref_fits["cur_act"], weights) - r2_from_sums(ref_fits["cur"], weights)
                    ref_dr2 = r2_from_sums(ref_fits["cur_act"]) - r2_from_sums(ref_fits["cur"])
                    entry["dr2_minus_ref"] = entry["dr2"] - ref_dr2
                    entry["dr2_minus_ref_ci"] = interval(boot - ref_boot)
                out[label][current][model] = entry
    return out


def decide(summary: dict, reference: str, current: str, model: str) -> dict:
    scores = {label: entry[current][model]["dr2"] for label, entry in summary.items()
              if model in entry.get(current, {})}
    best = max(scores, key=scores.get)
    if reference not in scores:
        return {"current": current, "model": model, "best": best, "chosen": best,
                "reason": f"reference {reference} was not checked; {best} has the largest dR2"}
    chosen, reason = reference, "reference kept: the best label is the reference"
    if best != reference:
        low = summary[best][current][model]["dr2_minus_ref_ci"][0]
        if low > 0:
            chosen, reason = best, f"{best} beats {reference}: dR2 difference interval starts at {low:+.4f}"
        else:
            reason = f"reference kept: {best} has the largest dR2 but its interval vs {reference} includes 0"
    return {"current": current, "model": model, "best": best, "chosen": chosen, "reason": reason}


def write_markdown(path: Path, result: dict) -> None:
    lines = ["# Label check for the action-conditioned head", "",
             f"Reference: `{result['reference']}`. Test samples: {result['n_test']}, "
             f"episodes: {result['n_test_episodes']}.", ""]
    decision = result["decision"]
    lines += [f"**Decision** ({decision['model']}, current = {decision['current']}): "
              f"`{decision['chosen']}`. {decision['reason']}.", ""]
    for current in result["currents"]:
        for model in result["models"]:
            lines += [f"## Forward, current = {current}, {model}", "",
                      "| label | R² cur | R² cur+act | ΔR² (95% CI) | ΔR² − ref (95% CI) | ΔR² / (1 − R² cur) |",
                      "|---|---|---|---|---|---|"]
            for label, entry in result["forward"].items():
                fit = entry.get(current, {}).get(model)
                if fit is None:
                    continue
                vs_ref = (f"{fit['dr2_minus_ref']:+.4f} [{fit['dr2_minus_ref_ci'][0]:+.4f}, "
                          f"{fit['dr2_minus_ref_ci'][1]:+.4f}]") if "dr2_minus_ref" in fit else "-"
                lines.append(f"| {label} | {fit['r2_cur']:.3f} | {fit['r2_cur_act']:.3f} | "
                             f"{fit['dr2']:+.4f} [{fit['dr2_ci'][0]:+.4f}, {fit['dr2_ci'][1]:+.4f}] | "
                             f"{vs_ref} | {fit['dr2_of_rest']:.3f} |")
            lines.append("")
    lines += ["## Inverse: action chunk from label + state (ridge, test nMSE)", "",
              "| input | nMSE | arm | gripper | gripper-event subset |", "|---|---|---|---|---|"]
    number = lambda value: "-" if value is None else f"{value:.4f}"  # noqa: E731
    for label, metrics in result["inverse"].items():
        lines.append(f"| {label} | {number(metrics['nmse'])} | {number(metrics['nmse_arm'])} | "
                     f"{number(metrics['nmse_gripper'])} | {number(metrics['nmse_event'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="writes <out>.json and <out>.md")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated name=expr")
    parser.add_argument("--current", default="V0,V1i", help="current-frame inputs; the first one decides")
    parser.add_argument("--reference", default="dino_delta")
    parser.add_argument("--device", default=None, help="MLP device; default cuda when available")
    parser.add_argument("--no_mlp", action="store_true")
    parser.add_argument("--mlp_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    models = ("ridge",) if args.no_mlp else ("ridge", "mlp")
    device = args.device
    if device is None and not args.no_mlp:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()

    def finished(variant: str) -> bool:
        if (args.features / variant / "stats.npz").is_file():
            return True
        print(f"skip {variant}: no finished features under {args.features / variant}", flush=True)
        return False

    labels = [parse_label(spec) for spec in args.labels.split(",") if spec]
    labels = [(name, terms) for name, terms in labels if all(finished(v) for v, _, _ in terms)]
    currents = [name for name in args.current.split(",") if name and finished(name)]
    if not labels or not currents:
        raise SystemExit("nothing to check: missing features")

    index = load_index(args.index)
    raw = dict(np.load(args.index / "labels.npz"))
    n_samples = len(index["sample_id"])
    ids = {split: split_ids(index, split) for split in SPLITS}
    state = by_split(raw["state_t"].astype(np.float32), ids)
    action = by_split(raw["action_chunk"].reshape(n_samples, -1).astype(np.float32), ids)
    test_keys = index["episode_key"][ids["test"]]
    events = index["gripper_event"][ids["test"]].astype(bool)
    episodes, codes = np.unique(test_keys, return_inverse=True)

    cache = StreamCache(args.features, n_samples,
                        [(name, "cur") for name in currents] + [(v, s) for _, terms in labels for v, s, _ in terms])
    current_inputs = {}
    for name in currents:
        current_inputs[name] = project(cache.get((name, "cur")).astype(np.float32), ids, INPUT_DIMS)
        cache.release((name, "cur"))

    inverse = {"state only": inverse_metrics([state], action, episode_keys=test_keys, events=events)}
    sums: dict = {}
    for name, terms in labels:
        label_started = time.time()
        values = None
        for variant, stream, sign in terms:
            term = cache.get((variant, stream)).astype(np.float32)
            values = sign * term if values is None else values + sign * term
            del term
            cache.release((variant, stream))
        target = project(values, ids, LABEL_DIMS)
        del values
        inverse[name] = inverse_metrics([target, state], action, episode_keys=test_keys, events=events)
        sums[name] = {current: forward_fits(current_inputs[current], state, action, target, codes=codes,
                                            n_episodes=len(episodes), models=models, device=device,
                                            seed=args.seed, epochs=args.mlp_epochs, patience=args.patience)
                      for current in currents}
        report = ", ".join(f"{current}/{model} dR2 "
                           f"{r2_from_sums(fits['cur_act']) - r2_from_sums(fits['cur']):+.4f}"
                           for current, by_model in sums[name].items() for model, fits in by_model.items())
        print(f"{name}: {report} ({time.time() - label_started:.0f}s)", flush=True)

    forward = summarize(sums, args.reference, args.bootstrap, len(episodes), args.seed)
    decision = decide(forward, args.reference, currents[0], models[-1])
    result = {"labels": {name: " ".join(f"{'+' if sign > 0 else '-'}{v}.{s}" for v, s, sign in terms)
                         for name, terms in labels},
              "reference": args.reference, "currents": currents, "models": list(models),
              "n_test": int(len(test_keys)), "n_test_episodes": int(len(episodes)),
              "forward": forward, "inverse": inverse, "decision": decision,
              "seconds": time.time() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_markdown(args.out.with_suffix(".md"), result)
    print(f"decision: {decision['chosen']} ({decision['reason']})")
    print(f"-> {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
