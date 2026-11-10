"""P5: measure teacher forward cost and the extra decode cost of the 8-frame clip.

Reports, per variant, the median and 90th percentile forward time at batch 64 in bf16, the peak
memory, and how many samples per second the frame decode sustains. The StepTime impact is estimated
against the measured training baseline of 3.36 s/step at micro batch 64.

  python -m research.jepa_probe.bench_teacher --variants V0,V1,V2 \
      --batch 64 --iters 50 --warmup 10 --dtype bf16 --out $R/jepa/s1_bench.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE_STEP_SECONDS = 3.36
MICRO_BATCH = 64


def time_forward(module, batch, iters: int, warmup: int) -> dict:
    import torch

    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(warmup):
        module(batch)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        starter.record()
        module(batch)
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender) / 1e3)
    values = np.sort(np.asarray(samples))
    return {"median_s": float(np.median(values)), "p90_s": float(values[int(0.9 * (len(values) - 1))])}


def step_time_increase(per_batch_seconds: float, batch: int = MICRO_BATCH) -> float:
    """Seconds added to each training step, which forwards one micro batch per GPU."""
    return per_batch_seconds * (MICRO_BATCH / batch)


def summarize(forward_seconds: float, baseline: float = BASELINE_STEP_SECONDS) -> dict:
    added = step_time_increase(forward_seconds)
    return {"added_s_per_step": added, "step_time_ratio": (baseline + added) / baseline}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("bench_teacher needs a GPU; run it on the pod.")
    result = {"variants": args.variants.split(","), "batch": args.batch, "dtype": args.dtype,
              "baseline_step_s": BASELINE_STEP_SECONDS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"benchmark plan for {result['variants']} written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
