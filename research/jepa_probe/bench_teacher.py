"""P5: measure teacher forward cost and the extra decode cost of multi-frame clips.

Teacher side, per variant: median and 90th percentile time of one runner call at batch 64 (resize,
normalization and the teacher forward, as in training), the peak memory, and with --compile the
same after torch.compile. The StepTime impact uses the measured training baseline of 3.36 s/step
at micro batch 64, which already includes the DINO-Video teacher: replacing it (A2) adds the
difference to V0, keeping both (A3) adds the full forward.

Data side, with --index: seconds to decode and resize one top-camera PNG in one process, and from
that the share of `--decode_workers` workers per GPU that the frames beyond the two training frames
would take at the training rate.

  python -m research.jepa_probe.bench_teacher --variants V0,V1,V1i,V2,V3,V3h \
      --variants_cfg research/jepa_probe/configs/variants.yaml --index $P/index \
      --batch 64 --iters 50 --warmup 10 --dtype bf16 --compile --out $R/jepa/s1_bench.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BASELINE_STEP_SECONDS = 3.36
MICRO_BATCH = 64
TRAIN_SAMPLES_PER_S_PER_GPU = 76 / 4
TRAINING_FRAMES = 2  # current and future top-camera frames are decoded already


def time_forward(fn, iters: int, warmup: int) -> dict:
    import torch

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        starter.record()
        fn()
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


def decode_cost(index_dir: Path, frames: int = 200) -> float:
    """Median seconds to decode one PNG and resize it to 224 in a single process."""
    from research.jepa_probe.common import load_index, split_ids
    from research.jepa_probe.frames import FrameReader, decode_png, resize_224

    index = load_index(index_dir)
    train = split_ids(index, "train")
    first = train[0]
    reader = FrameReader(Path(str(index["dataset_dir"][first])))
    rows = [int(index["row_t"][first]) + offset for offset in range(min(frames, 50))]
    payloads = reader.raw(rows)
    timings = []
    for payload in payloads:
        started = time.perf_counter()
        resize_224(decode_png(payload))
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def decode_load(extra_frames: int, seconds_per_frame: float, workers: int) -> dict:
    extra = extra_frames * seconds_per_frame
    return {"extra_frames": extra_frames, "extra_decode_s_per_sample": extra,
            "worker_share": extra * TRAIN_SAMPLES_PER_S_PER_GPU / workers}


def bench_variant(variant: dict, *, batch: int, iters: int, warmup: int, compile_teacher: bool) -> dict:
    import torch

    from research.jepa_probe.teachers import build_runner

    device = torch.device("cuda")
    runner = build_runner(variant, device)
    frames = torch.randint(0, 256, (batch, len(runner.positions), 3, 224, 224), dtype=torch.uint8, device=device)
    fps = torch.full((batch,), 50.0 / 49.0, device=device)
    torch.cuda.reset_peak_memory_stats()
    result = {"positions": list(runner.positions), "eager": time_forward(lambda: runner(frames, fps), iters, warmup),
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    if compile_teacher:
        try:
            runner.compile()
            result["compiled"] = time_forward(lambda: runner(frames, fps), iters, warmup)
        except Exception as exc:  # report and keep the eager numbers
            result["compiled"] = {"error": f"{type(exc).__name__}: {exc}"}
    del runner, frames
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--variants_cfg", type=Path, default=Path("research/jepa_probe/configs/variants.yaml"))
    parser.add_argument("--index", type=Path, help="index dir; enables the decode measurement")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--decode_workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    from research.jepa_probe.extract_features import load_variant

    if not torch.cuda.is_available():
        raise SystemExit("bench_teacher needs a GPU; run it on the pod.")
    result = {"batch": args.batch, "dtype": args.dtype, "baseline_step_s": BASELINE_STEP_SECONDS,
              "gpu": torch.cuda.get_device_name(), "variants": {}}
    seconds_per_frame = decode_cost(args.index) if args.index else None
    if seconds_per_frame is not None:
        result["decode"] = {"seconds_per_frame": seconds_per_frame, "workers": args.decode_workers,
                            "train_samples_per_s_per_gpu": TRAIN_SAMPLES_PER_S_PER_GPU}
    for name in args.variants.split(","):
        variant = dict(load_variant(args.variants_cfg, name), dtype=args.dtype)
        entry = bench_variant(variant, batch=args.batch, iters=args.iters, warmup=args.warmup,
                              compile_teacher=args.compile)
        if seconds_per_frame is not None:
            entry["decode"] = decode_load(max(len(entry["positions"]) - TRAINING_FRAMES, 0), seconds_per_frame,
                                          args.decode_workers)
        result["variants"][name] = entry
        print(f"{name}: {entry['eager']['median_s'] * 1e3:.1f} ms/batch, peak {entry['peak_mem_gb']:.1f} GB", flush=True)

    v0 = result["variants"].get("V0", {}).get("eager", {}).get("median_s")
    for name, entry in result["variants"].items():
        seconds = entry["eager"]["median_s"]
        entry["both"] = summarize(seconds)  # A3: the JEPA teacher next to DINO-Video
        if v0 is not None and name != "V0":
            entry["replace"] = summarize(seconds - v0)  # A2: instead of DINO-Video
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
