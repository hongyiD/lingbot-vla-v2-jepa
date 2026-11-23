"""rt_probe.py verdict: the B300 OIDN 2.0.1 failure (undenoised frames) must not pass."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("rt_probe", Path(__file__).resolve().parents[1] / "pod" / "rt_probe.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def frame(sigma, seed=0):
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(60, 200, 320), (240, 1))[..., None].repeat(3, axis=-1)
    return np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)


def result(sigma, **extra):
    return {"blank": False, "noise": [rp.noise(np, frame(sigma, s)) for s in range(3)], **extra}


def test_noise_ranks_path_tracing_noise():
    assert rp.noise(np, frame(0)) < 1 < rp.noise(np, frame(5)) < rp.noise(np, frame(20))


def test_denoised_frames_pass():
    ok, reasons = rp.verdict({"rt+oidn": result(1, oidn_errors=0), "rt+none": result(15, oidn_errors=0)})
    assert ok and not reasons


def test_b300_oidn_201_failure_is_caught():
    same = result(15)
    ok, reasons = rp.verdict({"rt+oidn": {**same, "oidn_errors": 40}, "rt+none": {**same, "oidn_errors": 0}})
    assert not ok
    assert any("OIDN Error" in r for r in reasons) and any("noise" in r for r in reasons)


def test_silent_undenoised_frames_are_caught():
    ok, reasons = rp.verdict({"rt+oidn": result(15, oidn_errors=0), "rt+none": result(14, oidn_errors=0)})
    assert not ok and reasons == [r for r in reasons if "noise" in r]


def test_render_failure_is_caught():
    ok, reasons = rp.verdict({"rt+oidn": {"error": "exit 1", "oidn_errors": 0}, "rt+none": result(15)})
    assert not ok and "non-blank" in reasons[0]
