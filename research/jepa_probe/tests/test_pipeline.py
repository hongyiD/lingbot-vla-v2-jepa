"""End to end on a fake dataset: index -> features (toy teachers) -> probes -> report, on CPU."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("yaml")
try:
    import lingbotvla.models.vla.vision_models.jepa.teacher  # noqa: F401
except Exception as exc:
    pytest.skip(f"toy teacher needs the lingbotvla package: {exc}", allow_module_level=True)

from research.jepa_probe import extract_features, report, ridge_probe, train_probe  # noqa: E402
from research.jepa_probe.build_index import build  # noqa: E402
from research.jepa_probe.tests.helpers import write_dataset, write_norm_stats, write_robot_config  # noqa: E402

OFFSET = 4
VARIANTS = """
weights_dir: /unused
variants:
  V0:
    teacher: toy
    embed_dim: 16
    input_size: 256
    clip_frames: [0, 0, 4, 4]
  V1:
    teacher: toy
    embed_dim: 16
    input_size: 256
    clip_frames: [0, 0]
    future_clip_frames: [4, 4]
  B-pixel:
    teacher: pixel
    clip_frames: [0, 4]
"""
SWEEP = """
tasks: [P1s, P2]
inputs: [B-mean, B-state, B-pixel, V0, V1]
lrs: [1.0e-3]
seeds: [0, 1]
train: {epochs: 2, batch_size: 16, patience: 2}
p4: {task: P1s, base: V0, top: 1, candidates: [V1]}
"""


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    dirs = [write_dataset(root / "data", f"{task}-aloha-agilex_clean_50-50", episodes=[12] * 6)
            for task in ("click_bell", "turn_switch")]
    train_list = root / "train.txt"
    train_list.write_text("".join(f"robotwin_c50 {d}\n" for d in dirs))
    norm, robot = root / "norm.json", root / "robot.yaml"
    write_norm_stats(norm)
    write_robot_config(robot)
    build(train_list, robot, norm, root / "index", stride=2, future_offset=OFFSET, chunk=OFFSET + 1,
          split_spec="train=0-3,val=4-4,test=5-5")
    (root / "variants.yaml").write_text(VARIANTS)
    (root / "sweep.yaml").write_text(SWEEP)
    for name in ("V0", "V1", "B-pixel"):
        variant = extract_features.load_variant(root / "variants.yaml", name)
        variant["dtype"] = "fp32"
        result = extract_features.extract(variant, root / "index", root / "features" / name, batch=8,
                                          num_workers=0, device="cpu", viz=10)
        assert result["remaining"] == 0
    return root


def test_features_are_complete_and_resumable(workspace):
    features = workspace / "features" / "V0"
    with np.load(features / "shard_0000000.npz") as data:
        assert data["cur"].shape == (48, 64, 16) and data["cur"].dtype == np.float16
        assert len(data["viz_ids"]) == 8 and data["viz_cur"].shape == (8, 256, 16)  # 2 tasks x 4 samples
    assert np.load(features / "stats.npz")["mean"].shape == (2, 16)
    meta = json.loads((features / "meta.json").read_text())
    assert meta["channels"] == 16 and meta["samples"] == 48
    with np.load(workspace / "features" / "B-pixel" / "shard_0000000.npz") as data:
        assert data["cur"].shape == (48, 64, 12)
    variant = extract_features.load_variant(workspace / "variants.yaml", "V0")
    variant["dtype"] = "fp32"
    again = extract_features.extract(variant, workspace / "index", features, batch=8, num_workers=0, device="cpu")
    assert again["remaining"] == 0 and again["seconds"] < 30


def test_multi_variant_parts_match_single_runs(workspace, monkeypatch, tmp_path):
    """Several variants in one pass, shards split over two parts, with one shard already done."""
    monkeypatch.setattr(extract_features, "SHARD_SIZE", 16)
    names = ["V0", "V1", "B-pixel"]
    variants = []
    for name in names:
        variant = extract_features.load_variant(workspace / "variants.yaml", name)
        variant["dtype"] = "fp32"
        variants.append(variant)
    single, multi = tmp_path / "single", tmp_path / "multi"
    for variant in variants:
        assert extract_features.extract(variant, workspace / "index", single / variant["name"], batch=8,
                                        num_workers=0, device="cpu", viz=10)["remaining"] == 0
    (multi / "V1").mkdir(parents=True)
    (multi / "V1" / "shard_0000032.npz").write_bytes((single / "V1" / "shard_0000032.npz").read_bytes())
    common = dict(batch=8, num_workers=0, device="cpu", viz=10)
    dirs = [multi / name for name in names]
    first = extract_features.extract_many(variants, dirs, workspace / "index", part=(0, 2), **common)
    assert all(result["remaining"] == 16 for result in first.values())
    assert not (multi / "V0" / "stats.npz").exists()
    second = extract_features.extract_many(variants, dirs, workspace / "index", part=(1, 2), **common)
    assert all(result["remaining"] == 0 for result in second.values())
    for name in names:
        shards = sorted(path.name for path in (single / name).glob("shard_*.npz"))
        assert len(shards) == 3 and shards == sorted(path.name for path in (multi / name).glob("shard_*.npz"))
        for shard in shards:
            with np.load(single / name / shard) as a, np.load(multi / name / shard) as b:
                assert set(a.files) == set(b.files)
                assert all(np.array_equal(a[key], b[key]) for key in a.files)
        with np.load(single / name / "stats.npz") as a, np.load(multi / name / "stats.npz") as b:
            assert np.allclose(a["mean"], b["mean"]) and np.allclose(a["std"], b["std"])

    assert extract_features.main(["--variant", "V0,V1", "--variants_cfg", str(workspace / "variants.yaml"),
                                  "--index", str(workspace / "index"), "--features", str(multi),
                                  "--device", "cpu", "--num_workers", "0", "--dtype", "fp32"]) == 0


def test_v1i_current_comes_from_its_own_clip(workspace):
    """V1 encodes the current frame alone; V0's toy encoder is per-tubelet, so both must agree."""
    with np.load(workspace / "features" / "V0" / "shard_0000000.npz") as v0, \
            np.load(workspace / "features" / "V1" / "shard_0000000.npz") as v1:
        assert np.allclose(v0["cur"], v1["cur"], atol=1e-3)
        assert np.allclose(v0["fut"], v1["fut"], atol=1e-3)


def test_probes_and_report(workspace):
    results = workspace / "results"
    common = ["--features", str(workspace / "features"), "--index", str(workspace / "index")]
    assert train_probe.main(["--sweep", str(workspace / "sweep.yaml"), "--out", str(results),
                             "--device", "cpu", *common]) == 0
    record = json.loads((results / "P1s" / "V0" / "lr0.001_s0.json").read_text())
    assert set(record["test"]["per_episode"]) == {"click_bell/5", "turn_switch/5"}
    assert record["test"]["nmse_gripper"] is not None and record["best_epoch"] >= 0
    assert (results / "P1s" / "B-mean" / "mean.json").is_file()
    assert (results / "P1s" / "V0+V1" / "lr0.001_s1.json").is_file()  # P4
    assert not (results / "P2" / "B-state").exists()

    assert ridge_probe.main(["--variants", "V0,V1,B-pixel,V9", "--out", str(results / "ridge"), *common]) == 0
    p3 = json.loads((results / "ridge" / "P3" / "V0.json").read_text())
    assert {"cur", "cur_act", "delta"} <= set(p3)
    assert (results / "ridge" / "P1s" / "B-pixel.json").is_file()

    bench = {"batch": 64, "dtype": "bf16", "variants": {
        "V0": {"eager": {"median_s": 0.05, "p90_s": 0.06}, "peak_mem_gb": 3.0, "both": {"step_time_ratio": 1.01}},
        "V1": {"eager": {"median_s": 0.08, "p90_s": 0.09}, "peak_mem_gb": 4.0, "both": {"step_time_ratio": 1.02},
               "replace": {"step_time_ratio": 1.01}}}}
    (workspace / "bench.json").write_text(json.dumps(bench))
    out = workspace / "report" / "stage1.md"
    assert report.main(["--results", str(results), "--bench", str(workspace / "bench.json"),
                        "--out", str(out), *common]) == 0
    text = out.read_text()
    for heading in ("### P1s", "### P2", "Δ vs V0", "P3", "P4", "P5", "Spearman", "PCA"):
        assert heading in text, heading
    assert list((workspace / "report" / "stage1_viz").glob("*.png"))


def test_decode_measurement(workspace):
    from research.jepa_probe.bench_teacher import decode_cost, decode_load

    seconds = decode_cost(workspace / "index", frames=5)
    assert 0 < seconds < 1
    load = decode_load(5, seconds, workers=8)
    assert load["extra_frames"] == 5 and load["worker_share"] > 0
