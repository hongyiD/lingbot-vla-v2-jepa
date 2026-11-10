"""Probe pieces that have to be right before any feature exists: pooling, nMSE, ridge, bootstrap."""
import numpy as np
import pytest

from research.jepa_probe.extract_features import compute_stats, save_shard
from research.jepa_probe.report import gap_to_baseline, paired_bootstrap, summarize_runs
from research.jepa_probe.ridge_probe import fit_ridge, p3_scores, predict, r2
from research.jepa_probe.train_probe import nmse, normalize_tokens


def test_pool_tokens_averages_blocks():
    torch = pytest.importorskip("torch")
    from research.jepa_probe.extract_features import pool_tokens

    tokens = torch.arange(2 * 4 * 4 * 1, dtype=torch.float32).reshape(2, 16, 1)
    pooled = pool_tokens(tokens, grid=4, pool=2)
    assert pooled.shape == (2, 4, 1)
    # First 2x2 block of batch 0 is tokens 0, 1, 4, 5.
    assert torch.allclose(pooled[0, 0], torch.tensor([2.5]))


def test_nmse_baseline_and_perfect():
    target = np.array([[0.0, 2.0], [2.0, 0.0]])
    mean = np.repeat(target.mean(axis=0, keepdims=True), len(target), axis=0)
    assert np.isclose(nmse(mean, target), 1.0)  # predicting the mean
    assert np.isclose(nmse(target, target), 0.0)


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    truth = np.array([1.5, -2.0, 0.5])
    target = features @ truth + 0.25
    weights = fit_ridge(features[:150], target[:150], penalty=1e-6)
    assert r2(predict(features[150:], weights), target[150:]) > 0.999


def test_p3_delta_is_zero_without_signal():
    rng = np.random.default_rng(1)
    current = rng.normal(size=(90, 64, 4)).astype(np.float32)
    future = rng.normal(size=(90, 64, 4)).astype(np.float32)
    action = rng.normal(size=(90, 5, 2)).astype(np.float32)
    ids = np.arange(60), np.arange(60, 75), np.arange(75, 90)
    scores = p3_scores(current, future, action, train_ids=ids[0], val_ids=ids[1], test_ids=ids[2], dims=8)
    assert abs(scores["delta"]) < 0.5  # no relationship, so the action adds nothing stable


def test_p3_delta_detects_action_signal():
    rng = np.random.default_rng(2)
    n = 300
    current = rng.normal(size=(n, 64, 4)).astype(np.float32)
    action = rng.normal(size=(n, 5, 2)).astype(np.float32)
    # Every future token is the matching current token plus a shift driven by the action.
    shift = action.reshape(n, -1) @ rng.normal(size=(10, 4)).astype(np.float32)
    future = current + shift[:, None, :]
    ids = np.arange(200), np.arange(200, 250), np.arange(250, n)
    scores = p3_scores(current, future, action, train_ids=ids[0], val_ids=ids[1], test_ids=ids[2],
                       dims=8, input_dims=64)
    assert scores["cur_act"] > 0.95 and scores["delta"] > 0.3


def test_randomized_pca_matches_full_svd():
    from research.jepa_probe.ridge_probe import pca_project

    rng = np.random.default_rng(3)
    data = rng.normal(size=(120, 40)) * np.linspace(5.0, 0.1, 40)
    (fast,) = pca_project(data, [], dims=6)
    centered = data - data.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    exact = centered @ vt[:6].T
    assert np.allclose(np.abs(fast), np.abs(exact), atol=1e-3)  # components agree up to sign


def test_load_variant_expands_weights_dir(tmp_path, monkeypatch):
    from research.jepa_probe.extract_features import load_variant

    config = tmp_path / "variants.yaml"
    config.write_text("weights_dir: ${D}/weights/vjepa\nvariants:\n  V1:\n    weights: vitl\n", encoding="utf-8")
    monkeypatch.setenv("D", "/mnt/work")
    assert load_variant(config, "V1")["weights"] == "/mnt/work/weights/vjepa/vitl"
    monkeypatch.delenv("D")
    with pytest.raises(ValueError):
        load_variant(config, "V1")


def test_bootstrap_interval_contains_mean():
    differences = np.array([0.1, -0.2, 0.3, 0.05])
    low, high = paired_bootstrap(differences, n_resamples=2000, seed=0)
    assert low < differences.mean() < high


def test_gap_flags_significance():
    baseline = np.zeros(50)
    better = np.full(50, -0.5)
    gap = gap_to_baseline(better, baseline)
    assert gap["significant"] and gap["ci95"][1] < 0
    noise = np.linspace(-0.2, 0.2, 50)
    assert not gap_to_baseline(noise, baseline)["significant"]


def test_summarize_runs():
    runs = [{"test": {"nmse": 0.4}}, {"test": {"nmse": 0.6}}]
    summary = summarize_runs(runs)
    assert summary == {"mean": 0.5, "std": np.std([0.4, 0.6], ddof=1), "seeds": 2}


def test_attention_probe_output_shape():
    torch = pytest.importorskip("torch")
    from research.jepa_probe.train_probe import build_attention_probe

    probe = build_attention_probe(in_dim=8, out_dim=14, d_model=32, n_heads=4, n_layers=2)
    tokens = torch.randn(5, 2, 64, 8)
    out = probe(tokens, state=torch.randn(5, 14))
    assert out.shape == (5, 14)
    single = probe(tokens[:, :1], state=None)
    assert single.shape == (5, 14)


def test_feature_stats(tmp_path):
    rng = np.random.default_rng(0)
    cur = rng.normal(loc=1.0, size=(6, 64, 4)).astype(np.float16)
    fut = rng.normal(loc=-1.0, size=(6, 64, 4)).astype(np.float16)
    save_shard(tmp_path / "shard_0000000.npz", cur, fut)
    stats = compute_stats(tmp_path, train_ids=np.array([0, 1, 2]))
    assert stats["mean"].shape == (2, 4)
    assert stats["mean"][0].mean() > 0 > stats["mean"][1].mean()
    assert np.all(stats["std"] > 0)


def test_token_normalization_centers_data():
    tokens = np.ones((4, 64, 2), dtype=np.float32) * np.array([2.0, 4.0])
    normed = normalize_tokens(tokens, mean=np.array([2.0, 4.0]), std=np.array([1.0, 2.0]))
    assert np.allclose(normed, 0.0)
