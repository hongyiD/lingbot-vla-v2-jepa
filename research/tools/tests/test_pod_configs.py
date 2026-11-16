"""make_train_configs.py arms and train_metrics.py action-conditioned columns."""
import importlib.util
from pathlib import Path

import pytest
import yaml

POD = Path(__file__).resolve().parents[1] / "pod"
LB = POD.parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, POD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mtc = load("make_train_configs")
tm = load("train_metrics")


def run(tmp_path, *args):
    out = tmp_path / "configs"
    argv = [*args, "--lingbot_root", str(LB), "--weights", str(tmp_path / "w"), "--cfg", str(tmp_path / "cfg"),
            "--work", str(tmp_path), "--out_dir", str(out), "--no_check"]
    assert mtc.main(argv) == 0
    return {p.stem: yaml.safe_load(p.read_text()) for p in out.glob("*.yaml")}


FORMAL = ("--micro_batch", "64", "--accum", "1", "--max_steps", "10735", "--save_steps", "2147")


def test_formal_arms(tmp_path):
    cfgs = run(tmp_path, "formal", "--arm", "B1", "--seed", "43", "--ac_weight", "0.002", *FORMAL)
    t = cfgs["jepa_B1_s43"]["train"]
    assert t["seed"] == 43 and t["output_dir"].endswith("runs/jepa_B1_s43")
    ac = t["align_params"]["action_cond"]
    assert ac["enabled"] is True and ac["use_actions"] is True and ac["loss_weight"] == 0.002

    b0 = run(tmp_path, "formal", "--arm", "B0", *FORMAL)["jepa_B0_s42"]["train"]["align_params"]["action_cond"]
    assert b0["enabled"] is True and b0["use_actions"] is False and b0["loss_weight"] == 0.004

    a0 = run(tmp_path, "formal", "--name", "train_c50_v1", *FORMAL)["train_c50_v1"]["train"]
    assert a0["align_params"]["action_cond"]["enabled"] is False and "seed" not in a0


def test_smoke_arm_suffixes_names_and_dirs(tmp_path):
    cfgs = run(tmp_path, "smoke", "--arm", "B1")
    assert set(cfgs) == {f"{n}_b1" for n in ("smoke_s1", "smoke_s2_ref", "smoke_s3", "smoke_s3b",
                                              "smoke_s3c_distmuon", "smoke_s5_resume")}
    assert cfgs["smoke_s1_b1"]["train"]["output_dir"] == cfgs["smoke_s5_resume_b1"]["train"]["output_dir"]
    assert cfgs["smoke_s3b_b1"]["train"]["output_dir"].endswith("smoke/s3b_b1")
    assert all(c["train"]["align_params"]["action_cond"]["enabled"] for c in cfgs.values())


def test_ac_weight_needs_a_b_arm(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, "formal", "--name", "x", "--ac_weight", "0.01", *FORMAL)
    with pytest.raises(ValueError):
        mtc.apply_arm({"train": {"align_params": {}}}, "A2")


LINE = ("11/16/2026 01:00:{s:02d} - INFO - __main__ - Step {step}/100, Epoch 1, Loss 0.5000, VLA_Loss 0.0500, "
        "Depth_Loss 0.2000, Future_Depth_Loss 0.3000, FutureVideo_Loss 0.0400, SeqWise_Loss 0.0010, "
        "RouterZ_Loss 0.0001, AC_Loss {ac:.4f}, AC_Shuffle_Gap {gap:.5f}, MaxVio 0.1000, GradNorm 1.0000, "
        "LR 1.00e-04, StepTime 3.000s, Depth_Forward_Time  0.100s, Ignore_Batch_Num 0\n")


def test_train_metrics_reads_ac_columns_and_calibrates(tmp_path, capsys):
    log = tmp_path / "train_0.log"
    log.write_text("".join(LINE.format(s=i, step=i + 1, ac=0.4, gap=-0.00012 + 0.001 * i) for i in range(20)))
    assert tm.main([str(log), "--bin", "10", "--ac_calibrate", "10", "--csv", str(tmp_path / "m.csv")]) == 0
    out = capsys.readouterr().out
    rec = tm.parse_log(log)["steps"][1]
    assert rec["AC_Loss"] == 0.4 and rec["AC_Shuffle_Gap"] == -0.00012
    w1 = 0.04 * 0.004 / 0.4
    assert f"1x {w1:.3e}" in out and "action-conditioned head" in out
    assert "AC_Loss" in (tmp_path / "m.csv").read_text().splitlines()[0]
