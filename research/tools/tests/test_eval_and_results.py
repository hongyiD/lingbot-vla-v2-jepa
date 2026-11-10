import json

import parse_eval_run
from helpers import episode_print, episodes_log
from lingbot_tools.io_utils import read_csv

BANNER = ("================================================================\n"
          "  [attempt {k}/3] 2026-10-20 10:00:00\n  task={task} slot=0 gpu=0 port=9000\n"
          "================================================================\n")


def _launcher_run(tmp_path, instruction="seen"):
    run = tmp_path / "runA_10000_demo_clean_20261020"
    (run / "eval_logs").mkdir(parents=True)
    log = (BANNER.format(k=1, task="click_bell") + episodes_log("click_bell", "demo_clean", [1, 1])
           + "Traceback: server died\n" + BANNER.format(k=2, task="click_bell")
           + episodes_log("click_bell", "demo_clean", [1, 0, 1, 1]))
    (run / "eval_logs" / "click_bell.log").write_text(log)
    (run / "eval_results" / "click_bell").mkdir(parents=True)
    (run / "eval_results" / "click_bell" / "_result.txt").write_text(f"Instruction Type: {instruction}\n\n0.75")
    return run


def test_launcher_layout_uses_last_attempt(tmp_path):
    run = _launcher_run(tmp_path)
    out = tmp_path / "parsed"
    assert parse_eval_run.main(["--run_dir", str(run), "--out_dir", str(out)]) == 0
    rows = read_csv(out / "episodes.csv")
    assert [r["success"] for r in rows] == ["1", "0", "1", "1"]
    assert [r["seed"] for r in rows] == ["100000", "100001", "100002", "100003"]
    assert {r["setting"] for r in rows} == {"clean"}
    assert {r["instruction_type"] for r in rows} == {"seen"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["per_setting"]["clean"]["successes"] == 3
    assert summary["errors"] == 0


def test_wrong_instruction_pool_is_an_error(tmp_path):
    run = _launcher_run(tmp_path, instruction="unseen")
    assert parse_eval_run.main(["--run_dir", str(run), "--out_dir", str(tmp_path / "parsed")]) == 1


def test_instruction_pool_follows_protocol_flag(tmp_path):
    run = _launcher_run(tmp_path, instruction="unseen")
    out = str(tmp_path / "parsed")
    assert parse_eval_run.main(["--run_dir", str(run), "--out_dir", out,
                                "--instruction_types", "clean=unseen,randomized=unseen"]) == 0
    assert parse_eval_run.main(["--run_dir", str(run), "--out_dir", out, "--instruction_types", "any"]) == 0


def test_benchmark_layout_and_final_line(tmp_path):
    run = tmp_path / "both_100x3_8gpu"
    (run / "logs").mkdir(parents=True)
    header = "============= Config =============\n\x1b[96mEval Instruction:\x1b[0m unseen\n"
    body = episodes_log("lift_pot", "demo_randomized", [1, 0, 1]) + "Final success rate: 2/3 = 66.7%\n"
    (run / "logs" / "eval_demo_randomized__lift_pot.log").write_text(header + body)
    parsed = parse_eval_run.parse_run(run, "r1")
    assert not parsed["problems"]
    rows = parsed["rows"]
    assert len(rows) == 3
    assert {r["setting"] for r in rows} == {"randomized"}
    assert {r["instruction_type"] for r in rows} == {"unseen"}


def test_counter_anomaly_and_batch_format():
    text = episode_print("click_bell", "demo_clean", 1, 1, 5) + episode_print("click_bell", "demo_clean", 1, 3, 7)
    parsed = parse_eval_run.parse_log_text(text)
    assert any("jumped" in a for a in parsed["anomalies"])
    batch = ("click_bell | episode=0 | worker=1 | seed=100003 | success=True | batch success: 1/1 => 100.0%\n"
             "click_bell | episode=1 | worker=0 | seed=100001 | success=False | batch success: 1/2 => 50.0%\n")
    parsed = parse_eval_run.parse_log_text(batch, config_hint="demo_clean")
    assert [(e["seed"], e["success"]) for e in parsed["episodes"]] == [(100003, 1), (100001, 0)]

