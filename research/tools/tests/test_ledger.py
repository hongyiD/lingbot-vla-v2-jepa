import hashlib

import ledger


def test_ledger_add_show_render(tmp_path, capsys):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("lr: 1e-4\n")
    d = str(tmp_path / "ledger")
    assert ledger.main(["--ledger_dir", d, "add", "--kind", "run", "--title", "runA", "--field", "steps=10000",
                        "--hash", str(cfg)]) == 0
    assert ledger.main(["--ledger_dir", d, "add", "--kind", "submit", "--title", "runA 100x50"]) == 0
    entries = ledger.load(tmp_path / "ledger")
    assert [e["id"] for e in entries] == [1, 2]
    assert entries[0]["fields"] == {"steps": "10000"}
    assert list(entries[0]["hashes"].values())[0] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    md = (tmp_path / "ledger" / "ledger.md").read_text()
    assert "submissions: 1" in md and "runA 100x50" in md
    assert ledger.main(["--ledger_dir", d, "show", "--kind", "run"]) == 0
    assert "runA" in capsys.readouterr().out

