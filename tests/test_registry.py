import re

import pytest

from dakd import deps
from dakd.registry import expand_stage

EXPECTED = {"s01": 2, "s03": 18, "s04": 8, "s05": 18, "s06": 6, "s07": 18, "s08": 12, "s09": 18, "s10": 18}


def test_counts_sum_118():
    n = {s: len(expand_stage(s, allow_missing=True)) for s in EXPECTED}
    assert n == EXPECTED and sum(n.values()) == 118


def test_run_id_rules():
    ids = {s.run_id for st in ("s01", "s03", "s04") for s in expand_stage(st, allow_missing=True)}
    assert "sst2__teacher__full__s42" in ids
    assert "sst2__student_ft__n10000__s87" in ids
    assert "ynat__std_kd__n2000__s42__T2_l1.0_g0_b0_none" in ids
    assert all(re.match(r"^(sst2|ynat)__\w+__(full|n\d+)__s\d+(__T.*)?$", i) for i in ids)


def test_unique_ids_per_stage():
    for st in EXPECTED:
        ids = [s.run_id for s in expand_stage(st, allow_missing=True)]
        assert len(ids) == len(set(ids))


def test_missing_selected_raises_unless_allowed():
    with pytest.raises(RuntimeError):
        expand_stage("s05")


def test_selected_injection(tmp_path, monkeypatch):
    import json
    from dakd import paths
    monkeypatch.setattr(paths, "selected_dir", lambda: tmp_path)
    (tmp_path / "sst2_std_kd.json").write_text(json.dumps({"T": 4, "lambda": 0.5}))
    s = [x for x in expand_stage("s05", ["sst2"]) if x.budget == 2000 and x.seed == 42][0]
    assert (s.T, s.lam, s.gamma, s.beta) == (4, 0.5, 0, 0)
    assert s.run_id == "sst2__std_kd__n2000__s42__T4_l0.5_g0_b0_none"


def test_deps_messages():
    msgs = deps.check_stage("s05", ["sst2"])
    assert any("teacher logits" in m for m in msgs) or any("선택값" in m for m in msgs)


def test_shard_deterministic():
    from dakd.scheduler import apply_shard
    specs = expand_stage("s03", allow_missing=True)
    parts = [apply_shard(specs, f"{k}/4") for k in range(1, 5)]
    assert sum(len(p) for p in parts) == 18
    assert len({s.run_id for p in parts for s in p}) == 18


def test_only_budget_seed_filters():
    from dakd.scheduler import build_jobs
    gate_runs = build_jobs(["s03"], ["sst2", "ynat"], True, None, "2000", "42")
    assert sorted(s.run_id for s in gate_runs) == ["sst2__student_ft__n2000__s42", "ynat__student_ft__n2000__s42"]
    assert len(build_jobs(["s03"], ["sst2"], True)) == 9
    assert len(build_jobs(["s03"], ["sst2"], True, None, "2000,5000", None)) == 6


def test_weights_check_gitkeep_only_is_missing(tmp_path):
    from dakd import weights
    d = tmp_path / "teachers" / "sst2"
    d.mkdir(parents=True)
    (d / ".gitkeep").touch()
    ok, why = weights.check(d)
    assert not ok and why == "폴더 없음"
    (d / "model.safetensors").touch()
    ok, why = weights.check(d)
    assert not ok and "WEIGHTS_OK" in why
