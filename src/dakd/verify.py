from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths, weights as W
from .config import all_tasks, load_base, load_task, parse_stage_list, parse_tasks
from .data import split_indices
from .registry import expand_stage

TRAIN_METRIC_KEYS = ["best_epoch", "best_dev", "primary_metric", "epochs", "train_time_s", "peak_mem_mb",
                     "world_size", "total_steps", "train_samples_per_s"]
EPOCH_KEYS = ["epoch", "train_loss", "train_ce", "epoch_time_s", "dev"]
EVAL_KEYS = ["accuracy", "macro_f1", "per_class_f1", "confusion_matrix", "ece", "nll", "brier",
             "mean_confidence", "n", "primary", "primary_name"]
CONFIG_KEYS = ["spec", "world_size", "per_rank_batch", "precision", "lr", "batch_size", "max_length", "gpu",
               "gpu_count", "git_commit", "torch", "params", "weights_dir", "n_layers", "epochs"]
KD_WEIGHT_KEYS = ["w_mean", "w_std", "c_mean", "d_mean", "w_mean_teacher_correct", "w_mean_teacher_wrong",
                  "n_teacher_wrong"]
EFF_COLS = ["task", "model", "precision", "params", "size_mb", "latency_ms_bs1", "throughput_samples_s_bs64",
            "peak_mem_mb_bs64", "gpu_name", "torch", "cuda", "seq_len", "latency_batch", "throughput_batch",
            "latency_warmup", "latency_iters"]
TABLES = ["main_results", "delta_vs_std_kd", "significance", "teacher_error_analysis", "confidence_bins",
          "calibration", "weight_diagnostics", "tuning_results"]


class Report:
    def __init__(self):
        self.fails: list[str] = []
        self.checked = 0
        self.incomplete: list[str] = []

    def fail(self, where: str, msg: str):
        self.fails.append(f"{where}: {msg}")

    def need_keys(self, where: str, d: dict, keys: list[str]):
        miss = [k for k in keys if k not in d]
        if miss:
            self.fail(where, f"키 누락 {miss}")


def _load(p: Path):
    return json.loads(p.read_text())


def _check_probs(rep: Report, where: str, p: Path, n_expected: int, K: int, labels: np.ndarray | None = None):
    if not p.exists():
        return rep.fail(where, f"{p.name} 없음")
    a = np.load(p)
    if a.shape != (n_expected, K):
        return rep.fail(where, f"{p.name} shape {a.shape} != ({n_expected}, {K})")
    if not np.isfinite(a).all():
        rep.fail(where, f"{p.name} 에 NaN/inf")
    if "preds" in p.name and not np.allclose(a.sum(1), 1, atol=1e-3):
        rep.fail(where, f"{p.name} 확률 합이 1 이 아님")
    return a


def verify_run(rep: Report, spec, is_tuning: bool, check_weights: bool = True) -> str:
    d, base, tc = spec.run_dir, load_base(), load_task(spec.task)
    where = f"{spec.stage}/{spec.run_id}"
    nfail = len(rep.fails)
    if not (d / "DONE_TRAIN").exists():
        rep.incomplete.append(where)
        return "incomplete"
    K, pm = tc["num_labels"], tc["primary_metric"]
    cfg = _load(d / "config.json")
    rep.need_keys(where + "/config.json", cfg, CONFIG_KEYS)
    if cfg.get("world_size") != base["gpu"]["world_size"]:
        rep.fail(where, f"world_size={cfg.get('world_size')} (본 실험은 {base['gpu']['world_size']}장이어야 함)")
    tm = _load(d / "train_metrics.json")
    rep.need_keys(where + "/train_metrics.json", tm, TRAIN_METRIC_KEYS)
    for e in tm.get("epochs", []):
        rep.need_keys(where + "/epoch", e, EPOCH_KEYS)
        rep.need_keys(where + "/epoch.dev", e.get("dev", {}), EVAL_KEYS[:-2])
    if len(tm.get("epochs", [])) != cfg.get("epochs") and not (tm.get("total_steps", 0) and cfg.get("epochs") is None):
        rep.fail(where, f"epoch 로그 수 {len(tm.get('epochs', []))} != 설정 {cfg.get('epochs')}")
    if spec.is_kd:
        ws = tm.get("weight_stats", [])
        if not ws:
            rep.fail(where, "KD run 인데 weight_stats 가 없음")
        else:
            rep.need_keys(where + "/weight_stats", ws[-1], KD_WEIGHT_KEYS)
        if not (d / "train_example_stats.npz").exists():
            rep.fail(where, "train_example_stats.npz 없음 (예제별 w/c/d/kd/ce)")
        for e in tm.get("epochs", []):
            if e.get("train_kd") is None:
                rep.fail(where, "KD run 의 epoch 에 train_kd 없음")
                break
    n_dev = len(split_indices(spec.task, "dev"))
    _check_probs(rep, where, d / "dev_preds.npy", n_dev, K)
    _check_probs(rep, where, d / "dev_logits.npy", n_dev, K)
    if not (d / "train_log.jsonl").exists():
        rep.fail(where, "train_log.jsonl 없음")
    wdir = paths.weights_dir(spec.stage, spec.run_id, spec.condition, spec.task)
    ok, why = W.check(wdir, verify_sha=True) if check_weights else (False, "skipped")
    if check_weights and not ok:
        rep.fail(where, f"weights 이상: {why} ({wdir})")
    elif ok:
        meta = _load(wdir / W.META_FILE)
        for k in ("run_id", "spec", "task", "max_length", "precision", "sha256", "best_epoch", "best_dev"):
            if k not in meta:
                rep.fail(where, f"weights_meta.json 키 누락: {k}")
    if is_tuning or not spec.test_inference:
        if (d / "test_preds.npy").exists() or (d / "DONE_INFER").exists():
            rep.fail(where, "튜닝 run 인데 test 추론 결과가 있음 (test 는 선택에 사용 금지)")
    else:
        if not (d / "DONE_INFER").exists():
            rep.incomplete.append(where + " (infer)")
        else:
            n_test = len(split_indices(spec.task, "test"))
            probs = _check_probs(rep, where, d / "test_preds.npy", n_test, K)
            _check_probs(rep, where, d / "test_logits.npy", n_test, K)
            m = _load(d / "test_metrics.json")
            rep.need_keys(where + "/test_metrics.json", m, EVAL_KEYS)
            if m.get("n") != n_test:
                rep.fail(where, f"test n={m.get('n')} != {n_test}")
            if isinstance(probs, np.ndarray) and "accuracy" in m:
                from .data import get_texts_labels
                y = np.asarray(json.loads(paths.splits_path(spec.task).read_text())["test_labels"])[:n_test]
                if abs((probs.argmax(1) == y).mean() - m["accuracy"]) > 1e-6:
                    rep.fail(where, "test_metrics.accuracy 가 test_preds 로 재계산한 값과 다름")
            if check_weights and ok and m.get("weights_sha256") != meta.get("sha256"):
                rep.fail(where, "추론에 사용된 가중치 sha256 이 weights/ 의 것과 다름")
    return "fail" if len(rep.fails) > nfail else "ok"


def verify_teacher_logits(rep: Report, task: str):
    out = paths.teacher_logits_dir(task)
    where = f"teacher_logits/{task}"
    if not (out / "DONE").exists():
        rep.incomplete.append(where)
        return "incomplete"
    K = load_task(task)["num_labels"]
    n_train = len(np.load(out / "train.npy"))
    for name, n in (("train", n_train), ("dev", len(split_indices(task, "dev"))), ("test", len(split_indices(task, "test")))):
        a = np.load(out / f"{name}.npy")
        if a.shape[1] != K or a.shape[0] != n:
            rep.fail(where, f"{name}.npy shape {a.shape}")
    c = np.load(out / "train_correct.npy")
    if (c >= 0).sum() == 0:
        rep.fail(where, "train_correct.npy 가 비어 있음")
    meta = _load(out / "meta.json")
    rep.need_keys(where + "/meta.json", meta, ["teacher_checkpoint_sha256", "created", "teacher_train_acc"])
    return "ok"


def verify_efficiency(rep: Report):
    import pandas as pd
    p = paths.results_dir() / "efficiency.csv"
    if not p.exists():
        rep.incomplete.append("results/efficiency.csv")
        return "incomplete"
    df = pd.read_csv(p)
    miss = [c for c in EFF_COLS if c not in df.columns]
    if miss:
        rep.fail("efficiency.csv", f"열 누락 {miss}")
    exp = len(all_tasks()) * 2 * 2
    if len(df) != exp:
        rep.fail("efficiency.csv", f"행 수 {len(df)} != {exp} (task x teacher/student x fp32/bf16)")
    return "ok"


def verify_analysis(rep: Report):
    res = paths.results_dir()
    if not (res / "RESULTS.md").exists():
        rep.incomplete.append("results/RESULTS.md")
        return "incomplete"
    for t in TABLES:
        if not (res / "tables" / f"{t}.csv").exists():
            rep.fail("analysis", f"tables/{t}.csv 없음")
    if not list((res / "figures").glob("*.png")):
        rep.fail("analysis", "figures/*.png 없음")
    return "ok"


def run_verify(stage_arg: str = "all", task_arg: str = "all", check_weights: bool = True) -> int:
    rep = Report()
    stages, tasks = parse_stage_list(stage_arg), parse_tasks(task_arg)
    print(f"[verify] workspace={paths.workspace()} | weights={paths.weights_root()}")
    if not check_weights:
        print("[verify] --no-weights: weights/ 검사 생략 (다른 서버에서 받은 결과를 합쳐 검증할 때)")
    print(f"{'stage':6}{'ok':>6}{'incomplete':>12}{'FAIL':>6}")
    for st in stages:
        c = {"ok": 0, "incomplete": 0, "fail": 0}
        for spec in expand_stage(st, tasks, allow_missing=True):
            r = verify_run(rep, spec, is_tuning=not spec.test_inference and st in ("s04", "s06", "s08"), check_weights=check_weights)
            c[r] += 1
        print(f"{st:6}{c['ok']:>6}{c['incomplete']:>12}{c['fail']:>6}")
    for t in tasks:
        verify_teacher_logits(rep, t)
    verify_efficiency(rep)
    verify_analysis(rep)
    print(f"\n미완료 항목 {len(rep.incomplete)}개" + (": " + ", ".join(rep.incomplete[:8]) + (" ..." if len(rep.incomplete) > 8 else "") if rep.incomplete else ""))
    if rep.fails:
        print(f"\n[FAIL] {len(rep.fails)}건")
        for f in rep.fails[:80]:
            print("  -", f)
        return 1
    print("\n[PASS] 검증된 항목에서 형식/누락/불일치 문제 없음")
    return 0
