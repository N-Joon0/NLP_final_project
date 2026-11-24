from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths
from ..config import all_tasks, load_task
from ..data import load_splits
from ..registry import expand_stage

MAIN_STAGES = ["s03", "s05", "s07", "s09", "s10"]
TUNE_STAGES = ["s04", "s06", "s08"]


def _read(p):
    return json.loads(p.read_text())


def collect_main() -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        td = paths.teacher_dir(task)
        if (td / "DONE_INFER").exists():
            m = _read(td / "test_metrics.json")
            rows.append({"task": task, "stage": "s01", "condition": "teacher", "budget": 0, "seed": 42,
                         "run_dir": str(td), **{k: m[k] for k in ("accuracy", "macro_f1", "ece", "primary")},
                         "T": None, "lam": None, "gamma": None, "beta": None, "weight_norm": None})
        for st in MAIN_STAGES:
            for s in expand_stage(st, [task], allow_missing=True):
                d = s.run_dir
                if not (d / "DONE_INFER").exists():
                    continue
                m = _read(d / "test_metrics.json")
                rows.append({"task": task, "stage": st, "condition": s.condition, "budget": s.budget, "seed": s.seed,
                             "run_dir": str(d), **{k: m[k] for k in ("accuracy", "macro_f1", "ece", "primary")},
                             "T": s.T, "lam": s.lam, "gamma": s.gamma, "beta": s.beta, "weight_norm": s.weight_norm})
    return pd.DataFrame(rows)


def collect_tuning() -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        for st in TUNE_STAGES:
            for s in expand_stage(st, [task], allow_missing=True):
                tm = s.run_dir / "train_metrics.json"
                if (s.run_dir / "DONE_TRAIN").exists() and tm.exists():
                    m = _read(tm)
                    rows.append({"task": task, "stage": st, "condition": s.condition, "T": s.T, "lam": s.lam,
                                 "gamma": s.gamma, "beta": s.beta, "weight_norm": s.weight_norm,
                                 "dev_score": m["best_dev"], "best_epoch": m["best_epoch"], "run_id": s.run_id})
    return pd.DataFrame(rows)


def collect_missing() -> list[str]:
    miss = []
    for task in all_tasks():
        if not (paths.teacher_dir(task) / "DONE_INFER").exists():
            miss.append(f"s01/{task}__teacher")
        for st in MAIN_STAGES + TUNE_STAGES:
            for s in expand_stage(st, [task], allow_missing=True):
                ok = (s.run_dir / "DONE_TRAIN").exists() and (not s.test_inference or (s.run_dir / "DONE_INFER").exists())
                if not ok:
                    miss.append(f"{st}/{s.run_id}")
    return miss


def load_preds(run_dir: str) -> np.ndarray:
    return np.load(f"{run_dir}/test_preds.npy")


def test_labels(task: str) -> np.ndarray:
    return np.asarray(load_splits(task)["test_labels"])


def teacher_test_probs(task: str):
    p = paths.teacher_dir(task) / "test_preds.npy"
    return np.load(p) if p.exists() else None


def label_names(task: str):
    return load_task(task)["label_names"]
