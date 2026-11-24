from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import all_tasks, load_base, load_task
from . import collect as C
from .stats import mean_std, paired_bootstrap

COND_ORDER = ["student_ft", "std_kd", "conf_kd", "ours", "ours_norm"]
COMPARISONS = [("ours", "std_kd"), ("ours", "conf_kd"), ("conf_kd", "std_kd"), ("ours", "student_ft"),
               ("ours_norm", "ours")]


def _students(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.condition != "teacher"]


def main_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        tc = load_task(task)
        pm, sm = tc["primary_metric"], tc["secondary_metric"]
        t = df[(df.task == task) & (df.condition == "teacher")]
        tp = float(t[pm].iloc[0]) if len(t) else np.nan
        sub = _students(df[df.task == task])
        for (budget, cond), g in sub.groupby(["budget", "condition"]):
            m, s = mean_std(g[pm])
            rows.append({"task": task, "budget": budget, "condition": cond, "n_seeds": len(g),
                         f"{pm}_mean": m, f"{pm}_std": s, f"{sm}_mean": g[sm].mean(), "primary_metric": pm,
                         "primary_mean": m, "primary_std": s, "teacher_primary": tp})
    out = pd.DataFrame(rows)
    if len(out):
        out["cond_rank"] = out.condition.map({c: i for i, c in enumerate(COND_ORDER)})
        out = out.sort_values(["task", "budget", "cond_rank"]).drop(columns="cond_rank")
    return out


def delta_vs_std_kd(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        pm = load_task(task)["primary_metric"]
        sub = _students(df[df.task == task])
        for budget in sorted(sub.budget.unique()):
            b = sub[sub.budget == budget]
            ref = b[b.condition == "std_kd"].set_index("seed")[pm]
            for cond in ("conf_kd", "ours", "ours_norm"):
                x = b[b.condition == cond].set_index("seed")[pm]
                common = ref.index.intersection(x.index)
                if len(common) == 0:
                    continue
                d = (x[common] - ref[common]) * 100
                m, s = mean_std(d)
                rows.append({"task": task, "budget": budget, "condition": cond, "metric": pm,
                             "delta_pp_mean": m, "delta_pp_std": s, "n_seeds": len(common)})
    return pd.DataFrame(rows)


def significance(df: pd.DataFrame) -> pd.DataFrame:
    base = load_base()
    rows = []
    for task in all_tasks():
        tc = load_task(task)
        pm, K = tc["primary_metric"], tc["num_labels"]
        y = C.test_labels(task)
        sub = _students(df[df.task == task])
        for budget in sorted(sub.budget.unique()):
            b = sub[sub.budget == budget]
            for a, c in COMPARISONS:
                ra, rc = b[b.condition == a], b[b.condition == c]
                seeds = sorted(set(ra.seed) & set(rc.seed))
                if not seeds:
                    continue
                pa = [C.load_preds(ra[ra.seed == s].run_dir.iloc[0]).argmax(1) for s in seeds]
                pc = [C.load_preds(rc[rc.seed == s].run_dir.iloc[0]).argmax(1) for s in seeds]
                r = paired_bootstrap(y, pa, pc, pm, K, B=base["eval"]["bootstrap_B"], seed=base["eval"]["bootstrap_seed"])
                rows.append({"task": task, "budget": budget, "comparison": f"{a} vs {c}", "metric": pm,
                             "n_seeds": len(seeds), **r})
    return pd.DataFrame(rows)


def _per_seed_preds(df, task, cond, budget):
    g = df[(df.task == task) & (df.condition == cond) & (df.budget == budget)]
    return [C.load_preds(d).argmax(1) for d in g.run_dir]


def teacher_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        tp = C.teacher_test_probs(task)
        if tp is None:
            continue
        y, tpred = C.test_labels(task), tp.argmax(1)
        right = tpred == y
        for (budget, cond), g in _students(df[df.task == task]).groupby(["budget", "condition"]):
            acc_r, acc_w, inh = [], [], []
            for d in g.run_dir:
                sp = C.load_preds(d).argmax(1)
                acc_r.append((sp[right] == y[right]).mean())
                if (~right).any():
                    acc_w.append((sp[~right] == y[~right]).mean())
                    inh.append((sp[~right] == tpred[~right]).mean())
            rows.append({"task": task, "budget": budget, "condition": cond, "n_seeds": len(g),
                         "n_teacher_correct": int(right.sum()), "n_teacher_wrong": int((~right).sum()),
                         "acc_teacher_correct": np.mean(acc_r),
                         "acc_teacher_wrong": np.mean(acc_w) if acc_w else np.nan,
                         "inheritance_rate": np.mean(inh) if inh else np.nan})
    return pd.DataFrame(rows)


def confidence_bins(df: pd.DataFrame, nbins: int = 5) -> pd.DataFrame:
    rows = []
    for task in all_tasks():
        tp = C.teacher_test_probs(task)
        if tp is None:
            continue
        y, conf = C.test_labels(task), tp.max(1)
        bins = pd.qcut(pd.Series(conf).rank(method="first"), nbins, labels=False)
        for (budget, cond), g in _students(df[df.task == task]).groupby(["budget", "condition"]):
            preds = [C.load_preds(d).argmax(1) for d in g.run_dir]
            for b in range(nbins):
                m = (bins == b).to_numpy()
                rows.append({"task": task, "budget": budget, "condition": cond, "conf_bin": b + 1,
                             "conf_mean": float(conf[m].mean()), "n": int(m.sum()),
                             "accuracy": float(np.mean([(p[m] == y[m]).mean() for p in preds]))})
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, budget, cond), g in df.groupby(["task", "budget", "condition"]):
        m, s = mean_std(g.ece)
        rows.append({"task": task, "budget": budget, "condition": cond, "n_seeds": len(g), "ece_mean": m, "ece_std": s})
    return pd.DataFrame(rows)


def weight_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df[df.condition.isin(["std_kd", "conf_kd", "ours", "ours_norm"])].iterrows():
        tm = Path(r.run_dir) / "train_metrics.json"
        if not tm.exists():
            continue
        ws = json.loads(tm.read_text()).get("weight_stats", [])
        if ws:
            rows.append({"task": r.task, "budget": r.budget, "condition": r.condition, "seed": r.seed, **ws[-1]})
    if not rows:
        return pd.DataFrame()
    g = pd.DataFrame(rows)
    keys = ["task", "budget", "condition"]
    return g.drop(columns=["seed", "epoch"]).groupby(keys, as_index=False).mean(numeric_only=True)


def build_all(df: pd.DataFrame, tune: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"main_results": main_results(df), "delta_vs_std_kd": delta_vs_std_kd(df),
              "significance": significance(df), "teacher_error_analysis": teacher_error_analysis(df),
              "confidence_bins": confidence_bins(df), "calibration": calibration(df),
              "weight_diagnostics": weight_diagnostics(df), "tuning_results": tune}
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    eff = paths.results_dir() / "efficiency.csv"
    if eff.exists():
        tables["efficiency"] = pd.read_csv(eff)
        if eff.resolve() != (out_dir / "efficiency.csv").resolve():
            shutil.copy(eff, out_dir / "efficiency.csv")
    return tables
