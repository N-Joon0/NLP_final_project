from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import all_tasks, load_task

COLORS = {"student_ft": "#888888", "std_kd": "#1f77b4", "conf_kd": "#2ca02c", "ours": "#d62728", "ours_norm": "#9467bd"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path.name


def budget_curve(main, out: Path) -> list[str]:
    files = []
    for task in all_tasks():
        t = main[main.task == task]
        if t.empty:
            continue
        pm = load_task(task)["primary_metric"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for cond, g in t.groupby("condition"):
            g = g.sort_values("budget")
            ax.plot(g.budget, g.primary_mean, "-o", label=cond, color=COLORS.get(cond))
            ax.fill_between(g.budget, g.primary_mean - g.primary_std, g.primary_mean + g.primary_std,
                            alpha=0.15, color=COLORS.get(cond))
        tp = t.teacher_primary.iloc[0]
        if not np.isnan(tp):
            ax.axhline(tp, ls="--", color="k", lw=1, label="teacher")
        ax.set_xscale("log")
        ax.set_xticks(sorted(t.budget.unique()))
        ax.set_xticklabels([f"{int(b) // 1000}k" for b in sorted(t.budget.unique())])
        ax.set_xlabel("student training examples")
        ax.set_ylabel(f"test {pm}")
        ax.set_title(f"{task}: budget curve")
        ax.legend(fontsize=8)
        files.append(_save(fig, out / f"budget_curve_{task}.png"))
    return files


def delta_bar(delta, sig, out: Path) -> list[str]:
    if delta.empty:
        return []
    fig, axes = plt.subplots(1, len(all_tasks()), figsize=(5.5 * len(all_tasks()), 4), squeeze=False)
    for ax, task in zip(axes[0], all_tasks()):
        d = delta[(delta.task == task) & (delta.condition.isin(["ours", "conf_kd"]))]
        budgets = sorted(d.budget.unique())
        w = 0.35
        for i, cond in enumerate(["conf_kd", "ours"]):
            ys, lo, hi = [], [], []
            for b in budgets:
                comp = f"{cond} vs std_kd"
                s = sig[(sig.task == task) & (sig.budget == b) & (sig.comparison == comp)] if len(sig) else sig
                r = d[(d.budget == b) & (d.condition == cond)]
                y = r.delta_pp_mean.iloc[0] if len(r) else np.nan
                ys.append(y)
                if len(s):
                    lo.append(y - s.ci_lo.iloc[0] * 100)
                    hi.append(s.ci_hi.iloc[0] * 100 - y)
                else:
                    lo.append(0), hi.append(0)
            ax.bar(np.arange(len(budgets)) + (i - 0.5) * w, ys, w, yerr=[lo, hi], capsize=3, label=f"{cond} − std_kd",
                   color=COLORS[cond])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(budgets)))
        ax.set_xticklabels([f"{int(b) // 1000}k" for b in budgets])
        ax.set_ylabel("Δ test primary metric (%p)")
        ax.set_title(f"{task} (95% bootstrap CI)")
        ax.legend(fontsize=8)
    return [_save(fig, out / "delta_bar.png")]


def confidence_bins_fig(cb, out: Path) -> list[str]:
    files = []
    for task in all_tasks():
        t = cb[cb.task == task] if len(cb) else cb
        if len(t) == 0:
            continue
        budget = t.budget.max()
        t = t[t.budget == budget]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for cond, g in t.groupby("condition"):
            g = g.sort_values("conf_bin")
            ax.plot(g.conf_bin, g.accuracy, "-o", label=cond, color=COLORS.get(cond))
        ax.set_xlabel("teacher confidence quintile (1=lowest)")
        ax.set_ylabel("student test accuracy")
        ax.set_title(f"{task}: accuracy by teacher confidence (n={int(budget) // 1000}k)")
        ax.legend(fontsize=8)
        files.append(_save(fig, out / f"confidence_bins_{task}.png"))
    return files


def teacher_error_fig(te, out: Path) -> list[str]:
    files = []
    for task in all_tasks():
        t = te[te.task == task] if len(te) else te
        if len(t) == 0:
            continue
        budget = t.budget.max()
        t = t[t.budget == budget]
        conds = [c for c in COLORS if c in set(t.condition)]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, col, title in ((axes[0], "acc_teacher_wrong", "accuracy on teacher-wrong examples"),
                               (axes[1], "inheritance_rate", "inheritance rate (same wrong answer as teacher)")):
            ax.bar(conds, [t[t.condition == c][col].iloc[0] for c in conds], color=[COLORS[c] for c in conds])
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"{task} (n={int(budget) // 1000}k)")
        files.append(_save(fig, out / f"teacher_error_{task}.png"))
    return files


def make_all(tables: dict, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    f = []
    f += budget_curve(tables["main_results"], out)
    f += delta_bar(tables["delta_vs_std_kd"], tables["significance"], out)
    f += confidence_bins_fig(tables["confidence_bins"], out)
    f += teacher_error_fig(tables["teacher_error_analysis"], out)
    return f
