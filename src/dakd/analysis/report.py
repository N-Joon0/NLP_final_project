from __future__ import annotations

import json
import platform
import time

import pandas as pd

from .. import paths
from ..config import all_tasks, load_base, load_task
from . import collect as C, figures, tables


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df is None or len(df) == 0:
        return "_(데이터 없음)_\n"
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _rq_summary(t: dict) -> str:
    out = []
    sig, delta = t["significance"], t["delta_vs_std_kd"]
    if len(sig) == 0:
        return "_(비교 가능한 run 이 아직 없습니다)_\n"
    for task in all_tasks():
        for comp in ("ours vs std_kd", "conf_kd vs std_kd", "ours vs conf_kd"):
            for _, r in sig[(sig.task == task) & (sig.comparison == comp)].iterrows():
                verdict = ("유의한 차이 없음" if r.ci_lo <= 0 <= r.ci_hi
                           else ("개선(95% CI > 0)" if r.ci_lo > 0 else "악화(95% CI < 0)"))
                out.append(f"- {task} n={int(r.budget) // 1000}k, {comp}: Δ={r.observed * 100:+.2f}%p "
                           f"[{r.ci_lo * 100:+.2f}, {r.ci_hi * 100:+.2f}], p(≤0)={r.p_one_sided:.3f}, seeds={int(r.n_seeds)} → {verdict}")
    return "\n".join(out) + "\n"


def run_analysis() -> None:
    res = paths.ensure(paths.results_dir())
    tdir, fdir = paths.ensure(res / "tables"), paths.ensure(res / "figures")
    df, tune, missing = C.collect_main(), C.collect_tuning(), C.collect_missing()
    if df.empty:
        print("[analyze] test 추론이 끝난 run 이 없습니다 (먼저 03, 05 단계 실행). 빈 리포트를 생성합니다.")
    tabs = tables.build_all(df, tune, tdir) if not df.empty else {k: pd.DataFrame() for k in (
        "main_results", "delta_vs_std_kd", "significance", "teacher_error_analysis", "confidence_bins",
        "calibration", "weight_diagnostics", "tuning_results")}
    figs = figures.make_all(tabs, fdir) if not df.empty else []

    base = load_base()
    sel = {}
    for task in all_tasks():
        for name in ("std_kd", "conf_kd", "ours"):
            p = paths.selected_path(task, name)
            if p.exists():
                j = json.loads(p.read_text())
                sel[f"{task}/{name}"] = {k: v for k, v in j.items() if k in ("T", "lambda", "gamma", "beta", "dev_score")}
    gate = json.loads(paths.gate_path().read_text()) if paths.gate_path().exists() else None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        tv = torch.__version__
    except Exception:
        gpu, tv = "n/a", "n/a"

    L = [f"# RESULTS ({time.strftime('%F %T')})\n"]
    if missing:
        L.append(f"## ⚠ 빠진 run ({len(missing)}개)\n")
        L += [f"- `{m}`" for m in missing[:60]] + ([f"- … 외 {len(missing) - 60}개"] if len(missing) > 60 else [])
        L.append("")
    L += ["## 환경\n", f"- host: {platform.node()}, GPU(분석 머신): {gpu}, torch {tv}, python {platform.python_version()}",
          f"- n_layers(student): {base['n_layers']}, seeds: {base['seeds']}, split_seed: {base['split_seed']}",
          f"- gate: {json.dumps({t: round(r['gap_pp'], 2) for t, r in gate['tasks'].items()}) if gate else '미실행'} (teacher−student 2k, %p)\n",
          "## 선택된 하이퍼파라미터 (dev 기준)\n", "```json", json.dumps(sel, indent=1, ensure_ascii=False), "```\n",
          "## 메인 결과 (test, seed 평균±표준편차)\n", _md(tabs["main_results"]),
          "## Standard KD 대비 차이 (%p, seed 쌍 평균)\n", _md(tabs["delta_vs_std_kd"], ".3f"),
          "## 유의성 (paired bootstrap, 10,000회)\n", _md(tabs["significance"], ".4f"),
          "## RQ 수치 요약 (과장 금지: CI 가 0 을 포함하면 '유의한 차이 없음')\n",
          "RQ1 (ours vs std_kd), RQ2 (budget 별 변화), RQ3 (task 간 일관성) 은 아래 수치로 판단한다.\n",
          _rq_summary(tabs),
          "## Teacher 정답/오답 분석\n", _md(tabs["teacher_error_analysis"]),
          "## Calibration (ECE)\n", _md(tabs["calibration"]),
          "## KD 가중치 진단 (마지막 epoch)\n", _md(tabs["weight_diagnostics"]),
          "## 튜닝 결과 (dev)\n", _md(tabs["tuning_results"]),
          "## 효율\n", _md(tabs.get("efficiency"), ".2f"),
          "## 그림\n"] + [f"![{f}](figures/{f})" for f in figs]
    (res / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[analyze] 표 → {tdir}, 그림 → {fdir}, 리포트 → {res / 'RESULTS.md'} (빠진 run {len(missing)}개)")
