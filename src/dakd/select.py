from __future__ import annotations

import json
import time

from . import paths
from .config import load_stage, parse_tasks
from .registry import expand_stage


def select_stage(stage: str, task_arg: str = "all") -> list[str]:
    cfg = load_stage(stage)
    if "select" not in cfg:
        raise ValueError(f"{stage} 는 튜닝 Stage 가 아닙니다 (select 설정 없음)")
    name, params = cfg["select"]["name"], cfg["select"]["params"]
    written = []
    for task in parse_tasks(task_arg):
        if task not in cfg["tasks"]:
            continue
        specs = expand_stage(stage, [task])
        rows, missing = [], []
        for order, s in enumerate(specs):
            tm = s.run_dir / "train_metrics.json"
            if not (s.run_dir / "DONE_TRAIN").exists() or not tm.exists():
                missing.append(s.run_id)
                continue
            m = json.loads(tm.read_text())
            rows.append({"order": order, "run_id": s.run_id, "dev_score": m["best_dev"], "best_epoch": m["best_epoch"],
                         **{p: getattr(s, p) for p in params}})
        if missing:
            raise RuntimeError(f"[{stage}/{task}] 튜닝 run 이 빠져 있어 선택하지 않습니다 ({len(missing)}개):\n  "
                               + "\n  ".join(missing))
        best = sorted(rows, key=lambda r: (-r["dev_score"], r["order"]))[0]
        out = {"task": task, "condition": cfg["condition"], "stage": stage,
               "primary_metric": "dev " + json.loads((specs[0].run_dir / "train_metrics.json").read_text())["primary_metric"],
               "dev_score": best["dev_score"], "selected_run_id": best["run_id"], "created": time.strftime("%F %T"),
               "table": rows}
        for p in params:
            out["lambda" if p == "lam" else p] = best[p]
        path = paths.selected_path(task, name)
        paths.ensure(path.parent)
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        written.append(str(path))
        sel = {("lambda" if p == "lam" else p): best[p] for p in params}
        print(f"[select] {task}/{name}: {sel} (dev {best['dev_score']:.4f}) → {path}")
    if written:
        print("★ 위 configs/selected/*.json 을 git 에 커밋해 팀원과 공유하세요.")
    return written
