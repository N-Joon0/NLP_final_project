from __future__ import annotations

from .config import all_stages, load_stage, parse_stage_list, parse_tasks
from .registry import expand_stage
from .train import is_done, running_elsewhere


def classify(spec) -> str:
    d = spec.run_dir
    if is_done(spec):
        return "done"
    if running_elsewhere(d):
        return "running"
    if (d / "FAILED").exists():
        return "failed"
    return "pending"


def show_status(stage_arg: str = "all", task_arg: str = "all") -> None:
    tasks = parse_tasks(task_arg)
    print(f"{'stage':6}{'name':14}{'total':>7}{'done':>7}{'running':>9}{'failed':>8}{'pending':>9}")
    tot = {"total": 0, "done": 0, "running": 0, "failed": 0, "pending": 0}
    for st in parse_stage_list(stage_arg):
        cfg = load_stage(st)
        specs = expand_stage(st, tasks, allow_missing=True)
        c = {"done": 0, "running": 0, "failed": 0, "pending": 0}
        for s in specs:
            c[classify(s)] += 1
        waiting = sum(1 for s in specs if s.missing)
        note = f"  (선택값 대기 {waiting}건)" if waiting else ""
        print(f"{st:6}{cfg['name']:14}{len(specs):>7}{c['done']:>7}{c['running']:>9}{c['failed']:>8}{c['pending']:>9}{note}")
        tot["total"] += len(specs)
        for k in c:
            tot[k] += c[k]
    print(f"{'합계':20}{tot['total']:>7}{tot['done']:>7}{tot['running']:>9}{tot['failed']:>8}{tot['pending']:>9}")
