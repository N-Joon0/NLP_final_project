from __future__ import annotations

from . import paths, weights as W
from .config import parse_stage_list, parse_tasks
from .registry import expand_stage


def check_weights(stage_arg: str = "all", task_arg: str = "all", verify_sha: bool = False) -> int:
    tasks = parse_tasks(task_arg)
    print(f"weights 디렉토리: {paths.weights_root()}")
    print(f"{'stage':6}{'total':>7}{'ok':>6}{'missing':>9}{'broken':>8}")
    bad_total = 0
    for st in parse_stage_list(stage_arg):
        ok = miss = broken = 0
        lines = []
        for spec in expand_stage(st, tasks, allow_missing=True):
            wdir = paths.weights_dir(spec.stage, spec.run_id, spec.condition, spec.task)
            good, why = W.check(wdir, verify_sha)
            if good:
                ok += 1
            elif why == "폴더 없음":
                miss += 1
            else:
                broken += 1
                lines.append(f"    ! {st}/{spec.run_id}: {why}")
        total = ok + miss + broken
        print(f"{st:6}{total:>7}{ok:>6}{miss:>9}{broken:>8}")
        for l in lines[:10]:
            print(l)
        bad_total += broken
    return 1 if bad_total else 0
