from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path

import yaml

from . import paths

REQUIRED_BASE = ["n_layers", "seeds", "tune_seed", "teacher_seed", "split_seed", "train", "data", "gpu"]
REQUIRED_STAGE = ["stage", "name", "condition", "tasks", "budgets", "seeds", "test_inference"]
STAGE_ORDER = ["s01", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"]


def _read(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_base() -> dict:
    cfg = _read(paths.configs_dir() / "base.yaml")
    miss = [k for k in REQUIRED_BASE if k not in cfg]
    if miss:
        raise ValueError(f"configs/base.yaml 에 필수 키가 없습니다: {miss}")
    return cfg


@lru_cache(maxsize=None)
def _task_cached(task: str) -> str:
    return str(paths.configs_dir() / "tasks" / f"{task}.yaml")


def load_task(task: str) -> dict:
    p = Path(_task_cached(task))
    if not p.exists():
        raise ValueError(f"알 수 없는 task: {task} (configs/tasks/ 확인)")
    return _read(p)


def all_tasks() -> list[str]:
    return sorted(p.stem for p in (paths.configs_dir() / "tasks").glob("*.yaml"))


def load_stage(stage: str) -> dict:
    files = list((paths.configs_dir() / "stages").glob(f"{stage}_*.yaml"))
    if len(files) != 1:
        raise ValueError(f"Stage '{stage}' 설정 파일을 찾을 수 없습니다 (configs/stages/{stage}_*.yaml)")
    cfg = _read(files[0])
    miss = [k for k in REQUIRED_STAGE if k not in cfg]
    if miss:
        raise ValueError(f"{files[0].name} 에 필수 키가 없습니다: {miss}")
    return cfg


def all_stages() -> list[str]:
    return list(STAGE_ORDER)


def parse_stage_list(arg: str) -> list[str]:
    if arg in ("all", "*"):
        return all_stages()
    out = []
    for s in arg.split(","):
        s = s.strip().lower()
        if s:
            out.append(s if s.startswith("s") else f"s{int(s):02d}")
    return out


def parse_tasks(arg: str) -> list[str]:
    return all_tasks() if arg in ("all", "*") else [t.strip() for t in arg.split(",") if t.strip()]


def effective_seeds(base: dict, which) -> list[int]:
    if isinstance(which, list):
        return which
    if which == "base":
        return list(base["seeds"])
    if which == "tune":
        return [base["tune_seed"]]
    if which == "teacher":
        return [base["teacher_seed"]]
    raise ValueError(f"seeds 값이 잘못되었습니다: {which}")


def effective_epochs(base: dict, is_teacher: bool) -> int:
    return base["train"]["teacher_epochs" if is_teacher else "student_epochs"]


def set_n_layers(n: int) -> None:
    p = paths.configs_dir() / "base.yaml"
    txt = p.read_text(encoding="utf-8")
    new, k = re.subn(r"^n_layers:\s*\d+", f"n_layers: {n}", txt, count=1, flags=re.M)
    if k != 1:
        raise ValueError("base.yaml 에서 n_layers 줄을 찾지 못했습니다")
    p.write_text(new, encoding="utf-8")


def deep(d):
    return copy.deepcopy(d)
