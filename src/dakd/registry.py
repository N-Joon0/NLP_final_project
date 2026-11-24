from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import paths
from .config import effective_seeds, load_base, load_stage

KD_CONDITIONS = {"std_kd", "conf_kd", "ours", "ours_norm"}


def _g(x) -> str:
    return "?" if x is None else f"{x:g}"


def _lam(x) -> str:
    return "?" if x is None else repr(float(x))


@dataclass
class RunSpec:
    stage: str
    task: str
    condition: str
    budget: Optional[int]
    seed: int
    T: Optional[float] = None
    lam: Optional[float] = None
    gamma: Optional[float] = None
    beta: Optional[float] = None
    weight_norm: str = "none"
    test_inference: bool = True
    missing: list = field(default_factory=list)

    @property
    def is_kd(self) -> bool:
        return self.condition in KD_CONDITIONS

    @property
    def run_id(self) -> str:
        b = "full" if self.budget is None else f"n{self.budget}"
        rid = f"{self.task}__{self.condition}__{b}__s{self.seed}"
        if self.is_kd:
            rid += f"__T{_g(self.T)}_l{_lam(self.lam)}_g{_g(self.gamma)}_b{_g(self.beta)}_{self.weight_norm}"
        return rid

    @property
    def run_dir(self):
        return paths.run_dir(self.stage, self.run_id, self.condition, self.task)

    def to_cli_args(self) -> list[str]:
        a = ["--stage", self.stage, "--task", self.task, "--condition", self.condition,
             "--budget", "full" if self.budget is None else str(self.budget),
             "--seed", str(self.seed), "--run-id", self.run_id]
        if self.is_kd:
            a += ["--T", str(self.T), "--lam", str(self.lam), "--gamma", str(self.gamma),
                  "--beta", str(self.beta), "--weight-norm", self.weight_norm]
        a += ["--test-inference", "1" if self.test_inference else "0"]
        return a

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_id"] = self.run_id
        return d


def load_selected(task: str, name: str) -> Optional[dict]:
    p = paths.selected_path(task, name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _resolve_selected(task: str, ref: str):
    name, key = ref.split(".", 1)
    sel = load_selected(task, name)
    return None if sel is None or key not in sel else sel[key]


def expand_stage(stage: str, tasks: list[str] | None = None, allow_missing: bool = False) -> list[RunSpec]:
    cfg = load_stage(stage)
    base = load_base()
    seeds = effective_seeds(base, cfg["seeds"])
    grid = cfg.get("grid", {})
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys])) if keys else [()]
    specs: list[RunSpec] = []
    for task in cfg["tasks"]:
        if tasks and task not in tasks:
            continue
        for budget in cfg["budgets"]:
            b = None if budget == "full" else int(budget)
            for seed in seeds:
                for combo in combos:
                    hp = dict(cfg.get("fixed", {}))
                    hp.update(dict(zip(keys, combo)))
                    missing = []
                    for k, ref in cfg.get("fixed_from_selected", {}).items():
                        v = _resolve_selected(task, ref)
                        if v is None:
                            missing.append(ref)
                        hp[k] = v
                    if missing and not allow_missing:
                        raise RuntimeError(f"[{stage}/{task}] 선택값이 없습니다: {missing}")
                    specs.append(RunSpec(
                        stage=stage, task=task, condition=cfg["condition"], budget=b, seed=seed,
                        T=hp.get("T"), lam=hp.get("lam"), gamma=hp.get("gamma"), beta=hp.get("beta"),
                        weight_norm=hp.get("weight_norm", "none"),
                        test_inference=bool(cfg["test_inference"]), missing=missing))
    return specs


def stage_summary_name(stage: str) -> str:
    return load_stage(stage)["name"]
