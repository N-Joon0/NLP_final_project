from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def workspace() -> Path:
    return ROOT / "workspace"


def hf_home() -> Path:
    return workspace() / "hf_cache"


def configs_dir() -> Path:
    return ROOT / "configs"


def splits_path(task: str) -> Path:
    return ROOT / "data" / "splits" / f"{task}.json"


def selected_dir() -> Path:
    return configs_dir() / "selected"


def selected_path(task: str, name: str) -> Path:
    return selected_dir() / f"{task}_{name}.json"


def gate_path() -> Path:
    return selected_dir() / "gate.json"


def teacher_dir(task: str) -> Path:
    return workspace() / "teachers" / task


def teacher_logits_dir(task: str) -> Path:
    return workspace() / "teacher_logits" / task


def runs_root() -> Path:
    return workspace() / "runs"


def weights_root() -> Path:
    return ROOT / "weights"


def weights_dir(stage: str, run_id: str, condition: str | None = None, task: str | None = None) -> Path:
    if condition == "teacher":
        return weights_root() / "teachers" / task
    return weights_root() / "runs" / stage / run_id


def run_dir(stage: str, run_id: str, condition: str | None = None, task: str | None = None) -> Path:
    if condition == "teacher":
        return teacher_dir(task)
    return runs_root() / stage / run_id


def jobs_dir() -> Path:
    return workspace() / "jobs"


def logs_dir() -> Path:
    return workspace() / "logs"


def results_dir() -> Path:
    return workspace() / "results"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
