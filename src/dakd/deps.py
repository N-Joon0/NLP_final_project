from __future__ import annotations

from . import paths, weights
from .config import load_stage


def _check_one(dep, task: str) -> str | None:
    if dep == "splits":
        if not paths.splits_path(task).exists():
            return (f"[{task}] 분할 파일 없음 ({paths.splits_path(task).relative_to(paths.ROOT)}) → "
                    f"`bash scripts/01_prepare_data.sh` 실행 또는 git pull")
    elif dep == "pretrained":
        from .config import load_task
        ck = load_task(task)["checkpoint"]
        hub = paths.hf_home() / "hub" / ("models--" + ck.replace("/", "--"))
        fb = load_task(task).get("checkpoint_fallback")
        fbh = paths.hf_home() / "hub" / ("models--" + fb.replace("/", "--")) if fb else None
        if not hub.exists() and not (fbh and fbh.exists()):
            return f"[{task}] 사전학습 모델 캐시 없음 ({ck}) → `bash scripts/02_download_models.sh`"
    elif dep == "teacher":
        ok, why = weights.check(paths.weights_dir("s01", "", "teacher", task))
        if not ok:
            return (f"[{task}] teacher 가중치가 없습니다 ({paths.weights_dir('s01', '', 'teacher', task)}: {why}). "
                    f"`bash scripts/03_train_teacher.sh` 를 실행하거나 공유받은 teacher 가중치를 "
                    f"`bash scripts/receive_shared.sh <공유폴더>` 로 {paths.weights_root()}/teachers/{task}/ 에 넣으세요.")
    elif dep == "teacher_logits":
        if not (paths.teacher_logits_dir(task) / "DONE").exists():
            return (f"[{task}] teacher logits 가 없습니다. `bash scripts/04_cache_teacher_logits.sh` 를 실행하거나 "
                    f"공유받은 teacher 폴더를 `bash scripts/receive_shared.sh <공유폴더>` 로 넣으세요.")
    elif dep == "gate":
        if not paths.gate_path().exists():
            return "Gate 결과(configs/selected/gate.json)가 없습니다 → `bash scripts/06_gate_check.sh` 또는 git pull"
    elif isinstance(dep, dict) and "selected" in dep:
        name = dep["selected"]
        if not paths.selected_path(task, name).exists():
            tune = {"std_kd": "07_tune_std_kd", "conf_kd": "09_tune_conf_kd", "ours": "11_tune_ours"}[name]
            return (f"[{task}] 선택값 없음: {paths.selected_path(task, name).name} → "
                    f"`bash scripts/{tune}.sh` 실행 또는 팀원이 커밋한 configs/selected/*.json 을 git pull")
    else:
        raise ValueError(f"알 수 없는 의존성 키: {dep}")
    return None


def check_stage(stage: str, tasks: list[str]) -> list[str]:
    cfg = load_stage(stage)
    msgs: list[str] = []
    for task in tasks:
        if task not in cfg["tasks"]:
            continue
        for dep in cfg.get("requires", []):
            if dep == "gate":
                m = _check_one(dep, task)
                if m and m not in msgs:
                    msgs.append(m)
                continue
            m = _check_one(dep, task)
            if m:
                msgs.append(m)
    return msgs
