from __future__ import annotations

import json
import shutil
import time

from . import paths
from .config import all_tasks, load_base, set_n_layers
from .registry import expand_stage

THRESHOLD_PP = 1.0
FALLBACK_LAYERS = 4


def run_gate() -> int:
    base = load_base()
    tune_seed = base["tune_seed"]
    rows = {}
    for task in all_tasks():
        tm = paths.teacher_dir(task) / "train_metrics.json"
        s = [x for x in expand_stage("s03", [task]) if x.budget == 2000 and x.seed == tune_seed]
        if not (paths.teacher_dir(task) / "DONE_TRAIN").exists() or not s or not (s[0].run_dir / "DONE_TRAIN").exists():
            print(f"[gate] {task}: teacher(s01) 와 student_ft 2k seed {tune_seed}(s03) 가 먼저 필요합니다 "
                  f"(03, 05 단계).")
            return 1
        t = json.loads(tm.read_text())["best_dev"]
        st = json.loads((s[0].run_dir / "train_metrics.json").read_text())["best_dev"]
        rows[task] = {"teacher_dev": t, "student_ft_2k_dev": st, "gap_pp": (t - st) * 100}
    n_layers = base["n_layers"]
    small = [t for t, r in rows.items() if r["gap_pp"] < THRESHOLD_PP]
    for t, r in rows.items():
        print(f"[gate] {t}: teacher {r['teacher_dev']:.4f} / student_ft(2k) {r['student_ft_2k_dev']:.4f} "
              f"→ gap {r['gap_pp']:.2f}%p")
    result = {"created": time.strftime("%F %T"), "threshold_pp": THRESHOLD_PP, "n_layers": n_layers,
              "tasks": rows}

    if small and n_layers != FALLBACK_LAYERS:
        set_n_layers(FALLBACK_LAYERS)
        arch = paths.runs_root() / "s03"
        if arch.exists():
            shutil.move(str(arch), str(arch.with_name(f"s03_nlayers{n_layers}_gap_too_small")))
        print(f"\n[gate] 차이가 {THRESHOLD_PP}%p 미만인 task: {small} → n_layers={FALLBACK_LAYERS} 로 변경했습니다 "
              f"(configs/base.yaml 갱신, 기존 s03 결과는 runs/s03_nlayers{n_layers}_gap_too_small 로 이동).\n"
              f"  → `bash scripts/05_train_student_ft.sh` 를 다시 실행한 뒤 `bash scripts/06_gate_check.sh` 를 한 번 더 실행하세요.\n"
              f"  (변경된 configs/base.yaml 을 팀원과 공유하려면 커밋하세요. gate.json 은 재실행 후에 생성됩니다.)")
        return 2
    if small:
        result["note"] = f"n_layers={n_layers} 에서도 gap<{THRESHOLD_PP}%p 이나 이미 {FALLBACK_LAYERS} 로 낮춘 상태라 진행"
    path = paths.gate_path()
    paths.ensure(path.parent)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[gate] 통과/확정 → {path}\n★ 이 파일을 git 에 커밋하세요.")
    return 0
