#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

SETS=("0,1,2,3" "4,5,6,7")
TASKS=(sst2 ynat)
SKIP_EFFICIENCY=0
SKIP_TUNING=0
USAGE="사용: bash scripts/run_all.sh [--sets 0,1,2,3:4,5,6,7] [--skip-efficiency] [--skip-tuning]
  GPU 4장 세트 2개(기본 0,1,2,3 / 4,5,6,7)로 sst2·ynat 를 동시에 진행한다.
  세트를 하나만 주면(--sets 0,1,2,3) 두 task 를 차례로 진행한다.
  --skip-tuning: teacher·gate·튜닝을 건너뛰고 공개된 값(configs/selected/, 받아 둔 teacher)으로 s03~s10 만 학습한다 (재현용)."
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --sets) IFS=: read -ra SETS <<<"${ARGS[i+1]:-}"; i=$((i+1)) ;;
    --skip-efficiency) SKIP_EFFICIENCY=1 ;;
    --skip-tuning) SKIP_TUNING=1 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) echo "알 수 없는 옵션: ${ARGS[i]}"; echo "$USAGE"; exit 2 ;;
  esac
done
[[ ${#SETS[@]} -ge 1 && ${#SETS[@]} -le 2 ]] || { echo "$USAGE"; exit 2; }
for g in "${SETS[@]}"; do
  [[ "$(awk -F, '{print NF}' <<<"$g")" -eq 4 ]] || { echo "GPU 세트는 정확히 4장이어야 합니다: '$g' (예: 0,1,2,3)"; exit 2; }
done

mkdir -p workspace/logs
exec > >(tee -a workspace/logs/run_all.log) 2>&1
RCDIR="$(mktemp -d)"
trap 'rm -rf "$RCDIR"' EXIT

log() { echo "[$(date +%T)] $*"; }
step() { local t="$1"; shift; log "[$t] 시작: ${*:2}"; "$@"; local rc=$?; log "[$t] 끝 (종료코드 $rc)"; return $rc; }

run_phase() {
  local fn="$1" fail=0 i t
  if [[ ${#SETS[@]} -ge 2 ]]; then
    local pids=()
    for i in 0 1; do
      t="${TASKS[i]}"
      ( set +e; "$fn" "$t" "${SETS[i]}"; echo $? > "$RCDIR/$fn.$t" ) 2>&1 | sed -u "s/^/[$t] /" &
      pids+=($!)
    done
    wait "${pids[@]}"
  else
    for t in "${TASKS[@]}"; do
      ( set +e; "$fn" "$t" "${SETS[0]}"; echo $? > "$RCDIR/$fn.$t" ) 2>&1 | sed -u "s/^/[$t] /"
    done
  fi
  for t in "${TASKS[@]}"; do
    [[ "$(cat "$RCDIR/$fn.$t" 2>/dev/null || echo 1)" == "0" ]] || { log "[$t] $fn 실패"; fail=1; }
  done
  return $fail
}

phase_base() {
  local t="$1" g="$2"
  step "$t" dakd run-stage --stage s01 --only-task "$t" --gpus "$g" || return 1
  step "$t" dakd cache-teacher-logits --task "$t" --gpu "${g%%,*}" || return 1
  step "$t" dakd run-stage --stage s03 --only-task "$t" --only-budget 2000 --only-seed 42 --gpus "$g" || return 1
}

phase_gate_rerun() {
  local t="$1" g="$2"
  step "$t" dakd run-stage --stage s03 --only-task "$t" --only-budget 2000 --only-seed 42 --gpus "$g" || return 1
}

phase_main() {
  local t="$1" g="$2"
  step "$t" dakd run-stage --stage s03 --only-task "$t" --gpus "$g" || return 1
  if [[ $SKIP_TUNING -eq 0 ]]; then
    step "$t" dakd run-stage --stage s04 --only-task "$t" --gpus "$g" || return 1
    step "$t" dakd select --stage s04 --task "$t" || return 1
  fi
  step "$t" dakd run-stage --stage s05 --only-task "$t" --gpus "$g" || return 1
  if [[ $SKIP_TUNING -eq 0 ]]; then
    step "$t" dakd run-stage --stage s06 --only-task "$t" --gpus "$g" || return 1
    step "$t" dakd select --stage s06 --task "$t" || return 1
  fi
  step "$t" dakd run-stage --stage s07 --only-task "$t" --gpus "$g" || return 1
  if [[ $SKIP_TUNING -eq 0 ]]; then
    step "$t" dakd run-stage --stage s08 --only-task "$t" --gpus "$g" || return 1
    step "$t" dakd select --stage s08 --task "$t" || return 1
  fi
  step "$t" dakd run-stage --stage s09 --only-task "$t" --gpus "$g" || return 1
  step "$t" dakd run-stage --stage s10 --only-task "$t" --gpus "$g" || return 1
}

log "[all] GPU 세트: ${SETS[*]} / task: ${TASKS[*]}"

log "[all] 0/6 데이터·모델 준비"
out="$(bash scripts/01_prepare_data.sh 2>&1)" || { echo "$out"; echo "[all] 중단: 01_prepare_data 실패"; exit 1; }
echo "$out" | grep -E "분할" || true
if echo "$out" | grep -q "created" || [[ "$(echo "$out" | grep -c "분할 verified")" -ne 2 ]]; then
  echo "[all] 중단: 분할이 'verified' 가 아닙니다 (data/splits 를 바꾸면 안 됩니다)."
  exit 1
fi
bash scripts/02_download_models.sh > /dev/null 2>&1 || { echo "[all] 중단: 02_download_models 실패"; exit 1; }

log "[all] 1/6 GPU 세트별 NCCL 점검"
for g in "${SETS[@]}"; do
  bash scripts/check_nccl.sh --gpus "$g" | tail -1
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[all] 중단: NCCL 점검 실패 (GPU $g)"; exit 1; }
done

if [[ $SKIP_TUNING -eq 1 ]]; then
  log "[all] 2~3/6 공개된 teacher·튜닝 값·gate 사용 (teacher, gate, 튜닝 단계 건너뜀)"
  python - <<'EOF' || { echo "[all] 중단: 재현에 필요한 값이 없습니다."; exit 1; }
import json
import sys

from dakd import paths
from dakd.config import all_tasks, load_base

errs = []
for t in all_tasks():
    if not (paths.teacher_logits_dir(t) / "DONE").exists():
        errs.append(f"teacher logits 없음: {paths.teacher_logits_dir(t)} → 공유받은 teacher 꾸러미를 scripts/receive_shared.sh 로 먼저 넣으세요")
    for name in ("std_kd", "conf_kd", "ours"):
        if not paths.selected_path(t, name).exists():
            errs.append(f"튜닝 값 없음: {paths.selected_path(t, name).relative_to(paths.ROOT)} → git pull 로 최신 커밋을 받으세요")
if not paths.gate_path().exists():
    errs.append("gate 결과 없음: configs/selected/gate.json → git pull")
else:
    n_gate, n_base = json.loads(paths.gate_path().read_text())["n_layers"], load_base()["n_layers"]
    if n_gate != n_base:
        errs.append(f"n_layers 불일치: gate.json={n_gate}, configs/base.yaml={n_base} → git pull 로 최신 커밋을 받으세요")
for e in errs:
    print("  -", e)
sys.exit(1 if errs else 0)
EOF
else
log "[all] 2/6 teacher → teacher logits → gate 용 student FT (2k, seed 42)"
run_phase phase_base || { echo "[all] 중단: 기반 단계 실패 — 로그(workspace/logs)를 확인하고 같은 명령으로 다시 실행하세요."; exit 1; }

log "[all] 3/6 gate"
set +e; dakd gate; rc=$?; set -e
if [[ $rc -eq 2 ]]; then
  log "[all] gate: n_layers 가 바뀌어 gate 용 s03 을 다시 실행합니다"
  run_phase phase_gate_rerun || { echo "[all] 중단: gate 용 s03 재실행 실패"; exit 1; }
  dakd gate || { echo "[all] 중단: gate 재확인 실패"; exit 1; }
elif [[ $rc -ne 0 ]]; then
  echo "[all] 중단: gate 실패"; exit $rc
fi
fi

log "[all] 4/6 student FT 전체 → 튜닝 → 본 실험 (s03~s10)"
MAIN_RC=0
run_phase phase_main || MAIN_RC=1

log "[all] 5/6 효율 측정 (빈 GPU)"
if [[ $SKIP_EFFICIENCY -eq 0 && $MAIN_RC -eq 0 ]]; then
  dakd efficiency --task all --gpu "${SETS[0]%%,*}" || MAIN_RC=1
else
  echo "  (건너뜀)"
fi

log "[all] 6/6 결과 종합·검증"
dakd analyze || MAIN_RC=1
dakd verify-outputs || MAIN_RC=1
dakd status

python - <<'EOF'
import json
from dakd import paths
print("\n[all] teacher 학습 시간")
for t in ("sst2", "ynat"):
    f = paths.teacher_dir(t) / "train_metrics.json"
    if f.exists():
        m = json.loads(f.read_text())
        ep = [round(e["epoch_time_s"], 1) for e in m["epochs"]]
        print(f"  {t}: epoch_time_s={ep} train_time_s={m['train_time_s']:.0f} peak_mem_mb={m['peak_mem_mb']:.0f} samples/s={m['train_samples_per_s']:.1f}")
EOF

echo
log "[all] 총 소요 $((SECONDS / 3600))시간 $((SECONDS % 3600 / 60))분"
if [[ $MAIN_RC -eq 0 ]]; then
  echo "[all] 완료. 결과: workspace/results/RESULTS.md · 공유용 정리: bash scripts/share_results.sh"
else
  echo "[all] 일부 실패 — workspace/logs/FAILED_RUNS.md 와 위 출력을 확인하고 같은 명령으로 다시 실행하면 완료된 run 은 건너뛰고 이어서 진행합니다."
  exit 1
fi
