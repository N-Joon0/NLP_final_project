#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

TEACHER_DIR="" RESULTS_DIR="" SETS=""
USAGE="사용: bash scripts/reproduce.sh --teacher <teacher 폴더> [--sets 0,1,2,3[:4,5,6,7]]     학습부터 재현
      bash scripts/reproduce.sh --results <results 폴더>                                  결과 재현 (분석·검증)
  (폴더는 Google Drive 에서 내려받은 teacher / results 폴더. 튜닝 값은 git clone 에 들어 있다.)
  학습 재현은 teacher 와 공개된 튜닝 값을 그대로 쓰고 s03~s10 만 다시 학습한다. 수 시간 걸린다."
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --teacher) TEACHER_DIR="${ARGS[i+1]:-}"; i=$((i+1)) ;;
    --results) RESULTS_DIR="${ARGS[i+1]:-}"; i=$((i+1)) ;;
    --sets) SETS="${ARGS[i+1]:-}"; i=$((i+1)) ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) echo "알 수 없는 옵션: ${ARGS[i]}"; echo "$USAGE"; exit 2 ;;
  esac
done
[[ -n "$TEACHER_DIR" || -n "$RESULTS_DIR" ]] || { echo "$USAGE"; exit 2; }
for d in "$TEACHER_DIR" "$RESULTS_DIR"; do
  [[ -z "$d" || -d "$d" ]] || { echo "폴더가 없습니다: $d"; exit 2; }
done

echo "[reproduce] 1/4 환경 설정 (도구 확인, .venv, 패키지)"
bash scripts/bootstrap.sh
source .venv/bin/activate

echo "[reproduce] 2/4 데이터·모델 준비"
out="$(bash scripts/01_prepare_data.sh 2>&1)" || { echo "$out"; echo "[reproduce] 중단: 01_prepare_data 실패"; exit 1; }
echo "$out" | grep -E "분할" || true
if echo "$out" | grep -q "created" || [[ "$(echo "$out" | grep -c "분할 verified")" -ne 2 ]]; then
  echo "[reproduce] 중단: 분할이 'verified' 가 아닙니다 (data/splits 를 바꾸면 안 됩니다)."
  exit 1
fi
bash scripts/02_download_models.sh > /dev/null 2>&1 || { echo "[reproduce] 중단: 02_download_models 실패"; exit 1; }

echo "[reproduce] 3/4 내려받은 꾸러미를 제 위치에 배치"
for d in "$TEACHER_DIR" "$RESULTS_DIR"; do
  [[ -z "$d" ]] || bash scripts/receive_shared.sh "$d"
done

if [[ -n "$RESULTS_DIR" ]]; then
  echo "[reproduce] 4/4 결과 재현: 분석 다시 생성 + 산출물 검증"
  bash scripts/99_analyze.sh
  bash scripts/verify_outputs.sh
  echo "[reproduce] 완료. 결과: workspace/results/RESULTS.md, tables/, figures/"
else
  echo "[reproduce] 4/4 학습부터 재현 (공개된 teacher·튜닝 값 사용)"
  bash scripts/check_nccl.sh
  if [[ -n "$SETS" ]]; then
    bash scripts/run_all.sh --skip-tuning --sets "$SETS"
  else
    bash scripts/run_all.sh --skip-tuning --sets 0,1,2,3
  fi
fi
