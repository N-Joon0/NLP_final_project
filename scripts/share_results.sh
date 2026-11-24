#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

FINAL_STAGES="s03,s05,s07,s09,s10"
TEACHER=0 RESULTS=0
USAGE="사용: bash scripts/share_results.sh [--teacher] [--results]   (둘 다 생략하면 둘 다 만든다)
  --teacher : teacher 가중치 + teacher logits            → workspace/share/teacher/
  --results : 학습된 student 가중치 + 전체 결과·분석      → workspace/share/results/"
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in
    --teacher) TEACHER=1 ;;
    --results) RESULTS=1 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) echo "알 수 없는 옵션: $a"; echo "$USAGE"; exit 2 ;;
  esac
done
[[ $TEACHER -eq 1 || $RESULTS -eq 1 ]] || { TEACHER=1; RESULTS=1; }

pack_weights() {
  local out="$1"; shift
  tar -cf - -C weights "$@" | gzip -1 > "$out"
  (cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
  echo "  $(du -h "$out" | cut -f1)  $out"
}

make_teacher() {
  local out="workspace/share/teacher" rc=0 t
  echo "[share] teacher 꾸러미: 검증"
  dakd verify-outputs --stage s01 || rc=1
  dakd check-weights --stage s01 --verify-sha || rc=1
  for t in sst2 ynat; do
    [[ -f workspace/teacher_logits/$t/DONE ]] || { echo "[FAIL] teacher logits 없음: workspace/teacher_logits/$t"; rc=1; }
  done
  [[ $rc -eq 0 ]] || { echo "[share] teacher 검증 실패 — 꾸러미를 만들지 않았습니다."; return 1; }
  rm -rf "$out"; mkdir -p "$out/weights" "$out/results/teacher_logits" "$out/results/teachers"
  pack_weights "$out/weights/teachers.tar.gz" teachers/sst2 teachers/ynat
  for t in sst2 ynat; do
    cp -r "workspace/teacher_logits/$t" "$out/results/teacher_logits/"
    cp -r "workspace/teachers/$t" "$out/results/teachers/"
  done
  echo "[share] teacher 꾸러미 완료 → $ROOT/$out"
}

make_results() {
  local out="workspace/share/results" rc=0 st n
  echo "[share] results 꾸러미: 검증"
  dakd verify-outputs || rc=1
  dakd check-weights --stage "$FINAL_STAGES" --verify-sha || rc=1
  [[ $rc -eq 0 ]] || { echo "[share] results 검증 실패 — 꾸러미를 만들지 않았습니다."; return 1; }
  rm -rf "$out"; mkdir -p "$out/weights" "$out/results" "$out/selected"
  for st in ${FINAL_STAGES//,/ }; do
    n=$(ls -d weights/runs/$st/*/ 2>/dev/null | wc -l)
    [[ $n -gt 0 ]] || { echo "  (건너뜀) $st: 가중치 없음"; continue; }
    pack_weights "$out/weights/$st.tar.gz" "runs/$st"
  done
  cp -r workspace/runs workspace/teachers workspace/teacher_logits workspace/results "$out/results/"
  cp -r configs/selected/. "$out/selected/" 2>/dev/null || true
  cp configs/base.yaml "$out/selected/base.yaml"
  echo "[share] results 꾸러미 완료 → $ROOT/$out ($(du -sh "$out/results" | cut -f1) 결과)"
}

RC=0
[[ $TEACHER -eq 0 ]] || make_teacher || RC=1
[[ $RESULTS -eq 0 ]] || make_results || RC=1

echo
if [[ $RC -eq 0 ]]; then
  echo "[share] 완료. 폴더를 통째로 Google Drive 공유 폴더에 올리세요 (weights/ 안은 압축 파일, results/ 안은 그대로):"
  [[ $TEACHER -eq 0 ]] || echo "  $ROOT/workspace/share/teacher   →   dakd_shared/teacher/"
  [[ $RESULTS -eq 0 ]] || echo "  $ROOT/workspace/share/results   →   dakd_shared/results/"
  exit 0
fi
echo "[share] 일부 실패 — 위 [FAIL] 항목을 먼저 해결하세요."
exit 1
