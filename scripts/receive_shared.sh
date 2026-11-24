#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

[[ ${#ARGS[@]} -eq 1 && -d "${ARGS[0]}" ]] || { echo "사용: bash scripts/receive_shared.sh <내려받은 공유 폴더>   (예: ~/Downloads/teacher, ~/Downloads/results)"; exit 2; }
SRC="$(cd "${ARGS[0]}" && pwd)"
[[ -d "$SRC/weights" || -d "$SRC/results" ]] || { echo "weights/ 또는 results/ 폴더가 없습니다: $SRC"; exit 2; }

if [[ -d "$SRC/weights" ]]; then
  echo "[receive] 가중치 압축 해제 → weights/"
  for f in "$SRC"/weights/*.tar.gz; do
    [[ -f "$f" ]] || continue
    if [[ -f "$f.sha256" ]]; then
      (cd "$(dirname "$f")" && sha256sum -c --quiet "$(basename "$f").sha256") || { echo "[FAIL] sha256 불일치(다운로드가 깨졌을 수 있음): $f"; exit 1; }
    fi
    tar -xzf "$f" -C weights --skip-old-files
    echo "  OK  $(basename "$f")"
  done
fi

if [[ -d "$SRC/results" ]]; then
  echo "[receive] 결과 복사 → workspace/ (이미 있는 파일은 덮어쓰지 않음)"
  mkdir -p workspace
  cp -rn "$SRC/results/." workspace/
  echo "  OK  $(ls "$SRC/results" | tr '\n' ' ')"
fi

echo "[receive] 가중치 점검"
dakd check-weights --verify-sha
echo "[receive] 완료"
