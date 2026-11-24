#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"

GPUS="0,1,2,3"
for ((i=0; i<${#ARGS[@]}; i++)); do
  [[ "${ARGS[i]}" == "--gpus" ]] && GPUS="${ARGS[i+1]:-$GPUS}"
done
N=$(awk -F, '{print NF}' <<<"$GPUS")
EXPECT=$(( N * (N + 1) / 2 ))

echo "[nccl] GPU $GPUS ($N장) all_reduce 점검, 기대값 ${EXPECT}.0"
out=$(CUDA_VISIBLE_DEVICES="$GPUS" timeout 180 python -m torch.distributed.run --standalone --nproc_per_node="$N" scripts/nccl_test.py 2>&1) || rc=$?
echo "$out" | grep -E "^rank |Error|error|NCCL" || true
ok=$(grep -o "all_reduce=${EXPECT}.0" <<<"$out" | wc -l)
if [[ "${rc:-0}" -eq 0 && "$ok" -eq "$N" ]]; then
  echo "[nccl] PASS: ${N}개 rank 모두 ${EXPECT}.0"
else
  echo "[nccl] FAIL (통과 rank ${ok}/${N}, 종료코드 ${rc:-0})"
  echo "  - 멈추거나 타임아웃이면 컨테이너를 --shm-size=32g 로 만들었는지 확인 (df -h /dev/shm)"
  echo "  - 임시 우회: NCCL_SHM_DISABLE=1 (성능이 떨어지니 원인 해결 전까지만 쓰고 팀에 공유)"
  exit 1
fi
