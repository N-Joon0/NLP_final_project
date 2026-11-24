#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_SETUP=1 MIN_DISK_GB=100
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-setup) RUN_SETUP=0; shift ;;
    --min-disk-gb) MIN_DISK_GB="${2:?--min-disk-gb 값 필요}"; shift 2 ;;
    -h|--help)
      echo "사용: bash scripts/bootstrap.sh [--no-setup] [--min-disk-gb 100]"
      exit 0 ;;
    *) echo "알 수 없는 옵션: $1"; exit 2 ;;
  esac
done

WARN=0
warn() { echo "  [경고] $*"; WARN=1; }

echo "== 1/3 시스템 패키지 =="
NEED=()
command -v git >/dev/null || NEED+=(git)
command -v curl >/dev/null || NEED+=(curl)
command -v tmux >/dev/null || NEED+=(tmux)
command -v python3 >/dev/null || NEED+=(python3)
python3 -c 'import venv, ensurepip' 2>/dev/null || NEED+=(python3-venv python3-pip)
if [[ ${#NEED[@]} -gt 0 ]]; then
  echo "  설치 필요: ${NEED[*]}"
  SUDO=""
  if [[ $EUID -ne 0 ]]; then
    command -v sudo >/dev/null || { echo "  root 도 sudo 도 없습니다. 서버 관리자에게 위 패키지 설치를 요청하세요."; exit 1; }
    SUDO="sudo"
  fi
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends "${NEED[@]}" >/dev/null
else
  echo "  필요한 도구가 이미 설치되어 있습니다 (apt 건너뜀)"
fi
python3 -c 'import sys; assert sys.version_info >= (3,10), f"Python 3.10+ 필요 (현재 {sys.version.split()[0]})"'
echo "  python $(python3 -c 'import sys; print(sys.version.split()[0])'), git $(git --version | cut -d' ' -f3), tmux $(tmux -V | cut -d' ' -f2)"

echo "== 2/3 GPU / 공유메모리 / 디스크 =="
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "  [오류] GPU 를 볼 수 없습니다. 컨테이너를 --gpus all 로 실행했는지 호스트에서 확인하세요."
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
NGPU=$(nvidia-smi -L | wc -l)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  드라이버 $DRIVER, GPU ${NGPU}장"
[[ $NGPU -ge 4 ]] || warn "GPU 가 4장 미만입니다. 모든 학습 run 은 4장(DDP)이 필요합니다."
if [[ "${DRIVER%%.*}" -lt 580 ]]; then
  warn "드라이버가 580 미만이라 기본 torch(CUDA 13) 가 GPU 를 못 볼 수 있습니다. 00_setup_env.sh 후 CUDA 가 False 면: "
  echo "         rm -rf .venv && TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 DAKD_IGNORE_LOCK=1 bash scripts/00_setup_env.sh"
fi
SHM_MB=$(df -m /dev/shm | awk 'NR==2{print $2}')
[[ $SHM_MB -ge 16000 ]] || warn "/dev/shm 이 ${SHM_MB}MB 입니다. 컨테이너를 --shm-size=32g 로 다시 만드세요 (NCCL/DataLoader 필요)."
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "  저장소 파티션 여유 ${AVAIL_GB}GB ($(df --output=source "$ROOT" | tail -1)), 공유메모리 ${SHM_MB}MB"
[[ $AVAIL_GB -ge $MIN_DISK_GB ]] || warn "여유 공간이 ${MIN_DISK_GB}GB 미만입니다. 가중치(약 30GB)·logits·로그·.venv 를 모두 저장소 안에 두므로 큰 디스크를 마운트하세요."
[[ "$(df --output=target "$ROOT" | tail -1)" != "/" ]] || ! mount | grep -q " on / type overlay" || warn "저장소가 컨테이너 루트(overlay) 위에 있습니다. 컨테이너를 지우면 결과가 사라집니다. 호스트 디스크를 마운트한 경로에서 작업하세요."

echo "== 3/3 파이썬 환경 =="
if [[ $RUN_SETUP -eq 1 ]]; then
  bash scripts/00_setup_env.sh
else
  echo "  (--no-setup: 건너뜀) 직접 실행: bash scripts/00_setup_env.sh"
fi

echo
[[ $WARN -eq 0 ]] && echo "[bootstrap] 완료. 다음: bash scripts/check_nccl.sh" || echo "[bootstrap] 완료 (경고 있음 — 위 [경고] 를 확인하세요). 다음: bash scripts/check_nccl.sh"
