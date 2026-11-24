#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYBIN="${PYTHON:-python3}"
"$PYBIN" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ 필요"'
if [[ ! -d .venv ]]; then echo "[setup] .venv 생성"; "$PYBIN" -m venv .venv; fi
source .venv/bin/activate
PY=python

if ! $PY -c "import dakd, torch, transformers" 2>/dev/null; then
  echo "[setup] 패키지 설치"
  $PY -m pip install -q --upgrade pip
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then $PY -m pip install -q torch --index-url "$TORCH_INDEX_URL"; fi
  if [[ -f requirements.lock && -z "${DAKD_IGNORE_LOCK:-}" ]]; then
    $PY -m pip install -q -r requirements.lock || { echo "[setup] lock 설치 실패(Python 버전 차이 등) → requirements.txt 로 대체 설치"; $PY -m pip install -q -r requirements.txt; }
  else
    $PY -m pip install -q -r requirements.txt
  fi
  $PY -m pip install -q -e .
else
  echo "[setup] 이미 설치됨 (건너뜀)"
fi

[[ -f requirements.lock ]] || { $PY -m pip freeze --exclude-editable > requirements.lock; echo "[setup] requirements.lock 생성"; }

$PY - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory/2**30:.0f}GB")
    print("  bf16 지원:", torch.cuda.is_bf16_supported(), "→ 정밀도:", "bf16" if torch.cuda.is_bf16_supported() else "fp16")
else:
    print("  [경고] GPU 가 보이지 않습니다. 학습 단계는 GPU 가 필요합니다.")
EOF
$PY -m dakd --help >/dev/null && echo "[setup] OK: python -m dakd --help"
