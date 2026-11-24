#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then source "$ROOT/.venv/bin/activate"; fi

ARGS=("$@")

TASKSEL="all"
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --only-task|--task) TASKSEL="${ARGS[i+1]:-all}" ;;
    --only-task=*|--task=*) TASKSEL="${ARGS[i]#*=}" ;;
  esac
done

dakd() { python -m dakd "$@"; }
run_stage() { local st="$1"; shift; dakd run-stage --stage "$st" "$@" ${ARGS[@]+"${ARGS[@]}"}; }
