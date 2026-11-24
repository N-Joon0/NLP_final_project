#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
dakd cache-teacher-logits --task all ${ARGS[@]+"${ARGS[@]}"}
