#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
dakd check-weights ${ARGS[@]+"${ARGS[@]}"}
