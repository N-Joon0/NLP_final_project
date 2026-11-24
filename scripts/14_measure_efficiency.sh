#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
dakd efficiency --task all ${ARGS[@]+"${ARGS[@]}"}
