#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
dakd download-models --task all ${ARGS[@]+"${ARGS[@]}"}
