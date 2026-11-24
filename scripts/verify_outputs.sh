#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
dakd verify-outputs ${ARGS[@]+"${ARGS[@]}"}
