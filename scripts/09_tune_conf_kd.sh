#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_stage s06
dakd select --stage s06 --task "$TASKSEL"
