#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_stage s08
dakd select --stage s08 --task "$TASKSEL"
