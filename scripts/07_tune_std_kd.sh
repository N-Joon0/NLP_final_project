#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
run_stage s04
dakd select --stage s04 --task "$TASKSEL"
