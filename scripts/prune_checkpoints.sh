#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
W="$(python -c 'from dakd import paths; print(paths.weights_root())')"
YES=0; [[ " ${ARGS[*]-} " == *" --yes "* ]] && YES=1
total=0
for st in s04 s06 s08; do
  [[ -d "$W/runs/$st" ]] || continue
  for d in "$W/runs/$st"/*/; do
    [[ -d "$d" ]] || continue
    sz=$(du -sm "$d" | cut -f1); total=$((total + sz))
    if [[ $YES -eq 1 ]]; then rm -rf "$d"; echo "삭제: $d (${sz}MB)"; else echo "대상: $d (${sz}MB)"; fi
  done
done
echo "합계 ${total}MB" $([[ $YES -eq 1 ]] && echo "삭제됨" || echo "(미삭제: --yes 를 붙이면 삭제)")
