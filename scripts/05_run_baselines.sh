#!/usr/bin/env bash
# 05: Run the prompt-based baselines bundled in `saber.baselines.run`.
# These read the same labelled behaviour files produced by
# `scripts/01_label_pk_ck.sh` and emit per-method answer JSONLs that
# `06_evaluate.sh` then scores.
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

mkdir -p "artifacts/$BACKBONE/baselines"

for method in closed_book dia internal_eval context_eval implicit_scr explicit_scr; do
  python -m saber.baselines.run \
    --backbone "$BACKBONE" \
    --method "$method" \
    --split-file "$SABER_DATA/splits/saber_split.json" \
    --datasets-dir "$SABER_DATA/datasets" \
    --behavior-dir "$SABER_DATA/evaluation/$BACKBONE" \
    --out-dir "artifacts/$BACKBONE/baselines/$method"
done
