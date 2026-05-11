#!/usr/bin/env bash
# 04: Train the two MLP predictors of SABER (PK-side and CK-side), each over
# [self-prior, source-conditional state]; then apply the 4-cell decision
# rule and write per-instance decisions.
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

mkdir -p "artifacts/$BACKBONE"

python -m saber.methods.saber_probe \
  --backbone "$BACKBONE" \
  --split-file "$SABER_DATA/splits/saber_split.json" \
  --behavior-dir "$SABER_DATA/evaluation/$BACKBONE" \
  --self-prior "artifacts/$BACKBONE/self_prior.pt" \
  --cond "artifacts/$BACKBONE/cond_traces.pt" \
  --out-probe "artifacts/$BACKBONE/saber_probe.pt" \
  --out-decisions "artifacts/$BACKBONE/saber_decisions.jsonl" \
  --tau 0.5
