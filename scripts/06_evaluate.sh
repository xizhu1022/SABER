#!/usr/bin/env bash
# 06: Aggregate per-(method, dataset) metrics:
# - Main comparison: Acc, CF, KF, MFS
# - Selective answering (SABER abstention): Score, Coverage, R_C, F_1
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

mkdir -p "artifacts/$BACKBONE"

python -m saber.metrics.joint_metrics \
  --backbone "$BACKBONE" \
  --split-file "$SABER_DATA/splits/saber_split.json" \
  --behavior-dir "$SABER_DATA/evaluation/$BACKBONE" \
  --baselines-dir "artifacts/$BACKBONE/baselines" \
  --saber-decisions "artifacts/$BACKBONE/saber_decisions.jsonl" \
  --out "artifacts/$BACKBONE/metrics.json"

python -m saber.metrics.cell_distribution \
  --backbone "$BACKBONE" \
  --split-file "$SABER_DATA/splits/saber_split.json" \
  --behavior-dir "$SABER_DATA/evaluation/$BACKBONE" \
  --out "artifacts/$BACKBONE/cell_distribution.json"
