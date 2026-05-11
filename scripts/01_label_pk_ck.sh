#!/usr/bin/env bash
# 01: Generate PK-only and CK-conditioned answer paths for each dataset and
# label them against the gold answer via the alias-match cascade.
# Output: $SABER_DATA/evaluation/<backbone>/behavior_<dataset>.jsonl
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

DATASETS=(confiqa-qa confiqa-mr confiqa-mc \
          conflictqa-popqa-llama2-7b conflictbank-pilot10k \
          triviaqa-huang nq-huang)

mkdir -p "$SABER_DATA/evaluation/$BACKBONE"

for ds in "${DATASETS[@]}"; do
  python -m saber.extract.label_pk_ck \
    --backbone "$BACKBONE" \
    --dataset "$ds" \
    --in-jsonl "$SABER_DATA/datasets/$ds.jsonl" \
    --out-jsonl "$SABER_DATA/evaluation/$BACKBONE/behavior_$ds.jsonl"
done
