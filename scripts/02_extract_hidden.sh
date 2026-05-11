#!/usr/bin/env bash
# 02: Extract the query-only self-prior representation from the frozen backbone.
# Output: artifacts/<backbone>/self_prior.pt
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

mkdir -p "artifacts/$BACKBONE"

python -m saber.extract.extract_hidden \
  --backbone "$BACKBONE" \
  --split-file "$SABER_DATA/splits/saber_split.json" \
  --datasets-dir "$SABER_DATA/datasets" \
  --out "artifacts/$BACKBONE/self_prior.pt"
