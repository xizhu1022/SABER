#!/usr/bin/env bash
# 03: Generate K=3 chain-of-thought reasoning traces per side (PK and CK),
# then run a self-evaluation forward pass and read the conditional-layer
# hidden states. Combines vLLM generation (fast) with HF post-extraction
# of hidden states from the cached traces.
set -euo pipefail

BACKBONE="${1:?usage: $0 <backbone-short-name>}"
: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

mkdir -p "artifacts/$BACKBONE"

# Stage A: vLLM-backed multi-trace sampling (PK and CK sides, K=3 each)
python -m saber.extract.multipath_vllm_gen \
  --backbone "$BACKBONE" \
  --split-file "$SABER_DATA/splits/saber_split.json" \
  --datasets-dir "$SABER_DATA/datasets" \
  --out "artifacts/$BACKBONE/multipath_traces.jsonl" \
  --k 3 --temperature 0.9 --top-p 1.0 --rep-pen 1.05 --max-new 256

# Stage B: HF forward pass for the self-evaluation hidden states
python -m saber.extract.multipath_hf_postgen \
  --backbone "$BACKBONE" \
  --traces-jsonl "artifacts/$BACKBONE/multipath_traces.jsonl" \
  --out "artifacts/$BACKBONE/cond_traces.pt"
