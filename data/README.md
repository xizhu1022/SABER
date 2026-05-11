# Data

This directory holds a 100-instance-per-dataset sample of the constructed benchmark, plus the full benchmark split and the matching PK/CK behaviour labels for the sampled instances on all four backbones.

## Layout

```
data/
├── splits/saber_split.json     full 80/10/10 train/val/test split, qid lists only (~5 MB)
├── datasets/                    100 first rows per dataset (~1.7 MB total)
│   ├── confiqa-qa.jsonl
│   ├── confiqa-mr.jsonl
│   ├── confiqa-mc.jsonl
│   ├── conflictqa-popqa-llama2-7b.jsonl
│   ├── conflictbank-pilot10k.jsonl
│   ├── triviaqa-huang.jsonl
│   ├── nq-huang.jsonl
│   └── _index.json
└── evaluation/<backbone>/      PK/CK answer paths labelled against gold
    └── behavior_<dataset>.jsonl
```

## Reconstructing the full benchmark

`splits/saber_split.json` enumerates all ~69K qids per backbone used in the paper. **The full benchmark will be released upon publication.** In the meantime, the full version can be regenerated locally as follows:

1. Download the upstream datasets (see the top-level `README.md` table) into `datasets/`, keeping the file names below.
2. Run `bash scripts/01_label_pk_ck.sh <backbone>` to relabel PK/CK paths via the alias-match cascade.
3. Repeat for each of the four backbones.

Expected full-data sizes: ~645 MB (raw datasets) + ~54 MB (per-backbone labels). The split file stays at ~5 MB.

## Schema

### `datasets/<name>.jsonl`
One JSON object per line with fields:

| Field | Description |
|---|---|
| `qid` | Unique instance id (string) |
| `question` | The natural-language question |
| `ck_text` | The retrieved context passage |
| `ground_truth` | Gold answer (string or list of aliases) |
| `dataset` | Source dataset short name |
| `meta` | Source-specific metadata (optional) |

### `evaluation/<backbone>/behavior_<dataset>.jsonl`
One JSON object per line with fields:

| Field | Description |
|---|---|
| `qid` | Same as in the dataset file |
| `pk_answer` | Backbone's closed-book (PK-only) answer string |
| `ck_answer` | Backbone's context-conditioned (CK) answer string |
| `y_pk` | 1 if `pk_answer` alias-matches the gold, else 0 |
| `y_ck` | 1 if `ck_answer` alias-matches the gold, else 0 |

The four-cell reliability outcome is `(y_pk, y_ck) ∈ {(0,0), (0,1), (1,0), (1,1)}`.

### `splits/saber_split.json`
Top-level keys:

| Field | Description |
|---|---|
| `version` | Split version tag (`saber_split`) |
| `partition.<dataset>.train` / `.val` / `.test` | Lists of qids in each partition |
| `seed` | Random seed used to construct the split |
