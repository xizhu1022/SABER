#!/usr/bin/env bash
# 00_smoke.sh -- repo health check that needs no GPU and no model weights.
#
# Runs in well under a minute. Verifies:
#   1. The `saber` package imports cleanly.
#   2. The bundled sample data (700 qids) parses as JSONL and contains the
#      expected fields.
#   3. The alias-match cascade produces the expected verdict on the
#      sampled instances against the ground truth.
#   4. The unit tests pass.
#   5. Each entry point referenced from scripts/01-06 accepts `--help`.
set -euo pipefail

: "${SABER_DATA:=$(pwd)/data}"
export PYTHONPATH=src

echo "==============================================================="
echo " SABER repo smoke test"
echo "==============================================================="
echo "SABER_DATA=$SABER_DATA"
echo ""

echo "[1/5] Imports"
python -c "
import importlib
mods = [
    'saber', 'saber.config',
    'saber.data.alias_match', 'saber.data.schema', 'saber.data.build_split',
    'saber.data.builders.confiqa', 'saber.data.builders.conflictbank',
    'saber.data.builders.conflictqa_popqa', 'saber.data.builders.huang_situated',
    'saber.models.registry',
    'saber.methods.probe_utils', 'saber.methods.saber_probe',
    'saber.metrics', 'saber.metrics.cell_counts',
    'saber.metrics.probe_helpers', 'saber.metrics.joint_metrics',
    'saber.metrics.cell_distribution', 'saber.metrics.score_with_judge',
    'saber.baselines.prompts', 'saber.baselines.evaluate',
    'saber.baselines.runners', 'saber.baselines.run',
    'saber.prompts',
]
for m in mods:
    importlib.import_module(m)
print(f'  {len(mods)} modules imported OK')
"

echo ""
echo "[2/5] Sample data parses"
python -c "
import json
from pathlib import Path
root = Path('$SABER_DATA')
datasets = ['confiqa-qa','confiqa-mr','confiqa-mc','conflictqa-popqa-llama2-7b',
            'conflictbank-pilot10k','triviaqa-huang','nq-huang']
required = {'qid','dataset','question','ck_text','ground_truth'}
total = 0
for ds in datasets:
    p = root / 'datasets' / f'{ds}.jsonl'
    n = 0
    for line in p.open():
        r = json.loads(line)
        missing = required - r.keys()
        assert not missing, f'{ds} row missing fields: {missing}'
        n += 1
    total += n
    print(f'  {ds:<35} {n:>4} rows')
print(f'  total: {total} rows')
assert total == 700, f'expected 700 sampled rows, got {total}'
"

echo ""
echo "[3/5] Behaviour labels align with the sampled qids"
python -c "
import json
from pathlib import Path
root = Path('$SABER_DATA')
backbones = ['llama-3.1-8b-instruct','llama-3.2-3b-instruct',
             'qwen2.5-7b-instruct','qwen2.5-3b-instruct']
datasets = ['confiqa-qa','confiqa-mr','confiqa-mc','conflictqa-popqa-llama2-7b',
            'conflictbank-pilot10k','triviaqa-huang','nq-huang']
sample_qids = set()
for ds in datasets:
    for line in (root / 'datasets' / f'{ds}.jsonl').open():
        sample_qids.add(json.loads(line)['qid'])
for bb in backbones:
    total = 0
    for ds in datasets:
        p = root / 'evaluation' / bb / f'behavior_{ds}.jsonl'
        n = 0
        for line in p.open():
            r = json.loads(line)
            assert r['qid'] in sample_qids, f'unexpected qid in {p}'
            assert {'pk_answer','ck_answer','pk_correct_alias','ck_correct_alias'} <= r.keys()
            n += 1
        total += n
    print(f'  {bb:<22} {total:>4} labels')
"

echo ""
echo "[4/5] alias_match reproduces behaviour labels on the sample"
python -c "
import json
from pathlib import Path
from saber.data.alias_match import alias_match
root = Path('$SABER_DATA')
bb = 'llama-3.1-8b-instruct'
n_total = n_pk_ok = n_ck_ok = 0
for line in (root / 'evaluation' / bb / 'behavior_confiqa-qa.jsonl').open():
    n_total += 1
    if n_total > 50: break  # sanity sample, not a full check
    r = json.loads(line)
    qid = r['qid']
    ds_row = None
    for l in (root / 'datasets' / 'confiqa-qa.jsonl').open():
        d = json.loads(l)
        if d['qid'] == qid: ds_row = d; break
    if ds_row is None: continue
    gt = ds_row['ground_truth']
    if isinstance(gt, str): gt = [gt]
    pk_re = alias_match(r['pk_answer'], gt)
    ck_re = alias_match(r['ck_answer'], gt)
    if pk_re == bool(r['pk_correct_alias']): n_pk_ok += 1
    if ck_re == bool(r['ck_correct_alias']): n_ck_ok += 1
print(f'  {n_pk_ok}/50 PK labels reproduced, {n_ck_ok}/50 CK labels reproduced')
assert n_pk_ok >= 48 and n_ck_ok >= 48, 'alias_match drift detected'
"

echo ""
echo "[5/5] Unit tests + --help for every pipeline entry point"
python -m pytest tests/ -q

for m in \
    saber.data.build_split \
    saber.methods.saber_probe \
    saber.metrics.joint_metrics \
    saber.metrics.cell_distribution; do
  python -m "$m" --help > /dev/null
  echo "  OK   $m --help"
done

echo ""
echo "==============================================================="
echo " ALL SMOKE CHECKS PASSED"
echo "==============================================================="
