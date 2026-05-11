"""Builder for ConFiQA-{QA, MR, MC} (Bi et al., 2025).

Each raw row has both an original-context (sup) and a counter-factual context
(conf), so we emit 2 DatasetRows per raw row.

GT origin: Wikidata triple's original answer + aliases  -> "wikidata+aliases"
ck_text origin: LLM-synthesised narrative around either orig or cf -> "llm_synth"
ck_relation derivation: hardcoded per branch (sup uses orig_context, conf
                       uses cf_context) -> "rule_hardcoded"

The three sub-tasks (QA, MR, MC) share an identical raw schema and an
identical builder; we expose them as three independent dataset names.
"""
from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path

from saber.config import DATA_ROOT
from saber.data.alias_match import alias_match
from saber.data.schema import DatasetRow

RAW_DIR = DATA_ROOT / "raw" / "confiqa_mquake"

VARIANT_TO_FILE = {
    "qa": "ConFiQA-QA.json",
    "mr": "ConFiQA-MR.json",
    "mc": "ConFiQA-MC.json",
}


def _hop_count(orig_path_str: str | None) -> int | None:
    if not orig_path_str:
        return None
    try:
        parsed = ast.literal_eval(orig_path_str)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return len(parsed)
    return None


def _build_one_variant(variant: str) -> list[DatasetRow]:
    """variant ∈ {"qa", "mr", "mc"}.  Returns sup+conf rows for that subset."""
    if variant not in VARIANT_TO_FILE:
        raise ValueError(f"unknown variant {variant!r}; must be one of {list(VARIANT_TO_FILE)}")
    path = RAW_DIR / VARIANT_TO_FILE[variant]
    with path.open() as f:
        raw = json.load(f)

    dataset_name = f"confiqa-{variant}"
    rows: list[DatasetRow] = []

    for i, r in enumerate(raw):
        question = r.get("question")
        orig = r.get("orig_answer")
        cf = r.get("cf_answer")
        orig_ctx = r.get("orig_context") or r.get("orig_context_piece")
        cf_ctx = r.get("cf_context") or r.get("cf_context_piece")
        if not (question and orig and cf and orig_ctx and cf_ctx):
            continue

        orig_aliases = r.get("orig_alias") or []
        cf_aliases = r.get("cf_alias") or []
        # Build dedup'd alias list, original answer first
        gt_list: list[str] = []
        for v in [orig, *orig_aliases]:
            if v and v not in gt_list:
                gt_list.append(str(v))

        hop = _hop_count(r.get("orig_path"))

        common_meta = {
            "orig_alias": list(orig_aliases),
            "cf_alias": list(cf_aliases),
            "cf_answer_aliases": [str(cf), *[str(a) for a in cf_aliases]],
            "orig_path_labeled": r.get("orig_path_labeled"),
            "cf_path_labeled": r.get("cf_path_labeled"),
        }

        # ── supportive row: passage supports orig (= GT)
        sup_target = str(orig)
        sup_matches_gt = alias_match(sup_target, gt_list)
        rows.append(DatasetRow(
            qid=f"{dataset_name}-{i:06d}-sup",
            dataset=dataset_name,
            dataset_family="confiqa",
            raw_idx=i,
            raw_split=None,
            question=question,
            question_format="open",
            options=None,
            ground_truth=gt_list,
            ground_truth_text=str(orig),
            ground_truth_origin="wikidata+aliases",
            ck_text=orig_ctx,
            ck_text_origin="llm_synth",
            ck_target_answer=sup_target,
            ck_target_matches_gt=sup_matches_gt,
            ck_relation="supportive",
            ck_relation_is_core=True,
            ck_relation_source="rule_hardcoded",
            ck_sub_type=None,
            hop_count=hop,
            source_model=None,
            metadata=common_meta,
        ))

        # ── conflicting row: passage supports cf (≠ GT)
        cf_target = str(cf)
        cf_matches_gt = alias_match(cf_target, gt_list)
        rows.append(DatasetRow(
            qid=f"{dataset_name}-{i:06d}-conf",
            dataset=dataset_name,
            dataset_family="confiqa",
            raw_idx=i,
            raw_split=None,
            question=question,
            question_format="open",
            options=None,
            ground_truth=gt_list,
            ground_truth_text=str(orig),
            ground_truth_origin="wikidata+aliases",
            ck_text=cf_ctx,
            ck_text_origin="llm_synth",
            ck_target_answer=cf_target,
            ck_target_matches_gt=cf_matches_gt,
            ck_relation="conflicting",
            ck_relation_is_core=True,
            ck_relation_source="rule_hardcoded",
            ck_sub_type=None,
            hop_count=hop,
            source_model=None,
            metadata=common_meta,
        ))

    return rows


def build() -> Iterator[tuple[str, list[DatasetRow]]]:
    """Yields (dataset_name, rows) for each of the 3 ConFiQA subsets."""
    for variant in ("qa", "mr", "mc"):
        yield f"confiqa-{variant}", _build_one_variant(variant)


# Convenience for testing one subset in isolation
def build_one(variant: str) -> list[DatasetRow]:
    return _build_one_variant(variant)
