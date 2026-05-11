"""Builder for Huang et al. 2025 (Situated Faithfulness) processed datasets.

Source: HF dataset `kkkevinkkk/SituatedFaithfulnessEval`, sub-configs `triviaqa`
and `naturalqa`. Each row carries:

    question, answers (list[str] aliases), correct_doc, wrong_doc,
    correct_answer, wrong_answer, question_id (or original_id), url

Per row we emit TWO DatasetRows:
  sup  → ck_text=correct_doc, ck_target_answer=correct_answer, ck_relation="supportive"
  conf → ck_text=wrong_doc,   ck_target_answer=wrong_answer,   ck_relation="conflicting"

Constraints check (datasets.md §1):
  A. GT native           — ✓ Original TriviaQA / NQ gold + aliases in `answers`
  B. (q, ck_text, ck_target, gt) — ✓ all 4 fields directly from raw schema
                            ck_target_answer comes from raw `correct_answer` /
                            `wrong_answer` (NOT inferred via LLM)
  C. answer-match        — alias_match works on TriviaQA out of the box;
                            NQ requires §4.3 alias∧judge fallback (paper documented
                            short-answer alias incomplete)
  D. 4-cell coverage     — ✓ each question contributes one sup row + one conf row
                            → 4-cell coverage emerges from PK x ck_correct interaction

ck_text_origin:
  correct_doc → "natural"        (retrieved web passage, NLI-verified by paper)
  wrong_doc   → "llm_synth"      (GPT-4o counterfactual rewrite of correct_doc;
                                  paper §3.2)
ck_relation_source: "rule_hardcoded" (per-row: correct_doc=sup, wrong_doc=conf)
ground_truth_origin: "source_document" (TriviaQA / NQ released aliases)
"""
from __future__ import annotations

from collections.abc import Iterator

from saber.data.alias_match import alias_match
from saber.data.schema import DatasetRow

# Lazy import datasets so module import doesn't pull in HF in unrelated paths.
HF_CACHE = "REDACTED_PATH"


def _build_dataset(cfg_name: str, ds_label: str) -> list[DatasetRow]:
    """Build DatasetRows for one Huang sub-config (triviaqa or naturalqa)."""
    from datasets import load_dataset

    ds = load_dataset("kkkevinkkk/SituatedFaithfulnessEval", cfg_name,
                      cache_dir=HF_CACHE)
    rows: list[DatasetRow] = []
    raw_idx = 0
    for split_name, split_ds in ds.items():  # 'test', 'dev'
        for raw in split_ds:
            # Pick id field: triviaqa uses `question_id`, naturalqa uses `original_id`
            qid_raw = raw.get("question_id") or raw.get("original_id") or f"row{raw_idx}"
            question = (raw["question"] or "").strip()
            answers = list(raw["answers"]) if raw["answers"] else []
            if not question or not answers:
                raw_idx += 1
                continue
            ground_truth_text = answers[0]
            # Common metadata
            base_meta = {
                "raw_qid": str(qid_raw),
                "raw_split": split_name,
                "url": raw.get("url"),
                "huang_correct_answer": raw.get("correct_answer"),
                "huang_wrong_answer": raw.get("wrong_answer"),
            }
            # ─── Supportive row ───
            sup_target = (raw.get("correct_answer") or "").strip()
            sup_match_gt = alias_match(sup_target, answers) if sup_target else False
            rows.append(DatasetRow(
                qid=f"{ds_label}-{raw_idx:06d}-sup",
                dataset=ds_label,
                dataset_family=ds_label.split("-")[0],   # "triviaqa" or "nq"
                raw_idx=raw_idx,
                raw_split=split_name,
                question=question,
                question_format="open",
                options=None,
                ground_truth=answers,
                ground_truth_text=ground_truth_text,
                ground_truth_origin="source_document",
                ck_text=(raw.get("correct_doc") or "").strip() or None,
                ck_text_origin="natural",
                ck_target_answer=sup_target or None,
                ck_target_matches_gt=sup_match_gt,
                ck_relation="supportive",
                ck_relation_is_core=True,
                ck_relation_source="rule_hardcoded",
                ck_sub_type=None,
                hop_count=1,
                source_model=None,
                metadata=dict(base_meta),
            ))
            # ─── Conflicting row ───
            conf_target = (raw.get("wrong_answer") or "").strip()
            conf_match_gt = alias_match(conf_target, answers) if conf_target else False
            rows.append(DatasetRow(
                qid=f"{ds_label}-{raw_idx:06d}-conf",
                dataset=ds_label,
                dataset_family=ds_label.split("-")[0],
                raw_idx=raw_idx,
                raw_split=split_name,
                question=question,
                question_format="open",
                options=None,
                ground_truth=answers,
                ground_truth_text=ground_truth_text,
                ground_truth_origin="source_document",
                ck_text=(raw.get("wrong_doc") or "").strip() or None,
                ck_text_origin="llm_synth",  # GPT-4o counterfactual rewrite
                ck_target_answer=conf_target or None,
                ck_target_matches_gt=conf_match_gt,
                ck_relation="conflicting",
                ck_relation_is_core=True,
                ck_relation_source="rule_hardcoded",
                ck_sub_type=None,
                hop_count=1,
                source_model=None,
                metadata=dict(base_meta),
            ))
            raw_idx += 1
    return rows


def build() -> Iterator[tuple[str, list[DatasetRow]]]:
    """Yield (dataset_name, rows) for triviaqa-huang and nq-huang."""
    yield "triviaqa-huang", _build_dataset("triviaqa", "triviaqa-huang")
    yield "nq-huang", _build_dataset("naturalqa", "nq-huang")
