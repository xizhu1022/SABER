"""Builder for ConflictBank CB_qa subsets (Su et al., NeurIPS 2024 D&B).

Two pilot sizes registered (same method, different target_n_raw):
  - pilot10k  : 10,000 raw × 4 = 40,000 processed rows   (~140 MB)
  - pilot100k : 100,000 raw × 4 = 400,000 processed rows (~1.4 GB)

All subsets use the same method:
  - Stratified by (relation × default_evidence_category)
  - Top-50 relations by raw frequency (covers ~95% of CB_qa)
  - Each raw row expands to 4 processed rows (1 sup + 3 conf-subtype)

Naming convention to avoid mixing with potential future full build:
  - dataset         = "conflictbank-pilot10k" / "conflictbank-pilot100k"
  - dataset_family  = "conflictbank"
  - (future full)   = "conflictbank-full" (not emitted here)

Per raw row, the 4 emitted rows are:
  - sup            : ck_text=default_evidence                         ck_target=object
  - conf-misinfo   : ck_text=misinformation_conflict_evidence_evidence ck_target=replaced_object
  - conf-temporal  : ck_text=temporal_conflict_evidence               ck_target=replaced_object
  - conf-semantic  : ck_text=semantic_conflict_evidence               ck_target=replaced_object

GT origin:          "wikidata"           (Wikidata triple object)
ck_text origin:     "llm_synth"          (all evidence variants are LLM-generated)
ck_relation_source: "rule_hardcoded"     (field-selection determines label)
ck_sub_type:        None / misinformation / temporal / semantic
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from saber.data.alias_match import alias_match
from saber.data.schema import DatasetRow

HF_CACHE = Path("REDACTED_PATH")

# Subset design parameters
PILOT_TOP_RELATIONS = 50  # use top-K relations by frequency (covers ~95% of rows)
PILOT_SEED = 42


def _stratified_sample(df, target_n: int, top_k_relations: int, seed: int):
    """Sample target_n rows stratified by (relation × default_evidence_category).

    Explicit per-stratum sampling (no groupby.apply — avoids pandas' tricky
    include_groups semantics). Returns a DataFrame with ALL original columns
    preserved and 'raw_idx_full' column holding each row's original index in
    the full CB_qa dataset.
    """
    import pandas as pd

    # Preserve original (full-dataset) index for traceability
    df = df.reset_index().rename(columns={"index": "raw_idx_full"})

    top_rel = df["relation"].value_counts().head(top_k_relations).index
    df_top = df[df["relation"].isin(top_rel)]

    # Build (relation, category) → sub-df mapping manually
    sampled_chunks: list = []
    groups = list(df_top.groupby(["relation", "default_evidence_category"], sort=False))
    n_strata = len(groups)
    per_stratum = max(1, (target_n + n_strata - 1) // n_strata)

    for (_rel, _cat), group in groups:
        take_n = min(per_stratum, len(group))
        sampled_chunks.append(group.sample(n=take_n, random_state=seed))
    sampled = pd.concat(sampled_chunks, ignore_index=True)

    # Top-up by random within top-rel pool if undershot
    if len(sampled) < target_n:
        remaining = df_top[~df_top["raw_idx_full"].isin(sampled["raw_idx_full"])]
        need = target_n - len(sampled)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(need, len(remaining)), random_state=seed + 1)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    # Cap at target_n (deterministic)
    sampled = sampled.head(target_n).reset_index(drop=True)
    return sampled


def _load_cb_qa_dataframe():
    """Load Warrieryes/CB_qa from HF cache → pandas DataFrame."""
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
    from datasets import load_dataset

    ds = load_dataset("Warrieryes/CB_qa", split="train", cache_dir=str(HF_CACHE))
    return ds.to_pandas()


def _build_one_raw_row(
    raw_idx: int,
    r,
    dataset_name: str,
) -> list[DatasetRow]:
    """Expand one CB_qa raw row into 1 sup + 3 conf rows."""
    rows: list[DatasetRow] = []

    question = r["question"]
    options = list(r["options"]) if r["options"] is not None else None
    if not question or not options:
        return rows

    gt_object = str(r["object"]).strip()
    replaced = str(r["replaced_object"]).strip()
    gt_list = [gt_object]  # object name from Wikidata triple
    if not gt_object:
        return rows

    n_opts = len(options)
    qfmt = f"mc-{n_opts}" if 2 <= n_opts <= 5 else "open"

    common_meta = {
        "relation": str(r["relation"]),
        "subject": str(r["subject"]),
        "subject_description": str(r.get("subject_description") or ""),
        "object_description": str(r.get("object_description") or ""),
        "replaced_description": str(r.get("replaced_description") or ""),
        "correct_option": str(r.get("correct_option") or ""),
        "replace_option": str(r.get("replace_option") or ""),
        "uncertain_option": str(r.get("uncertain_option") or ""),
        "semantic_description": str(r.get("semantic_description") or ""),
    }

    # ── Sup row: passage supports object (= GT)
    if r.get("default_evidence"):
        sup_match = alias_match(gt_object, gt_list)
        rows.append(DatasetRow(
            qid=f"{dataset_name}-{raw_idx:07d}-sup",
            dataset=dataset_name,
            dataset_family="conflictbank",
            raw_idx=raw_idx,
            raw_split=None,
            question=question,
            question_format=qfmt,
            options=options,
            ground_truth=gt_list,
            ground_truth_text=gt_object,
            ground_truth_origin="wikidata",
            ck_text=str(r["default_evidence"]),
            ck_text_origin="llm_synth",
            ck_target_answer=gt_object,
            ck_target_matches_gt=sup_match,
            ck_relation="supportive",
            ck_relation_is_core=True,
            ck_relation_source="rule_hardcoded",
            ck_sub_type=None,
            hop_count=1,
            source_model=None,
            metadata={
                **common_meta,
                "ck_claim": str(r.get("default_claim") or ""),
                "ck_evidence_category": str(r.get("default_evidence_category") or ""),
            },
        ))

    # ── 3 Conf rows by subtype
    conf_variants = [
        ("misinformation",
         r.get("misinformation_conflict_evidence_evidence"),
         r.get("misinformation_conflict_claim"),
         r.get("misinformation_conflict_evidence_category"),
         None),
        ("temporal",
         r.get("temporal_conflict_evidence"),
         r.get("temporal_conflict_claim"),
         r.get("temporal_conflict_evidence_category"),
         r.get("temporal_conflict_time_span")),
        ("semantic",
         r.get("semantic_conflict_evidence"),
         r.get("semantic_conflict_claim"),
         r.get("semantic_conflict_evidence_category"),
         None),
    ]
    for subtype, ev_text, claim, category, extra in conf_variants:
        if not ev_text:
            continue
        cf_match = alias_match(replaced, gt_list)
        meta_conf = {
            **common_meta,
            "ck_claim": str(claim or ""),
            "ck_evidence_category": str(category or ""),
        }
        if extra is not None:
            if hasattr(extra, "tolist"):
                extra_val = extra.tolist()
            else:
                extra_val = list(extra) if isinstance(extra, (list, tuple)) else extra
            meta_conf["ck_extra"] = extra_val
        rows.append(DatasetRow(
            qid=f"{dataset_name}-{raw_idx:07d}-conf_{subtype}",
            dataset=dataset_name,
            dataset_family="conflictbank",
            raw_idx=raw_idx,
            raw_split=None,
            question=question,
            question_format=qfmt,
            options=options,
            ground_truth=gt_list,
            ground_truth_text=gt_object,
            ground_truth_origin="wikidata",
            ck_text=str(ev_text),
            ck_text_origin="llm_synth",
            ck_target_answer=replaced,
            ck_target_matches_gt=cf_match,
            ck_relation="conflicting",
            ck_relation_is_core=True,
            ck_relation_source="rule_hardcoded",
            ck_sub_type=subtype,
            hop_count=1,
            source_model=None,
            metadata=meta_conf,
        ))

    return rows


def _build_subset(target_n_raw: int, suffix: str) -> list[DatasetRow]:
    """Shared builder: same stratification method, parameterized by target_n_raw."""
    print(f"[conflictbank] loading CB_qa from HF cache ...")
    df = _load_cb_qa_dataframe()
    print(f"[conflictbank] raw shape: {df.shape}")

    sampled = _stratified_sample(df, target_n=target_n_raw,
                                 top_k_relations=PILOT_TOP_RELATIONS,
                                 seed=PILOT_SEED)
    print(f"[conflictbank] sampled {len(sampled)} raw rows  "
          f"(top-{PILOT_TOP_RELATIONS} relations × 3 categories)")
    print(f"[conflictbank] relation diversity: {sampled['relation'].nunique()} distinct")
    print(f"[conflictbank] category distribution: "
          f"{sampled['default_evidence_category'].value_counts().to_dict()}")

    dataset_name = f"conflictbank-{suffix}"
    rows: list[DatasetRow] = []
    for _, r in sampled.iterrows():
        raw_idx = int(r["raw_idx_full"])
        rows.extend(_build_one_raw_row(raw_idx, r, dataset_name))
    return rows


def build() -> Iterator[tuple[str, list[DatasetRow]]]:
    """Default: pilot10k subset (registered under family='conflictbank')."""
    yield "conflictbank-pilot10k", _build_subset(10_000, "pilot10k")


def build_pilot100k() -> Iterator[tuple[str, list[DatasetRow]]]:
    """10x-scale subset (registered under family='conflictbank-100k')."""
    yield "conflictbank-pilot100k", _build_subset(100_000, "pilot100k")
