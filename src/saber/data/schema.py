"""SABER benchmark dataset schema (model-agnostic two-layer design).

Two-layer design:
  - DatasetRow      : model-agnostic; what the dataset is. Stored at
                      ``data/datasets/{dataset}.jsonl``.
  - ModelBehaviorRow: model-specific; how a particular backbone behaves on
                      each row. Stored at
                      ``data/evaluation/{backbone}/behavior_{dataset}.jsonl``
                      (extracted on demand only).

Hard requirements on DatasetRow (asserted by ``validate_dataset_row``):
  A. ``ground_truth_origin`` must be native (no ``llm_*``).
  B. ``ck_relation_source`` must be deterministic (no ``llm_judge``).
  C. No model-specific fields (``is_know_*``, ``pk_answer``, hidden states, ...).
  D. The dataset as a whole must contain both supportive and conflicting rows
     (asserted at builder level by ``validate_dataset``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Allowed enum values (single source of truth)
# ---------------------------------------------------------------------------

CK_RELATIONS = ("supportive", "conflicting", "irrelevant", "complementary")
CK_RELATIONS_CORE = ("supportive", "conflicting")

GROUND_TRUTH_ORIGINS = (
    "wikidata",
    "wikidata+aliases",
    "source_document",
    "human_annotation",
)

CK_TEXT_ORIGINS = (
    "natural",                       # source-document text, unchanged
    "natural_rule_perturbed",        # source-document with rule-based span swap
    "natural_noise",                 # source-document but noisy/user-generated
    "llm_synth",                     # LLM-generated, frozen at dataset build time
    "llm_synth_model_specific",      # LLM-generated targeting a specific source LLM
    "cross_question_swap",           # built by swapping ck_text across same-dataset questions
)

CK_RELATION_SOURCES = (
    "rule_hardcoded",        # hardcoded by builder per row (e.g., supportive vs conflicting branch)
    "rule_string_match",     # string equality on raw fields (e.g., ClashEval is_reference)
    "alias_match",           # our deterministic alias_match utility
    "dataset_native_label",  # label is a native dataset field/folder (e.g., EchoQA dir name)
    "cross_question_swap",   # built by swap construction (Archive / future)
)

QUESTION_FORMATS = ("open", "mc-2", "mc-3", "mc-4", "mc-5", "yes_no")

# Forbidden fields on DatasetRow (caught defensively by validator)
_FORBIDDEN_DATASET_FIELDS = frozenset({
    "is_know_pk", "is_know_ck",
    "pk_answer", "ck_answer",
    "pk_correct", "ck_correct", "pk_correct_lex", "ck_correct_lex",
    "first_token_entropy", "span_mean_entropy", "semantic_entropy",
    "confidence_gain", "entropy_no_ck", "entropy_with_ck",
    "hidden_states", "sampled_answers", "logp",
})


# ---------------------------------------------------------------------------
# DatasetRow (model-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    """Model-agnostic per-(query, ck_text) row. One dataset, many backbones."""

    # 1. Identity ----------------------------------------------------------
    qid: str
    dataset: str                    # fine-grained name: confiqa-qa, clasheval, ...
    dataset_family: str             # coarse family for reference (NOT for merging in training):
                                    # confiqa, clasheval, redditqa, conflictqa, conflictbank
    raw_idx: int                    # index in raw source file
    raw_split: Optional[str] = None  # original split if any

    # 2. Query + GT --------------------------------------------------------
    question: str = ""
    question_format: str = "open"   # open | mc-N | yes_no
    options: Optional[list[str]] = None
    ground_truth: list[str] = field(default_factory=list)  # accepted answer aliases
    ground_truth_text: str = ""
    ground_truth_origin: str = "wikidata"  # see GROUND_TRUTH_ORIGINS

    # 3. CK ---------------------------------------------------------------
    ck_text: Optional[str] = None
    ck_text_origin: str = "natural"  # see CK_TEXT_ORIGINS
    ck_target_answer: Optional[str] = None
    ck_target_matches_gt: Optional[bool] = None  # derived: alias_match(ck_target_answer, ground_truth)

    # 4. Label (model-agnostic; the project's Core target) ----------------
    ck_relation: str = "supportive"           # see CK_RELATIONS
    ck_relation_is_core: bool = True          # True iff ck_relation in {sup, conf}
    ck_relation_source: str = "rule_hardcoded"  # see CK_RELATION_SOURCES
    ck_sub_type: Optional[str] = None         # misinformation / temporal / semantic / explicit / implicit / None

    # 5. Task metadata -----------------------------------------------------
    hop_count: Optional[int] = None
    source_model: Optional[str] = None        # only set if dataset's sample selection was filtered by an LLM
                                              # (e.g., "llama2-7b" for ConflictQA-PopQA-llama2-7b)

    # 6. Extension --------------------------------------------------------
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ModelBehaviorRow (model-specific; minimal — extraction not run by default)
# ---------------------------------------------------------------------------

@dataclass
class ModelBehaviorRow:
    """Per-(qid, model_key) behavior. 4-cell labeling main store.
    See `context/notes/datasets.md` §5.3 / §4.

    Design: alias_match and judge results are stored as separate columns to
    allow lossless post-hoc analysis. The final pk_correct / ck_correct is
    `alias ∧ judge` (when judge available); otherwise alias_match only.
    """
    # FK
    qid: str
    model_key: str                # "llama-3.1-8b-instruct" / "qwen2.5-3b-instruct" / ...
    dataset: str                  # convenience denormalisation

    # Greedy generations (expensive — cache durably)
    pk_answer: str                # greedy(question)
    ck_answer: str                # greedy(question + ck_text); "" when ck_text is None
    pk_token_count: int = 0
    ck_token_count: int = 0

    # alias_match results (always computed)
    pk_correct_alias: bool = False
    ck_correct_alias: bool = False

    # Judge results (Optional — set only if judge ran on this row)
    pk_correct_judge: Optional[bool] = None
    ck_correct_judge: Optional[bool] = None
    pk_judge_response: Optional[str] = None     # raw "YES" / "NO" / error string
    ck_judge_response: Optional[str] = None
    judge_model: Optional[str] = None           # "gpt-4o-mini"

    # Final adopted correctness (alias ∧ judge if both present, else alias_only)
    pk_correct: bool = False
    ck_correct: bool = False
    pk_correct_source: str = "alias_only"       # "alias_only" | "alias_and_judge"
    ck_correct_source: str = "alias_only"

    # Provenance
    seed: int = 0
    max_new_tokens: int = 64
    pk_extracted_at: str = ""
    ck_extracted_at: str = ""
    judge_extracted_at: Optional[str] = None

    metadata: dict = field(default_factory=dict)


# Derived fields (NOT stored; computed on load from pk_correct + ck_correct):
#   cell                = f"C{int(pk_correct)}{int(ck_correct)}"  ∈ {C00, C01, C10, C11}
#   routing_target_3way = {C11:"either", C10:"use_PK", C01:"use_CK", C00:"abstain"}[cell]


def derive_cell(row: "ModelBehaviorRow | dict") -> str:
    """Compute 4-cell label from a ModelBehaviorRow or its dict form."""
    pk = row.pk_correct if hasattr(row, "pk_correct") else row["pk_correct"]
    ck = row.ck_correct if hasattr(row, "ck_correct") else row["ck_correct"]
    return f"C{int(bool(pk))}{int(bool(ck))}"


def derive_routing_target(row: "ModelBehaviorRow | dict") -> str:
    """3-way routing target derived from cell."""
    return {"C11": "either", "C10": "use_PK",
            "C01": "use_CK", "C00": "abstain"}[derive_cell(row)]


# ---------------------------------------------------------------------------
# Validators (enforce model-agnostic guarantees)
# ---------------------------------------------------------------------------

def validate_dataset_row(row: DatasetRow) -> None:
    """Strict per-row validation; raises AssertionError on any violation."""
    # Enum membership
    assert row.ck_relation in CK_RELATIONS, \
        f"qid={row.qid}: ck_relation={row.ck_relation!r} not in {CK_RELATIONS}"
    assert row.ck_relation_is_core == (row.ck_relation in CK_RELATIONS_CORE), \
        f"qid={row.qid}: ck_relation_is_core={row.ck_relation_is_core} inconsistent with ck_relation={row.ck_relation!r}"

    assert row.ground_truth_origin in GROUND_TRUTH_ORIGINS, \
        f"qid={row.qid}: ground_truth_origin={row.ground_truth_origin!r} not in {GROUND_TRUTH_ORIGINS}"
    assert "llm" not in row.ground_truth_origin.lower(), \
        f"qid={row.qid}: GT origin contains 'llm': {row.ground_truth_origin!r}"

    assert row.ck_text_origin in CK_TEXT_ORIGINS, \
        f"qid={row.qid}: ck_text_origin={row.ck_text_origin!r} not in {CK_TEXT_ORIGINS}"

    assert row.ck_relation_source in CK_RELATION_SOURCES, \
        f"qid={row.qid}: ck_relation_source={row.ck_relation_source!r} not in {CK_RELATION_SOURCES}"
    assert "llm" not in row.ck_relation_source.lower() and "judge" not in row.ck_relation_source.lower(), \
        f"qid={row.qid}: ck_relation_source forbidden: {row.ck_relation_source!r}"

    # Question format
    assert row.question_format in QUESTION_FORMATS, \
        f"qid={row.qid}: question_format={row.question_format!r} not in {QUESTION_FORMATS}"
    if row.question_format.startswith("mc-"):
        assert row.options is not None and len(row.options) > 0, \
            f"qid={row.qid}: mc question_format requires non-empty options"

    # Row-type consistency: warn (not error) on sup rows where alias_match misses.
    # Some builder sources (e.g. Huang 2025 TriviaQA) have paper-verified supportive
    # passages whose `correct_answer` is a paraphrase that fails our deterministic
    # alias_match (e.g., "Of" vs aliases ["of"] — alias_match drops "of" as stopword).
    # The pipeline's downstream judge step (datasets.md §4.3) catches these.
    if row.ck_relation == "supportive" and row.ck_target_answer is not None:
        if row.ck_target_matches_gt is False:
            import warnings
            warnings.warn(
                f"qid={row.qid}: supportive row with ck_target_matches_gt=False "
                f"(alias_match miss; will rely on judge fallback)",
                UserWarning, stacklevel=2,
            )

    # Required non-empty fields
    assert row.ground_truth, f"qid={row.qid}: ground_truth list is empty"
    assert row.ground_truth_text, f"qid={row.qid}: ground_truth_text is empty"
    assert row.question, f"qid={row.qid}: question is empty"

    # Defensive: catch any model-behavior field accidentally set in metadata
    leaked = _FORBIDDEN_DATASET_FIELDS & set(row.metadata.keys())
    assert not leaked, f"qid={row.qid}: metadata leaks model-behavior fields: {leaked}"


def validate_dataset(rows: list[DatasetRow], dataset_name: str) -> dict:
    """Validate a full dataset; enforces §0 D (must have both sup and conf).
    Returns a small summary dict for logging.
    """
    if not rows:
        raise AssertionError(f"dataset {dataset_name}: empty")

    counts: dict[str, int] = {}
    for r in rows:
        validate_dataset_row(r)
        assert r.dataset == dataset_name, \
            f"row dataset={r.dataset!r} != dataset_name={dataset_name!r}"
        counts[r.ck_relation] = counts.get(r.ck_relation, 0) + 1

    n_sup = counts.get("supportive", 0)
    n_conf = counts.get("conflicting", 0)
    assert n_sup > 0 and n_conf > 0, (
        f"dataset {dataset_name}: violates §0 constraint D — "
        f"need both supportive and conflicting rows, got "
        f"sup={n_sup}, conf={n_conf}"
    )

    return {
        "dataset": dataset_name,
        "n_rows": len(rows),
        "n_supportive": n_sup,
        "n_conflicting": n_conf,
        "n_irrelevant": counts.get("irrelevant", 0),
        "n_complementary": counts.get("complementary", 0),
    }


# ---------------------------------------------------------------------------
# Convenience: DatasetRow <-> dict (for parquet IO)
# ---------------------------------------------------------------------------

def row_to_dict(row: DatasetRow) -> dict[str, Any]:
    return {
        "qid": row.qid,
        "dataset": row.dataset,
        "dataset_family": row.dataset_family,
        "raw_idx": row.raw_idx,
        "raw_split": row.raw_split,
        "question": row.question,
        "question_format": row.question_format,
        "options": row.options,
        "ground_truth": row.ground_truth,
        "ground_truth_text": row.ground_truth_text,
        "ground_truth_origin": row.ground_truth_origin,
        "ck_text": row.ck_text,
        "ck_text_origin": row.ck_text_origin,
        "ck_target_answer": row.ck_target_answer,
        "ck_target_matches_gt": row.ck_target_matches_gt,
        "ck_relation": row.ck_relation,
        "ck_relation_is_core": row.ck_relation_is_core,
        "ck_relation_source": row.ck_relation_source,
        "ck_sub_type": row.ck_sub_type,
        "hop_count": row.hop_count,
        "source_model": row.source_model,
        "metadata": row.metadata,
    }
