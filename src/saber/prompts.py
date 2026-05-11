"""Centralised prompt templates used by SABER.

Each use case exposes (i) a module-level TEMPLATE string for reference and
(ii) a function ``(row: dict) -> str | messages`` that builds the final
prompt. All prompts target the backbone itself; no third-party LLM is
invoked.

USE CASE INDEX
==============

1. ``HIDDEN_STATE_PROBE`` -- read hidden state, no generation. Used by
   ``saber.extract.extract_hidden``.
2. ``PK_ANSWER_GEN``      -- greedy-generate ``pk_answer`` with no CK in
   the prompt. Used to label whether PK alone is correct.
3. ``CK_ANSWER_GEN``      -- greedy-generate ``ck_answer`` with CK in the
   prompt. Used to label whether the CK-conditioned path is correct.
"""
from __future__ import annotations

from typing import Any


# ============================================================================
# USE CASE 1 — HIDDEN_STATE_PROBE
# ============================================================================
# Used by: extract_hidden.py
# Goal: have the LLM "read" the query (optionally with CK) in one forward pass,
# extract a hidden state at a specific (layer, position). Minimal format -- no
# system prompt, no chat template, no "Answer:" trigger -- to avoid biasing
# the state.

HIDDEN_STATE_PROBE_TEMPLATE = "{question}\n{ck_text}"
HIDDEN_STATE_PROBE_TEMPLATE_NOCK = "{question}"  # query-only forward


def hidden_state_probe_prompt(row: dict[str, Any], include_ck: bool = True) -> str:
    """Build the prompt for hidden-state extraction.

    Args:
        row: DatasetRow dict (must have 'question'; may have 'ck_text')
        include_ck: if False, returns question only (for no-CK baseline forward)

    Returns:
        Plain text prompt (no chat template applied; tokenizer handles BOS).
    """
    q = str(row["question"]).strip()
    if not include_ck:
        return HIDDEN_STATE_PROBE_TEMPLATE_NOCK.format(question=q)
    ck = str(row.get("ck_text") or "").strip()
    if not ck:
        return q
    return HIDDEN_STATE_PROBE_TEMPLATE.format(question=q, ck_text=ck)


# ============================================================================
# USE CASE 2/3 — PK_ANSWER_GEN / CK_ANSWER_GEN (symmetric, neutral)
# ============================================================================
# Used by: extract_model_behavior.py (4-cell labeling — datasets.md §3)
# Design rule: PK and CK prompts differ ONLY by the presence of the Context
# block. No instruction tells the model to "use your memory" or "trust the
# passage" — wording is held constant so any change in the answer is
# attributable to context, not to prompt phrasing.
# Reference: CK-PLUG (Bi 2025 App. C.3.1: Background/Q/A), SeaKR (Yao 2025
#            App. D.1: Context/Question/Answer), Knowledgeable-R1 (Lin 2025
#            App. A: identical except retrieved-info block). EchoQA (Cheng)
#            documents that asymmetric "trust your memory" wording inflates
#            parametric reliance ~2× — exactly the bias we avoid.

ANSWER_SYSTEM = "Answer the question with a short phrase."
ANSWER_SYSTEM_ABSTAIN = (
    "Answer the question with a short phrase. If you cannot answer the "
    "question accurately based on either your own knowledge or the provided "
    "context, respond exactly with: I don't know."
)
PK_ANSWER_USER_TEMPLATE = "Question: {question}\nAnswer:"
CK_ANSWER_USER_TEMPLATE = "Context: {ck_text}\n\nQuestion: {question}\nAnswer:"


def pk_answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """OpenAI-style message list for query-only greedy decode."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": PK_ANSWER_USER_TEMPLATE.format(question=row["question"])},
    ]


def pk_answer_abstain_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Closed-book + abstain instruction (Closed-Book + abstain baseline)."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_ABSTAIN},
        {"role": "user", "content": PK_ANSWER_USER_TEMPLATE.format(question=row["question"])},
    ]


def ck_answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """OpenAI-style message list for query+context greedy decode."""
    ck = str(row.get("ck_text") or "").strip()
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": CK_ANSWER_USER_TEMPLATE.format(
            ck_text=ck, question=row["question"],
        )},
    ]


# ============================================================================
# Future use cases will be appended here as new §2.N experiments are designed.
# ============================================================================
