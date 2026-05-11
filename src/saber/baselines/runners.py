"""Per-baseline async runner functions.

Each `run_<baseline>(pool, model, row, behavior)` returns a dict that gets
written to `data/baselines/<method>/<backbone>/answers_<dataset>.jsonl`.

Common output schema:
    qid, method, backbone, final_answer, predicted_source, raw, latency_sec,
    cost_usd, error
"""
from __future__ import annotations

import re

from . import prompts as P
from .openrouter_client import OpenRouterPool


# ─── Output parsers ───────────────────────────────────────────────────────────

# Markdown bold (** … **) and common Huang-style prefixes that wrap the final
# answer. We strip these to obtain a clean answer string. Order matters:
# strip outer wrappers first, then the prefix.

_PREFIX_RE = re.compile(
    r"^\s*\**\s*(?:final\s+answer|final\s+judgment|final\s+judgement|answer)\s*:?\s*\**\s*",
    flags=re.IGNORECASE,
)


def _strip_markdown(text: str) -> str:
    """Remove leading/trailing ** markdown bold and stray asterisks."""
    t = text.strip()
    # Strip wrapping ** ... **
    while t.startswith("**") and t.endswith("**") and len(t) > 4:
        t = t[2:-2].strip()
    # Strip leading **
    if t.startswith("**"):
        t = t[2:].strip()
    if t.endswith("**"):
        t = t[:-2].strip()
    return t


def _clean_answer(line: str) -> str:
    """Strip markdown bold + 'Final Answer:' / 'Answer:' prefix + trailing punct."""
    s = _strip_markdown(line)
    s = _PREFIX_RE.sub("", s)
    s = _strip_markdown(s)  # in case prefix was inside ** **
    s = s.strip().rstrip(".!?,;:")
    return s


def _first_line(text: str) -> str:
    """Take first non-empty line; clean prefix + markdown."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return _clean_answer(line)
    return ""


def _last_line(text: str) -> str:
    """Take last non-empty line; clean prefix + markdown.

    For ExplicitSCR CoT — paper says "last line should contain only the final
    answer". If the last line is empty after cleaning (e.g. just "**Final
    Answer:**" with the answer on a previous line), fall back to the line
    above.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        cleaned = _clean_answer(line)
        if cleaned:
            return cleaned
    return ""


def _parse_true_false(text: str) -> bool | None:
    """Robust parser: lowercase contains 'true' before 'false' → True."""
    t = (text or "").lower()
    has_true = "true" in t
    has_false = "false" in t
    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False
    if has_true and has_false:
        # Both present — first occurrence wins
        return t.find("true") < t.find("false")
    return None


# Answer extraction is deterministic only (`_clean_answer`, `_first_line`,
# `_last_line` + alias_match), to avoid letting any LLM-based extractor leak
# the gold answer into the post-processed text.


# ─── Baselines ────────────────────────────────────────────────────────────────

# Closed-Book and DIA are zero-call — they read from existing behavior_*.jsonl.
# Implemented as a pass-through.

def run_closed_book(behavior: dict) -> dict:
    return {
        "method": "closed_book",
        "qid": behavior["qid"],
        "final_answer": behavior["pk_answer"],
        "predicted_source": "use_PK",  # by definition
        "cost_usd": 0.0,
        "latency_sec": 0.0,
        "error": None,
    }


def run_dia(behavior: dict) -> dict:
    return {
        "method": "dia",
        "qid": behavior["qid"],
        "final_answer": behavior["ck_answer"],
        "predicted_source": "use_CK",  # by definition
        "cost_usd": 0.0,
        "latency_sec": 0.0,
        "error": None,
    }


# ─── Prompted baselines (async OpenRouter) ────────────────────────────────────

async def run_internal_eval(
    pool: OpenRouterPool, model: str, row: dict, behavior: dict,
) -> dict:
    """Judge whether pk_answer is correct → True: use PK, False: use CK."""
    pk_ans = behavior["pk_answer"]
    ck_ans = behavior["ck_answer"]
    msgs = P.render_internal_eval_messages(row["question"], pk_ans)
    res = await pool.call(model, msgs, max_tokens=8)
    judge = _parse_true_false(res.text)
    if judge is True:
        final = pk_ans; src = "use_PK"
    elif judge is False:
        final = ck_ans; src = "use_CK"
    else:
        final = ck_ans; src = "use_CK"  # parse-fail fallback
    return {
        "method": "internal_eval",
        "qid": row["qid"],
        "final_answer": final,
        "predicted_source": src,
        "judge_response": res.text,
        "judge_decision": judge,
        "cost_usd": res.cost,
        "latency_sec": res.latency_sec,
        "error": res.error,
    }


async def run_context_eval(
    pool: OpenRouterPool, model: str, row: dict, behavior: dict,
) -> dict:
    """Judge whether doc gives correct answer → True: use CK, False: use PK."""
    pk_ans = behavior["pk_answer"]
    ck_ans = behavior["ck_answer"]
    msgs = P.render_context_eval_messages(row["question"], row["ck_text"])
    res = await pool.call(model, msgs, max_tokens=8)
    judge = _parse_true_false(res.text)
    if judge is True:
        final = ck_ans; src = "use_CK"
    elif judge is False:
        final = pk_ans; src = "use_PK"
    else:
        final = ck_ans; src = "use_CK"  # parse-fail fallback
    return {
        "method": "context_eval",
        "qid": row["qid"],
        "final_answer": final,
        "predicted_source": src,
        "judge_response": res.text,
        "judge_decision": judge,
        "cost_usd": res.cost,
        "latency_sec": res.latency_sec,
        "error": res.error,
    }


async def run_implicit_scr(
    pool: OpenRouterPool, model: str, row: dict, behavior: dict,
) -> dict:
    msgs = P.render_implicit_scr_messages(row["question"], row["ck_text"])
    # max_tokens=128 (was 64): some backbones produce 1-2 line preamble before
    # answer; 64 was cutting off the actual answer for Llama-8B / Qwen-7B.
    res = await pool.call(model, msgs, max_tokens=128)
    raw = res.text
    final = _first_line(raw)  # deterministic post-processing only
    return {
        "method": "implicit_scr",
        "qid": row["qid"],
        "final_answer": final,
        "predicted_source": None,  # implicit; derive at eval time via string match
        "raw_response": raw,
        "cost_usd": res.cost or 0.0,
        "latency_sec": res.latency_sec or 0.0,
        "error": res.error,
    }


async def run_explicit_scr(
    pool: OpenRouterPool, model: str, row: dict, behavior: dict,
) -> dict:
    msgs = P.render_explicit_scr_messages(
        question=row["question"],
        internal_answer=behavior["pk_answer"],
        doc=row["ck_text"],
        doc_answer=behavior["ck_answer"],
    )
    res = await pool.call(model, msgs, max_tokens=512)
    raw = res.text
    final = _last_line(raw)  # deterministic: Huang's prompt requires answer on last line
    return {
        "method": "explicit_scr",
        "qid": row["qid"],
        "final_answer": final,
        "predicted_source": None,  # CoT; derive via string match at eval
        "raw_response": raw,
        "cost_usd": res.cost or 0.0,
        "latency_sec": res.latency_sec or 0.0,
        "error": res.error,
    }


async def run_tacs_lr(
    pool: OpenRouterPool, model: str, row: dict, behavior: dict,
) -> dict:
    """Two-step: filter doc → re-answer with filtered doc via DIA."""
    # Step 1: filter
    f_msgs = P.render_filter_doc_messages(row["question"], row["ck_text"])
    f_res = await pool.call(model, f_msgs, max_tokens=512)
    filtered_doc = (f_res.text or "").strip()
    # Step 2: DIA on filtered doc
    a_msgs = P.render_dia_messages(row["question"], filtered_doc or row["ck_text"])
    # max_tokens=96: Qwen-7B tends to give statement-style answer.
    a_res = await pool.call(model, a_msgs, max_tokens=96)
    raw = a_res.text
    final = _first_line(raw)  # deterministic post-processing only
    err = f_res.error or a_res.error
    return {
        "method": "tacs_lr",
        "qid": row["qid"],
        "final_answer": final,
        "predicted_source": None,  # filtered context; derive via string match
        "filtered_doc": filtered_doc,
        "raw_answer_response": raw,
        "cost_usd": (f_res.cost or 0.0) + (a_res.cost or 0.0),
        "latency_sec": (f_res.latency_sec or 0.0) + (a_res.latency_sec or 0.0),
        "error": err,
    }


# ─── Registry ─────────────────────────────────────────────────────────────────

# Methods that need OpenRouter (async).
ASYNC_BASELINES = {
    "internal_eval": run_internal_eval,
    "context_eval": run_context_eval,
    "implicit_scr": run_implicit_scr,
    "explicit_scr": run_explicit_scr,
    "tacs_lr": run_tacs_lr,
}

# Methods that read from local behavior file (zero-call).
LOCAL_BASELINES = {
    "closed_book": run_closed_book,
    "dia": run_dia,
}
