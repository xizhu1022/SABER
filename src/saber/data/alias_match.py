"""Answer correctness via alias-match against a ground-truth alias list.

Used by Stage 1 to assign KNOW/UNKNOWN labels (Llama-3 greedy output vs
PopQA ``possible_answers``). Also used by Stage 2 to build post-hoc
stratification flags for S3.

Matching layers (in order — first True wins):

1. Exact match of normalised strings.
2. Whole-word containment: GT appears as a space-delimited token run inside
   the answer (or vice versa). This catches "answer: London" vs "London"
   without matching "pol" inside "political" (the false-positive that the
   previous substring-based ``is_correct`` would accept).
3. Token-set Jaccard >= 0.8 after normalisation, as a gentle fallback for
   multi-word aliases with word-order shuffle.

Normalisation: lower-case, strip punctuation, collapse whitespace, drop
a small stop-word set (articles, "and", "of", "the"). Articles are dropped
because PopQA aliases routinely include / exclude them inconsistently.
"""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_STOP = frozenset({"a", "an", "the", "and", "of"})


def normalise(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t and t not in _STOP]
    return " ".join(tokens)


def _word_contains(haystack: str, needle: str) -> bool:
    """Whole-word substring check: ``needle`` occurs as a space-delimited run."""
    if not needle:
        return False
    return (
        haystack == needle
        or haystack.startswith(needle + " ")
        or haystack.endswith(" " + needle)
        or (" " + needle + " ") in haystack
    )


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def alias_match(answer: str, aliases: list[str]) -> bool:
    """Return True iff ``answer`` matches any of ``aliases`` under the layered rules."""
    if answer is None or not aliases:
        return False
    ans = normalise(answer)
    if not ans:
        return False
    for raw in aliases:
        if raw is None:
            continue
        gt = normalise(str(raw))
        if not gt:
            continue
        if ans == gt:
            return True
        if _word_contains(ans, gt) or _word_contains(gt, ans):
            return True
        if len(gt.split()) >= 2 and _jaccard(ans, gt) >= 0.8:
            return True
    return False


_YES_TOKENS = frozenset({"yes", "true", "correct", "right", "affirmative", "definitely", "indeed"})
_NO_TOKENS = frozenset({"no", "false", "incorrect", "wrong", "negative", "not"})


def yes_no_alias_match(answer: str, ground_truth: list[str]) -> bool:
    """Relaxed matcher for open-ended yes/no responses.

    Accepts the model's first meaningful token; maps synonyms to True/False.
    Returns True iff the polarity of the answer's first word matches any
    element of ``ground_truth`` (expected members: "True" or "False").
    """
    if not answer or not ground_truth:
        return False
    ans = normalise(answer).split()
    if not ans:
        return False
    first = ans[0]
    if first in _YES_TOKENS:
        polarity = "True"
    elif first in _NO_TOKENS:
        polarity = "False"
    else:
        return False
    return any(polarity.lower() == str(g).lower() for g in ground_truth)


# Back-compat alias so existing call sites (``extractor.is_correct``) keep working.
def is_correct(answer: str, ground_truth: list[str]) -> bool:
    return alias_match(answer, ground_truth)


# ---------------------------------------------------------------------------
# Semantic (paraphrase-aware) matcher — used for open-ended answers
# (e.g., RedditQA bare variant) where lexical alias_match fails on paraphrase.
# ---------------------------------------------------------------------------

_SBERT_CACHE: dict = {}


def _get_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _SBERT_CACHE:
        from sentence_transformers import SentenceTransformer
        _SBERT_CACHE[model_name] = SentenceTransformer(model_name)
    return _SBERT_CACHE[model_name]


def semantic_alias_match_batch(
    answers: list[str],
    ground_truth_lists: list[list[str]],
    threshold: float = 0.65,
    fallback_to_lexical: bool = True,
) -> list[bool]:
    """Paraphrase-aware KNOW labeling.

    For each (answer, ground_truth_list) pair, returns True iff:
      * lexical alias_match succeeds (cheap whole-word / Jaccard), OR
      * cosine similarity between the answer and any ground-truth alias
        is >= ``threshold`` (default 0.65 — tuned on PopQA validation
        to keep KNOW rate within ±5pp of lexical matching on short answers).
    """
    import numpy as np

    n = len(answers)
    out = [False] * n

    need_semantic: list[int] = []
    for i, (a, gts) in enumerate(zip(answers, ground_truth_lists)):
        if not a or not gts:
            continue
        if fallback_to_lexical and alias_match(a, list(gts)):
            out[i] = True
            continue
        need_semantic.append(i)

    if not need_semantic:
        return out

    sbert = _get_sbert()
    # Encode answers + flattened GTs; track offsets.
    ans_texts = [answers[i] for i in need_semantic]
    gt_flat: list[str] = []
    gt_offsets: list[tuple[int, int]] = []
    for i in need_semantic:
        lo = len(gt_flat)
        gt_flat.extend(str(g) for g in ground_truth_lists[i] if g)
        gt_offsets.append((lo, len(gt_flat)))

    ans_emb = sbert.encode(ans_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    gt_emb = sbert.encode(gt_flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False) if gt_flat else np.zeros((0, ans_emb.shape[1]))

    for k, i in enumerate(need_semantic):
        lo, hi = gt_offsets[k]
        if hi == lo:
            continue
        sims = gt_emb[lo:hi] @ ans_emb[k]
        if float(sims.max()) >= threshold:
            out[i] = True
    return out
