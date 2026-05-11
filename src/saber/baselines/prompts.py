"""Prompts for Huang 2025 baselines (verbatim from §H.1-§H.6 + Huang repo templates).

No modifications: prompts are kept exactly as in the original paper and code, per
the "zero prompt-change" principle (baselines.md §10.1).
"""
from __future__ import annotations

# ─── Shared 3-shot examples (Huang 2025 §H.1) ─────────────────────────────────
# Used by DIA, ImplicitSCR, TACS-LR step-2 (re-answer with filtered doc).

THREE_SHOT_EXAMPLES = [
    {
        "doc": "From the moment Sadie Frost and Jude Law met on the set of 1992 Brit flick, Shopping, she felt it was her destiny to \"spend the rest of my life\" with him. Married to Spandau Ballet star Gary Kemp, Sadie, then 25, tried to \"crush her unwelcome ideas\" about Jude, knowing they were \"jeopardising an idyllic home life.\"",
        "question": "Who was married to Spandau Ballet's Gary Kemp and later to Jude Law?",
        "answer": "Sadie Frost",
    },
    {
        "doc": "Allegra Kent (CBA '19), ballerina and muse of George Balanchine and Joseph Cornell, started studying ballet at 11 with Bronislava Nijinska and Carmelita Maracci. In 1952, Balanchine invited her to New York City Ballet, where she danced for the next 30 years.",
        "question": "In which branch of the arts does Allegra Kent work?",
        "answer": "Ballet",
    },
    {
        "doc": "The magnificent tiger, Panthera tigris is a striped animal. It has a thick yellow coat of fur with dark stripes. The combination of grace, strength, agility and enormous power has earned the tiger its pride of place as the national animal of India.",
        "question": "Which animal is the national emblem of India?",
        "answer": "The Tiger",
    },
]

# ─── DIA / Vanilla RAG (Huang 2025 §H.1) ──────────────────────────────────────

DIA_INSTRUCTION = (
    "You will be given a question and a document. Utilize the information in "
    "the document to assist you in answering the question."
)

# Abstention suffix. Appended to a method's system / instruction message; the
# few-shot examples are NOT changed (no IDK exemplars), so the model abstains
# only via the explicit instruction.
ABSTAIN_SUFFIX = (
    " If you cannot answer the question accurately based on either your own "
    "knowledge or the provided document, respond exactly with: I don't know."
)

DIA_INSTRUCTION_ABSTAIN = DIA_INSTRUCTION + ABSTAIN_SUFFIX


def render_dia_messages(question: str, doc: str) -> list[dict]:
    """3-shot DIA, rendered as multi-turn chat messages.

    Order: instruction (system) + 3 user/assistant pairs + final user.
    """
    msgs = [{"role": "system", "content": DIA_INSTRUCTION}]
    for ex in THREE_SHOT_EXAMPLES:
        msgs.append({
            "role": "user",
            "content": f"Document: {ex['doc']}\nQuestion: {ex['question']}\nAnswer:",
        })
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({
        "role": "user",
        "content": f"Document: {doc}\nQuestion: {question}\nAnswer:",
    })
    return msgs


def render_dia_abstain_messages(question: str, doc: str) -> list[dict]:
    """3-shot DIA with abstain instruction appended to the system message."""
    msgs = [{"role": "system", "content": DIA_INSTRUCTION_ABSTAIN}]
    for ex in THREE_SHOT_EXAMPLES:
        msgs.append({
            "role": "user",
            "content": f"Document: {ex['doc']}\nQuestion: {ex['question']}\nAnswer:",
        })
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({
        "role": "user",
        "content": f"Document: {doc}\nQuestion: {question}\nAnswer:",
    })
    return msgs


# ─── ImplicitSCR (Huang 2025 §H.2) ────────────────────────────────────────────
# Same 3-shot examples as DIA but Question-before-Document order.

IMPLICIT_SCR_INSTRUCTION = (
    "You will be given a question and a document. The document may not be "
    "trustworthy. Use your judgment to assess the reliability of the document. "
    "Then, based on both your assessment and your own knowledge, provide the "
    "best possible answer."
)

IMPLICIT_SCR_INSTRUCTION_ABSTAIN = IMPLICIT_SCR_INSTRUCTION + ABSTAIN_SUFFIX


def render_implicit_scr_messages(question: str, doc: str) -> list[dict]:
    msgs = [{"role": "system", "content": IMPLICIT_SCR_INSTRUCTION}]
    for ex in THREE_SHOT_EXAMPLES:
        msgs.append({
            "role": "user",
            "content": f"Question: {ex['question']}\nDocument: {ex['doc']}\nAnswer:",
        })
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({
        "role": "user",
        "content": f"Question: {question}\nDocument: {doc}\nAnswer:",
    })
    return msgs


def render_implicit_scr_abstain_messages(question: str, doc: str) -> list[dict]:
    msgs = [{"role": "system", "content": IMPLICIT_SCR_INSTRUCTION_ABSTAIN}]
    for ex in THREE_SHOT_EXAMPLES:
        msgs.append({
            "role": "user",
            "content": f"Question: {ex['question']}\nDocument: {ex['doc']}\nAnswer:",
        })
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({
        "role": "user",
        "content": f"Question: {question}\nDocument: {doc}\nAnswer:",
    })
    return msgs


# ─── InternalEval (Huang 2025 §H.4) ───────────────────────────────────────────

INTERNAL_EVAL_TEMPLATE = (
    "Your task is to evaluate the model's response to a question. You will be "
    "provided with a question, the model's answer. Your job is to determine "
    "whether the model's answer is true or false.\n\n"
    "Question: {question}\n\n"
    "Model Answer: {model_answer}\n\n"
    "Is the model's answer true or false?\n\n"
    'Return "True" if the model\'s answer is correct, and "False" if the '
    "model's answer is incorrect."
)

# Abstain variant: judge can return Unsure → routed to "I don't know".
INTERNAL_EVAL_TEMPLATE_ABSTAIN = (
    "Your task is to evaluate the model's response to a question. You will be "
    "provided with a question and the model's answer. Your job is to determine "
    "whether the model's answer is true, false, or whether you cannot reliably "
    "decide.\n\n"
    "Question: {question}\n\n"
    "Model Answer: {model_answer}\n\n"
    "Is the model's answer true, false, or unsure?\n\n"
    'Return "True" if the model\'s answer is correct, "False" if it is incorrect, '
    'or "Unsure" if you cannot reliably decide.'
)


def render_internal_eval_messages(question: str, model_answer: str) -> list[dict]:
    return [{
        "role": "user",
        "content": INTERNAL_EVAL_TEMPLATE.format(
            question=question, model_answer=model_answer,
        ),
    }]


def render_internal_eval_abstain_messages(question: str, model_answer: str) -> list[dict]:
    return [{
        "role": "user",
        "content": INTERNAL_EVAL_TEMPLATE_ABSTAIN.format(
            question=question, model_answer=model_answer,
        ),
    }]


# ─── ContextEval (Huang 2025 §H.5) ────────────────────────────────────────────

CONTEXT_EVAL_TEMPLATE = (
    "You will be given a question and a document that answers the question. "
    "Your task is to evaluate whether the document provides a correct answer "
    "to the question. If the document's answer is correct, return \"True\"; "
    "otherwise, return \"False\".\n\n"
    "Question: {question}\n\n"
    "Document: {doc}\n\n"
    "Is the document correct?\n\n"
    'Return "True" if the document\'s answer is correct, and "False" if the '
    "document's answer is incorrect."
)

CONTEXT_EVAL_TEMPLATE_ABSTAIN = (
    "You will be given a question and a document that answers the question. "
    "Your task is to evaluate whether the document provides a correct answer "
    "to the question. If the document's answer is correct, return \"True\"; "
    "if it is incorrect, return \"False\"; if you cannot reliably decide, "
    "return \"Unsure\".\n\n"
    "Question: {question}\n\n"
    "Document: {doc}\n\n"
    "Is the document correct?\n\n"
    'Return "True" / "False" / "Unsure".'
)


def render_context_eval_messages(question: str, doc: str) -> list[dict]:
    return [{
        "role": "user",
        "content": CONTEXT_EVAL_TEMPLATE.format(question=question, doc=doc),
    }]


def render_context_eval_abstain_messages(question: str, doc: str) -> list[dict]:
    return [{
        "role": "user",
        "content": CONTEXT_EVAL_TEMPLATE_ABSTAIN.format(question=question, doc=doc),
    }]


# ─── TACS-LR / Filter Context (Huang 2025 §H.6) ───────────────────────────────

FILTER_DOC_INSTRUCTION = (
    "You will be given a document and a question. You need to remove the "
    "sentence which you think is not correct. You can only do removal and "
    "you can not add any new information or change the existing information. "
    "Only return the filtered document as your output."
)

FILTER_DOC_EXAMPLES = [
    {
        "doc": "The Eiffel Tower is located in Paris, France. It is the tallest structure in Paris. The Eiffel Tower was built in the 19th century and is made of wood.",
        "question": "Where is the Eiffel Tower located?",
        "filtered": "The Eiffel Tower is located in Paris, France. It is the tallest structure in Paris. The Eiffel Tower was built in the 19th century.",
    },
    {
        "doc": "Donald Trump is the President of the United States. He was elected in 2016 as a Democrat. He is the 45th President of the United States. Donald Trump was born in New York City.",
        "question": "Who is the President of the United States?",
        "filtered": "Donald Trump is the President of the United States. He was elected in 2016. He is the 45th President of the United States. Donald Trump was born in New York City.",
    },
    {
        "doc": "Taylor Swift is a famous singer. She was born in 1989. Taylor Swift has won multiple Grammy Awards. She is known for her country music.",
        "question": "When was Taylor Swift born?",
        "filtered": "Taylor Swift is a famous singer. She was born in 1989. Taylor Swift has won multiple Grammy Awards. She is known for her country music.",
    },
]


def render_filter_doc_messages(question: str, doc: str) -> list[dict]:
    """Filter step of TACS-LR: model removes incorrect sentence(s)."""
    msgs = [{"role": "system", "content": FILTER_DOC_INSTRUCTION}]
    for ex in FILTER_DOC_EXAMPLES:
        msgs.append({
            "role": "user",
            "content": f"Document: {ex['doc']}\nQuestion: {ex['question']}",
        })
        msgs.append({"role": "assistant", "content": ex["filtered"]})
    msgs.append({
        "role": "user",
        "content": f"Document: {doc}\nQuestion: {question}",
    })
    return msgs


# ─── ExplicitSCR (Huang 2025 §H.3) ────────────────────────────────────────────
# CoT reasoning: model gets question + internal_answer + doc + doc_answer →
# reasons whether doc is deceptive → final answer on last line.

EXPLICIT_SCR_INSTRUCTION = (
    "Task Overview: You will be given a question along with your internal "
    "answer, a document that may contain either true or false information, "
    "and the document's answer to the same question. Your task is to evaluate "
    "the reliability of the document and determine whether the document is "
    "deceptive or not.\n"
    "Steps:\n"
    "1.Internal Reasoning: Reflect on how you arrived at your internal answer "
    "using your own knowledge. Break down your reasoning process and assess "
    "the confidence level of your original answer, explaining why you believe "
    "your answer is correct.\n"
    "2. Document Evaluation: Analyze the document and cross-reference the "
    "information provided with the known facts you used to form your internal "
    "answer. Determine whether the document contains deceptive or unreliable "
    "information, considering possible contradictions or inconsistencies.\n"
    "3. Final Judgment: Based on your analysis, decide which answer (your "
    "internal answer or the document's answer) is more likely to be correct. "
    "Clearly state your final answer."
)

EXPLICIT_SCR_POST = (
    "Please provide a detailed reasoning process, followed by your final "
    "judgment. Ensure the last line of your response contains only the final "
    "answer without any additional explanation or details."
)

EXPLICIT_SCR_POST_ABSTAIN = (
    "Please provide a detailed reasoning process, followed by your final "
    "judgment. Ensure the last line of your response contains only the final "
    "answer without any additional explanation or details. If you cannot "
    "decide between the two and the question cannot be answered reliably, "
    "make the last line exactly: I don't know."
)


def render_explicit_scr_messages(
    question: str, internal_answer: str, doc: str, doc_answer: str,
) -> list[dict]:
    """Single-turn ExplicitSCR (no in-context examples; instruction-only).

    Note: paper §H.3 lists 3 in-context examples. For latency / cost we use
    the zero-shot variant which Huang's released code also supports. If
    accuracy looks low we can add the examples later.
    """
    user_content = (
        f"{EXPLICIT_SCR_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Your answer: {internal_answer}\n"
        f"The document to judge: {doc}\n"
        f"The document answer: {doc_answer}\n"
        f"{EXPLICIT_SCR_POST}"
    )
    return [{"role": "user", "content": user_content}]


def render_explicit_scr_abstain_messages(
    question: str, internal_answer: str, doc: str, doc_answer: str,
) -> list[dict]:
    user_content = (
        f"{EXPLICIT_SCR_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Your answer: {internal_answer}\n"
        f"The document to judge: {doc}\n"
        f"The document answer: {doc_answer}\n"
        f"{EXPLICIT_SCR_POST_ABSTAIN}"
    )
    return [{"role": "user", "content": user_content}]


# Answer extraction is purely deterministic post-processing (see `_clean_answer`,
# `_first_line`, `_last_line` in runners.py) combined with alias_match-based
# evaluation. No LLM-based extractor is invoked.
