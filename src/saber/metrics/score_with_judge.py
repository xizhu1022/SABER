"""Optional LLM-judge scorer over labelled behaviour files.

For each row with a ``pk_answer`` / ``ck_answer``, queries a strong LLM
judge for an answer-equivalence YES/NO verdict, then updates the
``behavior_{dataset}.jsonl`` in place with the judge fields and the final
``pk_correct = alias and judge`` / ``ck_correct = alias and judge``.

Usage::

    PYTHONPATH=src python -m saber.metrics.score_with_judge \\
        --behavior-files data/evaluation/llama-3.1-8b-instruct/behavior_*.jsonl \\
        --judge-model gpt-4o-mini --concurrency 24
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

JUDGE_PROMPT = """You are an answer-equivalence judge. Decide whether the CANDIDATE answer expresses the same fact as ANY of the GOLD answers.

Treat as equivalent: paraphrases, alias names, abbreviations/acronyms, partial matches that uniquely identify the entity, language variations, formatting differences. Treat as NOT equivalent: different entities, off-by-one numbers, contradictory facts.

GOLD answers (any one match suffices): {gold_list}

CANDIDATE answer: {candidate}

Reply with exactly one word: YES or NO."""


def _judge_one(client: OpenAI, judge_model: str, candidate: str,
               gold_aliases: list[str]) -> tuple[bool | None, str]:
    """Call judge once. Return (judge_bool_or_None_on_error, raw_response).

    Relies on OpenAI SDK's built-in exponential-backoff retry (max_retries set
    on client). Outer retry loop handles unexpected non-retried failures.
    """
    if not candidate or not candidate.strip():
        return False, "empty_candidate"
    gold_list = " | ".join(str(g) for g in gold_aliases if g)
    prompt = JUDGE_PROMPT.format(gold_list=gold_list, candidate=candidate.strip())
    last_err = ""
    for attempt in range(8):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            return text.startswith("YES"), text
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            # Long backoff for rate limit (SDK retries failed already)
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
    return None, f"ERROR:{last_err}"


def _build_qid_to_gt(dataset_jsonl: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with dataset_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["qid"]] = row["ground_truth"]
    return out


def score_file(behavior_path: Path, dataset_jsonl: Path | None,
               judge_model: str, concurrency: int) -> dict:
    """Score one behavior_{dataset}.jsonl in place. Returns summary dict."""
    rows = []
    with behavior_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"[{behavior_path.name}] empty; skipping")
        return {"file": str(behavior_path), "n": 0}

    if dataset_jsonl is None:
        from saber.config import DATA_ROOT
        ds_name = rows[0]["dataset"]
        dataset_jsonl = DATA_ROOT / "datasets" / f"{ds_name}.jsonl"
    qid_to_gt = _build_qid_to_gt(dataset_jsonl)

    # Skip rows that already have BOTH pk and ck judge responses (idempotent re-runs).
    # A row needs work if pk_judge_response is None OR (ck_answer is non-empty AND ck_judge_response is None).
    def _needs(r):
        if r.get("pk_judge_response") is None: return True
        if r.get("ck_answer") and r.get("ck_judge_response") is None: return True
        return False
    todo_idx = [i for i, r in enumerate(rows) if _needs(rows[i])]
    if not todo_idx:
        print(f"[{behavior_path.name}] all rows already judged; computing summary only")
    else:
        print(f"[{behavior_path.name}] judging {len(todo_idx)} rows × 2 axes")

    client = OpenAI(max_retries=10, timeout=60.0)

    def _score_row(idx: int) -> tuple[int, dict]:
        r = rows[idx]
        gt = qid_to_gt[r["qid"]]
        pk_judge, pk_resp = _judge_one(client, judge_model, r["pk_answer"], gt)
        # ck_answer might be empty if ck_text is None — skip ck judge in that case
        if r["ck_answer"]:
            ck_judge, ck_resp = _judge_one(client, judge_model, r["ck_answer"], gt)
        else:
            ck_judge, ck_resp = False, "empty_candidate"
        return idx, {"pk_judge": pk_judge, "pk_resp": pk_resp,
                     "ck_judge": ck_judge, "ck_resp": ck_resp}

    if todo_idx:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_score_row, i): i for i in todo_idx}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"judge {behavior_path.stem}"):
                idx, res = fut.result()
                r = rows[idx]
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                r["pk_correct_judge"] = res["pk_judge"]
                r["ck_correct_judge"] = res["ck_judge"]
                r["pk_judge_response"] = res["pk_resp"]
                r["ck_judge_response"] = res["ck_resp"]
                r["judge_model"] = judge_model
                r["judge_extracted_at"] = ts
                # Final correct = alias ∧ judge
                #   - if judge is None (API failure), fall back to alias
                #   - if alias is False, judge cannot resurrect (alias rules out semantically wrong)
                #     but we still record it for analysis
                if res["pk_judge"] is not None:
                    r["pk_correct"] = bool(r["pk_correct_alias"]) and bool(res["pk_judge"])
                    r["pk_correct_source"] = "alias_and_judge"
                if res["ck_judge"] is not None:
                    r["ck_correct"] = bool(r["ck_correct_alias"]) and bool(res["ck_judge"])
                    r["ck_correct_source"] = "alias_and_judge"

    # Atomic rewrite
    tmp_path = behavior_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(behavior_path)

    # Summary
    n = len(rows)
    n_aT_jT = sum(1 for r in rows if r["pk_correct_alias"] and r.get("pk_correct_judge"))
    n_aT_jF = sum(1 for r in rows if r["pk_correct_alias"] and r.get("pk_correct_judge") is False)
    n_aF_jT = sum(1 for r in rows if not r["pk_correct_alias"] and r.get("pk_correct_judge"))
    pk_n_alias_T = sum(1 for r in rows if r["pk_correct_alias"])
    pk_n_judge_T = sum(1 for r in rows if r.get("pk_correct_judge"))
    pk_n_final_T = sum(1 for r in rows if r["pk_correct"])
    ck_n_alias_T = sum(1 for r in rows if r["ck_correct_alias"])
    ck_n_judge_T = sum(1 for r in rows if r.get("ck_correct_judge"))
    ck_n_final_T = sum(1 for r in rows if r["ck_correct"])
    cells = {"C00": 0, "C01": 0, "C10": 0, "C11": 0}
    for r in rows:
        c = f"C{int(bool(r['pk_correct']))}{int(bool(r['ck_correct']))}"
        cells[c] += 1

    summary = {
        "file": str(behavior_path),
        "n": n,
        "pk_alias_True": pk_n_alias_T,
        "pk_judge_True": pk_n_judge_T,
        "pk_final_True": pk_n_final_T,
        "ck_alias_True": ck_n_alias_T,
        "ck_judge_True": ck_n_judge_T,
        "ck_final_True": ck_n_final_T,
        "pk_aT_jT": n_aT_jT,
        "pk_aT_jF": n_aT_jF,
        "pk_aF_jT": n_aF_jT,
        "cells_final": cells,
        "judge_model": judge_model,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior-files", nargs="+", required=True,
                    help="One or more behavior_{dataset}.jsonl files (glob patterns OK)")
    ap.add_argument("--dataset-jsonl-dir", type=Path, default=None,
                    help="Directory containing dataset jsonl files for GT lookup. "
                         "Default: derived from behavior file's dataset field.")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY env var required"

    files: list[Path] = []
    for pattern in args.behavior_files:
        p = Path(pattern)
        if "*" in pattern or "?" in pattern:
            files.extend(sorted(p.parent.glob(p.name)))
        else:
            files.append(p)
    print(f"[judge] {len(files)} behavior files to score")

    summaries = []
    for fp in files:
        if not fp.exists():
            print(f"[judge] missing: {fp}")
            continue
        ds_jsonl = None
        if args.dataset_jsonl_dir:
            with fp.open() as f:
                ds_name = json.loads(f.readline())["dataset"]
            ds_jsonl = args.dataset_jsonl_dir / f"{ds_name}.jsonl"
        s = score_file(fp, ds_jsonl, args.judge_model, args.concurrency)
        summaries.append(s)
        print(json.dumps(s, indent=2, ensure_ascii=False))

    # Write joint summary
    out = {
        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summaries": summaries,
    }
    print("\n" + json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
