"""Re-parse final_answer from saved raw_response without re-running generation.

This is the safety net for our pipeline: every prompted baseline saves its raw
LLM response. If we change the parser (e.g. add more aggressive prefix
stripping), we can re-parse from saved raw without paying for generation again.

Pure deterministic — no LLM calls. Fast (seconds even for full pool).

What this updates:
- implicit_scr / tacs_lr: re-parse from `raw_response` / `raw_answer_response`
  via `_first_line` (matches runner's current parser)
- explicit_scr: re-parse from `raw_response` via `_last_line`
- internal_eval / context_eval: re-derive from `judge_decision` + behavior
  pk/ck_answer
- closed_book / dia: no-op (final = pk/ck_answer)

Usage:
    python -m saber.baselines.re_extract --run-name pool2_full
    python -m saber.baselines.re_extract --run-name pool2_full --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from saber.baselines.runners import _first_line, _last_line


DATASET_DIR = Path("$SABER_DATA/datasets")
EVAL_DIR = Path("$SABER_DATA/evaluation")
OUT_BASE = Path("$SABER_DATA/baselines")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reparse_one_method_file(
    method: str, backbone: str, dataset: str, answers_path: Path,
    behavior_by_qid: dict, dry_run: bool = False,
) -> dict:
    records = load_jsonl(answers_path)
    n_changed = 0
    n_unchanged = 0
    n_skipped = 0
    for r in records:
        qid = r.get("qid")
        if not qid:
            n_skipped += 1
            continue
        if method == "implicit_scr":
            raw = r.get("raw_response") or ""
            new_final = _first_line(raw)
        elif method == "explicit_scr":
            raw = r.get("raw_response") or ""
            new_final = _last_line(raw)
        elif method == "tacs_lr":
            raw = r.get("raw_answer_response") or ""
            new_final = _first_line(raw)
        elif method in {"internal_eval", "context_eval"}:
            beh = behavior_by_qid.get(qid)
            if not beh:
                n_skipped += 1; continue
            judge = r.get("judge_decision")
            # Match runner fallbacks (runners.py): when judge is None (parse fail),
            # internal_eval and context_eval both fall back to ck_answer (use_CK).
            if method == "internal_eval":
                if judge is True:
                    new_final = beh["pk_answer"]
                elif judge is False:
                    new_final = beh["ck_answer"]
                else:  # parse fail → match runner fallback
                    new_final = beh["ck_answer"]
            else:  # context_eval
                if judge is True:
                    new_final = beh["ck_answer"]
                elif judge is False:
                    new_final = beh["pk_answer"]
                else:  # parse fail → match runner fallback
                    new_final = beh["ck_answer"]
        elif method in {"closed_book", "dia"}:
            n_unchanged += 1
            continue
        else:
            n_skipped += 1; continue
        if new_final != r.get("final_answer"):
            r["final_answer"] = new_final
            r["reparsed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            n_changed += 1
        else:
            n_unchanged += 1
    if not dry_run and n_changed > 0:
        bak = answers_path.with_suffix(answers_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(answers_path, bak)
        with answers_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"changed": n_changed, "unchanged": n_unchanged, "skipped": n_skipped, "total": len(records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--backbones", nargs="+",
                    default=["llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct"])
    ap.add_argument("--baselines", nargs="+",
                    default=["implicit_scr", "explicit_scr", "tacs_lr",
                             "internal_eval", "context_eval"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.datasets is None:
        datasets = set()
        for method in args.baselines:
            for bb in args.backbones:
                d = OUT_BASE / args.run_name / method / bb
                if d.exists():
                    for f in d.glob("answers_*.jsonl"):
                        datasets.add(f.stem.replace("answers_", ""))
        args.datasets = sorted(datasets)
    print(f"[reparse] run-name={args.run_name} datasets={args.datasets}")

    beh_cache = {}
    for bb in args.backbones:
        for ds in args.datasets:
            beh = load_jsonl(EVAL_DIR / bb / f"behavior_{ds}.jsonl")
            beh_cache[(bb, ds)] = {b["qid"]: b for b in beh}

    summary = {}
    for method in args.baselines:
        for bb in args.backbones:
            for ds in args.datasets:
                p = OUT_BASE / args.run_name / method / bb / f"answers_{ds}.jsonl"
                if not p.exists(): continue
                stats = reparse_one_method_file(
                    method, bb, ds, p, beh_cache[(bb, ds)], dry_run=args.dry_run,
                )
                summary[(method, bb, ds)] = stats
                if stats["changed"] > 0 or args.dry_run:
                    print(f"  {method:<14} {bb:<22} {ds:<28} | "
                          f"changed={stats['changed']:>3}/{stats['total']:>3} "
                          f"unchanged={stats['unchanged']:>3} skipped={stats['skipped']}")
    n_changed = sum(s["changed"] for s in summary.values())
    n_total = sum(s["total"] for s in summary.values())
    print(f"\n[reparse] {n_changed}/{n_total} records updated"
          f" ({'DRY RUN' if args.dry_run else 'written; .bak preserved'})")


if __name__ == "__main__":
    main()
