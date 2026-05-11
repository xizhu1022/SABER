"""Phase 6 + 7: Cell distribution report + audit sampling.

Produces:
  data/evaluation/{model_key}/cell_distribution_summary.json    per backbone
  data/evaluation/cell_distribution_2backbones.json             cross-backbone
  data/evaluation/{model_key}/audit_{dataset}.jsonl             alias=T ∧ judge=NO samples (50 each)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from saber.config import DATA_ROOT


def _load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _cell_summary(rows: list[dict], by_relation: bool = True) -> dict:
    n = len(rows)
    counts = {"C00": 0, "C01": 0, "C10": 0, "C11": 0}
    counts_alias = {"C00": 0, "C01": 0, "C10": 0, "C11": 0}
    by_rel: dict[str, dict] = {}
    for r in rows:
        c_final = f"C{int(bool(r['pk_correct']))}{int(bool(r['ck_correct']))}"
        c_alias = f"C{int(bool(r['pk_correct_alias']))}{int(bool(r['ck_correct_alias']))}"
        counts[c_final] += 1
        counts_alias[c_alias] += 1
        if by_relation:
            rel = (r.get("metadata") or {}).get("ck_relation") or "unknown"
            d = by_rel.setdefault(rel, {"C00": 0, "C01": 0, "C10": 0, "C11": 0, "n": 0})
            d[c_final] += 1
            d["n"] += 1
    pk_T_final = counts["C10"] + counts["C11"]
    ck_T_final = counts["C01"] + counts["C11"]
    pk_T_alias = counts_alias["C10"] + counts_alias["C11"]
    ck_T_alias = counts_alias["C01"] + counts_alias["C11"]
    n_judged = sum(1 for r in rows if r.get("pk_correct_judge") is not None)
    return {
        "n_rows": n,
        "n_judged": n_judged,
        "cell_final": counts,
        "cell_final_pct": {k: round(v/n*100, 2) for k, v in counts.items()} if n else {},
        "cell_alias_only": counts_alias,
        "pk_correct_final": pk_T_final,
        "ck_correct_final": ck_T_final,
        "pk_correct_alias": pk_T_alias,
        "ck_correct_alias": ck_T_alias,
        "pk_rate_final": round(pk_T_final / n, 4) if n else 0,
        "ck_rate_final": round(ck_T_final / n, 4) if n else 0,
        "informative_C01_C10_pct": round((counts["C01"] + counts["C10"]) / n * 100, 2) if n else 0,
        "by_ck_relation": by_rel,
    }


def report_backbone(model_key: str, eval_root: Path) -> dict:
    bb_dir = eval_root / model_key
    out: dict = {"model_key": model_key, "datasets": {}}
    files = [f for f in sorted(bb_dir.glob("behavior_*.jsonl"))
             if not f.name.endswith("_judge.jsonl")]
    for fp in files:
        rows = _load_jsonl(fp)
        ds_name = fp.stem.replace("behavior_", "")
        out["datasets"][ds_name] = _cell_summary(rows)

    # Pool aggregate across all datasets
    all_rows = []
    for fp in files:
        all_rows.extend(_load_jsonl(fp))
    out["pool_all_datasets"] = _cell_summary(all_rows, by_relation=False)
    return out


def audit_sample(model_key: str, eval_root: Path, n_per_axis: int = 50,
                 seed: int = 42) -> None:
    """For each dataset: sample alias=T ∧ judge=False rows separately for pk and ck."""
    rng = random.Random(seed)
    bb_dir = eval_root / model_key
    files = [f for f in sorted(bb_dir.glob("behavior_*.jsonl"))
             if not f.name.endswith("_judge.jsonl")]
    for fp in files:
        rows = _load_jsonl(fp)
        ds_name = fp.stem.replace("behavior_", "")
        pk_disagree = [r for r in rows
                       if r["pk_correct_alias"] and r.get("pk_correct_judge") is False]
        ck_disagree = [r for r in rows
                       if r["ck_correct_alias"] and r.get("ck_correct_judge") is False]
        if not pk_disagree and not ck_disagree:
            continue
        # Random sample for human spot-check
        pk_sample = rng.sample(pk_disagree, min(n_per_axis, len(pk_disagree)))
        ck_sample = rng.sample(ck_disagree, min(n_per_axis, len(ck_disagree)))
        out_path = bb_dir / f"audit_{ds_name}.jsonl"
        with out_path.open("w") as f:
            for r in pk_sample:
                f.write(json.dumps({"axis": "pk", **r}, ensure_ascii=False) + "\n")
            for r in ck_sample:
                f.write(json.dumps({"axis": "ck", **r}, ensure_ascii=False) + "\n")
        print(f"[{model_key} {ds_name}] audit: {len(pk_sample)} pk + {len(ck_sample)} ck "
              f"disagreements → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["llama-3.1-8b-instruct", "qwen2.5-3b-instruct"])
    ap.add_argument("--audit-n", type=int, default=50)
    args = ap.parse_args()

    eval_root = DATA_ROOT / "evaluation"
    cross = {"backbones": {}}
    for bb in args.backbones:
        rep = report_backbone(bb, eval_root)
        out_path = eval_root / bb / "cell_distribution_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"\n=== {bb} ===")
        print(json.dumps({k: v for k, v in rep["datasets"].items()},
                         indent=2, ensure_ascii=False))
        print("--- pool ---")
        print(json.dumps(rep["pool_all_datasets"], indent=2, ensure_ascii=False))
        cross["backbones"][bb] = rep
        audit_sample(bb, eval_root, args.audit_n)

    cross_path = eval_root / "cell_distribution_2backbones.json"
    cross_path.write_text(json.dumps(cross, indent=2, ensure_ascii=False))
    print(f"\nWrote cross-backbone summary → {cross_path}")


if __name__ == "__main__":
    main()
