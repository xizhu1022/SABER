"""HF batched prefill that reads the conditional-layer hidden state for each
generated reasoning trace and writes per-trace condition vectors.

Pipeline:
  - Reads:  artifacts/{backbone}/traces[suffix]/{dataset}.K{K}.traces.jsonl
  - Writes: artifacts/{backbone}/cond[suffix]/
            {dataset}.K{K}.cond_pk.npz, .cond_ck.npz, .meta.jsonl

Both ``prompt_1_reasoning_*`` and ``prompt_2_judgment`` follow the SABER
self-evaluation template. The trace itself comes from
``multipath_vllm_gen.py``; this script only re-runs the prefill to read
hidden states at the conditional layer.

Resume-safe: skips a ``(ds, K)`` shard when ``meta.jsonl + cond_*.npz``
already cover all qids that have both PK and CK traces in the source jsonl.
Crash-safe via shard files.

Usage::
    PYTHONPATH=src \\
      python -u -m saber.extract.multipath_hf_postgen \\
        --backbone llama-3.1-8b-instruct \\
        --gpu-ids 0 1 2 3 \\
        --K-list 1 2 3 \\
        --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import yaml


BACKBONE_PATHS = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


def _ab_token_ids(tokenizer):
    A_variants = [" A", "A", " (A", "(A", "A)", " A)", "**A", " **A"]
    B_variants = [" B", "B", " (B", "(B", "B)", " B)", "**B", " **B"]
    A_ids, B_ids = set(), set()
    for v in A_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            A_ids.add(ids[0])
    for v in B_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            B_ids.add(ids[0])
    return sorted(A_ids), sorted(B_ids)


def _heal_shards(out_dir: Path, ds_k_prefix: str):
    """Merge per-checkpoint shards into canonical .npz, aligned with meta.jsonl.
    `ds_k_prefix` is e.g. 'confiqa-qa.K1' — the file stem before .cond_pk.shard.
    """
    for tmp in out_dir.glob(f"{ds_k_prefix}.cond_*.shard*.tmp.npz"):
        tmp.unlink()

    shard_pks = sorted(out_dir.glob(f"{ds_k_prefix}.cond_pk.shard*.npz"))
    if not shard_pks:
        return

    final_pk = out_dir / f"{ds_k_prefix}.cond_pk.npz"
    final_ck = out_dir / f"{ds_k_prefix}.cond_ck.npz"
    meta_path = out_dir / f"{ds_k_prefix}.meta.jsonl"

    canon_pk = canon_ck = canon_jpk = canon_jck = None
    if final_pk.exists():
        z = np.load(final_pk); canon_pk = z["trace_mp"]; canon_jpk = z["judge"]
        z = np.load(final_ck); canon_ck = z["trace_mp"]; canon_jck = z["judge"]

    meta_qids: list = []
    if meta_path.exists():
        with meta_path.open() as f:
            for line in f:
                meta_qids.append(json.loads(line)["qid"])

    n_canon = int(canon_pk.shape[0]) if canon_pk is not None else 0
    if n_canon > len(meta_qids):
        raise RuntimeError(
            f"[multipath-postgen] {ds_k_prefix}: canonical has {n_canon} rows but "
            f"meta.jsonl has only {len(meta_qids)} lines — alignment broken")

    new_meta_qids = meta_qids[n_canon:]
    new_idx = {q: i for i, q in enumerate(new_meta_qids)}
    out_pk = [None] * len(new_meta_qids)
    out_ck = [None] * len(new_meta_qids)
    out_jpk = [None] * len(new_meta_qids)
    out_jck = [None] * len(new_meta_qids)

    print(f"[multipath-postgen] {ds_k_prefix}: healing {len(shard_pks)} shards into "
          f"{len(new_meta_qids)} new rows + {n_canon} existing", flush=True)

    for sp in shard_pks:
        sc = out_dir / sp.name.replace("cond_pk", "cond_ck")
        if not sc.exists():
            print(f"[multipath-postgen] {ds_k_prefix}: orphan PK shard {sp.name}, removing",
                  flush=True)
            sp.unlink()
            continue
        zp = np.load(sp); zc = np.load(sc)
        if "qids" not in zp.files:
            raise RuntimeError(
                f"[multipath-postgen] {ds_k_prefix}: shard {sp.name} missing qids array")
        shard_qids = [str(q) for q in zp["qids"]]
        for i, q in enumerate(shard_qids):
            idx = new_idx.get(q)
            if idx is None:
                continue
            if out_pk[idx] is not None:
                continue
            out_pk[idx] = zp["trace_mp"][i]
            out_jpk[idx] = zp["judge"][i]
            out_ck[idx] = zc["trace_mp"][i]
            out_jck[idx] = zc["judge"][i]

    missing = [i for i, x in enumerate(out_pk) if x is None]
    if missing:
        raise RuntimeError(
            f"[multipath-postgen] {ds_k_prefix}: heal failed — {len(missing)} meta lines "
            f"unmatched (first qid={new_meta_qids[missing[0]]})")

    new_pk = np.stack(out_pk); new_ck = np.stack(out_ck)
    new_jpk = np.stack(out_jpk); new_jck = np.stack(out_jck)
    if canon_pk is not None:
        full_pk = np.concatenate([canon_pk, new_pk], axis=0)
        full_ck = np.concatenate([canon_ck, new_ck], axis=0)
        full_jpk = np.concatenate([canon_jpk, new_jpk], axis=0)
        full_jck = np.concatenate([canon_jck, new_jck], axis=0)
    else:
        full_pk, full_ck, full_jpk, full_jck = new_pk, new_ck, new_jpk, new_jck

    tmp = final_pk.with_suffix(".tmp.npz")
    with tmp.open("wb") as f:
        np.savez_compressed(f, trace_mp=full_pk, judge=full_jpk)
    tmp.replace(final_pk)
    tmp = final_ck.with_suffix(".tmp.npz")
    with tmp.open("wb") as f:
        np.savez_compressed(f, trace_mp=full_ck, judge=full_jck)
    tmp.replace(final_ck)

    for sp in shard_pks:
        sc = out_dir / sp.name.replace("cond_pk", "cond_ck")
        sp.unlink()
        if sc.exists():
            sc.unlink()
    print(f"[multipath-postgen] {ds_k_prefix}: healed → {full_pk.shape[0]} canonical rows",
          flush=True)


def _worker_loop(gpu_id, model_path, prompt_1_ck, prompt_1_pk, prompt_2,
                 batch_size, in_q, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    A_ids, B_ids = _ab_token_ids(tok)
    out_q.put(("READY", gpu_id, n_layers))

    while True:
        item = in_q.get()
        if item is None:
            out_q.put(("BYE", gpu_id))
            return
        text_1 = [p1 + tr for (_, _, _, p1, tr) in item]
        try:
            with torch.no_grad():
                enc1 = tok(text_1, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to("cuda")
                fwd1 = model(**enc1, output_hidden_states=True, use_cache=False)

                p1_tok_lens = []
                for (_, _, _, p1_text, _) in item:
                    p1_ids = tok(p1_text, add_special_tokens=False).input_ids
                    p1_tok_lens.append(len(p1_ids))
                enc1_mask = enc1.attention_mask
                B = len(item)
                max_len = enc1_mask.shape[1]
                real_lens = enc1_mask.sum(dim=1).tolist()

                trace_mp_per_row = []
                for i in range(B):
                    real_len = int(real_lens[i])
                    real_start = int(max_len - real_len)
                    p1_len = min(p1_tok_lens[i], real_len)
                    trace_start = real_start + p1_len
                    trace_end = real_start + real_len
                    if trace_end <= trace_start:
                        trace_mp_per_row.append(None)
                        continue
                    mp_layers = np.stack([
                        fwd1.hidden_states[li][i, trace_start:trace_end, :]
                            .mean(dim=0).to(torch.float16).cpu().numpy()
                        for li in range(1, n_layers + 1)
                    ])
                    trace_mp_per_row.append(mp_layers)
                del fwd1, enc1
                torch.cuda.empty_cache()

                text_2 = [
                    prompt_2.format(question=q, reasoning=tr)
                    for (_, q, _, _, tr) in item
                ]
                enc2 = tok(text_2, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to("cuda")
                last_pos = (enc2.attention_mask.sum(dim=1) - 1).tolist()
                fwd2 = model(**enc2, output_hidden_states=True, use_cache=False)
                judge_per_row = []
                p_A_per_row, p_B_per_row = [], []
                for i in range(B):
                    lp = int(last_pos[i])
                    judge_layers = np.stack([
                        fwd2.hidden_states[li][i, lp, :].to(torch.float16).cpu().numpy()
                        for li in range(1, n_layers + 1)
                    ])
                    judge_per_row.append(judge_layers)
                    logits = fwd2.logits[i, lp, :].to(torch.float32)
                    probs = torch.softmax(logits, dim=-1)
                    p_A_per_row.append(float(probs[A_ids].sum().item()) if A_ids else 0.0)
                    p_B_per_row.append(float(probs[B_ids].sum().item()) if B_ids else 0.0)
                del fwd2, enc2
                torch.cuda.empty_cache()

            for i, (task_id, _, side, _, _) in enumerate(item):
                if trace_mp_per_row[i] is None:
                    out_q.put(("RES", task_id, side, None))
                    continue
                out_q.put(("RES", task_id, side, {
                    "h_trace_mp_all": trace_mp_per_row[i],
                    "h_judge_all": judge_per_row[i],
                    "p_A": p_A_per_row[i], "p_B": p_B_per_row[i],
                }))
        except Exception as e:
            for task_id, _, side, _, _ in item:
                out_q.put(("ERR", task_id, side, f"{type(e).__name__}: {e}"))


def _process_one(ds, K, traces_dir, out_dir, eval_dir, dataset_dir,
                 rec, in_q, out_q, batch_size, checkpoint_every, n_workers):
    """Process a single (dataset, K) shard end-to-end via the worker pool."""
    ds_k = f"{ds}.K{K}"
    traces_path = traces_dir / f"{ds_k}.traces.jsonl"
    if not traces_path.exists():
        print(f"[multipath-postgen] {ds_k}: no traces.jsonl, skip", flush=True)
        return

    traces_by_qid: dict = {}
    with traces_path.open() as f:
        for line in f:
            t = json.loads(line)
            traces_by_qid.setdefault(t["qid"], {})[t["side"]] = {
                "trace": t["trace"], "n_tok": t["n_tok"],
            }

    ds_path = dataset_dir / f"{ds}.jsonl"
    beh_path = eval_dir / f"behavior_{ds}.jsonl"
    if not ds_path.exists() or not beh_path.exists():
        print(f"[multipath-postgen] {ds_k}: missing dataset/behavior files", flush=True)
        return
    ds_by_qid = {}
    with ds_path.open() as f:
        for line in f:
            d = json.loads(line)
            ds_by_qid[d["qid"]] = d
    beh_by_qid = {}
    with beh_path.open() as f:
        for line in f:
            b = json.loads(line)
            beh_by_qid[b["qid"]] = b

    rows = []
    for qid, sides in traces_by_qid.items():
        if "PK" not in sides or "CK" not in sides:
            continue
        if qid not in ds_by_qid or qid not in beh_by_qid:
            continue
        rows.append({
            "qid": qid,
            "question": ds_by_qid[qid]["question"],
            "ck_text": ds_by_qid[qid]["ck_text"],
            "pk_correct_alias": beh_by_qid[qid].get("pk_correct_alias"),
            "ck_correct_alias": beh_by_qid[qid].get("ck_correct_alias"),
            "pk_trace": sides["PK"]["trace"],
            "ck_trace": sides["CK"]["trace"],
            "n_tok_pk": sides["PK"]["n_tok"],
            "n_tok_ck": sides["CK"]["n_tok"],
        })
    print(f"[multipath-postgen] {ds_k}: {len(rows)} rows with both PK+CK traces", flush=True)
    if not rows:
        return

    meta_path = out_dir / f"{ds_k}.meta.jsonl"
    existing_qids: set = set()
    if meta_path.exists():
        with meta_path.open() as f:
            for line in f:
                existing_qids.add(json.loads(line)["qid"])
        rows = [r for r in rows if r["qid"] not in existing_qids]
        print(f"[multipath-postgen] {ds_k}: resume — {len(existing_qids)} existing, "
              f"{len(rows)} new", flush=True)
    if not rows:
        print(f"[multipath-postgen] {ds_k}: nothing new", flush=True)
        return

    def build_task(tid, row, side):
        # Raw-completion prompt (no chat template) so the conditional-layer
        # hidden state reflects the same prefix layout used for the self-prior.
        if side == "PK":
            p1 = rec["prompt_1_reasoning_pk"].format(question=row["question"])
            tr = row["pk_trace"]
        else:
            p1 = rec["prompt_1_reasoning_ck"].format(
                question=row["question"], ck_text=row["ck_text"])
            tr = row["ck_trace"]
        return (tid, row["question"], side, p1, tr)

    all_tasks = []
    for tid, row in enumerate(rows):
        all_tasks.append(build_task(tid, row, "PK"))
        all_tasks.append(build_task(tid, row, "CK"))
    for i in range(0, len(all_tasks), batch_size):
        in_q.put(all_tasks[i:i + batch_size])

    results: dict = {tid: {} for tid in range(len(rows))}
    n_done = 0
    n_total = len(all_tasks)
    next_unconsumed_tid = 0
    cond_pk_running, cond_ck_running = [], []
    judg_pk_running, judg_ck_running = [], []
    meta_running: list = []
    n_rows_saved = 0
    next_shard_idx = len(list(out_dir.glob(f"{ds_k}.cond_pk.shard*.npz")))
    t0 = time.time()

    def write_state():
        nonlocal n_rows_saved, next_shard_idx
        if not cond_pk_running:
            return
        pk_new = np.stack(cond_pk_running)
        ck_new = np.stack(cond_ck_running)
        jpk_new = np.stack(judg_pk_running)
        jck_new = np.stack(judg_ck_running)
        new_meta = meta_running[n_rows_saved:]
        qids_new = np.array([m["qid"] for m in new_meta])

        shard_pk = out_dir / f"{ds_k}.cond_pk.shard{next_shard_idx:04d}.npz"
        shard_ck = out_dir / f"{ds_k}.cond_ck.shard{next_shard_idx:04d}.npz"
        tmp = shard_pk.with_suffix(".tmp.npz")
        with tmp.open("wb") as f:
            np.savez(f, trace_mp=pk_new, judge=jpk_new, qids=qids_new)
        tmp.replace(shard_pk)
        tmp = shard_ck.with_suffix(".tmp.npz")
        with tmp.open("wb") as f:
            np.savez(f, trace_mp=ck_new, judge=jck_new, qids=qids_new)
        tmp.replace(shard_ck)

        with meta_path.open("a") as f:
            for m in new_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        n_rows_saved = len(meta_running)
        next_shard_idx += 1

        cond_pk_running.clear()
        cond_ck_running.clear()
        judg_pk_running.clear()
        judg_ck_running.clear()

        print(f"[multipath-postgen] {ds_k}: shard{next_shard_idx-1:04d} written "
              f"({pk_new.shape[0]} new, total committed={n_rows_saved})",
              flush=True)

    while n_done < n_total:
        msg = out_q.get()
        if msg[0] == "ERR":
            _, task_id, side, err = msg
            results[task_id][f"err_{side}"] = err
            n_done += 1
            print(f"[multipath-postgen] ERR {ds_k} tid={task_id} side={side}: {err[:200]}",
                  flush=True)
            continue
        _, task_id, side, payload = msg
        if payload is None:
            results[task_id][f"err_{side}"] = "empty"
            n_done += 1
            continue
        results[task_id][side] = payload
        n_done += 1
        while next_unconsumed_tid < len(rows):
            r = results[next_unconsumed_tid]
            pk_settled = "PK" in r or "err_PK" in r
            ck_settled = "CK" in r or "err_CK" in r
            if not (pk_settled and ck_settled):
                break
            tid = next_unconsumed_tid
            if "PK" in r and "CK" in r:
                cond_pk_running.append(r["PK"]["h_trace_mp_all"])
                cond_ck_running.append(r["CK"]["h_trace_mp_all"])
                judg_pk_running.append(r["PK"]["h_judge_all"])
                judg_ck_running.append(r["CK"]["h_judge_all"])
                row = rows[tid]
                meta_running.append({
                    "qid": row["qid"], "dataset": ds, "k": K,
                    "pk_correct_alias": row.get("pk_correct_alias"),
                    "ck_correct_alias": row.get("ck_correct_alias"),
                    "p_A_pk": r["PK"]["p_A"], "p_B_pk": r["PK"]["p_B"],
                    "p_A_ck": r["CK"]["p_A"], "p_B_ck": r["CK"]["p_B"],
                    "n_tok_pk": row["n_tok_pk"], "n_tok_ck": row["n_tok_ck"],
                })
            del results[tid]
            next_unconsumed_tid += 1
            if (checkpoint_every > 0
                and (len(meta_running) - n_rows_saved) >= checkpoint_every):
                write_state()
        if n_done % 200 == 0 or n_done == n_total:
            rate = n_done / max(time.time() - t0, 1e-9)
            eta_min = (n_total - n_done) / max(rate, 1e-9) / 60
            print(f"[multipath-postgen] {ds_k}: {n_done}/{n_total} ({rate:.2f}/s, "
                  f"ETA {eta_min:.0f}m)", flush=True)

    write_state()
    skipped = sum(1 for tid in results if "PK" not in results[tid] or "CK" not in results[tid])
    n_total_after = len(meta_running) + len(existing_qids)
    print(f"[multipath-postgen] {ds_k}: wrote {n_total_after} total ({len(meta_running)} new), "
          f"skipped {skipped}", flush=True)
    (out_dir / f"{ds_k}.post_done").touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=list(BACKBONE_PATHS.keys()))
    ap.add_argument("--datasets", nargs="+", default=[
        "confiqa-qa", "confiqa-mr", "confiqa-mc",
        "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
        "triviaqa-huang", "nq-huang", "redditqa",
    ])
    ap.add_argument("--K-list", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--traces-dir-suffix", default="")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--prompts-yaml", default="prompts/multipath_prompts.yaml")
    args = ap.parse_args()

    with open(args.prompts_yaml) as f:
        prompts = yaml.safe_load(f)
    rec = prompts["multipath"]

    from saber.config import DATA_ROOT, ARTIFACTS_DIR
    EVAL = DATA_ROOT / "evaluation" / args.backbone
    DATASET_DIR = DATA_ROOT / "datasets"
    TRACES_DIR = ARTIFACTS_DIR / args.backbone / f"traces{args.traces_dir_suffix}"
    OUT_DIR = ARTIFACTS_DIR / args.backbone / f"cond{args.out_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[multipath-postgen] backbone={args.backbone}", flush=True)
    print(f"[multipath-postgen] traces_dir={TRACES_DIR}", flush=True)
    print(f"[multipath-postgen] out={OUT_DIR}", flush=True)
    print(f"[multipath-postgen] K-list={args.K_list}  datasets={args.datasets}", flush=True)

    model_path = BACKBONE_PATHS[args.backbone]
    ctx = mp.get_context("spawn")
    in_q, out_q = ctx.Queue(), ctx.Queue()
    workers = [
        ctx.Process(target=_worker_loop,
                    args=(g, model_path, rec["prompt_1_reasoning_ck"],
                          rec["prompt_1_reasoning_pk"], rec["prompt_2_judgment"],
                          args.batch_size, in_q, out_q),
                    daemon=True)
        for g in args.gpu_ids
    ]
    for w in workers:
        w.start()
    n_ready = 0
    n_layers_seen = None
    while n_ready < len(workers):
        msg = out_q.get()
        if msg[0] == "READY":
            n_ready += 1
            n_layers_seen = msg[2]
    print(f"[multipath-postgen] {n_ready} workers ready, GPUs={args.gpu_ids}, "
          f"n_layers={n_layers_seen}", flush=True)

    # NB: we deliberately do NOT call _heal_shards in this loop — gzip is
    # single-threaded and blocks the dispatcher, leaving all GPU workers idle
    # for 30s-2min per (ds, K). Shards stay on disk during the run;
    # meta.jsonl is the truth source for what's committed. After all
    # extraction completes, merge shards into the canonical compressed
    # .npz files in a separate offline pass.
    for K in args.K_list:
        for ds in args.datasets:
            done_marker = OUT_DIR / f"{ds}.K{K}.post_done"
            if done_marker.exists():
                print(f"[multipath-postgen] {ds}.K{K} already done, skip", flush=True)
                continue
            _process_one(ds, K, TRACES_DIR, OUT_DIR, EVAL, DATASET_DIR,
                         rec, in_q, out_q,
                         args.batch_size, args.checkpoint_every, len(workers))

    for _ in workers:
        in_q.put(None)
    for w in workers:
        w.join(timeout=10)
    print("[multipath-postgen] all done", flush=True)


if __name__ == "__main__":
    main()
