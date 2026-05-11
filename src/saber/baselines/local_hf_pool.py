"""Local-GPU pool: same `.call()` interface as OpenRouterPool, but routes to
N HF transformers worker processes (one per GPU).

Each worker pins a single GPU (CUDA_VISIBLE_DEVICES set before torch import),
loads its own copy of the model, and consumes (task_id, messages, max_tokens,
temperature) from a shared input queue. An asyncio bridge in the parent
process exposes the same `await pool.call(...)` API the existing
`baselines/runners.py` expects, so prompts + post-processing are unchanged
between OpenRouter and local backends.

Concurrency model:
- 8 workers × 1 model copy each (3B model fp16 ≈ 6GB on a 23GB A5000)
- mp.Queue auto-balances load (faster GPUs pull more tasks)
- Within each worker, generation is sequential (one request at a time);
  throughput comes from running 8 workers in parallel
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from queue import Empty

from saber.baselines.openrouter_client import CallResult, CallStats


def _worker_loop(gpu_id: int, model_name: str, in_q, out_q,
                 batch_size: int | None = None, batch_timeout: float = 0.4,
                 max_input_len: int = 4096) -> None:
    """One worker process: load model on assigned GPU, serve **batched** generate.

    Collects up to `batch_size` requests within `batch_timeout` seconds before
    issuing one `model.generate` call. All requests in a batch share
    `max_new_tokens = max(batch.max_tokens)` and `temperature` (0.0 throughout
    our setup). Padding is left-padded so causal generation continues from the
    rightmost prompt token of each row.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if batch_size is None:
        batch_size = int(os.environ.get("LOCAL_HF_BATCH_SIZE", "32"))
    # Import torch / transformers AFTER setting CUDA_VISIBLE_DEVICES so they
    # only see the one assigned GPU as device 0.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from queue import Empty as QEmpty

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched causal LM generation
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    out_q.put(("READY", gpu_id))

    poison = False
    while not poison:
        # ── collect a batch ────────────────────────────────────────────────
        batch: list[tuple] = []
        deadline = time.time() + batch_timeout
        first = in_q.get()  # block until we have at least one task
        if first is None:
            out_q.put(("BYE", gpu_id))
            return
        batch.append(first)
        while len(batch) < batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = in_q.get(timeout=remaining)
            except QEmpty:
                break
            if item is None:
                poison = True  # propagate AFTER processing this batch
                break
            batch.append(item)

        # ── batched generate ───────────────────────────────────────────────
        t0 = time.time()
        try:
            tids = [b[0] for b in batch]
            messages_list = [b[1] for b in batch]
            max_tok_batch = max(b[2] for b in batch)
            temp = batch[0][3]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages_list]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_input_len).to("cuda")
            attn_lens = enc["attention_mask"].sum(dim=1)  # actual tokens per row
            prompt_padded_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                kwargs = dict(
                    **enc, max_new_tokens=max_tok_batch,
                    pad_token_id=tok.pad_token_id,
                )
                if temp and temp > 0:
                    kwargs.update(do_sample=True, temperature=temp)
                else:
                    kwargs.update(do_sample=False)
                out = model.generate(**kwargs)
            for i, tid in enumerate(tids):
                new_ids = out[i, prompt_padded_len:]
                # strip pad/eos at end
                eos_id = tok.eos_token_id
                if eos_id is not None:
                    eos_pos = (new_ids == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos):
                        new_ids = new_ids[: int(eos_pos[0].item())]
                text = tok.decode(new_ids, skip_special_tokens=True)
                in_len = int(attn_lens[i].item())
                ct = int(new_ids.shape[0])
                out_q.put((tid, {
                    "text": text,
                    "prompt_tokens": in_len,
                    "completion_tokens": ct,
                    "total_tokens": in_len + ct,
                    "cost": 0.0,
                    "latency_sec": (time.time() - t0) / max(1, len(batch)),
                    "error": None,
                }))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            for tid in (b[0] for b in batch):
                out_q.put((tid, {
                    "text": "", "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "cost": 0.0,
                    "latency_sec": 0.0, "error": err,
                }))

    out_q.put(("BYE", gpu_id))


class LocalHFPool:
    """8-worker HF transformers pool with asyncio-compatible `.call()`."""

    def __init__(self, model_name: str, gpu_ids: list[int] | None = None):
        if gpu_ids is None:
            gpu_ids = list(range(8))
        self.model_name = model_name
        self.gpu_ids = gpu_ids

        ctx = mp.get_context("spawn")
        self.in_q = ctx.Queue()
        self.out_q = ctx.Queue()
        self.workers = [
            ctx.Process(
                target=_worker_loop,
                args=(g, model_name, self.in_q, self.out_q),
                daemon=True,
            )
            for g in gpu_ids
        ]
        for w in self.workers:
            w.start()

        # Wait for all workers to finish loading the model.
        n_ready = 0
        t0 = time.time()
        while n_ready < len(self.workers):
            tag, gid = self.out_q.get()
            if tag == "READY":
                n_ready += 1
                print(f"[LocalHFPool] worker gpu={gid} ready ({n_ready}/{len(self.workers)}, "
                      f"{time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"[LocalHFPool] unexpected init message: {tag} {gid}", flush=True)

        # Async bookkeeping
        self._futures: dict[int, asyncio.Future] = {}
        self._task_counter = 0
        self._reader_task: asyncio.Task | None = None
        self._reader_lock = asyncio.Lock()

        # Stats (mirror OpenRouterPool)
        self.stats: dict[str, CallStats] = {}
        self.server_stats: list[CallStats] = [CallStats() for _ in gpu_ids]

    async def _ensure_reader(self) -> None:
        async with self._reader_lock:
            if self._reader_task is None or self._reader_task.done():
                self._reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            tid, res = await loop.run_in_executor(None, self.out_q.get)
            fut = self._futures.pop(tid, None)
            if fut is not None and not fut.done():
                fut.set_result(res)

    async def call(
        self, model: str, messages: list[dict],
        max_tokens: int = 64, temperature: float = 0.0,
    ) -> CallResult:
        await self._ensure_reader()
        loop = asyncio.get_running_loop()
        tid = self._task_counter
        self._task_counter += 1
        fut: asyncio.Future = loop.create_future()
        self._futures[tid] = fut
        # Round-robin GPU is automatic via mp.Queue (workers pull as they free).
        self.in_q.put((tid, messages, max_tokens, temperature))
        d = await fut
        result = CallResult(
            text=d["text"], prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            total_tokens=d["total_tokens"], cost=d["cost"],
            latency_sec=d["latency_sec"], error=d["error"],
        )
        st = self.stats.setdefault(model, CallStats())
        st.n_calls += 1
        if result.error:
            st.n_errors += 1
        else:
            st.total_prompt_tokens += result.prompt_tokens
            st.total_completion_tokens += result.completion_tokens
            st.latencies.append(result.latency_sec)
        return result

    async def close(self) -> None:
        # Send poison pills, drain reader, join workers.
        for _ in self.workers:
            self.in_q.put(None)
        if self._reader_task is not None:
            self._reader_task.cancel()
        for w in self.workers:
            w.join(timeout=15)

    def summary(self) -> dict:
        out: dict = {}
        for model, st in self.stats.items():
            lats = st.latencies or [0.0]
            out[model] = {
                "calls": st.n_calls,
                "errors": st.n_errors,
                "prompt_tokens": st.total_prompt_tokens,
                "completion_tokens": st.total_completion_tokens,
                "cost_usd": 0.0,
                "p50_latency_sec": round(sorted(lats)[len(lats) // 2], 2),
                "p95_latency_sec": round(sorted(lats)[int(len(lats) * 0.95)], 2),
                "mean_latency_sec": round(sum(lats) / len(lats), 2),
            }
        return out

    def key_summary(self) -> list[dict]:
        return [
            {"key_idx": i, "key_prefix": f"gpu={g}",
             "calls": s.n_calls, "errors": s.n_errors, "cost_usd": 0.0}
            for i, (g, s) in enumerate(zip(self.gpu_ids, self.server_stats))
        ]
