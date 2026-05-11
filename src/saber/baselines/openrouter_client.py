"""Async OpenRouter client with concurrent dispatch + retry + cost tracking.

Supports multiple API keys: rate limits are per-key on OpenRouter, so providing
N keys increases effective concurrency by N×. Each key has its own semaphore.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import time
from dataclasses import dataclass, field

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class CallResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_sec: float = 0.0
    error: str | None = None
    raw: dict | None = None


@dataclass
class CallStats:
    n_calls: int = 0
    n_errors: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 64,
    temperature: float = 0.0,
    max_retries: int = 4,
    timeout: float = 90.0,
) -> CallResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {"include": True},
    }
    delay = 1.0
    last_err: str | None = None
    t0 = time.time()
    for attempt in range(max_retries + 1):
        try:
            r = await client.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                if "choices" not in d:
                    last_err = f"no choices: {d}"
                else:
                    text = d["choices"][0]["message"]["content"] or ""
                    usage = d.get("usage", {}) or {}
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    tt = int(usage.get("total_tokens") or pt + ct)
                    cost = float(usage.get("cost") or 0.0)
                    return CallResult(
                        text=text, prompt_tokens=pt, completion_tokens=ct,
                        total_tokens=tt, cost=cost,
                        latency_sec=time.time() - t0, raw=d,
                    )
            elif r.status_code in (429, 502, 503, 504):
                last_err = f"http {r.status_code}: {r.text[:200]}"
            else:
                last_err = f"http {r.status_code}: {r.text[:300]}"
                # non-retryable client error → break
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    break
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_retries:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    return CallResult(text="", error=last_err or "unknown error",
                      latency_sec=time.time() - t0)


class OpenRouterPool:
    """Multi-key pool: each key has its own semaphore (N keys → N× concurrency).

    Rate limit on OpenRouter is per-account (per-key). Round-robin dispatch
    spreads load across keys. Total in-flight calls bounded by
    `concurrency_per_key × len(api_keys)`.

    Auto-fallback: when a request to a `:free` model is rate-limited (429
    with "rate-limit" in error), retries with the corresponding paid model.
    Use when caller explicitly specifies a `:free` model id; saves cost when
    free tier has capacity, falls back to paid when exhausted.
    """

    # Maps `:free` variant → paid variant for auto-fallback on rate limit.
    AUTO_FALLBACK = {
        "meta-llama/llama-3.2-3b-instruct:free": "meta-llama/llama-3.2-3b-instruct",
    }

    def __init__(
        self,
        api_keys: list[str] | str | None = None,
        concurrency_per_key: int = 16,
    ):
        if api_keys is None:
            env = os.environ.get("OPENROUTER_API_KEYS") or os.environ.get("OPENROUTER_API_KEY", "")
            api_keys = [k.strip() for k in env.split(",") if k.strip()]
        elif isinstance(api_keys, str):
            api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        if not api_keys:
            raise ValueError("No OpenRouter API keys provided. Set OPENROUTER_API_KEYS=k1,k2,... or pass api_keys=")
        self.api_keys = api_keys
        self.concurrency_per_key = concurrency_per_key
        self.semaphores = [asyncio.Semaphore(concurrency_per_key) for _ in api_keys]
        self._round_robin = itertools.cycle(range(len(api_keys)))
        self._dispatch_lock = asyncio.Lock()  # to safely advance round-robin
        self.client = httpx.AsyncClient()
        self.stats: dict[str, CallStats] = {}
        # Per-key stats for monitoring per-account quota
        self.key_stats: list[CallStats] = [CallStats() for _ in api_keys]

    async def _next_key_idx(self) -> int:
        async with self._dispatch_lock:
            return next(self._round_robin)

    async def _call_with_model(self, model: str, messages, max_tokens, temperature):
        """Single attempt with model+key rotation, no fallback."""
        idx = await self._next_key_idx()
        async with self.semaphores[idx]:
            res = await call_openrouter(
                self.client, self.api_keys[idx], model, messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        # Update per-model stats
        st = self.stats.setdefault(model, CallStats())
        st.n_calls += 1
        if res.error:
            st.n_errors += 1
        else:
            st.total_prompt_tokens += res.prompt_tokens
            st.total_completion_tokens += res.completion_tokens
            st.total_cost += res.cost
            st.latencies.append(res.latency_sec)
        # Per-key stats
        ks = self.key_stats[idx]
        ks.n_calls += 1
        if res.error:
            ks.n_errors += 1
        else:
            ks.total_cost += res.cost
        return res

    async def call(
        self, model: str, messages: list[dict],
        max_tokens: int = 64, temperature: float = 0.0,
    ) -> CallResult:
        """Call model. If model is in AUTO_FALLBACK and hits rate limit, retry on fallback."""
        res = await self._call_with_model(model, messages, max_tokens, temperature)
        # Auto-fallback: :free → paid when rate-limited
        if res.error and model in self.AUTO_FALLBACK:
            err_low = (res.error or "").lower()
            if "429" in err_low or "rate-limit" in err_low or "rate limit" in err_low \
                    or "temporarily rate" in err_low:
                fallback_model = self.AUTO_FALLBACK[model]
                res2 = await self._call_with_model(fallback_model, messages, max_tokens, temperature)
                if not res2.error:
                    return res2
                # Both failed; return the fallback's error (paid is more reliable signal)
                return res2
        return res

    async def close(self):
        await self.client.aclose()

    def summary(self) -> dict:
        out = {}
        for model, st in self.stats.items():
            lats = st.latencies or [0.0]
            out[model] = {
                "calls": st.n_calls,
                "errors": st.n_errors,
                "prompt_tokens": st.total_prompt_tokens,
                "completion_tokens": st.total_completion_tokens,
                "cost_usd": round(st.total_cost, 6),
                "p50_latency_sec": round(sorted(lats)[len(lats) // 2], 2),
                "p95_latency_sec": round(sorted(lats)[int(len(lats) * 0.95)], 2),
                "mean_latency_sec": round(sum(lats) / len(lats), 2),
            }
        return out

    def key_summary(self) -> list[dict]:
        return [
            {
                "key_idx": i,
                "key_prefix": self.api_keys[i][:18] + "...",
                "calls": s.n_calls,
                "errors": s.n_errors,
                "cost_usd": round(s.total_cost, 6),
            }
            for i, s in enumerate(self.key_stats)
        ]
