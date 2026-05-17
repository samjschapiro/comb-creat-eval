"""Minimal OpenAI-compatible server with dynamic batching.

Replaces vLLM for serving HF checkpoints on Lambda's driver-570 hosts
(vLLM's pinned torch is too new there). Runs on the project env
(torch 2.6.0+cu124, verified on driver 570).

Concurrent incoming requests are collected into batches and run through
one model.generate call (left padding), then dispatched back to each
waiting request. This is many times faster than serializing one request
at a time. Implements GET /v1/models and POST /v1/chat/completions,
enough for the dat_eval / new_tests harnesses (they speak the OpenAI
chat API via LLM_BASE_URL).

Usage:
    uv run python src/creativity_rl/scripts/serve_hf_openai.py \\
        --model <hf_id> [--port 8000] [--max-batch 24]
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_WAIT_S = 0.15  # how long to let a batch fill before running it


@dataclass
class Req:
    messages: list
    temperature: float
    max_tokens: int
    top_p: float
    future: asyncio.Future = field(default=None)


def build_app(model_id: str, max_batch: int) -> FastAPI:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # correct for batched decoder generation
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    app = FastAPI()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_batch(reqs: list[Req]) -> list[str]:
        texts = [
            tok.apply_chat_template(
                r.messages, tokenize=False, add_generation_prompt=True
            )
            for r in reqs
        ]
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1536
        ).to("cuda")
        # All reqs in a bucket share sampling params (bucketed by caller).
        r0 = reqs[0]
        do_sample = r0.temperature is not None and r0.temperature > 0
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=int(r0.max_tokens or 512),
                do_sample=do_sample,
                temperature=r0.temperature if do_sample else None,
                top_p=r0.top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]  # left-padded: same for all rows
        return [tok.decode(o[in_len:], skip_special_tokens=True) for o in out]

    async def worker():
        loop = asyncio.get_event_loop()
        while True:
            first: Req = await queue.get()
            batch = [first]
            t0 = loop.time()
            while len(batch) < max_batch:
                remaining = MAX_WAIT_S - (loop.time() - t0)
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            # Bucket by sampling params so one generate call is valid.
            buckets: dict[Any, list[Req]] = {}
            for r in batch:
                key = (r.temperature, r.max_tokens, r.top_p)
                buckets.setdefault(key, []).append(r)
            for reqs in buckets.values():
                try:
                    outs = await loop.run_in_executor(None, _run_batch, reqs)
                    for r, txt in zip(reqs, outs):
                        if not r.future.done():
                            r.future.set_result(txt)
                except Exception as e:  # surface to each waiting request
                    for r in reqs:
                        if not r.future.done():
                            r.future.set_exception(e)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(worker())

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        req = Req(
            messages=body["messages"],
            temperature=body.get("temperature", 1.0),
            max_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 512,
            top_p=body.get("top_p", 1.0),
            future=asyncio.get_event_loop().create_future(),
        )
        await queue.put(req)
        content = await req.future
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-batch", type=int, default=8)
    a = p.parse_args()
    print(f"loading {a.model} ...", flush=True)
    app = build_app(a.model, a.max_batch)
    print(f"READY: {a.model} on :{a.port} (max_batch={a.max_batch})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
