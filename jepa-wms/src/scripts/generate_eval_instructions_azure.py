#!/usr/bin/env python3

"""Generate episode-fixed instructions from init images (no goal leakage).

Input: manifest.jsonl with fields: task, ep_seed, init_image_path
Output: instructions.jsonl with fields: task, ep_seed, instruction, model, prompt_version

Uses Azure OpenAI deployment (e.g., gpt-5.2) with vision.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any


PROMPT_VERSION = "eval_init_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _img_data_url(path: Path) -> str:
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _cache_key(task: str, ep_seed: int, img_bytes: bytes, deployment: str) -> str:
    h = hashlib.sha256()
    h.update(img_bytes)
    h.update(task.encode("utf-8"))
    h.update(str(ep_seed).encode("utf-8"))
    h.update(deployment.encode("utf-8"))
    h.update(PROMPT_VERSION.encode("utf-8"))
    return h.hexdigest()


def _prompt(task: str) -> str:
    if task == "mw-reach-wall":
        must = "Every instruction MUST explicitly mention avoiding a wall/obstacle or going through a gap/opening."
    else:
        must = "Do NOT mention walls or obstacles."
    return (
        "You are writing a single robot control instruction in imperative English.\n"
        "You are given ONLY the initial observation. Do not assume you know the goal image.\n"
        "Your instruction should be generally correct for the task.\n"
        "Do NOT mention images, cameras, pictures, or frames.\n"
        f"Task: {task}\n"
        f"{must}\n"
        'Return ONLY JSON: {"instruction": "..."}'
    )


def _extract_instruction(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        j = json.loads(s)
    except Exception:
        i = s.find("{")
        j0 = s.rfind("}")
        if i >= 0 and j0 > i:
            try:
                j = json.loads(s[i : j0 + 1])
            except Exception:
                return None
        else:
            return None
    ins = j.get("instruction") if isinstance(j, dict) else None
    if not isinstance(ins, str):
        return None
    ins = ins.strip()
    if not ins:
        return None
    return ins


async def _sleep_backoff(attempt: int) -> None:
    base = min(60.0, 1.8**attempt)
    await asyncio.sleep(base + random.random() * 0.25)


async def main_async(args) -> None:
    from openai import AsyncAzureOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2")

    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=float(args.timeout_s),
    )

    manifest = _read_jsonl(Path(args.manifest))
    out_path = Path(args.out)
    cache_path = Path(args.cache) if args.cache else None

    done = set()
    if out_path.exists():
        for r in _read_jsonl(out_path):
            done.add((str(r.get("task")), int(r.get("ep_seed"))))

    cache = {}
    if cache_path and cache_path.exists():
        for r in _read_jsonl(cache_path):
            ck = r.get("cache_key")
            if isinstance(ck, str):
                cache[ck] = r

    sem = asyncio.Semaphore(int(args.concurrency))
    write_lock = asyncio.Lock()
    processed = 0
    total = len(manifest)
    print(
        f"gen start: total={total} concurrency={args.concurrency} deployment={deployment}",
        flush=True,
    )

    async def handle(rec: dict[str, Any]) -> None:
        nonlocal processed
        task = str(rec["task"])
        ep_seed = int(rec["ep_seed"])
        key = (task, ep_seed)
        if key in done:
            return
        img_path = Path(rec["init_image_path"])
        img_bytes = img_path.read_bytes()
        ck = _cache_key(task, ep_seed, img_bytes, deployment)
        if ck in cache:
            out = {
                "task": task,
                "ep_seed": ep_seed,
                "instruction": cache[ck]["instruction"],
                "model": deployment,
                "prompt_version": PROMPT_VERSION,
                "cache_hit": True,
            }
            async with write_lock:
                _append_jsonl(out_path, out)
            done.add(key)
            processed += 1
            if processed % int(args.log_every) == 0:
                print(f"[{processed}/{total}] {task} seed={ep_seed} cache", flush=True)
            return

        data_url = _img_data_url(img_path)
        attempt = 0
        instruction = None
        async with sem:
            while instruction is None and attempt <= int(args.max_retries):
                try:
                    resp = await client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": _prompt(task)},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_url},
                                    },
                                ],
                            }
                        ],
                        max_completion_tokens=200,
                    )
                    instruction = _extract_instruction(
                        resp.choices[0].message.content or ""
                    )
                except Exception:
                    instruction = None
                if instruction is None:
                    await _sleep_backoff(attempt)
                attempt += 1

        if instruction is None:
            instruction = (
                "Reach the target."
                if task == "mw-reach"
                else "Reach the target while avoiding the wall."
            )

        out = {
            "task": task,
            "ep_seed": ep_seed,
            "instruction": instruction,
            "model": deployment,
            "prompt_version": PROMPT_VERSION,
        }
        async with write_lock:
            _append_jsonl(out_path, out)
            if cache_path:
                c = dict(out)
                c["cache_key"] = ck
                _append_jsonl(cache_path, c)
                cache[ck] = c

        done.add(key)
        processed += 1
        if processed % int(args.log_every) == 0:
            print(f"[{processed}/{total}] {task} seed={ep_seed}", flush=True)

    tasks = [asyncio.create_task(handle(r)) for r in manifest]
    for t in asyncio.as_completed(tasks):
        await t

    print(f"gen done: wrote={len(done)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate eval instructions from init images (Azure)"
    )
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cache", type=str, default="")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout-s", type=float, default=120)
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
