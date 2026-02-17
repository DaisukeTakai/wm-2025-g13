#!/usr/bin/env python3

"""Annotate (init, goal) image pairs with Azure OpenAI (vision), concurrently.

Input JSONL must contain at least:
  {"id": "...", "task": "mw-reach"|"mw-reach-wall", "init_path": "...", "goal_path": "..."}

Output JSONL (append-only):
  {"id": ..., "task": ..., "instruction_variants": [...], "model": ..., "prompt_version": ...}

This version uses `openai.AsyncAzureOpenAI` to run concurrent requests safely.
Writes are serialized with a single lock to avoid file corruption.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_API_VERSION = "2024-06-01"
PROMPT_VERSION = "v2_async"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _load_image_data_url(path: Path) -> str:
    data = path.read_bytes()
    ext = path.suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext in {"png"}:
        mime = "image/png"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _normalize_text(s: str) -> str:
    s = " ".join((s or "").strip().lower().split())
    while s.endswith((".", "!")):
        s = s[:-1].rstrip()
    return s


def _filter_variants(
    task: str, variants: list[str], min_words: int, max_words: int
) -> list[str]:
    banned = ("image", "camera", "picture", "frame")
    must_wall = ("wall", "obstacle", "avoid", "gap", "opening", "around")

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if not isinstance(v, str):
            continue
        t = v.strip()
        if not t:
            continue
        lt = t.lower()
        if any(b in lt for b in banned):
            continue
        wc = len(t.split())
        if wc < min_words or wc > max_words:
            continue
        if task == "mw-reach-wall":
            if not any(k in lt for k in must_wall):
                continue
        if task == "mw-reach":
            if "wall" in lt or "obstacle" in lt:
                continue
        key = _normalize_text(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _extract_json_from_text(s: str) -> dict[str, Any] | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                return None
        return None


def _cache_key(sample: dict[str, Any], prompt_version: str, model: str, k: int) -> str:
    init_path = Path(sample["init_path"])
    goal_path = Path(sample["goal_path"])
    h = hashlib.sha256()
    h.update(init_path.read_bytes())
    h.update(goal_path.read_bytes())
    h.update(str(sample.get("task", "")).encode("utf-8"))
    h.update(prompt_version.encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(str(k).encode("utf-8"))
    return h.hexdigest()


def _make_prompt(task: str, k: int) -> str:
    if task == "mw-reach-wall":
        constraint = (
            "Every instruction MUST explicitly mention avoiding a wall/obstacle "
            "or going through a gap/opening."
        )
    else:
        constraint = (
            "Do NOT mention walls or obstacles. Focus on reaching the target/goal."
        )
    return (
        "You are writing robot control instructions.\n"
        "Write concise imperative English instructions that transform the initial state into the goal state.\n"
        "Do NOT mention images, cameras, pictures, or frames.\n"
        "Do NOT invent objects beyond: gripper/end-effector, target/goal, wall/obstacle (only if applicable).\n"
        f"Task: {task}\n"
        f"{constraint}\n"
        f'Return ONLY JSON with the exact schema: {{"instruction_variants": [<string>...]}}\n'
        f"Generate exactly {k} distinct instruction_variants."
    )


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


async def _sleep_backoff(attempt: int) -> None:
    base = min(60.0, 1.5**attempt)
    await asyncio.sleep(base + random.random() * 0.25)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate init/goal pairs with Azure OpenAI vision (async)"
    )
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--cache", type=str, default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=25)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--print-errors", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--api-version",
        type=str,
        default=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
    )
    ap.add_argument(
        "--deployment", type=str, default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    )
    ap.add_argument(
        "--endpoint", type=str, default=os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    )
    args = ap.parse_args()

    endpoint = args.endpoint
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = args.deployment
    if not endpoint or not api_key or not deployment:
        raise SystemExit(
            "Missing Azure config. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT."
        )

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    cache_path = Path(args.cache) if args.cache else None

    pairs = _read_jsonl(pairs_path)

    cache_map: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        for rec in _read_jsonl(cache_path):
            ck = rec.get("cache_key")
            if isinstance(ck, str):
                cache_map[ck] = rec

    done_ids: set[str] = set()
    if out_path.exists():
        for rec in _read_jsonl(out_path):
            rid = rec.get("id")
            if isinstance(rid, str):
                done_ids.add(rid)

    # Worklist.
    work: list[dict[str, Any]] = []
    for s in pairs:
        sid = str(s.get("id"))
        if not sid or sid in done_ids:
            continue
        work.append(s)
        if args.limit and len(work) >= int(args.limit):
            break

    total = len(work)
    print(
        f"Annotate start: total={total} k={args.k} concurrency={args.concurrency} deployment={deployment} api_version={args.api_version} "
        f"out={out_path} cache={cache_path if cache_path else '-'}",
        flush=True,
    )

    # Import openai async client.
    try:
        from openai import AsyncAzureOpenAI
    except Exception as e:
        raise SystemExit(
            "Could not import openai AsyncAzureOpenAI. Ensure 'openai' is installed in this environment. "
            f"Import error: {e}"
        )

    cfg = AzureConfig(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        api_version=args.api_version,
    )

    start_time = time.time()
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    write_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0
    cache_hits = 0

    def _fmt_eta(done: int) -> str:
        if done <= 0:
            return "?"
        elapsed = time.time() - start_time
        rate = done / max(elapsed, 1e-9)
        remaining = max(total - done, 0)
        eta_s = remaining / max(rate, 1e-9)
        if eta_s < 60:
            return f"{int(eta_s)}s"
        if eta_s < 3600:
            return f"{int(eta_s // 60)}m{int(eta_s % 60)}s"
        return f"{int(eta_s // 3600)}h{int((eta_s % 3600) // 60)}m"

    async def _annotate_one(client: Any, sample: dict[str, Any]) -> None:
        nonlocal completed, cache_hits
        sid = str(sample.get("id"))
        task = str(sample.get("task"))
        if task not in {"mw-reach", "mw-reach-wall"}:
            raise ValueError(f"Unsupported task: {task} for sample id={sid}")

        init_path = Path(sample["init_path"])
        goal_path = Path(sample["goal_path"])
        if not init_path.exists() or not goal_path.exists():
            raise FileNotFoundError(
                f"Missing init/goal images for id={sid}: {init_path} {goal_path}"
            )

        ck = _cache_key(sample, PROMPT_VERSION, cfg.deployment, args.k)
        if ck in cache_map:
            rec = cache_map[ck]
            variants = rec.get("instruction_variants", [])
            if isinstance(variants, list) and variants:
                out_rec = {
                    "id": sid,
                    "task": task,
                    "instruction_variants": variants,
                    "model": cfg.deployment,
                    "prompt_version": PROMPT_VERSION,
                    "cache_hit": True,
                }
                async with write_lock:
                    _append_jsonl(out_path, out_rec)
                cache_hits += 1
                async with progress_lock:
                    completed += 1
                    if args.log_every > 0 and (
                        completed == 1 or completed % int(args.log_every) == 0
                    ):
                        print(
                            f"[{completed}/{total}] id={sid} task={task} cache_hit=true eta={_fmt_eta(completed)}",
                            flush=True,
                        )
                return

        if args.dry_run:
            out_rec = {
                "id": sid,
                "task": task,
                "instruction_variants": [],
                "model": cfg.deployment,
                "prompt_version": PROMPT_VERSION,
                "dry_run": True,
            }
            async with write_lock:
                _append_jsonl(out_path, out_rec)
            async with progress_lock:
                completed += 1
            return

        init_url = _load_image_data_url(init_path)
        goal_url = _load_image_data_url(goal_path)

        variants: list[str] = []
        attempt = 0
        async with sem:
            while len(variants) < args.k and attempt <= args.max_retries:
                need = args.k - len(variants)
                prompt = _make_prompt(task, k=max(need, 3))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": init_url}},
                            {"type": "image_url", "image_url": {"url": goal_url}},
                        ],
                    }
                ]
                try:
                    resp = await client.chat.completions.create(
                        model=cfg.deployment,
                        messages=messages,
                        max_completion_tokens=512,
                    )
                    content = resp.choices[0].message.content or ""
                    obj = _extract_json_from_text(content)
                    new: list[str] = []
                    if isinstance(obj, dict):
                        raw = obj.get("instruction_variants")
                        if isinstance(raw, list):
                            new = [str(x) for x in raw]
                    merged = variants + new
                    variants = _filter_variants(
                        task, merged, min_words=args.min_words, max_words=args.max_words
                    )
                except Exception as e:
                    if args.print_errors:
                        print(
                            f"retry id={sid} task={task} attempt={attempt}/{args.max_retries} got_k={len(variants)}/{args.k} err={type(e).__name__}",
                            flush=True,
                        )
                    await _sleep_backoff(attempt)
                attempt += 1

        if len(variants) < args.k:
            rec_out = {
                "id": sid,
                "task": task,
                "instruction_variants": variants,
                "model": cfg.deployment,
                "prompt_version": PROMPT_VERSION,
                "incomplete": True,
            }
        else:
            rec_out = {
                "id": sid,
                "task": task,
                "instruction_variants": variants[: args.k],
                "model": cfg.deployment,
                "prompt_version": PROMPT_VERSION,
            }

        async with write_lock:
            _append_jsonl(out_path, rec_out)
            if cache_path:
                rec_cache = dict(rec_out)
                rec_cache["cache_key"] = ck
                rec_cache["init_path"] = str(init_path)
                rec_cache["goal_path"] = str(goal_path)
                _append_jsonl(cache_path, rec_cache)
                cache_map[ck] = rec_cache

        async with progress_lock:
            completed += 1
            if args.log_every > 0 and (
                completed == 1 or completed % int(args.log_every) == 0
            ):
                status = "incomplete" if rec_out.get("incomplete") else "ok"
                print(
                    f"[{completed}/{total}] id={sid} task={task} status={status} k={len(rec_out.get('instruction_variants', []))}/{args.k} eta={_fmt_eta(completed)}",
                    flush=True,
                )

    async def _run() -> None:
        client = AsyncAzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            timeout=args.timeout_s,
        )
        tasks = [asyncio.create_task(_annotate_one(client, s)) for s in work]
        for t in asyncio.as_completed(tasks):
            await t

    asyncio.run(_run())
    elapsed = time.time() - start_time
    print(
        f"Annotate done: completed={completed}/{total} cache_hits={cache_hits} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
