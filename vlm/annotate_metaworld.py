from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_int_list(csv: str) -> list[int]:
    items: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _jsonl_iter(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_existing_episode_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # allow resuming even if the last line is truncated
                continue
            ep = row.get("episode_idx")
            if isinstance(ep, int):
                seen.add(ep)
    return seen


def _maybe_import_hf() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `datasets`. Install with: pip/uv install datasets pyarrow"
        ) from e
    return load_dataset, None


def _decode_mp4_bytes_to_frames(video_bytes: bytes) -> list[Any]:
    """Return a list of numpy frames (H,W,C uint8)."""
    import io

    import imageio  # type: ignore

    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    return frames


def _extract_frames(row_video: Any, frame_indices: list[int]) -> list[Any]:
    """Extract selected frames from HF `video` column.

    `row_video` can be:
    - dict with key 'bytes'
    - a torchcodec VideoDecoder-like object (supports __len__ and __getitem__ returning C,H,W)
    """
    if isinstance(row_video, dict) and "bytes" in row_video:
        frames = _decode_mp4_bytes_to_frames(row_video["bytes"])
        return [frames[i] for i in frame_indices]

    # torchcodec auto-decoded
    frames_out: list[Any] = []
    for i in frame_indices:
        f = row_video[i]
        # VideoDecoder[i] returns an object with `.data` tensor in C,H,W
        tensor = getattr(f, "data", f)
        frame_np = tensor.permute(1, 2, 0).cpu().numpy()
        frames_out.append(frame_np)
    return frames_out


def _as_pil_images(frames: list[Any]) -> list[Any]:
    from PIL import Image

    images = []
    for f in frames:
        images.append(Image.fromarray(f).convert("RGB"))
    return images


def _safe_json_extract(text: str) -> str | None:
    """Extract the first JSON object from text, if present."""
    # naive but robust for typical LLM outputs
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        json.loads(candidate)
    except Exception:
        return None
    return candidate


def _normalize_label_choice(label: str, labels: list[str]) -> str:
    label = label.strip()
    if label in labels:
        return label
    # allow quoting
    label2 = label.strip('"').strip("'")
    if label2 in labels:
        return label2
    # common JSON / dict fallback
    if label.upper() == "UNKNOWN":
        return "UNKNOWN"
    return "UNKNOWN"


def _extract_label_from_text(text: str, labels: list[str]) -> str:
    """Heuristic label extraction when the model doesn't follow JSON."""
    # Prefer exact label mentions in the raw text.
    for lab in labels:
        if lab == "UNKNOWN":
            continue
        if lab in text:
            return lab
    # Common patterns: label: xxx
    m = re.search(r"\blabel\s*[:=]\s*([\w\-]+)", text, flags=re.IGNORECASE)
    if m:
        return _normalize_label_choice(m.group(1), labels)
    return "UNKNOWN"


def _extract_first_int(text: str) -> int | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        if re.fullmatch(r"\d{1,3}", lines[0]):
            return int(lines[0])
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    return int(m.group(1))


@dataclasses.dataclass(frozen=True)
class ClosedSetResult:
    label: str
    confidence: float | None
    top_k: list[dict[str, Any]] | None
    raw_initial: str
    raw_refined: str | None


@dataclasses.dataclass(frozen=True)
class OpenSetResult:
    instruction: str
    confidence: float | None
    raw_initial: str
    raw_refined: str | None


class Qwen3VLMoeAnnotator:
    def __init__(
        self,
        model_id: str,
        attn_implementation: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing inference deps. Install: transformers torch qwen-vl-utils"
            ) from e

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dtype": "auto",
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model_id = model_id
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        # batch generation: left padding
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"

        self._torch = torch

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing qwen-vl-utils. Install: pip/uv install qwen-vl-utils"
            ) from e

        self.process_vision_info = process_vision_info

    def _generate(
        self,
        messages: list[list[dict[str, Any]]],
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        # messages: batch of conversations
        # qwen-vl-utils: handle images/videos; for Qwen3-VL set image_patch_size=16 and return metadata
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = self.process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
        }
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        # Bias toward short/format-following outputs.
        eos_id = getattr(self.processor, "tokenizer", None)
        if eos_id is not None and getattr(self.processor.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id

        # Mild repetition penalty helps reduce verbose "thinking".
        gen_kwargs.setdefault("repetition_penalty", 1.05)

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # trim prompt
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return texts

    def _generate_text_only(
        self,
        text_prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = [[{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]]
        return self._generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)[0]

    def infer_goal_summary(self, images: list[Any], max_new_tokens: int) -> str:
        """Get a short, text-only goal summary from frames.

        This is used because the Thinking variants may ignore strict output formatting;
        a second-stage text-only normalization is more reliable.
        """
        prompt = (
            "Summarize the episode goal (task) from the sequence of frames.\n"
            "Output ONLY one short English sentence describing the intended goal.\n"
            "No preamble (e.g., no 'Got it'). No quotes."
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": images, "sample_fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]
        raw = self._generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)[0].strip()
        first_line = raw.splitlines()[0].strip().strip('"').strip("'")
        bad_prefixes = ("got it", "okay", "sure", "the user", "we are")
        if first_line.lower().startswith(bad_prefixes):
            refine_prompt = (
                "Rewrite the text below into ONE short English sentence describing the episode goal/task.\n"
                "Output ONLY the sentence. No preamble. No quotes.\n\n"
                f"TEXT: {first_line}\n"
            )
            refined = self._generate_text_only(refine_prompt, max_new_tokens=32).strip()
            first_line = refined.splitlines()[0].strip().strip('"').strip("'")

        return first_line

    def infer_goal_summaries(self, images_batch: list[list[Any]], max_new_tokens: int) -> list[str]:
        """Batch version of infer_goal_summary (vision-heavy; batching helps throughput)."""
        prompt = (
            "Summarize the episode goal (task) from the sequence of frames.\n"
            "Output ONLY one short English sentence describing the intended goal.\n"
            "No preamble (e.g., no 'Got it'). No quotes."
        )
        messages: list[list[dict[str, Any]]] = []
        for imgs in images_batch:
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": imgs, "sample_fps": 1.0},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        raws = self._generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)
        out: list[str] = []
        for raw in raws:
            first_line = raw.strip().splitlines()[0].strip().strip('"').strip("'")
            bad_prefixes = ("got it", "okay", "sure", "the user", "we are")
            if first_line.lower().startswith(bad_prefixes):
                refine_prompt = (
                    "Rewrite the text below into ONE short English sentence describing the episode goal/task.\n"
                    "Output ONLY the sentence. No preamble. No quotes.\n\n"
                    f"TEXT: {first_line}\n"
                )
                refined = self._generate_text_only(refine_prompt, max_new_tokens=32).strip()
                first_line = refined.splitlines()[0].strip().strip('"').strip("'")
            out.append(first_line)
        return out

    def infer_closed_set(
        self,
        images: list[Any],
        labels: list[str],
        goal_summary: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> ClosedSetResult:
        # NOTE: The "Thinking" variants often ignore strict JSON-only instructions.
        # To make parsing robust, we ask for a single integer choice.
        # Output may still contain extra text; we extract the first integer token.
        numbered = [f"{i+1}: {lab}" for i, lab in enumerate(labels)]
        labels_block = "\n".join(numbered)
        prompt = (
            "You are a robotics task classifier.\n"
            "Given the goal summary, choose the best matching task label by ID.\n\n"
            "Rules:\n"
            "- Focus on the goal/task, not low-level motion.\n"
            "- If unclear, choose the ID for UNKNOWN.\n\n"
            f"TASK_LABELS (choose ONE ID):\n{labels_block}\n\n"
            f"GOAL_SUMMARY: {goal_summary}\n\n"
            "Output format:\n"
            "- First line: the integer ID only (e.g., 17)\n"
            f"- Second line (optional): up to {top_k} alternative IDs separated by commas\n"
        )

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]
        # Nudge the decoder to be concise (helps "Thinking" variants follow the format).
        out_text = self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)[0]

        # parse first integer as label id
        first_int = _extract_first_int(out_text)

        # Keep for parsing alternative IDs
        lines = [ln.strip() for ln in out_text.splitlines() if ln.strip()]

        refined_text: str | None = None
        if first_int is None or not (1 <= first_int <= len(labels)):
            refine_prompt = (
                "Extract the chosen TASK label ID from the assistant output below.\n"
                "Output ONLY the integer ID on the first line. No other text.\n\n"
                "TASK_LABELS:\n"
                f"{labels_block}\n\n"
                "ASSISTANT_OUTPUT:\n"
                f"{out_text}\n"
            )
            refined_text = self._generate_text_only(refine_prompt, max_new_tokens=16)
            first_int = _extract_first_int(refined_text)

        if first_int is None or not (1 <= first_int <= len(labels)):
            label = _extract_label_from_text(refined_text or out_text, labels)
            return ClosedSetResult(
                label=label,
                confidence=None,
                top_k=None,
                raw_initial=out_text,
                raw_refined=refined_text,
            )

        label = labels[first_int - 1]

        # parse optional second-line alternatives
        alt_ids: list[int] = []
        if len(lines) >= 2:
            for part in re.split(r"\s*,\s*", lines[1]):
                if not part:
                    continue
                m2 = re.fullmatch(r"\d{1,3}", part)
                if not m2:
                    continue
                v = int(m2.group(0))
                if 1 <= v <= len(labels) and v != first_int:
                    alt_ids.append(v)
                if len(alt_ids) >= top_k:
                    break

        topk = [{"label": labels[i - 1], "confidence": None} for i in alt_ids] if alt_ids else None
        return ClosedSetResult(
            label=label,
            confidence=None,
            top_k=topk,
            raw_initial=out_text,
            raw_refined=refined_text,
        )

    def infer_open_set(
        self,
        goal_summary: str,
        max_new_tokens: int,
        temperature: float,
    ) -> OpenSetResult:
        # Text-only conversion; keep it simple and store raw text for later filtering.
        prompt = (
            "Convert the goal summary into ONE English imperative instruction (<= 12 words).\n"
            "Output ONLY the instruction. No preamble. No explanations.\n"
            "Do NOT mention GOAL_SUMMARY. Do NOT quote the input.\n\n"
            f"GOAL: {goal_summary}\n"
        )
        out_text = self._generate_text_only(prompt, max_new_tokens=max_new_tokens).strip()
        instruction = out_text.splitlines()[0].strip().strip('"').strip("'")

        def _looks_bad(s: str) -> bool:
            sl = s.lower()
            return (
                sl.startswith(("got it", "okay", "sure", "we are", "the user", "input"))
                or "goal_summary" in sl
                or "we are given" in sl
            )

        refined: str | None = None
        if _looks_bad(instruction) or len(instruction.split()) > 14:
            refine_prompt = (
                "Rewrite into ONE English imperative instruction (<= 12 words).\n"
                "Output ONLY the instruction.\n\n"
                f"TEXT: {goal_summary}\n"
            )
            refined = self._generate_text_only(refine_prompt, max_new_tokens=max_new_tokens).strip()
            instruction = refined.splitlines()[0].strip().strip('"').strip("'")

        return OpenSetResult(
            instruction=instruction,
            confidence=None,
            raw_initial=out_text,
            raw_refined=refined,
        )


def _build_labels(ds: Any) -> list[str]:
    # Prefer using the dataset's task column if present.
    if "task" not in ds.column_names:
        return ["UNKNOWN"]
    # Avoid ds[i] access here: it may trigger video decoding (slow + extra deps).
    try:
        labels_raw = ds.unique("task")
    except Exception:
        labels_raw = ds["task"]

    tasks = [x for x in labels_raw if isinstance(x, str)]
    labels = sorted(set(tasks))
    labels.append("UNKNOWN")
    return labels


def _get_row_task_gt(row: dict[str, Any]) -> str | None:
    v = row.get("task")
    return v if isinstance(v, str) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate Metaworld parquet with VLM outputs (jsonl).")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to datasets/Metaworld/data")
    parser.add_argument("--out", type=str, required=True, help="Output jsonl path")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="HF model id (BF16, Transformers-loadable)",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        help="Attention implementation: sdpa | flash_attention_2 | none",
    )
    parser.add_argument("--frames", type=str, default="0,24,49,74,98", help="Comma-separated frame indices")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy")
    parser.add_argument("--top-k", type=int, default=5, help="closed-set top_k")
    parser.add_argument("--batch-size", type=int, default=1, help="episodes per batch (currently 1 recommended)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N episodes (0=all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "closed", "open"],
        help="Which annotations to produce",
    )
    args = parser.parse_args(argv)

    # Basic safety: vjepa2/.venv or other giant dirs can confuse env;
    # ensure we don't accidentally run with system python missing deps.
    if os.environ.get("VLM_QUIET") != "1":
        _eprint(f"python: {sys.executable}")
        _eprint(f"cwd: {os.getcwd()}")

    if args.batch_size < 1:
        _eprint("--batch-size must be >= 1")
        return 2

    attn: str | None
    if args.attn == "none":
        attn = None
    else:
        attn = args.attn

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _eprint(f"data-dir not found: {data_dir}")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = _parse_int_list(args.frames)
    if not frame_indices:
        _eprint("--frames is empty")
        return 2

    load_dataset, _ = _maybe_import_hf()
    ds = load_dataset("parquet", data_dir=str(data_dir), split="train")

    labels = _build_labels(ds) if args.mode in ("both", "closed") else ["UNKNOWN"]
    _eprint(f"Loaded dataset: {len(ds)} episodes")
    if args.mode in ("both", "closed"):
        _eprint(f"Closed-set labels: {len(labels)} (includes UNKNOWN)")

    seen = _load_existing_episode_ids(out_path) if args.resume else set()
    if seen:
        _eprint(f"Resume enabled: will skip {len(seen)} already-annotated episodes")

    annotator = Qwen3VLMoeAnnotator(model_id=args.model_id, attn_implementation=attn)

    prompt_version = "v1"
    n_total = len(ds) if args.limit <= 0 else min(len(ds), args.limit)

    with out_path.open("a", encoding="utf-8") as out_f:
        idx = 0
        processed = 0
        while idx < n_total:
            # Build a batch of indices, skipping already-seen ones.
            batch_indices: list[int] = []
            while idx < n_total and len(batch_indices) < args.batch_size:
                if idx not in seen:
                    batch_indices.append(idx)
                idx += 1

            if not batch_indices:
                continue

            rows: list[dict[str, Any]] = []
            pil_images_batch: list[list[Any]] = []

            for bi in batch_indices:
                row = ds[bi]
                video = row.get("video")
                if video is None:
                    _eprint(f"episode {bi}: missing video column; skipping")
                    continue
                try:
                    frames = _extract_frames(video, frame_indices)
                    pil_images = _as_pil_images(frames)
                except Exception as e:
                    _eprint(f"episode {bi}: failed to decode frames: {e}")
                    continue
                rows.append(row)
                pil_images_batch.append(pil_images)

            if not rows:
                continue

            # Vision-heavy step: batch goal summary.
            goal_summaries = annotator.infer_goal_summaries(pil_images_batch, max_new_tokens=32)

            for bi, row, pil_images, goal_summary in zip(batch_indices, rows, pil_images_batch, goal_summaries):
                task_gt = _get_row_task_gt(row)
                episode_id = row.get("episode") if isinstance(row.get("episode"), int) else None
                seed = row.get("seed") if isinstance(row.get("seed"), int) else None

                closed_res: ClosedSetResult | None = None
                open_res: OpenSetResult | None = None

                if args.mode in ("both", "closed"):
                    closed_res = annotator.infer_closed_set(
                        images=pil_images,
                        labels=labels,
                        goal_summary=goal_summary,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )

                if args.mode in ("both", "open"):
                    open_res = annotator.infer_open_set(
                        goal_summary=goal_summary,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )

                record: dict[str, Any] = {
                    "episode_idx": bi,
                    "episode": episode_id,
                    "seed": seed,
                    "task_gt": task_gt,
                    "frame_indices": frame_indices,
                    "goal_summary": goal_summary,
                    "model": {"id": args.model_id, "prompt_version": prompt_version},
                }
                if closed_res is not None:
                    record["closed"] = {
                        "label": closed_res.label,
                        "confidence": closed_res.confidence,
                        "top_k": closed_res.top_k,
                        "raw_initial": closed_res.raw_initial,
                        "raw_refined": closed_res.raw_refined,
                    }
                if open_res is not None:
                    record["open"] = {
                        "instruction": open_res.instruction,
                        "confidence": open_res.confidence,
                        "raw_initial": open_res.raw_initial,
                        "raw_refined": open_res.raw_refined,
                    }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                if processed % 10 == 0:
                    _eprint(f"processed {processed}/{n_total}")
                if processed % 100 == 0:
                    out_f.flush()
                    os.fsync(out_f.fileno())

            out_f.flush()

    _eprint("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
