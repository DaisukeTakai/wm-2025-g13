from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _task_name_to_tokens(task_name: str) -> List[str]:
    # Example: "mw-reach-wall" -> ["reach", "wall"]
    # Keep it intentionally simple and deterministic.
    if task_name.startswith("mw-"):
        task_name = task_name[len("mw-") :]
    task_name = task_name.replace("_", "-")
    toks = [t for t in task_name.split("-") if t]
    return toks


@dataclass(frozen=True)
class TaskTokenizer:
    """Minimal tokenizer for task-name templates.

    This intentionally avoids external language models: it maps task-name tokens
    (e.g. "mw-reach-wall" -> "reach wall") into integer ids.
    """

    vocab: List[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def from_task_set(cls, task_set: str) -> "TaskTokenizer":
        # Keep the import local to avoid coupling plan_common to evals at import time.
        from evals.simu_env_planning.planning.common import TASK_SET

        if task_set not in TASK_SET:
            raise KeyError(
                f"Unknown task_set: {task_set}. Available: {list(TASK_SET.keys())}"
            )
        tokens = set()
        for t in TASK_SET[task_set]:
            tokens.update(_task_name_to_tokens(t))
        vocab = ["<pad>", "<unk>"] + sorted(tokens)
        return cls(vocab=vocab)

    @classmethod
    def from_tasks(cls, tasks: Sequence[str]) -> "TaskTokenizer":
        tokens = set()
        for t in tasks:
            tokens.update(_task_name_to_tokens(t))
        vocab = ["<pad>", "<unk>"] + sorted(tokens)
        return cls(vocab=vocab)

    @classmethod
    def from_vocab(cls, vocab: Sequence[str]) -> "TaskTokenizer":
        vocab = list(vocab)
        if not vocab or vocab[0] != "<pad>" or (len(vocab) > 1 and vocab[1] != "<unk>"):
            raise ValueError("Expected vocab to start with ['<pad>', '<unk>', ...]")
        return cls(vocab=vocab)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def token_to_id(self, token: str) -> int:
        try:
            return self.vocab.index(token)
        except ValueError:
            return self.unk_id

    def encode(self, task_name: str, max_len: int) -> List[int]:
        toks = _task_name_to_tokens(task_name)
        ids = [self.token_to_id(t) for t in toks[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def batch_encode(self, task_names: Iterable[str], max_len: int) -> List[List[int]]:
        return [self.encode(t, max_len=max_len) for t in task_names]
