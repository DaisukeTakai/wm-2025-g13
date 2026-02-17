from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokenize_instruction(text: str) -> List[str]:
    """Tokenize free-form English instructions.

    This is intentionally simple and deterministic (no external LMs).
    """
    text = (text or "").strip().lower()
    text = _NON_ALNUM.sub(" ", text)
    toks = [t for t in text.split() if t]
    return toks


@dataclass(frozen=True)
class InstructionTokenizer:
    """Whitespace-ish tokenizer for free-form instruction text."""

    vocab: List[str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        max_vocab: int = 20000,
        min_freq: int = 1,
    ) -> "InstructionTokenizer":
        if max_vocab < 2:
            raise ValueError("max_vocab must be >= 2")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")

        c: Counter[str] = Counter()
        for t in texts:
            c.update(_tokenize_instruction(t))

        # Reserve slots for special tokens.
        vocab = ["<pad>", "<unk>"]
        for tok, freq in c.most_common(max(0, max_vocab - len(vocab))):
            if freq < min_freq:
                continue
            vocab.append(tok)
            if len(vocab) >= max_vocab:
                break
        return cls(vocab=vocab)

    @classmethod
    def from_vocab(cls, vocab: Sequence[str]) -> "InstructionTokenizer":
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
        # Keep it simple; linear scan is fine for small vocab.
        try:
            return self.vocab.index(token)
        except ValueError:
            return self.unk_id

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = _tokenize_instruction(text)
        ids = [self.token_to_id(t) for t in toks[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def batch_encode(self, texts: Iterable[str], max_len: int) -> List[List[int]]:
        return [self.encode(t, max_len=max_len) for t in texts]
