from __future__ import annotations

from dataclasses import dataclass


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>"]


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    pad_id: int
    unk_id: int
    mask_id: int

    @classmethod
    def amino_acid(cls) -> "Vocab":
        tokens = SPECIAL_TOKENS + list(AMINO_ACIDS) + ["X", "B", "Z", "U", "O", "-"]
        mapping = {token: idx for idx, token in enumerate(tokens)}
        return cls(mapping, mapping["<pad>"], mapping["<unk>"], mapping["<mask>"])

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: str, *, max_len: int) -> tuple[list[int], list[int]]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in sequence[:max_len]]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(self.pad_id)
            mask.append(0)
        return ids, mask


def make_context(peptide: str, context: str, *, flank: int = 6) -> str:
    """Return a fixed 2*flank context string, using X when context is missing."""
    if not context or set(context) == {"X"}:
        return "X" * (2 * flank)
    if len(context) >= 2 * flank:
        left = context[:flank]
        right = context[-flank:]
        return left + right
    padded = context + ("X" * (2 * flank))
    return padded[: 2 * flank]

