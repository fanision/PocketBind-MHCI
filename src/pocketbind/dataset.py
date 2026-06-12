from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pocketbind.data import normalize_allele, read_pocketbind_table
from pocketbind.encoding import Vocab, make_context


class PocketBindFrameBuilder:
    def __init__(self, pseudoseq_path: Path) -> None:
        self.pseudoseqs = self._read_pseudoseqs(pseudoseq_path)

    @staticmethod
    def _read_pseudoseqs(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                allele, pseudo = line.split()[:2]
                normalized = normalize_allele(allele)
                if normalized:
                    out[normalized] = pseudo
        return out

    def load(
        self,
        path: Path,
        *,
        task: str,
        has_context: bool,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        df = read_pocketbind_table(path, has_context=has_context, nrows=nrows)
        df = df.loc[df["has_valid_allele"]].copy()
        df["task"] = task
        df["hla_pseudoseq"] = df["allele"].map(self.pseudoseqs)
        df = df.loc[df["hla_pseudoseq"].notna()].copy()
        return df.reset_index(drop=True)


class EncodedPocketBindDataset:
    """A torch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        vocab: Vocab | None = None,
        max_peptide_len: int = 15,
        max_hla_len: int = 34,
        max_context_len: int = 12,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab or Vocab.amino_acid()
        self.max_peptide_len = max_peptide_len
        self.max_hla_len = max_hla_len
        self.max_context_len = max_context_len
        self.task_to_id = {"ba": 0, "el": 1, "iedb": 2, "cedar": 3}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        peptide_ids, peptide_mask = self.vocab.encode(row.peptide, max_len=self.max_peptide_len)
        hla_ids, hla_mask = self.vocab.encode(row.hla_pseudoseq, max_len=self.max_hla_len)
        context = make_context(row.peptide, row.context, flank=self.max_context_len // 2)
        context_ids, context_mask = self.vocab.encode(context, max_len=self.max_context_len)
        return {
            "peptide_ids": peptide_ids,
            "peptide_mask": peptide_mask,
            "hla_ids": hla_ids,
            "hla_mask": hla_mask,
            "context_ids": context_ids,
            "context_mask": context_mask,
            "task_id": self.task_to_id[row.task],
            "label": float(row.label),
            "allele": row.allele,
            "peptide": row.peptide,
        }
