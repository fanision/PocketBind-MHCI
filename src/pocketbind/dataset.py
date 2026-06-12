from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pocketbind.data import normalize_allele, read_pocketbind_table, sample_pocketbind_table
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
        sample_rows: bool = False,
        seed: int = 13,
    ) -> pd.DataFrame:
        if sample_rows and nrows is not None:
            df = sample_pocketbind_table(path, has_context=has_context, nrows=nrows, seed=seed)
        else:
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
        self.cache = self._build_cache()

    def _build_cache(self) -> dict[str, Any]:
        peptide_ids = []
        peptide_mask = []
        hla_ids = []
        hla_mask = []
        context_ids = []
        context_mask = []
        task_ids = []
        labels = []
        peptides = []
        alleles = []

        for row in self.frame.itertuples(index=False):
            pep_ids, pep_mask = self.vocab.encode(row.peptide, max_len=self.max_peptide_len)
            mhc_ids, mhc_mask = self.vocab.encode(row.hla_pseudoseq, max_len=self.max_hla_len)
            context = make_context(row.peptide, row.context, flank=self.max_context_len // 2)
            ctx_ids, ctx_mask = self.vocab.encode(context, max_len=self.max_context_len)
            peptide_ids.append(pep_ids)
            peptide_mask.append(pep_mask)
            hla_ids.append(mhc_ids)
            hla_mask.append(mhc_mask)
            context_ids.append(ctx_ids)
            context_mask.append(ctx_mask)
            task_ids.append(self.task_to_id[row.task])
            labels.append(float(row.label))
            peptides.append(row.peptide)
            alleles.append(row.allele)

        cache = {
            "peptide_ids": peptide_ids,
            "peptide_mask": peptide_mask,
            "hla_ids": hla_ids,
            "hla_mask": hla_mask,
            "context_ids": context_ids,
            "context_mask": context_mask,
            "task_id": task_ids,
            "label": labels,
            "peptide": peptides,
            "allele": alleles,
        }
        try:
            import torch
        except ImportError:
            return cache

        for key in [
            "peptide_ids",
            "peptide_mask",
            "hla_ids",
            "hla_mask",
            "context_ids",
            "context_mask",
            "task_id",
        ]:
            cache[key] = torch.tensor(cache[key], dtype=torch.long)
        cache["label"] = torch.tensor(cache["label"], dtype=torch.float32)
        return cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "peptide_ids": self.cache["peptide_ids"][idx],
            "peptide_mask": self.cache["peptide_mask"][idx],
            "hla_ids": self.cache["hla_ids"][idx],
            "hla_mask": self.cache["hla_mask"][idx],
            "context_ids": self.cache["context_ids"][idx],
            "context_mask": self.cache["context_mask"][idx],
            "task_id": self.cache["task_id"][idx],
            "label": self.cache["label"][idx],
            "allele": self.cache["allele"][idx],
            "peptide": self.cache["peptide"][idx],
        }
