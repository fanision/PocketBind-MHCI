from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


HLA_SUFFIX_RE = re.compile(r"(?:^|__)([ABCabc])(\d{2})(\d{2})$")
STANDARD_HLA_RE = re.compile(r"^HLA-([ABCabc])(\d{2}):(\d{2})$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    task: str
    has_context: bool


def normalize_allele(value: str) -> str | None:
    """Return NetMHCpan-style HLA allele when the input is unambiguous."""
    value = value.strip()
    standard = STANDARD_HLA_RE.match(value)
    if standard:
        locus, group, protein = standard.groups()
        return f"HLA-{locus.upper()}{group}:{protein}"

    suffix = HLA_SUFFIX_RE.search(value)
    if suffix:
        locus, group, protein = suffix.groups()
        return f"HLA-{locus.upper()}{group}:{protein}"

    return None


def read_pocketbind_table(path: Path, *, has_context: bool, nrows: int | None = None) -> pd.DataFrame:
    names = ["peptide", "label", "allele_raw", "context"]
    if not has_context:
        names = ["peptide", "label", "allele_raw", "context"]

    df = pd.read_csv(path, sep=r"\s+", names=names, engine="python", nrows=nrows)
    df["source_path"] = str(path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["allele"] = df["allele_raw"].map(normalize_allele)
    df["has_valid_allele"] = df["allele"].notna()
    return df


def sample_pocketbind_table(
    path: Path,
    *,
    has_context: bool,
    nrows: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[list[str]] = []
    seen = 0
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 4:
                continue
            seen += 1
            if len(rows) < nrows:
                rows.append(parts[:4])
                continue
            j = rng.randrange(seen)
            if j < nrows:
                rows[j] = parts[:4]

    names = ["peptide", "label", "allele_raw", "context"]
    df = pd.DataFrame(rows, columns=names)
    df["source_path"] = str(path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["allele"] = df["allele_raw"].map(normalize_allele)
    df["has_valid_allele"] = df["allele"].notna()
    return df


def write_netmhcpan_input(df: pd.DataFrame, path: Path, *, include_context: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["peptide"]
    if include_context:
        cols.append("context")
    cols.extend(["allele", "label"])
    df.loc[:, cols].to_csv(path, sep="\t", header=False, index=False)


def load_predictions_tsv(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    if not lines:
        return pd.DataFrame()
    from io import StringIO

    return pd.read_csv(StringIO("".join(lines)), sep="\t")
