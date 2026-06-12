#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pocketbind.metrics import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged NetMHCpan/PocketBind predictions.")
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["ba", "el", "iedb", "cedar"])
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.merged, sep="\t")
    score_col = args.score_col
    if score_col is None:
        score_col = {
            "ba": "netmhcpan_BA_score",
            "el": "netmhcpan_EL_score",
            "iedb": "netmhcpan_Pathogen_score",
            "cedar": "netmhcpan_Neo_score",
        }[args.task]
    metrics = evaluate(df, label_col="label", score_col=score_col, task=args.task)
    out = pd.DataFrame([metrics])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, sep="\t", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
